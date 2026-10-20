import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import cv2
import numpy as np

from app.core.config import settings, CameraConfig
from app.core.logging import logger
from app.stream.capture import RTSPCapture
from app.detector.engine import detector_engine
from app.detector.tracker import CameraTrackerManager

class StreamManager:
    """
    Orquestrador central de streaming e inferência de IA.
    Coordena as threads de captura RTSP, o pipeline de inferência YOLO e a
    distribuição de frames e eventos para clientes conectados via WebSockets.
    """
    def __init__(self):
        self.cameras_config: List[CameraConfig] = []
        self.captures: Dict[str, RTSPCapture] = {}
        self.trackers: Dict[str, CameraTrackerManager] = {}

        # Armazenamento do último frame processado (JPEG e raw)
        self.latest_annotated_jpeg: Dict[str, bytes] = {}
        self.latest_telemetry: Dict[str, Dict[str, Any]] = {}
        self.inference_fps: Dict[str, float] = {}

        # Clientes WebSocket conectados para receber eventos em tempo real
        self._event_listeners: Set[asyncio.Queue] = set()

        self._running = False
        self._process_task: Optional[asyncio.Task] = None

    def load_cameras_config(self) -> List[CameraConfig]:
        """Carrega lista de câmeras do arquivo JSON ou usa fallback padrão."""
        config_path = Path(settings.config_path)
        cameras: List[CameraConfig] = []

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cameras = [CameraConfig(**item) for item in data]
                logger.info(f"Carregadas {len(cameras)} câmeras de {config_path}")
            except Exception as e:
                logger.error(f"Erro ao carregar {config_path}: {e}")

        if not cameras:
            cameras = [
                CameraConfig(
                    id="cam-0",
                    name="Câmera do Portão",
                    url="rtsp://etbt:dx4ana@192.168.68.53:554/stream1",
                    active=True,
                    confidence_threshold=0.65,
                    priority=True
                ),
                CameraConfig(
                    id="cam-1",
                    name="Câmera da Garagem",
                    url="rtsp://afey:dm24t4@192.168.68.57:554/stream1",
                    active=True,
                    confidence_threshold=0.65,
                    priority=False
                ),
                CameraConfig(
                    id="cam-2",
                    name="Câmera Área do Fundo",
                    url="rtsp://wnsj:7kr3kw@192.168.68.56:554/stream1",
                    active=True,
                    confidence_threshold=0.65,
                    priority=False
                )
            ]

        self.cameras_config = cameras
        return cameras

    async def _on_event_triggered(self, event: Dict[str, Any]) -> None:
        """Notifica todos os listeners de WebSocket sobre novas detecções."""
        dead_queues = set()
        for q in self._event_listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead_queues.add(q)

        self._event_listeners -= dead_queues

    def register_event_listener(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._event_listeners.add(q)
        return q

    def unregister_event_listener(self, q: asyncio.Queue) -> None:
        self._event_listeners.discard(q)

    def start(self) -> None:
        """Inicializa captura e processamento assíncrono."""
        if self._running:
            return

        self._running = True
        self.load_cameras_config()

        # Inicia captura para câmeras ativas
        for cam in self.cameras_config:
            if cam.active:
                capture = RTSPCapture(cam.id, cam.name, cam.url)
                capture.start()
                self.captures[cam.id] = capture

                tracker = CameraTrackerManager(
                    camera_id=cam.id,
                    camera_name=cam.name,
                    event_callback=self._on_event_triggered
                )
                self.trackers[cam.id] = tracker
                self.inference_fps[cam.id] = 0.0

        # Dispara loop de inferência como tarefa assíncrona
        self._process_task = asyncio.create_task(self._inference_loop())
        logger.info("StreamManager e pipeline de inferência iniciados.")

    def stop(self) -> None:
        """Para todas as capturas e encerra o pipeline."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()

        for capture in self.captures.values():
            capture.stop()

        self.captures.clear()
        self.trackers.clear()
        logger.info("StreamManager finalizado.")

    async def _inference_loop(self) -> None:
        """
        Loop assíncrono que lê frames das câmeras, submete à inferência YOLO,
        atualiza rastreadores e codifica frames para entrega via WebSocket.
        """
        # Contadores de FPS de inferência por câmera
        fps_counters: Dict[str, int] = {cam.id: 0 for cam in self.cameras_config}
        fps_timers: Dict[str, float] = {cam.id: time.time() for cam in self.cameras_config}

        while self._running:
            had_frame = False

            for cam in self.cameras_config:
                if not cam.active or cam.id not in self.captures:
                    continue

                capture = self.captures[cam.id]
                raw_frame = capture.get_latest_frame()

                if raw_frame is None:
                    continue

                had_frame = True

                # Executa detecção YOLO em thread separada
                detections = await asyncio.to_thread(
                    detector_engine.detect,
                    raw_frame,
                    cam.confidence_threshold,
                    settings.iou_threshold
                )

                # Processa rastreamento, reconhecimento facial e anotação do frame
                tracker = self.trackers.get(cam.id)
                if tracker:
                    annotated_frame, _ = await tracker.process_tracks(raw_frame, detections)
                else:
                    annotated_frame = detector_engine.draw_annotations(raw_frame, detections)

                # Redimensiona no servidor para resolução leve antes de codificar (alivia CPU/RAM do cliente fraco)
                h_ann, w_ann = annotated_frame.shape[:2]
                if settings.stream_max_width > 0 and w_ann > settings.stream_max_width:
                    scale = settings.stream_max_width / w_ann
                    stream_frame = cv2.resize(
                        annotated_frame,
                        (settings.stream_max_width, int(h_ann * scale)),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    stream_frame = annotated_frame

                # Codifica o frame anotado para JPEG otimizado
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), settings.stream_quality]
                success, buffer = cv2.imencode(".jpg", stream_frame, encode_param)
                if success:
                    jpeg_bytes = buffer.tobytes()
                    self.latest_annotated_jpeg[cam.id] = jpeg_bytes

                # Atualiza métricas de FPS de inferência
                fps_counters[cam.id] += 1
                now = time.time()
                elapsed = now - fps_timers[cam.id]
                if elapsed >= 1.0:
                    self.inference_fps[cam.id] = round(fps_counters[cam.id] / elapsed, 1)
                    fps_counters[cam.id] = 0
                    fps_timers[cam.id] = now

                # Atualiza telemetria da câmera
                self.latest_telemetry[cam.id] = {
                    "camera_id": cam.id,
                    "camera_name": cam.name,
                    "connected": capture.is_connected,
                    "capture_fps": capture.fps,
                    "inference_fps": self.inference_fps.get(cam.id, 0.0),
                    "resolution": capture.resolution,
                    "detections_count": len(detections),
                    "timestamp": now
                }

            # Pequeno intervalo para ceder controle ao event loop caso não haja frames
            if not had_frame:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.002)

    def get_camera_statuses(self) -> List[Dict[str, Any]]:
        """Retorna status consolidado de todas as câmeras."""
        statuses = []
        for cam in self.cameras_config:
            capture = self.captures.get(cam.id)
            is_connected = capture.is_connected if capture else False
            capture_fps = capture.fps if capture else 0.0
            inf_fps = self.inference_fps.get(cam.id, 0.0)
            res = capture.resolution if capture else (0, 0)

            statuses.append({
                "id": cam.id,
                "name": cam.name,
                "url": cam.url,
                "active": cam.active,
                "priority": cam.priority,
                "confidence_threshold": cam.confidence_threshold,
                "connected": is_connected,
                "capture_fps": capture_fps,
                "inference_fps": inf_fps,
                "resolution": f"{res[0]}x{res[1]}" if res[0] > 0 else "N/A"
            })
        return statuses


stream_manager = StreamManager()
