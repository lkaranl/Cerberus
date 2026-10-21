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
        self.latest_detections: Dict[str, Dict[str, Any]] = {}
        self.latest_telemetry: Dict[str, Dict[str, Any]] = {}
        self.inference_fps: Dict[str, float] = {}
        self.stream_fps: Dict[str, float] = {}

        # Armazenamento e listeners de metadados para overlay WebRTC
        self.latest_metadata: Dict[str, Dict[str, Any]] = {}
        self._meta_listeners: Dict[str, Set[asyncio.Queue]] = {}

        # Clientes WebSocket conectados para receber eventos em tempo real
        self._event_listeners: Set[asyncio.Queue] = set()
        self._running = False
        self._camera_tasks: Dict[str, asyncio.Task] = {}

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
        """Notifica todos os listeners de WebSocket e despacha alerta ao Telegram."""
        dead_queues = set()
        for q in self._event_listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead_queues.add(q)

        self._event_listeners -= dead_queues

        # Notificação assíncrona desacoplada para os bots do Telegram
        try:
            from app.bot import telegram_bot, telegram_critical_bot
            if telegram_bot.is_active:
                asyncio.create_task(telegram_bot.send_alert(event))
            if telegram_critical_bot.is_active:
                asyncio.create_task(telegram_critical_bot.send_alert(event))
        except Exception as err:
            logger.debug(f"Erro ao encaminhar evento para Telegram: {err}")

    def register_event_listener(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._event_listeners.add(q)
        return q

    def unregister_event_listener(self, q: asyncio.Queue) -> None:
        self._event_listeners.discard(q)

    def register_meta_listener(self, camera_id: str) -> asyncio.Queue:
        if camera_id not in self._meta_listeners:
            self._meta_listeners[camera_id] = set()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._meta_listeners[camera_id].add(q)
        return q

    def unregister_meta_listener(self, camera_id: str, q: asyncio.Queue) -> None:
        if camera_id in self._meta_listeners:
            self._meta_listeners[camera_id].discard(q)

    def _broadcast_metadata(self, camera_id: str, payload: Dict[str, Any]) -> None:
        listeners = self._meta_listeners.get(camera_id, set())
        dead = set()
        for q in listeners:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    pass
            except Exception:
                dead.add(q)
        listeners -= dead

    def start(self) -> None:
        """Inicializa captura e processamento assíncrono paralelo por câmera."""
        if self._running:
            return

        self._running = True
        self.load_cameras_config()

        # Inicia captura e loops dedicados de stream e inferência para cada câmera ativa
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
                self.stream_fps[cam.id] = 0.0

                # Task 1: Streaming Web contínuo de alta taxa (20-25 FPS) com anotações visuais
                stream_task = asyncio.create_task(
                    self._camera_stream_worker(cam),
                    name=f"stream-{cam.id}"
                )
                self._camera_tasks[f"stream-{cam.id}"] = stream_task

                # Task 2: Inferência pesada de IA desacoplada (YOLO + ArcFace + OSNet + Alertas)
                infer_task = asyncio.create_task(
                    self._camera_inference_worker(cam),
                    name=f"infer-{cam.id}"
                )
                self._camera_tasks[f"infer-{cam.id}"] = infer_task

        logger.info(f"StreamManager iniciado com {len(self._camera_tasks)} workers assíncronos desacoplados.")

    def stop(self) -> None:
        """Para todas as capturas e encerra o pipeline."""
        self._running = False
        for task in self._camera_tasks.values():
            task.cancel()
        self._camera_tasks.clear()

        for capture in self.captures.values():
            capture.stop()

        self.captures.clear()
        self.trackers.clear()
        self.latest_detections.clear()
        logger.info("StreamManager finalizado.")

    async def _camera_stream_worker(self, cam: CameraConfig) -> None:
        """
        Worker de streaming contínuo de alta taxa (20-25 FPS).
        Gera os frames JPEG com as anotações visuais atualizadas sem esperar pela IA,
        garantindo fluidez máxima no navegador e nas conexões WebSocket.
        """
        capture = self.captures.get(cam.id)
        if not capture:
            return

        fps_counter = 0
        fps_timer = time.time()
        last_frame_id = -1

        while self._running:
            try:
                raw_frame, frame_id = capture.get_latest_frame_and_id()
                if raw_frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Evita recodificar o mesmo frame se a câmera física ainda não enviou um novo
                if frame_id == last_frame_id:
                    await asyncio.sleep(0.008)
                    continue

                last_frame_id = frame_id

                # Aplica as últimas anotações visuais geradas pelo tracker (válidas por até 1.2s)
                det_info = self.latest_detections.get(cam.id)
                now = time.time()
                if det_info and (now - det_info.get("timestamp", 0.0) <= 1.2) and det_info.get("detections"):
                    annotated_frame = detector_engine.draw_annotations(raw_frame.copy(), det_info["detections"])
                else:
                    annotated_frame = raw_frame

                # Redimensiona no servidor com interpolação linear rápida
                h_ann, w_ann = annotated_frame.shape[:2]
                if settings.stream_max_width > 0 and w_ann > settings.stream_max_width:
                    scale = settings.stream_max_width / w_ann
                    stream_frame = cv2.resize(
                        annotated_frame,
                        (settings.stream_max_width, int(h_ann * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    stream_frame = annotated_frame

                # Codifica para JPEG otimizado
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), settings.stream_quality]
                success, buffer = cv2.imencode(".jpg", stream_frame, encode_param)
                if success:
                    self.latest_annotated_jpeg[cam.id] = buffer.tobytes()

                # Contabilidade de FPS do stream web
                fps_counter += 1
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    self.stream_fps[cam.id] = round(fps_counter / elapsed, 1)
                    fps_counter = 0
                    fps_timer = now

                # Intervalo suave para cravar ~25 FPS
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no stream worker da câmera {cam.name}: {e}")
                await asyncio.sleep(0.05)

    async def _camera_inference_worker(self, cam: CameraConfig) -> None:
        """
        Worker de inferência pesada de IA desacoplada.
        Processa detecção YOLO, reconhecimento facial InsightFace, Re-ID OSNet,
        leitura de placas e alertas sem impactar o FPS de vídeo da interface web.
        """
        capture = self.captures.get(cam.id)
        tracker = self.trackers.get(cam.id)
        if not capture:
            return

        fps_counter = 0
        fps_timer = time.time()
        last_inferred_id = -1

        while self._running:
            try:
                raw_frame, frame_id = capture.get_latest_frame_and_id()
                if raw_frame is None or frame_id == last_inferred_id:
                    await asyncio.sleep(0.01)
                    continue

                last_inferred_id = frame_id

                # Executa detecção YOLO/TensorRT em thread pool separada
                detections = await asyncio.to_thread(
                    detector_engine.detect,
                    raw_frame,
                    cam.confidence_threshold,
                    settings.iou_threshold
                )

                # Processa rastreamento, consenso biométrico e disparo de alertas
                if tracker:
                    await tracker.process_tracks(raw_frame, detections)

                # Compartilha as detecções enriquecidas (com nome, status, cor) para o stream_worker
                now = time.time()
                self.latest_detections[cam.id] = {
                    "detections": detections,
                    "timestamp": now
                }

                # Monta e publica metadados ultraleves para o overlay WebRTC
                h_frame, w_frame = raw_frame.shape[:2]
                clean_dets = []
                for det in detections:
                    clean_dets.append({
                        "tracking_id": det.get("tracking_id"),
                        "bbox": [int(v) for v in det["bbox"]] if "bbox" in det else None,
                        "entity_type": det.get("entity_type", "person"),
                        "class_label": det.get("class_label", "Objeto"),
                        "name": det.get("person_name", "Desconhecido"),
                        "is_known": det.get("is_known", False),
                        "confidence": round(float(det.get("confidence", 0.0)), 2),
                        "license_plate": det.get("license_plate", ""),
                        "is_stationary": det.get("is_stationary", False)
                    })

                meta_payload = {
                    "camera_id": cam.id,
                    "timestamp": now,
                    "width": w_frame,
                    "height": h_frame,
                    "detections": clean_dets
                }
                self.latest_metadata[cam.id] = meta_payload
                self._broadcast_metadata(cam.id, meta_payload)

                # Métricas de FPS de inferência da IA
                fps_counter += 1
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    self.inference_fps[cam.id] = round(fps_counter / elapsed, 1)
                    fps_counter = 0
                    fps_timer = now

                # Telemetria consolidada em tempo real
                self.latest_telemetry[cam.id] = {
                    "camera_id": cam.id,
                    "camera_name": cam.name,
                    "connected": capture.is_connected,
                    "capture_fps": capture.fps,
                    "stream_fps": self.stream_fps.get(cam.id, 0.0),
                    "inference_fps": self.inference_fps.get(cam.id, 0.0),
                    "resolution": capture.resolution,
                    "detections_count": len(detections),
                    "nvdec_active": getattr(capture, "is_nvdec_active", False),
                    "timestamp": now
                }

                # Micro-pausa para ceder controle ao event loop
                await asyncio.sleep(0.005)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no pipeline de IA da câmera {cam.name}: {e}")
                await asyncio.sleep(0.05)

    def get_camera_statuses(self) -> List[Dict[str, Any]]:
        """Retorna status consolidado de todas as câmeras."""
        statuses = []
        for cam in self.cameras_config:
            capture = self.captures.get(cam.id)
            is_connected = capture.is_connected if capture else False
            capture_fps = capture.fps if capture else 0.0
            stream_fps = self.stream_fps.get(cam.id, 0.0)
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
                "stream_fps": stream_fps,
                "inference_fps": inf_fps,
                "resolution": f"{res[0]}x{res[1]}" if res[0] > 0 else "N/A",
                "nvdec_active": getattr(capture, "is_nvdec_active", False) if capture else False
            })
        return statuses


stream_manager = StreamManager()
