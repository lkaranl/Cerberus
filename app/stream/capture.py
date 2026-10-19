import os
import time
import queue
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

from app.core.logging import logger

# Força transporte TCP para todos os streams RTSP no OpenCV/FFmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPCapture:
    """
    Worker desacoplado em thread dedicada para captura RTSP sem latência acumulada.
    Utiliza buffer de 1 frame e reconexão automática com transporte TCP.
    """
    def __init__(self, camera_id: str, camera_name: str, rtsp_url: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Fila de tamanho 1 para manter sempre e apenas o frame mais recente
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)

        # Telemetria e métricas
        self.is_connected: bool = False
        self.last_frame_time: float = 0.0
        self.fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self.resolution: Tuple[int, int] = (0, 0)

        # Timeout de inatividade para disparar reconexão
        self.read_timeout: float = 3.0
        self.reconnect_delay: float = 2.0

    def start(self) -> None:
        """Inicia a thread de captura em segundo plano."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"RTSP-{self.camera_id}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"Thread de captura RTSP iniciada para {self.camera_name} ({self.camera_id})")

    def stop(self) -> None:
        """Encerra a captura e libera recursos."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_cap()
        logger.info(f"Captura RTSP finalizada para {self.camera_name} ({self.camera_id})")

    def _open_stream(self) -> bool:
        """Abre a conexão RTSP com timeout otimizado."""
        self._release_cap()
        logger.info(f"Conectando a {self.camera_name} via RTSP TCP...")

        try:
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            # Reduz o buffer interno do OpenCV para 1 frame
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._cap.isOpened():
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.resolution = (width, height)
                self.is_connected = True
                self.last_frame_time = time.time()
                logger.info(f"✅ Conectado a {self.camera_name}: {width}x{height}")
                return True
            else:
                logger.warning(f"Falha ao abrir stream RTSP para {self.camera_name}")
                self.is_connected = False
                return False
        except Exception as e:
            logger.error(f"Exceção ao conectar stream {self.camera_name}: {e}")
            self.is_connected = False
            return False

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.is_connected = False

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._fps_timer = now

    def _capture_loop(self) -> None:
        """Loop contínuo de captura e auto-reconexão."""
        while self._running:
            if not self.is_connected or self._cap is None or not self._cap.isOpened():
                if not self._open_stream():
                    time.sleep(self.reconnect_delay)
                    continue

            # cap.grab() descarta frames pendentes no buffer do driver sem decodificar tudo
            grabbed = self._cap.grab()
            now = time.time()

            if grabbed:
                ret, frame = self._cap.retrieve()
                if ret and frame is not None and frame.size > 0:
                    self.last_frame_time = now
                    self._update_fps()

                    # Atualiza a fila mantendo sempre apenas o frame mais novo
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            _ = self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._frame_queue.put_nowait(frame)
                else:
                    grabbed = False

            # Verificação de timeout de leitura
            if not grabbed or (now - self.last_frame_time) > self.read_timeout:
                logger.warning(
                    f"⚠️ Timeout de leitura (> {self.read_timeout}s) em {self.camera_name}. "
                    "Reconectando..."
                )
                self.is_connected = False
                self._release_cap()
                time.sleep(self.reconnect_delay)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Retorna o frame mais recente capturado ou None se indisponível."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
