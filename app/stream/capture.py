import os
import time
import queue
import shutil
import subprocess
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

# Força transporte TCP para todos os streams RTSP no OpenCV/FFmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPCapture:
    """
    Worker desacoplado em thread dedicada para captura RTSP sem latência acumulada.
    Possui suporte duplo:
    1. Aceleração Nativa por Hardware NVIDIA NVDEC (via FFmpeg -hwaccel cuda na GPU);
    2. Fallback Transparente para OpenCV (cv2.VideoCapture) caso a GPU não esteja disponível.
    """
    def __init__(self, camera_id: str, camera_name: str, rtsp_url: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._proc: Optional[subprocess.Popen] = None

        # Fila de tamanho 1 para manter sempre e apenas o frame mais recente
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)

        # Telemetria e métricas
        self.is_connected: bool = False
        self.is_nvdec_active: bool = False
        self.last_frame_time: float = 0.0
        self.fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self.resolution: Tuple[int, int] = (0, 0)
        self._frame_bytes: int = 0

        # Timeout de inatividade para disparar reconexão
        self.read_timeout: float = 3.5
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
        self._release_resources()
        logger.info(f"Captura RTSP finalizada para {self.camera_name} ({self.camera_id})")

    # Cache em memória para resolução das câmeras (evita probes lentos repetidos)
    _RESOLUTION_CACHE: dict = {}

    def _probe_resolution(self) -> Tuple[int, int]:
        """Obtém a resolução da câmera com cache imediato para conexões subsequentes."""
        if self.rtsp_url in RTSPCapture._RESOLUTION_CACHE:
            return RTSPCapture._RESOLUTION_CACHE[self.rtsp_url]

        # Resolução padrão das câmeras Tapo stream1 (2K / 3MP)
        if "stream1" in self.rtsp_url:
            RTSPCapture._RESOLUTION_CACHE[self.rtsp_url] = (2304, 1296)
            return (2304, 1296)

        try:
            temp_cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if temp_cap.isOpened():
                w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                temp_cap.release()
                if w > 0 and h > 0:
                    RTSPCapture._RESOLUTION_CACHE[self.rtsp_url] = (w, h)
                    return (w, h)
        except Exception:
            pass

        default_res = (2304, 1296)
        RTSPCapture._RESOLUTION_CACHE[self.rtsp_url] = default_res
        return default_res

    def _open_stream(self) -> bool:
        """Abre a conexão RTSP priorizando NVIDIA NVDEC com fallback para OpenCV."""
        self._release_resources()
        logger.info(f"Conectando a {self.camera_name} via RTSP TCP...")

        ffmpeg_bin = shutil.which("ffmpeg")
        can_use_nvdec = (
            settings.use_nvdec
            and ffmpeg_bin is not None
            and settings.resolve_device().startswith("cuda")
        )

        if can_use_nvdec:
            # Tenta inicializar decodificação por hardware NVDEC na GTX 1660
            try:
                res_w, res_h = self._probe_resolution()
                self.resolution = (res_w, res_h)
                self._frame_bytes = res_w * res_h * 3

                cmd = [
                    ffmpeg_bin,
                    "-nostdin",
                    "-loglevel", "error",
                    "-rtsp_transport", "tcp",
                    "-flags", "low_delay",
                    "-fflags", "nobuffer",
                    "-analyzeduration", "500000",
                    "-probesize", "500000",
                    "-hwaccel", "cuda",
                    "-hwaccel_device", str(settings.nvdec_device),
                    "-i", self.rtsp_url,
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-"
                ]

                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=self._frame_bytes * 2
                )

                # Verifica se o primeiro frame é lido com sucesso no timeout
                raw = self._proc.stdout.read(self._frame_bytes)
                if len(raw) == self._frame_bytes:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((res_h, res_w, 3))
                    self._update_queue(frame)
                    self.is_connected = True
                    self.is_nvdec_active = True
                    self.last_frame_time = time.time()
                    logger.info(
                        f"🚀 [NVDEC Hardware] Conectado a {self.camera_name}: {res_w}x{res_h} "
                        f"(GPU GTX 1660 - 0% CPU)"
                    )
                    return True
                else:
                    logger.warning(f"NVDEC não retornou quadro inicial para {self.camera_name}. Revertendo para OpenCV.")
                    self._release_resources()
            except Exception as e:
                logger.warning(f"Falha ao iniciar NVDEC para {self.camera_name} ({e}). Revertendo para OpenCV.")
                self._release_resources()

        # Fallback para OpenCV padrão
        try:
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._cap.isOpened():
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.resolution = (width, height)
                self.is_connected = True
                self.is_nvdec_active = False
                self.last_frame_time = time.time()
                logger.info(f"✅ [OpenCV Fallback] Conectado a {self.camera_name}: {width}x{height}")
                return True
            else:
                logger.warning(f"Falha ao abrir stream RTSP para {self.camera_name}")
                self.is_connected = False
                return False
        except Exception as e:
            logger.error(f"Exceção ao conectar stream {self.camera_name}: {e}")
            self.is_connected = False
            return False

    def _release_resources(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1)
            except Exception:
                pass
            self._proc = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self.is_connected = False
        self.is_nvdec_active = False

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._fps_timer = now

    def _update_queue(self, frame: np.ndarray) -> None:
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(frame)

    def _capture_loop(self) -> None:
        """Loop contínuo de captura e auto-reconexão com suporte a NVDEC e OpenCV."""
        while self._running:
            if not self.is_connected:
                if not self._open_stream():
                    time.sleep(self.reconnect_delay)
                    continue

            now = time.time()
            grabbed = False

            # Caminho 1: Leitura via FFmpeg NVDEC Hardware
            if self.is_nvdec_active and self._proc is not None and self._proc.stdout is not None:
                try:
                    raw = self._proc.stdout.read(self._frame_bytes)
                    if len(raw) == self._frame_bytes:
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.resolution[1], self.resolution[0], 3))
                        self.last_frame_time = now
                        self._update_fps()
                        self._update_queue(frame)
                        grabbed = True
                    else:
                        grabbed = False
                except Exception:
                    grabbed = False

            # Caminho 2: Leitura via OpenCV Fallback
            elif self._cap is not None and self._cap.isOpened():
                if self._cap.grab():
                    ret, frame = self._cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        self.last_frame_time = now
                        self._update_fps()
                        self._update_queue(frame)
                        grabbed = True

            # Verificação de timeout de leitura
            if not grabbed or (now - self.last_frame_time) > self.read_timeout:
                backend = "NVDEC" if self.is_nvdec_active else "OpenCV"
                logger.warning(
                    f"⚠️ Timeout de leitura (> {self.read_timeout}s) em {self.camera_name} [{backend}]. "
                    "Reconectando..."
                )
                self.is_connected = False
                self._release_resources()
                time.sleep(self.reconnect_delay)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Retorna o frame mais recente capturado ou None se indisponível."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
