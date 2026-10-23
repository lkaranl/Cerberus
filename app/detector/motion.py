from typing import Tuple
import cv2
import numpy as np
from app.core.logging import logger

class MotionDetector:
    """
    Detector de movimento ultraleve baseado no algoritmo MOG2 do OpenCV.
    Projetado para atuar como filtro pré-IA (Motion Gating), descartando
    frames estáticos antes do processamento pesado no YOLOv8/TensorRT.
    """
    def __init__(
        self,
        min_area_ratio: float = 0.005,
        history: int = 500,
        var_threshold: int = 25,
        target_width: int = 320
    ):
        self.min_area_ratio = min_area_ratio
        self.target_width = target_width
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._warmup_frames = 5
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Analisa o frame e retorna se houve variação significativa de pixels.
        Retorna (has_motion: bool, motion_ratio: float).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False, 0.0

        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return False, 0.0

        # Redução proporcional ultraleve para target_width (ex: 320x180)
        scale = self.target_width / float(w)
        target_h = max(1, int(h * scale))
        small_frame = cv2.resize(frame, (self.target_width, target_h), interpolation=cv2.INTER_NEAREST)

        # Converte para tons de cinza
        if len(small_frame.shape) == 3 and small_frame.shape[2] >= 3:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = small_frame

        # Suavização Gaussiana rápida para atenuar ruídos de alta frequência
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Subtração adaptativa de fundo
        fg_mask = self.subtractor.apply(blurred)

        # Dilatação para conectar pixels contíguos de movimento
        dilated = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # Proporção de pixels ativos
        total_pixels = float(target_h * self.target_width)
        active_pixels = cv2.countNonZero(dilated)
        motion_ratio = active_pixels / total_pixels if total_pixels > 0 else 0.0

        self._frame_count += 1
        # Nos primeiros frames (aquecimento do fundo), considera presença para evitar perder início de eventos
        if self._frame_count <= self._warmup_frames:
            return True, motion_ratio

        has_motion = motion_ratio >= self.min_area_ratio
        return has_motion, motion_ratio

    def reset(self) -> None:
        """Reinicia o contador de frames do modelo de fundo."""
        self._frame_count = 0
