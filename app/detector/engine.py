import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

class YOLODetectorEngine:
    """
    Engine de inferência e rastreamento de IA utilizando Ultralytics YOLO (v8n).
    Configurado para aceleração em GPU NVIDIA (GTX 1660) com precisão mista FP16,
    com suporte transparente para Apple Silicon (MPS) e CPU no ambiente de desenvolvimento.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.model_path
        self.device = settings.resolve_device()
        self.is_cuda = self.device.startswith("cuda")
        self.use_half = self.is_cuda  # FP16 apenas para CUDA (GTX 1660)
        self.model = None

    def initialize(self) -> None:
        """Carrega o modelo YOLO na memória e envia para o dispositivo correto."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.error("Ultralytics não está instalado. Instale os requisitos para inferência.")
            raise e

        # Garante diretório pai do modelo
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"🚀 Inicializando YOLO [{self.model_path}] no dispositivo: {self.device} "
            f"(FP16 half={self.use_half})"
        )

        self.model = YOLO(self.model_path)
        logger.info("Modelo YOLO carregado com sucesso.")

    def track(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executa inferência e rastreamento no frame fornecido.
        Filtra estritamente a classe 0 (pessoa).
        Retorna uma cópia do frame com bounding boxes estilizadas e a lista de detecções.
        """
        if self.model is None:
            self.initialize()

        conf = confidence_threshold or settings.confidence_threshold
        iou = iou_threshold or settings.iou_threshold

        try:
            # Inferência com ByteTrack integrado da Ultralytics
            results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],  # Apenas 'person' do dataset COCO
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Erro durante inferência YOLO: {e}")
            return frame.copy(), []

        annotated_frame = frame.copy()
        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return annotated_frame, detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return annotated_frame, detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id != 0:
                continue

            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Tracking ID (pode ser None no primeiro frame antes de associação)
            tracking_id = None
            if box.id is not None:
                tracking_id = int(box.id[0].item())

            detections.append({
                "tracking_id": tracking_id,
                "confidence": conf_val,
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            })

            # Desenho da Bounding Box (Verde Ciano / Neon, 2px)
            box_color = (120, 240, 0) if tracking_id is not None else (0, 200, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

            # Rótulo superior: Pessoa #ID (89%)
            label_id = f"#{tracking_id}" if tracking_id is not None else "?"
            label_text = f"Pessoa {label_id} ({int(conf_val * 100)}%)"

            # Fundo sólido escuro para o rótulo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            label_y1 = max(0, y1 - text_h - 8)
            label_y2 = y1
            label_x1 = x1
            label_x2 = x1 + text_w + 10

            cv2.rectangle(
                annotated_frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                (20, 24, 30),
                -1
            )
            # Borda do rótulo
            cv2.rectangle(
                annotated_frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                box_color,
                1
            )

            # Texto do rótulo
            cv2.putText(
                annotated_frame,
                label_text,
                (label_x1 + 5, label_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

        return annotated_frame, detections


detector_engine = YOLODetectorEngine()
