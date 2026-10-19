import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.detector.recognizer import face_recognizer
from app.detector.visual_matcher import visual_matcher

# Mapeamento de Classes COCO suportadas
CLASS_MAP = {
    0: ("person", "Pessoa"),
    1: ("vehicle", "Bicicleta"),
    2: ("vehicle", "Carro"),
    3: ("vehicle", "Moto"),
    5: ("vehicle", "Ônibus"),
    7: ("vehicle", "Caminhão"),
    15: ("pet", "Gato"),
    16: ("pet", "Cachorro")
}
TARGET_CLASS_IDS = list(CLASS_MAP.keys())


class YOLODetectorEngine:
    """
    Engine de inferência e rastreamento de IA utilizando Ultralytics YOLO (v8n).
    Configurado para detectar simultaneamente Pessoas, Veículos e Animais (Pets)
    com anotações visuais diferenciadas por categoria e status de identificação.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.model_path
        self.device = settings.resolve_device()
        self.is_cuda = self.device.startswith("cuda")
        self.model = None

    def initialize(self) -> None:
        """Carrega o modelo YOLO e inicializa os reconhecedores faciais e visuais."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.error("Ultralytics não está instalado. Instale os requisitos para inferência.")
            raise e

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"🚀 Inicializando YOLO [{self.model_path}] no dispositivo: {self.device}...")
        self.model = YOLO(self.model_path)
        logger.info("Modelo YOLO carregado com sucesso.")

        # Inicializa o FaceNet para reconhecimento facial em GPU
        face_recognizer.initialize()

        # Inicializa o extrator de características para Veículos e Pets
        visual_matcher.initialize()

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Executa rastreamento YOLO e extrai bounding boxes e IDs para pessoas, veículos e pets."""
        if self.model is None:
            self.initialize()

        conf = confidence_threshold or settings.confidence_threshold
        iou = iou_threshold or settings.iou_threshold

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=TARGET_CLASS_IDS,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Erro durante inferência YOLO: {e}")
            return []

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in CLASS_MAP:
                continue

            entity_type, class_label = CLASS_MAP[cls_id]
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            tracking_id = None
            if box.id is not None:
                tracking_id = int(box.id[0].item())

            detections.append({
                "tracking_id": tracking_id,
                "confidence": conf_val,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "entity_type": entity_type,
                "class_label": class_label,
                "person_name": f"{class_label} Desconhecido",
                "is_known": False,
                "face_confidence": 0.0,
                "license_plate": ""
            })

        return detections

    def draw_annotations(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Desenha bounding boxes com esquemas de cores específicos por tipo de entidade."""
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf_val = det["confidence"]
            tracking_id = det["tracking_id"]
            entity_type = det.get("entity_type", "person")
            class_label = det.get("class_label", "Objeto")
            name = det.get("person_name", "Desconhecido")
            is_known = det.get("is_known", False)
            plate = det.get("license_plate", "")

            # Esquema de Cores (BGR):
            # Pessoa Conhecida -> Verde Neon (118, 230, 0)
            # Veículo Conhecido -> Azul Ciano (255, 240, 0)
            # Pet Conhecido -> Magenta/Roxo (255, 0, 180)
            # Qualquer Desconhecido -> Âmbar/Laranja (0, 180, 255)
            if is_known:
                if entity_type == "person":
                    box_color = (118, 230, 0)
                elif entity_type == "vehicle":
                    box_color = (255, 240, 0)
                elif entity_type == "pet":
                    box_color = (255, 0, 180)
                else:
                    box_color = (118, 230, 0)
                label_text = f"{name} ({int(conf_val * 100)}%)"
            else:
                box_color = (0, 180, 255)
                id_tag = f"#{tracking_id}" if tracking_id is not None else "?"
                plate_tag = f" [{plate}]" if plate else ""
                label_text = f"{class_label} {id_tag}{plate_tag} ({int(conf_val * 100)}%)"

            # Desenha a caixa
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            label_y1 = max(0, y1 - text_h - 8)
            label_y2 = y1
            label_x1 = x1
            label_x2 = x1 + text_w + 10

            # Fundo escuro do rótulo
            cv2.rectangle(
                annotated_frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                (15, 20, 28),
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
            # Texto
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

        return annotated_frame


detector_engine = YOLODetectorEngine()
