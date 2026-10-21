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
        self.engine_path = settings.engine_path
        self.device = settings.resolve_device()
        self.is_cuda = self.device.startswith("cuda")
        self.model = None
        self.using_tensorrt = False

    def initialize(self) -> None:
        """Carrega o modelo YOLO (TensorRT ou PyTorch) e inicializa reconhecedores."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.error("Ultralytics não está instalado. Instale os requisitos para inferência.")
            raise e

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        # 1. Tenta carregar a engine compilada do TensorRT (FP16 nativo)
        target_engine = Path(self.engine_path)
        # Se yolo11n.engine ainda não foi gerado, aceita yolov8n.engine como transição temporária
        if not target_engine.exists() and Path("models/yolov8n.engine").exists():
            target_engine = Path("models/yolov8n.engine")

        if self.is_cuda and settings.use_tensorrt and target_engine.exists():
            try:
                logger.info(f"⚡ Inicializando YOLO com aceleração máxima TensorRT FP16 [{target_engine}]...")
                self.model = YOLO(str(target_engine), task="detect")
                self.using_tensorrt = True
                logger.info(f"🚀 Engine TensorRT ({target_engine.name}) carregada com sucesso na GPU NVIDIA GTX 1660.")
            except Exception as trt_err:
                logger.warning(f"Falha ao carregar TensorRT ({trt_err}). Revertendo para PyTorch CUDA.")
                self.using_tensorrt = False

        # 2. Fallback para modelo PyTorch (.pt) padrão
        if self.model is None:
            pt_target = self.model_path if Path(self.model_path).exists() else "yolo11n.pt"
            logger.info(f"🚀 Inicializando YOLO PyTorch [{pt_target}] no dispositivo: {self.device}...")
            self.model = YOLO(pt_target)
            self.using_tensorrt = False
            logger.info("Modelo YOLO PyTorch carregado com sucesso.")

            if self.is_cuda and settings.use_tensorrt and not Path(self.engine_path).exists():
                logger.info(
                    "💡 Dica de Desempenho: Para ativar aceleração máxima TensorRT no YOLO11 (~3ms na GTX 1660), "
                    "execute no servidor: python3 scripts/export_yolo11_tensorrt.py"
                )

        # Inicializa o FaceNet para reconhecimento facial em GPU
        face_recognizer.initialize()

        # Inicializa o extrator de características para Veículos e Pets
        visual_matcher.initialize()

        # Inicializa o extrator biométrico corporal OSNet (Person Re-ID SOTA)
        from app.detector.person_reid import person_reid
        person_reid.initialize()

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
            track_kwargs = {
                "source": frame,
                "persist": True,
                "tracker": "bytetrack.yaml",
                "classes": TARGET_CLASS_IDS,
                "conf": conf,
                "iou": iou,
                "verbose": False
            }
            if not self.using_tensorrt:
                track_kwargs["device"] = self.device

            results = self.model.track(**track_kwargs)
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

            # --- Filtros Físicos e Biomecânicos Anti-Falsos Positivos (Anti-Gatos / Pássaros / Sombras) ---
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            box_area = box_w * box_h

            # 1. Filtros de Confiança Mínima Diferenciada por Classe:
            # Pessoas exigem maior certeza (>= 0.50) para evitar alucinação com animais ou reflexos
            if entity_type == "person" and conf_val < 0.50:
                continue
            elif entity_type == "vehicle" and conf_val < 0.42:
                continue
            elif entity_type == "pet" and conf_val < 0.35:
                continue

            # 2. Filtro Biomecânico Anatômico Humano (Aspect Ratio):
            if entity_type == "person":
                # Em câmeras 2K (2304x1296), um ser humano adulto em pé tem altura considerável.
                # Gatos, cães pequenos e animais no chão raramente passam de 85px de altura
                if box_h < 95 or box_area < 2200:
                    continue

                # Postura Humana: Altura SEMPRE maior que largura (aspecto estritamente vertical).
                # Animais quadrúpedes andando têm o corpo horizontal (largura > altura).
                # Se largura for maior ou quase igual à altura, é um animal ou falso positivo no chão!
                if box_w >= (box_h * 0.95):
                    # Se for largo e baixo, NUNCA é uma pessoa em pé; reclassifica para Pet ou descarta
                    if box_w >= 50 and box_h >= 35:
                        entity_type = "pet"
                        class_label = "Animal"
                    else:
                        continue

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

        h_img, w_img = annotated_frame.shape[:2]
        # Escala adaptativa para resoluções altas (1080p, 2K e 4K)
        # Em 2304x1296, scale_factor é ~1.8, gerando texto e bordas 100% nítidos e legíveis
        scale_factor = max(1.0, w_img / 1280.0)
        box_thickness = max(2, int(round(2.5 * scale_factor)))
        font_scale = 0.58 * scale_factor
        font_thickness = max(1, int(round(1.8 * scale_factor)))
        pad_x = int(8 * scale_factor)
        pad_y = int(6 * scale_factor)
        border_thick = max(1, int(round(1.2 * scale_factor)))

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf_val = det["confidence"]
            tracking_id = det["tracking_id"]
            entity_type = det.get("entity_type", "person")
            class_label = det.get("class_label", "Objeto")
            name = det.get("person_name", "Desconhecido")
            is_known = det.get("is_known", False)
            plate = det.get("license_plate", "")
            is_stationary = det.get("is_stationary", False)

            status_tag = ""
            if is_stationary:
                if entity_type == "vehicle":
                    status_tag = " [Estacionado]"
                elif entity_type == "pet":
                    status_tag = " [Parado]"

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
                label_text = f"{name}{status_tag} ({int(conf_val * 100)}%)"
            else:
                box_color = (0, 180, 255)
                id_tag = f"#{tracking_id}" if tracking_id is not None else "?"
                plate_tag = f" [{plate}]" if plate else ""
                label_text = f"{class_label} {id_tag}{plate_tag}{status_tag} ({int(conf_val * 100)}%)"

            # Desenha a caixa com espessura proporcional à resolução
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, box_thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            # Posicionamento inteligente para evitar corte no topo da imagem
            if (y1 - text_h - (pad_y * 2)) < 0:
                label_y1 = y1
                label_y2 = y1 + text_h + (pad_y * 2)
                text_y = label_y2 - pad_y
            else:
                label_y1 = y1 - text_h - (pad_y * 2)
                label_y2 = y1
                text_y = label_y2 - pad_y

            label_x1 = max(0, x1)
            label_x2 = min(w_img, label_x1 + text_w + (pad_x * 2))

            # Fundo escuro contrastante (Deep Slate #0F141C)
            cv2.rectangle(
                annotated_frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                (15, 20, 28),
                -1
            )
            # Borda do rótulo na cor da categoria
            cv2.rectangle(
                annotated_frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                box_color,
                border_thick
            )
            # Texto nítido com anti-aliasing
            cv2.putText(
                annotated_frame,
                label_text,
                (label_x1 + pad_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

        return annotated_frame


detector_engine = YOLODetectorEngine()
