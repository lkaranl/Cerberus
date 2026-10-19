import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

class VisualEntityMatcher:
    """
    Extrator de características visuais e comparador baseado em PyTorch/Torchvision (MobileNetV3).
    Permite identificar veículos e animais de estimação (pets) por fotos de referência
    via comparação de embeddings de similaridade de cosseno.
    """
    def __init__(
        self,
        vehicles_dir: str = "data/known_vehicles",
        pets_dir: str = "data/known_pets",
        threshold: float = 0.68
    ):
        self.vehicles_dir = Path(vehicles_dir)
        self.pets_dir = Path(pets_dir)
        self.threshold = threshold
        self.device = settings.resolve_device()

        # Dicionários de embeddings médios
        self.known_vehicles: Dict[str, Dict[str, Any]] = {} # nome -> {embedding, plate}
        self.known_pets: Dict[str, Any] = {} # nome -> embedding

        self.model = None
        self.transform = None
        self.torch = None
        self.is_ready = False

    def initialize(self) -> bool:
        """Inicializa o extrator visual leve da Torchvision."""
        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

            self.torch = torch

            logger.info(f"Carregando MobileNetV3 para reconhecimento visual em: {self.device}...")
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base_model = mobilenet_v3_small(weights=weights)

            # Usa a camada extratora de características global
            base_model.classifier = self.torch.nn.Identity()
            base_model.eval().to(self.device)
            self.model = base_model

            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.is_ready = True
            self.load_all()
            return True
        except Exception as e:
            logger.warning(f"Extrator visual de veículos/pets operando em modo fallback: {e}")
            self.is_ready = False
            return False

    def _extract_embedding(self, img_bgr: np.ndarray) -> Optional[Any]:
        if not self.is_ready or img_bgr is None or img_bgr.size == 0:
            return None

        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

            with self.torch.no_grad():
                feats = self.model(tensor)
                norm_feats = self.torch.nn.functional.normalize(feats, p=2, dim=1)
                return norm_feats.squeeze(0)
        except Exception as err:
            logger.debug(f"Erro ao extrair vetor visual: {err}")
            return None

    def load_vehicles(self) -> int:
        """Carrega e calcula embeddings dos veículos cadastrados."""
        if not self.is_ready or not self.vehicles_dir.exists():
            return 0

        self.known_vehicles.clear()
        count = 0

        for folder in self.vehicles_dir.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue

            folder_name = folder.name
            # Procura por placa entre colchetes no nome da pasta, ex: Meu_Carro_[BRA2E19]
            plate_match = re.search(r"\[([A-Z0-9\-]+)\]", folder_name, re.IGNORECASE)
            registered_plate = plate_match.group(1).upper().replace("-", "") if plate_match else ""

            # Nome limpo para exibição
            display_name = re.sub(r"\[.*?\]", "", folder_name).replace("_", " ").strip()

            embs = []
            for img_file in folder.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                img = cv2.imread(str(img_file))
                emb = self._extract_embedding(img)
                if emb is not None:
                    embs.append(emb)

            if embs:
                mean_emb = self.torch.stack(embs).mean(dim=0)
                mean_norm = self.torch.nn.functional.normalize(mean_emb, p=2, dim=0)
                self.known_vehicles[display_name] = {
                    "embedding": mean_norm,
                    "plate": registered_plate
                }
                count += 1
                plate_info = f" (Placa: {registered_plate})" if registered_plate else ""
                logger.info(f"🚗 Veículo cadastrado: {display_name}{plate_info} com {len(embs)} fotos")

        return count

    def load_pets(self) -> int:
        """Carrega e calcula embeddings dos pets (gatos e cães)."""
        if not self.is_ready or not self.pets_dir.exists():
            return 0

        self.known_pets.clear()
        count = 0

        for folder in self.pets_dir.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue

            display_name = folder.name.replace("_", " ").strip()
            embs = []

            for img_file in folder.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                img = cv2.imread(str(img_file))
                emb = self._extract_embedding(img)
                if emb is not None:
                    embs.append(emb)

            if embs:
                mean_emb = self.torch.stack(embs).mean(dim=0)
                mean_norm = self.torch.nn.functional.normalize(mean_emb, p=2, dim=0)
                self.known_pets[display_name] = mean_norm
                count += 1
                logger.info(f"🐾 Pet cadastrado: {display_name} com {len(embs)} fotos")

        return count

    def load_all(self) -> None:
        self.load_vehicles()
        self.load_pets()

    def identify_vehicle(self, crop_bgr: np.ndarray, detected_plate: str = "") -> Tuple[str, float, bool, str]:
        """
        Identifica um veículo.
        Prioridade 1: Match exato de placa.
        Prioridade 2: Match de similaridade visual por foto.
        Retorna: (nome_do_veiculo, confianca, is_conhecido, placa_final)
        """
        norm_detected_plate = detected_plate.upper().replace("-", "").strip()

        # 1. Checagem por placa cadastrada
        if norm_detected_plate:
            for name, data in self.known_vehicles.items():
                if data["plate"] and data["plate"] == norm_detected_plate:
                    return f"{name} [{norm_detected_plate}]", 0.99, True, norm_detected_plate

        # 2. Checagem por embedding visual
        probe_emb = self._extract_embedding(crop_bgr)
        if probe_emb is None or not self.known_vehicles:
            plate_tag = f" [{norm_detected_plate}]" if norm_detected_plate else ""
            return f"Veículo Desconhecido{plate_tag}", 0.0, False, norm_detected_plate

        best_name = "Veículo Desconhecido"
        best_sim = 0.0
        matched_plate = norm_detected_plate

        for name, data in self.known_vehicles.items():
            sim = self.torch.dot(probe_emb, data["embedding"]).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name
                if data["plate"]:
                    matched_plate = data["plate"]

        if best_sim >= self.threshold:
            plate_tag = f" [{matched_plate}]" if matched_plate else ""
            return f"{best_name}{plate_tag}", round(best_sim, 3), True, matched_plate
        else:
            plate_tag = f" [{norm_detected_plate}]" if norm_detected_plate else ""
            return f"Veículo Desconhecido{plate_tag}", round(best_sim, 3), False, norm_detected_plate

    def identify_pet(self, crop_bgr: np.ndarray, animal_type: str = "Animal") -> Tuple[str, float, bool]:
        """Identifica um gato ou cachorro com base na aparência visual."""
        probe_emb = self._extract_embedding(crop_bgr)
        if probe_emb is None or not self.known_pets:
            return f"{animal_type} Desconhecido", 0.0, False

        best_name = f"{animal_type} Desconhecido"
        best_sim = 0.0

        for name, emb in self.known_pets.items():
            sim = self.torch.dot(probe_emb, emb).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self.threshold:
            return best_name, round(best_sim, 3), True
        else:
            return f"{animal_type} Desconhecido", round(best_sim, 3), False


visual_matcher = VisualEntityMatcher()
