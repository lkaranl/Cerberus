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
        pets_dir: str = "data/known_pets"
    ):
        self.vehicles_dir = Path(vehicles_dir)
        self.pets_dir = Path(pets_dir)
        self.device = settings.resolve_device()

        # Dicionários de características cadastradas
        # nome -> {"embedding": tensor, "color_hist": np.ndarray, "plate": str}
        self.known_vehicles: Dict[str, Dict[str, Any]] = {}
        # nome -> {"embedding": tensor, "color_hist": np.ndarray}
        self.known_pets: Dict[str, Dict[str, Any]] = {}

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

    def _extract_color_histogram(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai histograma de cor 2D (H-S) normalizado focado na região central da lataria ou corpo,
        evitando interferência de para-brisas pretos, pneus e asfalto do chão.
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        try:
            h, w = img_bgr.shape[:2]
            y1, y2 = int(h * 0.25), int(h * 0.85)
            x1, x2 = int(w * 0.20), int(w * 0.80)
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                roi = img_bgr

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # 18 faixas de matiz (Hue) e 16 faixas de saturação (Saturation)
            hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        except Exception as err:
            logger.debug(f"Erro ao extrair histograma de cores: {err}")
            return None

    def _compare_color_hist(self, hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
        """Compara dois histogramas de cor usando correlação Pearson."""
        if hist1 is None or hist2 is None:
            return 0.5
        try:
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, float(score))
        except Exception:
            return 0.5

    def _auto_crop_vehicle(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Se a imagem de referência for uma cena ampla (com chão/cadeiras),
        tenta isolar o veículo principal da imagem para não poluir o embedding.
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        try:
            h, w = img_bgr.shape[:2]
            # Se for muito grande (imagem inteira de câmera), recorta as bordas externas se o centro contiver o carro
            if w > 400 and h > 400:
                # Recorte central suave para focar no carro
                crop_y1 = int(h * 0.05)
                crop_y2 = int(h * 0.95)
                crop_x1 = int(w * 0.05)
                crop_x2 = int(w * 0.95)
                return img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        except Exception:
            pass
        return img_bgr

    def load_vehicles(self) -> int:
        """Carrega e calcula embeddings e histogramas de cores dos veículos cadastrados."""
        from app.detector.plate_reader import plate_reader

        if not self.is_ready or not self.vehicles_dir.exists():
            return 0

        self.known_vehicles.clear()
        count = 0

        for folder in self.vehicles_dir.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue

            folder_name = folder.name
            # Procura por placa entre colchetes no nome da pasta, ex: Meu_Carro_[NEC4J87]
            plate_match = re.search(r"\[([A-Z0-9\-]+)\]", folder_name, re.IGNORECASE)
            registered_plate = plate_match.group(1).upper().replace("-", "") if plate_match else ""

            # Nome limpo para exibição
            display_name = re.sub(r"\[.*?\]", "", folder_name).replace("_", " ").strip()

            embs = []
            color_hists = []

            for img_file in folder.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                raw_img = cv2.imread(str(img_file))
                if raw_img is None:
                    continue

                clean_img = self._auto_crop_vehicle(raw_img)

                # Se a placa não foi informada na pasta, tenta ler da foto de referência
                if not registered_plate:
                    found_plate = plate_reader.read_plate(clean_img)
                    if found_plate:
                        registered_plate = found_plate
                        logger.info(f"🔍 Placa lida automaticamente na foto de referência: {registered_plate} para '{display_name}'")

                emb = self._extract_embedding(clean_img)
                if emb is not None:
                    embs.append(emb)

                chist = self._extract_color_histogram(clean_img)
                if chist is not None:
                    color_hists.append(chist)

            if embs:
                mean_emb = self.torch.stack(embs).mean(dim=0)
                mean_norm = self.torch.nn.functional.normalize(mean_emb, p=2, dim=0)
                avg_color_hist = np.mean(color_hists, axis=0) if color_hists else None

                self.known_vehicles[display_name] = {
                    "embedding": mean_norm,
                    "color_hist": avg_color_hist,
                    "plate": registered_plate
                }
                count += 1
                plate_info = f" [Placa: {registered_plate}]" if registered_plate else ""
                logger.info(f"🚗 Veículo cadastrado: {display_name}{plate_info} com {len(embs)} foto(s)")

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
            color_hists = []

            for img_file in folder.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                emb = self._extract_embedding(img)
                if emb is not None:
                    embs.append(emb)

                chist = self._extract_color_histogram(img)
                if chist is not None:
                    color_hists.append(chist)

            if embs:
                mean_emb = self.torch.stack(embs).mean(dim=0)
                mean_norm = self.torch.nn.functional.normalize(mean_emb, p=2, dim=0)
                avg_color_hist = np.mean(color_hists, axis=0) if color_hists else None

                self.known_pets[display_name] = {
                    "embedding": mean_norm,
                    "color_hist": avg_color_hist
                }
                count += 1
                logger.info(f"🐾 Pet cadastrado: {display_name} com {len(embs)} foto(s)")

        return count

    def load_all(self) -> None:
        self.load_vehicles()
        self.load_pets()

    def identify_vehicle(self, crop_bgr: np.ndarray, detected_plate: str = "") -> Tuple[str, float, bool, str]:
        """
        Identifica um veículo com fusão multimodal:
        Prioridade 1: Match de placa oficial (LPR/ANPR) -> 99% confiança.
        Prioridade 2: Fusão de Deep Embedding + Histograma de Cor HSV com veto de discrepância.
        Retorna: (nome_do_veiculo, confianca, is_conhecido, placa_final)
        """
        norm_detected_plate = detected_plate.upper().replace("-", "").strip()

        # 1. Checagem por placa cadastrada
        if norm_detected_plate:
            for name, data in self.known_vehicles.items():
                known_plate = data.get("plate", "")
                if known_plate:
                    # Match exato de 7 caracteres
                    if known_plate == norm_detected_plate:
                        return f"{name} [{known_plate}]", 0.99, True, known_plate
                    # Match de prefixo de 6 caracteres (caso o último dígito esteja cortado na borda)
                    if len(norm_detected_plate) == 6 and known_plate.startswith(norm_detected_plate):
                        return f"{name} [{known_plate}]", 0.98, True, known_plate

        # 2. Checagem visual por fusão de Deep Features + Histograma de Cor
        probe_emb = self._extract_embedding(crop_bgr)
        probe_hist = self._extract_color_histogram(crop_bgr)

        if probe_emb is None or not self.known_vehicles:
            plate_tag = f" [{norm_detected_plate}]" if norm_detected_plate else ""
            return f"Veículo Desconhecido{plate_tag}", 0.0, False, norm_detected_plate

        best_name = "Veículo Desconhecido"
        best_sim = 0.0
        matched_plate = norm_detected_plate

        for name, data in self.known_vehicles.items():
            deep_sim = self.torch.dot(probe_emb, data["embedding"]).item()
            color_sim = self._compare_color_hist(probe_hist, data.get("color_hist"))

            # Se a cor for drasticamente divergente (ex: carro prata vs carro preto/vermelho),
            # aplica veto estrito para impedir falsos positivos
            if color_sim < 0.25:
                combined_sim = deep_sim * color_sim
            else:
                combined_sim = (deep_sim * 0.60) + (color_sim * 0.40)

            if combined_sim > best_sim:
                best_sim = combined_sim
                best_name = name
                if data["plate"]:
                    matched_plate = data["plate"]

        thresh = settings.vehicle_similarity_threshold
        if best_sim >= thresh:
            plate_tag = f" [{matched_plate}]" if matched_plate else ""
            return f"{best_name}{plate_tag}", round(best_sim, 3), True, matched_plate
        else:
            plate_tag = f" [{norm_detected_plate}]" if norm_detected_plate else ""
            return f"Veículo Desconhecido{plate_tag}", round(best_sim, 3), False, norm_detected_plate

    def identify_pet(self, crop_bgr: np.ndarray, animal_type: str = "Animal") -> Tuple[str, float, bool]:
        """Identifica um gato ou cachorro usando embedding e colorimetria."""
        probe_emb = self._extract_embedding(crop_bgr)
        probe_hist = self._extract_color_histogram(crop_bgr)

        if probe_emb is None or not self.known_pets:
            return f"{animal_type} Desconhecido", 0.0, False

        best_name = f"{animal_type} Desconhecido"
        best_sim = 0.0

        for name, data in self.known_pets.items():
            deep_sim = self.torch.dot(probe_emb, data["embedding"]).item()
            color_sim = self._compare_color_hist(probe_hist, data.get("color_hist"))

            if color_sim < 0.25:
                combined_sim = deep_sim * color_sim
            else:
                combined_sim = (deep_sim * 0.60) + (color_sim * 0.40)

            if combined_sim > best_sim:
                best_sim = combined_sim
                best_name = name

        thresh = settings.pet_similarity_threshold
        if best_sim >= thresh:
            return best_name, round(best_sim, 3), True
        else:
            return f"{animal_type} Desconhecido", round(best_sim, 3), False


visual_matcher = VisualEntityMatcher()
