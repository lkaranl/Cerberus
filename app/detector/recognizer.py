import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

class FaceRecognizer:
    """
    Módulo de Reconhecimento Facial baseado em Embeddings de Redes Neurais Profundas.
    Utiliza FaceNet (Inception-ResNet) pré-treinada para extrair vetores faciais (512-d)
    e comparar via similaridade de cosseno com as fotos cadastradas em data/known_faces/.
    """
    def __init__(self, known_faces_dir: str = "data/known_faces", similarity_threshold: float = 0.65):
        self.known_faces_dir = Path(known_faces_dir)
        self.similarity_threshold = similarity_threshold
        self.device = settings.resolve_device()
        self.known_embeddings: Dict[str, Any] = {} # nome -> tensor de embedding médio
        self.is_ready = False

        self.mtcnn = None
        self.resnet = None
        self.torch = None

    def initialize(self) -> bool:
        """Inicializa os modelos de detecção e extração facial em GPU/CPU."""
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.torch = torch

            logger.info(f"Carregando FaceNet (InceptionResnetV1) no dispositivo: {self.device}...")
            # MTCNN leve para recorte preciso do rosto
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                keep_all=False,
                select_largest=True,
                device=self.device
            )
            # Modelo de extração de embeddings (512 dimensões)
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.is_ready = True
            self.load_known_faces()
            return True
        except ImportError:
            logger.warning(
                "facenet-pytorch não instalado. Reconhecimento facial operando em modo fallback "
                "(identificará todas as detecções como 'Desconhecido')."
            )
            self.is_ready = False
            return False
        except Exception as e:
            logger.error(f"Erro ao inicializar FaceRecognizer: {e}")
            self.is_ready = False
            return False

    def load_known_faces(self) -> int:
        """
        Varre as subpastas de data/known_faces/<Nome_da_Pessoa>/,
        extrai os embeddings faciais e armazena a média vetorial por pessoa.
        """
        if not self.is_ready or not self.known_faces_dir.exists():
            return 0

        self.known_embeddings.clear()
        person_count = 0
        total_photos = 0

        logger.info(f"Escaneando rostos conhecidos em: {self.known_faces_dir}")

        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue

            person_name = person_dir.name
            embeddings_list = []

            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    face_tensor = self.mtcnn(img_rgb)

                    if face_tensor is not None:
                        with self.torch.no_grad():
                            face_batch = face_tensor.unsqueeze(0).to(self.device)
                            emb = self.resnet(face_batch)
                            # Normaliza o vetor para comparação de cosseno
                            emb_norm = self.torch.nn.functional.normalize(emb, p=2, dim=1)
                            embeddings_list.append(emb_norm.squeeze(0))
                            total_photos += 1
                except Exception as err:
                    logger.warning(f"Erro ao processar imagem {img_path}: {err}")

            if embeddings_list:
                # Calcula o vetor médio representativo dessa pessoa
                mean_emb = self.torch.stack(embeddings_list).mean(dim=0)
                mean_emb_norm = self.torch.nn.functional.normalize(mean_emb, p=2, dim=0)
                self.known_embeddings[person_name] = mean_emb_norm
                person_count += 1
                logger.info(f"✅ Pessoa cadastrada: {person_name} ({len(embeddings_list)} fotos processadas)")

        logger.info(f"Carregamento concluído: {person_count} pessoas ({total_photos} fotos).")
        return person_count

    def identify_person(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float, bool]:
        """
        Recebe o frame completo e a bounding box do corpo da pessoa detectada pelo YOLO.
        Recorta a porção superior (cabeça/rosto), extrai o embedding e compara com a base conhecida.
        Retorna: (nome_da_pessoa, confianca_similaridade, is_conhecido)
        """
        if not self.is_ready or not self.known_embeddings:
            return "Desconhecido", 0.0, False

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        # Foca nos primeiros 45% superiores da bounding box (onde fica a cabeça/rosto)
        box_h = y2 - y1
        head_y2 = min(y2, y1 + int(box_h * 0.45))

        # Adiciona pequena margem horizontal
        box_w = x2 - x1
        pad_x = int(box_w * 0.1)
        crop_x1 = max(0, x1 - pad_x)
        crop_x2 = min(w, x2 + pad_x)
        crop_y1 = max(0, y1 - 10)
        crop_y2 = min(h, head_y2 + 10)

        head_crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if head_crop_bgr.size == 0:
            return "Desconhecido", 0.0, False

        try:
            head_crop_rgb = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2RGB)
            face_tensor = self.mtcnn(head_crop_rgb)

            if face_tensor is None:
                # Rosto não suficientemente visível ou de costas
                return "Desconhecido", 0.0, False

            with self.torch.no_grad():
                face_batch = face_tensor.unsqueeze(0).to(self.device)
                probe_emb = self.resnet(face_batch)
                probe_norm = self.torch.nn.functional.normalize(probe_emb, p=2, dim=1).squeeze(0)

                best_name = "Desconhecido"
                best_sim = 0.0

                for name, known_emb in self.known_embeddings.items():
                    sim = self.torch.dot(probe_norm, known_emb).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

                if best_sim >= self.similarity_threshold:
                    return best_name, round(best_sim, 3), True
                else:
                    return "Desconhecido", round(best_sim, 3), False

        except Exception as e:
            logger.debug(f"Falha ao identificar rosto: {e}")
            return "Desconhecido", 0.0, False


face_recognizer = FaceRecognizer()
