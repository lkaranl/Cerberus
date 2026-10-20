import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

# URLs oficiais e mirrors confiáveis do modelo pré-treinado OSNet x0.25 (MSMT17)
OSNET_ONNX_URLS = [
    "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx",
    "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v8.0/osnet_x0_25_msmt17.onnx",
    "https://github.com/mikel-brostrom/boxmot/releases/download/v10.0.81/osnet_x0_25_msmt17.pt"
]

class PersonReID:
    """
    Extrator biométrico de características corporais de corpo inteiro (Person Re-Identification).
    Baseado na arquitetura SOTA OSNet (Omni-Scale Network).
    
    Gera um vetor métrico L2-normalizado de 512 dimensões para cada pessoa detectada.
    Permite:
    1. Manter a identidade de moradores mesmo de costas, com capuz, boné ou em câmeras sem visão frontal.
    2. Rastreamento e associação de indivíduos entre múltiplas câmeras (Cross-Camera Tracking).
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        similarity_threshold: float = 0.70,
        memory_ttl_seconds: float = 3600.0  # Mantém assinatura corporal do morador por 1 hora
    ):
        self.model_path = Path(model_path or getattr(settings, "osnet_model_path", "models/osnet_x0_25_msmt17.onnx"))
        self.similarity_threshold = similarity_threshold
        self.memory_ttl_seconds = memory_ttl_seconds

        self.session = None
        self.input_name = None
        self.output_name = None
        self.is_ready = False

        # Pool Dinâmico de Assinaturas Corporais Ativas:
        # nome_morador -> {"embedding": np.ndarray (512,), "last_seen": float, "camera": str}
        self.active_residents: Dict[str, Dict[str, Any]] = {}

        # Histórico recente de pessoas desconhecidas para correlação entre câmeras
        # [{"embedding": np.ndarray, "timestamp": float, "camera": str, "track_id": int}]
        self.recent_unknowns: List[Dict[str, Any]] = []

    def _download_model_if_needed(self) -> bool:
        """Garante a presença do arquivo ONNX do OSNet, realizando download automático se necessário."""
        if self.model_path.exists() and self.model_path.stat().st_size > 1_000_000:
            return True

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"📥 Baixando modelo neural de Re-ID (OSNet x0.25 ONNX) em {self.model_path}...")

        import requests
        headers = {"User-Agent": "Cerberus-Vision-Agent/2.0 (Linux; Nobara; Python 3.11)"}
        for url in OSNET_ONNX_URLS:
            try:
                logger.info(f"Tentando download de {url}...")
                response = requests.get(url, stream=True, timeout=60, headers=headers, allow_redirects=True)
                if response.status_code == 200:
                    with open(self.model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    
                    file_size = self.model_path.stat().st_size
                    if file_size > 1_000_000:
                        logger.info(f"✅ Download do OSNet x0.25 concluído com sucesso ({file_size / 1024 / 1024:.2f} MB)!")
                        return True
                    else:
                        logger.warning(f"Arquivo baixado tem tamanho insuficiente ({file_size} bytes). Descartando...")
                        self.model_path.unlink(missing_ok=True)
                else:
                    logger.warning(f"HTTP {response.status_code} ao baixar de {url}")
            except Exception as e:
                logger.warning(f"Falha ao baixar OSNet de {url}: {e}")

        return False

    def initialize(self) -> bool:
        """Inicializa a sessão do OSNet via ONNX Runtime com aceleração GPU CUDA."""
        try:
            import onnxruntime as ort

            # Garante que o arquivo ONNX exista
            if not self._download_model_if_needed():
                logger.warning(f"Arquivo OSNet não encontrado em {self.model_path}.")
                self.is_ready = False
                return False

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            active_provider = self.session.get_providers()[0]

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            self.is_ready = True
            logger.info(f"⚡ Módulo Person Re-ID (OSNet x0.25) inicializado com sucesso via {active_provider}!")
            return True

        except Exception as e:
            logger.warning(f"Módulo Person Re-ID (OSNet) operando em modo inativo ou erro ao carregar: {e}")
            self.is_ready = False
            return False

    def extract_embedding(self, person_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai o vetor métrico (512 dimensões L2-normalizado) do corpo inteiro de uma pessoa.
        Entrada padrão Re-ID: 256 (altura) x 128 (largura).
        """
        if not self.is_ready or self.session is None or person_crop_bgr is None or person_crop_bgr.size == 0:
            return None

        try:
            h, w = person_crop_bgr.shape[:2]
            if h < 20 or w < 10:
                return None

            # 1. Redimensionamento para o padrão de entrada do OSNet (256x128)
            resized = cv2.resize(person_crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # 2. Normalização ImageNet (média e desvio padrão padrão do PyTorch/OSNet)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rgb - mean) / std

            # 3. Transposição para formato Tensor (B, C, H, W) -> (1, 3, 256, 128)
            blob = np.transpose(normalized, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            # 4. Inferência via ONNX Runtime
            outputs = self.session.run([self.output_name], {self.input_name: blob})
            raw_emb = outputs[0].flatten().astype(np.float32)

            # 5. Normalização L2 (para comparação direta via produto escalar / cosseno)
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                emb = raw_emb / norm
            else:
                emb = raw_emb

            return emb

        except Exception as e:
            logger.debug(f"Erro ao extrair embedding de Re-ID com OSNet: {e}")
            return None

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula a similaridade de cosseno entre dois embeddings biométricos corporais."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))

    def register_resident_sighting(self, name: str, person_crop_bgr: np.ndarray, camera: str) -> bool:
        """
        Registra/atualiza dinamicamente a assinatura corporal do morador
        logo após uma identificação facial positiva com o ArcFace.
        """
        if not name or "Desconhecido" in name or person_crop_bgr is None:
            return False

        emb = self.extract_embedding(person_crop_bgr)
        if emb is None:
            return False

        now = time.time()
        # Se já existe uma assinatura recente, atualiza via média móvel ponderada (80% histórico, 20% novo)
        if name in self.active_residents:
            old_emb = self.active_residents[name]["embedding"]
            blended = (old_emb * 0.75) + (emb * 0.25)
            norm = np.linalg.norm(blended)
            updated_emb = blended / norm if norm > 0 else blended
        else:
            updated_emb = emb

        self.active_residents[name] = {
            "embedding": updated_emb,
            "last_seen": now,
            "camera": camera
        }
        logger.info(f"👤 [Re-ID] Assinatura corporal do morador '{name}' atualizada em {camera}.")
        return True

    def match_resident(self, person_crop_bgr: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[str], float]:
        """
        Compara o corpo da pessoa atual contra a base dinâmica de moradores ativos.
        Retorna: (nome_morador, confianca_similaridade).
        """
        if not self.is_ready or not self.active_residents:
            return None, 0.0

        emb = self.extract_embedding(person_crop_bgr)
        if emb is None:
            return None, 0.0

        now = time.time()
        effective_thresh = threshold or self.similarity_threshold

        best_name = None
        best_sim = 0.0

        # Purga assinaturas expiradas (TTL) e busca a melhor correspondência
        expired_keys = []
        for name, data in self.active_residents.items():
            if now - data["last_seen"] > self.memory_ttl_seconds:
                expired_keys.append(name)
                continue

            sim = self.compute_similarity(emb, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_name = name

        for k in expired_keys:
            del self.active_residents[k]

        if best_name and best_sim >= effective_thresh:
            return best_name, round(best_sim, 3)

        return None, round(best_sim, 3)


person_reid = PersonReID()
