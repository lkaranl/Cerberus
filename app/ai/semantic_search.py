import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.storage.database import db

TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL_NAME = "clip-ViT-B-32"

class SemanticSearchEngine:
    """
    Motor Multimodal de Busca Semântica em Linguagem Natural (Zero-Shot Cross-Modal).
    Utiliza arquitetura CLIP/SigLIP com suporte multilíngue nativo (Português do Brasil).
    
    Permite indexar eventos por conteúdo visual e pesquisar por descrições livres:
    Exemplos:
    - "homem de jaqueta vermelha"
    - "carro prata estacionado"
    - "entregador de moto com capacete"
    - "cachorro correndo perto do portão"
    """
    def __init__(self, text_model_name: str = TEXT_MODEL_NAME, img_model_name: str = IMAGE_MODEL_NAME):
        self.text_model_name = text_model_name
        self.img_model_name = img_model_name
        self.device = settings.resolve_device()
        self.text_model = None
        self.img_model = None
        self.is_ready = False
        self._initializing = False

    def initialize(self) -> bool:
        """Carrega os encoders multimodais de imagem e texto CLIP na GPU."""
        if self.is_ready:
            return True
        if self._initializing:
            return False

        self._initializing = True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"⚡ Inicializando Encoders de Busca Semântica (CLIP Multilíngue + Visual) em {self.device}...")
            t0 = time.time()
            self.text_model = SentenceTransformer(self.text_model_name, device=self.device)
            self.img_model = SentenceTransformer(self.img_model_name, device=self.device)
            self.is_ready = True
            self._initializing = False
            logger.info(f"🚀 Busca Semântica Multimodal pronta em {time.time() - t0:.1f}s na GPU!")
            return True
        except ImportError:
            logger.warning("sentence-transformers não instalado. Busca semântica operando em modo fallback textual.")
            self.is_ready = False
            self._initializing = False
            return False
        except Exception as e:
            logger.warning(f"Erro ao inicializar modelo multimodal de busca: {e}")
            self.is_ready = False
            self._initializing = False
            return False

    def extract_image_embedding(self, image_input: Any) -> Optional[np.ndarray]:
        """
        Gera um vetor métrico L2-normalizado de 512 dimensões a partir de uma imagem
        (caminho em disco, PIL Image ou array OpenCV BGR).
        """
        if not self.is_ready:
            self.initialize()
        if not self.is_ready or self.img_model is None or image_input is None:
            return None

        try:
            if isinstance(image_input, (str, Path)):
                if not os.path.exists(str(image_input)):
                    return None
                pil_img = Image.open(str(image_input)).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    return None
                rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            else:
                return None

            # Extração via encoder visual oficial do CLIP (compartilha o mesmo espaço vetorial)
            emb = self.img_model.encode(
                pil_img,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return emb.astype(np.float32)

        except Exception as e:
            logger.debug(f"Erro ao gerar embedding visual CLIP: {e}")
            return None

    def extract_image_embedding_bytes(self, image_input: Any) -> Optional[bytes]:
        """Retorna o embedding em formato de bytes binários para armazenamento direto em coluna BLOB do SQLite."""
        emb = self.extract_image_embedding(image_input)
        if emb is not None:
            return emb.tobytes()
        return None

    def encode_text_query(self, query: str) -> Optional[np.ndarray]:
        """Converte uma pergunta ou descrição em texto (ex: 'homem de casaco azul') em embedding multimodal."""
        if not self.is_ready:
            self.initialize()
        if not self.is_ready or self.text_model is None or not query:
            return None

        try:
            emb = self.text_model.encode(
                query.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return emb.astype(np.float32)
        except Exception as e:
            logger.error(f"Erro ao codificar texto de busca: {e}")
            return None

    async def search_events(self, query: str, top_k: int = 5, min_score: float = 0.18) -> List[Dict[str, Any]]:
        """
        Executa busca semântica no banco de eventos.
        Se o modelo neural estiver ativo, calcula a similaridade de cosseno vetorial.
        Caso contrário, realiza busca textual por metadados.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()

        # 1. Rota Neural SOTA (CLIP Multimodal)
        if self.is_ready or self.initialize():
            q_emb = self.encode_text_query(clean_query)
            if q_emb is not None:
                matches = await db.search_events_by_similarity(q_emb, top_k=top_k, min_score=min_score)
                if matches:
                    return matches

        # 2. Rota Fallback Textual (se CLIP não estiver instalado ou eventos sem embedding)
        logger.info(f"Executando busca textual de contingência para: '{clean_query}'")
        return await db.search_events_by_text(clean_query, limit=top_k)


semantic_search = SemanticSearchEngine()
