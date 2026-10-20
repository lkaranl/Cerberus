import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np
from PIL import Image

import torch

from app.core.config import settings
from app.core.logging import logger

class EdgeVLM:
    """
    Motor de Visão & Linguagem Local de Borda (Edge VLM - Moondream2 SOTA).
    Executa 100% offline diretamente na GPU local com quantização FP16,
    sem enviar frames ou imagens para serviços de terceiros ou nuvem.
    
    Capacidades:
    1. Descrição detalhada de cena e recortes (captioning em tempo real).
    2. Visual Question Answering (VQA) livre (responde qualquer dúvida sobre a imagem).
    3. Auditoria visual profunda de segurança (detecta ferramentas, capacetes, uniformes, atitudes suspeitas).
    """
    def __init__(
        self,
        model_id: str = settings.edge_vlm_model,
        revision: str = settings.edge_vlm_revision
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = settings.resolve_device()
        self.model = None
        self.tokenizer = None
        self.is_ready = False
        self._initializing = False

    def initialize(self) -> bool:
        """Carrega os pesos neurais do Moondream2 na GPU."""
        if not settings.enable_edge_vlm:
            return False
        if self.is_ready:
            return True
        if self._initializing:
            return False

        self._initializing = True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"⚡ Carregando Edge VLM [{self.model_id}] em {self.device} (FP16)...")
            t0 = time.time()

            torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype
            }
            tokenizer_kwargs = {"trust_remote_code": True}

            if self.revision and self.revision.strip():
                model_kwargs["revision"] = self.revision.strip()
                tokenizer_kwargs["revision"] = self.revision.strip()

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            except Exception as load_err:
                # Se a revisão falhar, tenta carregar sem revisão
                if "revision" in model_kwargs:
                    logger.warning(f"Revisão '{self.revision}' falhou ({load_err}). Carregando branch principal...")
                    model_kwargs.pop("revision", None)
                    tokenizer_kwargs.pop("revision", None)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
                else:
                    raise load_err

            # Aloca diretamente no hardware sem exigir a biblioteca 'accelerate'
            self.model.to(self.device)
            self.model.eval()
            self.is_ready = True
            self._initializing = False
            logger.info(f"🚀 Edge VLM (Moondream2) pronto em {time.time() - t0:.1f}s na GPU!")
            return True

        except ImportError:
            logger.warning("Dependências do VLM não instaladas (transformers, einops, timm).")
            self.is_ready = False
            self._initializing = False
            return False
        except Exception as e:
            logger.error(f"Erro ao inicializar Moondream2 Edge VLM: {e}")
            self.is_ready = False
            self._initializing = False
            return False

    def _prepare_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Optional[Image.Image]:
        """Converte diferentes tipos de entrada para PIL Image RGB."""
        if image_input is None:
            return None
        try:
            if isinstance(image_input, (str, Path)):
                p = Path(image_input)
                if not p.exists():
                    return None
                return Image.open(str(p)).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    return None
                rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            elif isinstance(image_input, Image.Image):
                return image_input.convert("RGB")
        except Exception as err:
            logger.debug(f"Erro ao processar imagem para VLM: {err}")
        return None

    def caption(self, image_input: Union[str, Path, np.ndarray, Image.Image], length: str = "normal") -> Optional[str]:
        """
        Gera uma descrição textual em linguagem natural do que está visível na imagem.
        length: 'short' ou 'normal'.
        """
        if not self.is_ready and not self.initialize():
            return None

        pil_img = self._prepare_image(image_input)
        if pil_img is None or self.model is None:
            return None

        try:
            with torch.inference_mode():
                # API Moondream moderna
                if hasattr(self.model, "caption"):
                    try:
                        res = self.model.caption(pil_img, length=length)
                        if isinstance(res, dict) and "caption" in res:
                            return res["caption"].strip()
                    except TypeError:
                        res = self.model.caption(pil_img)
                        if isinstance(res, dict) and "caption" in res:
                            return res["caption"].strip()
                
                # Fallback via query direta
                return self.query(pil_img, "Describe what you see in this image in detail.")
        except Exception as e:
            logger.error(f"Erro ao gerar caption via Edge VLM: {e}")
            return None

    def query(self, image_input: Union[str, Path, np.ndarray, Image.Image], question: str) -> Optional[str]:
        """
        Responde a uma pergunta em linguagem natural sobre a imagem (VQA).
        Exemplos:
        - "What is the person wearing and holding?"
        - "Is this person wearing a helmet or a mask?"
        - "What color and type is this vehicle?"
        """
        if not self.is_ready and not self.initialize():
            return None

        pil_img = self._prepare_image(image_input)
        if pil_img is None or self.model is None or not question:
            return None

        try:
            with torch.inference_mode():
                # Moondream query API
                if hasattr(self.model, "query"):
                    res = self.model.query(pil_img, question)
                    if isinstance(res, dict) and "answer" in res:
                        return res["answer"].strip()
                    elif isinstance(res, str):
                        return res.strip()

                # Moondream classic API
                if hasattr(self.model, "encode_image") and hasattr(self.model, "answer_question"):
                    enc_img = self.model.encode_image(pil_img)
                    return self.model.answer_question(enc_img, question, self.tokenizer)

        except Exception as e:
            logger.error(f"Erro ao executar query no Edge VLM: {e}")
        return None

    def audit_security_scene(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        entity_type: str = "person"
    ) -> Dict[str, Any]:
        """
        Realiza uma auditoria visual completa de segurança sobre a evidência.
        100% offline e privada.
        """
        desc = self.caption(image_input, length="normal")
        if not desc:
            return {
                "descricao": "Não foi possível analisar a imagem no momento.",
                "risco": "baixo",
                "motivo": "Análise visual indisponível.",
                "acao": "Manter monitoramento rotineiro."
            }

        # Perguntas dirigidas ao VLM dependendo do tipo de alvo
        if entity_type == "person":
            q_threat = (
                "Is the person in this image wearing a helmet, face mask, holding a tool, "
                "climbing, or acting suspiciously? Answer in one short sentence."
            )
            threat_details = self.query(image_input, q_threat) or ""

            q_wear = "What is the person wearing? Describe colors and clothing style."
            wear_details = self.query(image_input, q_wear) or ""

            # Avaliação de risco baseada nos achados visuais
            lower_threat = threat_details.lower()
            is_suspicious = any(k in lower_threat for k in ["helmet", "mask", "tool", "weapon", "climbing", "suspicious", "pry", "backpack"])

            return {
                "descricao": desc,
                "vestimentas": wear_details,
                "indicadores_risco": threat_details,
                "risco": "medio" if is_suspicious else "baixo",
                "motivo": f"Auditoria visual local: {threat_details if is_suspicious else desc}",
                "acao": "Verifique o vídeo ao vivo." if is_suspicious else "Monitoramento de rotina.",
                "avaliador": "edge_vlm_moondream2"
            }

        elif entity_type == "vehicle":
            q_vehicle = "What type, make and color is this vehicle? Is it a car, motorcycle, van, or truck?"
            veh_details = self.query(image_input, q_vehicle) or ""
            return {
                "descricao": desc,
                "detalhes_veiculo": veh_details,
                "risco": "baixo",
                "motivo": f"Veículo identificado localmente: {veh_details}",
                "acao": "Monitoramento padrão.",
                "avaliador": "edge_vlm_moondream2"
            }

        return {
            "descricao": desc,
            "risco": "baixo",
            "motivo": desc,
            "acao": "Nenhuma ação necessária.",
            "avaliador": "edge_vlm_moondream2"
        }


edge_vlm = EdgeVLM()
