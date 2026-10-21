"""
Cerberus AI — Cliente de Comunicação Assíncrono para o Worker AMD RX 5600 XT
Permite ao processo principal do Cerberus (NVIDIA CUDA) delegar tarefas analíticas
pesadas para a GPU AMD secundária com timeout rígido e fallback resiliente.
"""

import base64
import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("cerberus.amd_client")

class AmdWorkerClient:
    """
    Cliente HTTP ultraleve assíncrono para consumir o Worker da AMD na porta 8001.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8001", timeout_sec: float = 1.5):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._is_online = False
        self._last_health_check = 0.0

    async def check_health(self) -> bool:
        """Verifica se o microsserviço da AMD está respondendo."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=0.6) as client:
                res = await client.get(f"{self.base_url}/health")
                if res.status_code == 200:
                    self._is_online = True
                    return True
        except Exception:
            self._is_online = False
        return False

    async def restore_face(self, face_bgr: np.ndarray, upscale: int = 2) -> Tuple[np.ndarray, bool]:
        """
        Envia o corte facial para super-resolução na AMD RX 5600 XT.
        Retorna (imagem_restaurada, sucesso_no_worker_amd).
        Em caso de timeout ou falha de rede, devolve a imagem original intacta.
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr, False

        try:
            import cv2
            import httpx

            # Codifica recorte em JPEG base64
            _, buffer = cv2.imencode(".jpg", face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            b64_payload = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "image_b64": b64_payload,
                "upscale": upscale
            }

            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(f"{self.base_url}/api/v1/restore-face", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    restored_b64 = data.get("restored_b64")
                    if restored_b64:
                        raw_bytes = base64.b64decode(restored_b64)
                        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                        restored_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if restored_bgr is not None:
                            return restored_bgr, True

        except Exception as e:
            # Fallback transparente sem interromper fluxo de detecção
            logger.debug(f"Worker AMD não disponível ou timeout ({e}), utilizando corte original.")

        return face_bgr, False

    async def get_face_embedding(self, face_bgr: np.ndarray) -> Tuple[Optional[List[float]], bool]:
        """
        Envia o corte facial para extração de vetor biométrico de alta precisão na AMD.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None, False

        try:
            import cv2
            import httpx

            _, buffer = cv2.imencode(".jpg", face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            b64_payload = base64.b64encode(buffer).decode("utf-8")

            payload = {"image_b64": b64_payload}

            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(f"{self.base_url}/api/v1/face-embedding", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = data.get("embedding")
                    if embedding and len(embedding) == 512:
                        return embedding, True

        except Exception as e:
            logger.debug(f"Falha ao obter embedding no worker AMD: {e}")

        return None, False

# Instância global compartilhada pelo Cerberus
amd_worker_client = AmdWorkerClient()
