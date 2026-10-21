#!/usr/bin/env python3
"""
Cerberus — Módulo 2: Super-Resolução e Restauração Facial Neural (AI Face Restoration)
Foco: Reconstrução de detalhes anatômicos (olhos, nariz, boca) em rostos distantes (< 60px)
capturados por câmeras de segurança residenciais 2K com lentes grande-angular.

Arquitetura:
- Motor Primário: Rede Neural de Super-Resolução ONNX (FSRCNN / Real-ESRGAN Compact Face 4x)
  acelerada via ONNX Runtime GPU (CUDA/TensorRT).
- Motor de Fallback: Algoritmo Fotométrico de Interpolação Lanczos-4 + Unsharp Masking de Alta Frequência.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

FACE_SR_MODEL_URLS = [
    "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
    "https://huggingface.co/qualcomm/FSRCNN/resolve/main/FSRCNN.onnx",
    "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/fsrcnn_x4.onnx"
]

class FaceSuperResolver:
    """
    Restaura e amplia a resolução de recortes faciais pequenos ou degradados por compressão H.264.
    Permite que o ArcFace extraia embeddings estáveis mesmo para pessoas a 6-8 metros da câmera.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or "models/face_sr_x4.onnx")
        self.session = None
        self.input_name = None
        self.output_name = None
        self.is_neural_ready = False
        self._initialize_neural_engine()

    def _initialize_neural_engine(self) -> bool:
        """Inicializa o modelo neural de super-resolução via ONNX Runtime na GPU."""
        try:
            import onnxruntime as ort

            # Verifica e cria pasta models
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            if not self.model_path.exists():
                logger.debug(f"Modelo neural SR {self.model_path} não encontrado localmente. Usando modo fotométrico adaptativo Lanczos.")
                return False

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(str(self.model_path), sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            active_provider = self.session.get_providers()[0]

            self.is_neural_ready = True
            logger.info(f"🚀 Face Super-Resolution Neural (ONNX / {active_provider}) ativo em {self.model_path}!")
            return True
        except Exception as err:
            logger.debug(f"Super-Resolução Neural em modo de fallback analítico: {err}")
            self.is_neural_ready = False
            return False

    def enhance(self, face_bgr: np.ndarray, target_size: Optional[Tuple[int, int]] = (112, 112)) -> np.ndarray:
        """
        Restaura e amplia o recorte facial.
        Se o modelo neural ONNX estiver disponível, executa a inferência 4x na GPU.
        Caso contrário, aplica restauração analítica Lanczos-4 com realce de micro-arestas (Unsharp Masking).
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr

        h, w = face_bgr.shape[:2]

        # Se a face já for suficientemente grande e nítida (> 80px), apenas garante o tamanho alvo
        if min(h, w) >= settings.face_super_res_threshold and target_size is not None:
            if (w, h) != target_size:
                return cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_LINEAR)
            return face_bgr

        # 1. Inferência Neural (se carregado)
        if self.is_neural_ready and self.session is not None:
            try:
                # Normalização para a rede [0, 1] ou [0, 255]
                inp = face_bgr.astype(np.float32) / 255.0
                inp = np.transpose(inp, (2, 0, 1))  # (C, H, W)
                inp = np.expand_dims(inp, axis=0)   # (1, C, H, W)

                out = self.session.run([self.output_name], {self.input_name: inp})[0]
                out = np.squeeze(out, axis=0)       # (C, H, W)
                out = np.transpose(out, (1, 2, 0))  # (H, W, C)
                out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

                if target_size is not None:
                    out = cv2.resize(out, target_size, interpolation=cv2.INTER_LANCZOS4)
                return out
            except Exception as e:
                logger.debug(f"Falha na inferência neural SR: {e}. Aplicando restauração analítica.")

        # 2. Restauração Fotométrica Analítica de Alta Resolução (Lanczos-4 + Unsharp Masking)
        out_w = target_size[0] if target_size else max(112, w * 4)
        out_h = target_size[1] if target_size else max(112, h * 4)

        # Interpolação Lanczos-4 (a mais nítida para reconstrução de bordas)
        upscaled = cv2.resize(face_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # Unsharp Masking Adaptativo para reconstruir micro-arestas de olhos e nariz
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(upscaled, 1.4, gaussian, -0.4, 0)

        # Realce sutil de contraste no canal L para compensar desfoque óptico
        try:
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            l_enh = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception:
            return sharpened

face_super_resolver = FaceSuperResolver()
