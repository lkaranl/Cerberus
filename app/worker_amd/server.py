#!/usr/bin/env python3
"""
Cerberus AI — Servidor Microsserviço do Worker AMD RX 5600 XT
Ouve na porta local 127.0.0.1:8001 e atende requisições analíticas pesadas
do processo principal do Cerberus de forma totalmente assíncrona e desacoplada.
"""

import sys
import os
import base64
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

# Adiciona raiz do repositório ao sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uvicorn

from app.worker_amd.engine_amd import AmdInferenceEngine

app = FastAPI(
    title="Cerberus AMD Specialist Worker",
    description="Microsserviço de Inferência Neural Dedicado à AMD Radeon RX 5600 XT",
    version="1.0.0"
)

# Inicializa o motor de inferência da AMD
engine = AmdInferenceEngine()

# Semáforo para controlar concorrência e proteger VRAM
SEMAPHORE = asyncio.Semaphore(4)

class RestoreFaceRequest(BaseModel):
    image_b64: str = Field(..., description="Imagem do rosto codificada em base64 (JPEG ou PNG)")
    upscale: int = Field(default=2, ge=1, le=4, description="Fator de aumento da super-resolução")

class FaceEmbeddingRequest(BaseModel):
    image_b64: str = Field(..., description="Corte do rosto codificado em base64")

class PlateRequest(BaseModel):
    image_b64: str = Field(..., description="Corte da placa veicular codificado em base64")

def b64_to_cv2(b64_str: str) -> Optional[np.ndarray]:
    """Decodifica string base64 para matriz numpy BGR OpenCV."""
    try:
        import cv2
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def cv2_to_b64(img_bgr: np.ndarray, quality: int = 92) -> str:
    """Codifica imagem OpenCV BGR para string base64 JPEG."""
    import cv2
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode("utf-8")

@app.get("/health")
async def health_check():
    """Retorna o estado operacional e telemetria da GPU AMD."""
    info = engine.get_device_info()
    return {
        "status": "online",
        "service": "cerberus-worker-amd",
        "telemetry": info
    }

@app.post("/api/v1/restore-face")
async def restore_face_endpoint(req: RestoreFaceRequest):
    """Executa restauração e super-resolução de corte facial na GPU AMD."""
    async with SEMAPHORE:
        img_bgr = b64_to_cv2(req.image_b64)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Imagem base64 inválida ou corrompida.")

        restored_bgr, latency_ms = engine.restore_face(img_bgr, upscale=req.upscale)
        restored_b64 = cv2_to_b64(restored_bgr)

        return {
            "success": True,
            "latency_ms": round(latency_ms, 2),
            "original_shape": list(img_bgr.shape[:2]),
            "restored_shape": list(restored_bgr.shape[:2]),
            "restored_b64": restored_b64,
            "device": engine.device_name
        }

@app.post("/api/v1/face-embedding")
async def face_embedding_endpoint(req: FaceEmbeddingRequest):
    """Extrai vetor biométrico de 512 dimensões na GPU AMD."""
    async with SEMAPHORE:
        img_bgr = b64_to_cv2(req.image_b64)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Imagem base64 inválida ou corrompida.")

        embedding, latency_ms = engine.extract_face_embedding(img_bgr)

        return {
            "success": True,
            "latency_ms": round(latency_ms, 2),
            "dimensions": len(embedding),
            "embedding": embedding,
            "device": engine.device_name
        }

@app.post("/api/v1/license-plate")
async def license_plate_endpoint(req: PlateRequest):
    """Executa processamento neural de placa de veículo na GPU AMD."""
    async with SEMAPHORE:
        img_bgr = b64_to_cv2(req.image_b64)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Imagem base64 inválida ou corrompida.")

        plate_data, latency_ms = engine.read_license_plate(img_bgr)

        return {
            "success": True,
            "latency_ms": round(latency_ms, 2),
            "plate": plate_data.get("plate", ""),
            "confidence": plate_data.get("confidence", 0.0),
            "device": engine.device_name
        }

def start_server(host: str = "127.0.0.1", port: int = 8001):
    """Inicia o servidor Uvicorn do Worker AMD."""
    print(f"\n=======================================================")
    print(f" ⚡ CERBERUS AMD WORKER — RADEON RX 5600 XT")
    print(f" Endereço Local: http://{host}:{port}")
    print(f" Dispositivo   : {engine.device_name}")
    print(f" Backend       : {engine.backend}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_server()
