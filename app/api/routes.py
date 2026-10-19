import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.stream.manager import stream_manager
from app.storage.database import db

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/cameras")
async def get_cameras():
    """Retorna lista de câmeras com status de conexão e FPS."""
    return stream_manager.get_camera_statuses()


@router.get("/events")
async def get_events(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    camera_id: Optional[str] = Query(default=None)
):
    """Retorna histórico de eventos de detecção de pessoas com paginação."""
    events = await db.get_recent_events(limit=limit, offset=offset, camera_id=camera_id)
    total = await db.get_total_events_count(camera_id=camera_id)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": events
    }


@router.get("/events/{event_id}/crop")
async def get_event_crop(event_id: str):
    """Retorna a imagem recortada (crop) da pessoa detectada."""
    event = await db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado")

    crop_path = Path(event["crop_path"])
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo de recorte não encontrado")

    return FileResponse(crop_path, media_type="image/jpeg")


@router.get("/events/{event_id}/full")
async def get_event_full(event_id: str):
    """Retorna o snapshot completo com bounding boxes da detecção."""
    event = await db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado")

    full_path = Path(event["full_path"])
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo de snapshot completo não encontrado")

    return FileResponse(full_path, media_type="image/jpeg")


@router.get("/stats")
async def get_system_stats():
    """Retorna métricas de hardware, aceleração e resumo do sistema."""
    device = settings.resolve_device()
    gpu_info = {"device": device, "type": "CPU"}

    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            mem_alloc = round(torch.cuda.memory_allocated(0) / 1024**2, 2)
            mem_res = round(torch.cuda.memory_reserved(0) / 1024**2, 2)
            total_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)
            gpu_info = {
                "device": device,
                "type": "NVIDIA CUDA",
                "name": dev_name,
                "allocated_mb": mem_alloc,
                "reserved_mb": mem_res,
                "total_vram_mb": total_mem
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = {
                "device": "mps",
                "type": "Apple Silicon Metal (MPS)",
                "name": "Apple M4 / Metal"
            }
    except Exception:
        pass

    total_events = await db.get_total_events_count()

    return {
        "status": "online",
        "hardware": gpu_info,
        "cameras_active": len([c for c in stream_manager.cameras_config if c.active]),
        "total_events_logged": total_events
    }
