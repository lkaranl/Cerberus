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
    camera_id: Optional[str] = Query(default=None),
    is_known: Optional[bool] = Query(default=None),
    entity_type: Optional[str] = Query(default=None)
):
    """Retorna histórico de eventos de detecção (pessoas, veículos, pets) com filtros."""
    events = await db.get_recent_events(
        limit=limit,
        offset=offset,
        camera_id=camera_id,
        is_known=is_known,
        entity_type=entity_type
    )
    total = await db.get_total_events_count(camera_id=camera_id)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "events": events
    }


@router.get("/events/{event_id}/crop")
async def get_event_crop(event_id: str):
    """Retorna a imagem recortada (crop) da pessoa detectada com cache HTTP."""
    event = await db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado")

    crop_path = Path(event["crop_path"])
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo de recorte não encontrado")

    return FileResponse(
        crop_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800, immutable"}
    )


@router.get("/events/{event_id}/full")
async def get_event_full(event_id: str, raw: bool = False):
    """
    Retorna o snapshot completo com bounding boxes da detecção.
    Por padrão entrega versão otimizada para web (max 1280px) com cache HTTP,
    garantindo carregamento instantâneo no navegador.
    Use ?raw=true para obter o arquivo original em resolução nativa da câmera.
    """
    event = await db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado")

    full_path = Path(event["full_path"])
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo de snapshot completo não encontrado")

    cache_headers = {"Cache-Control": "public, max-age=604800, immutable"}

    if raw:
        return FileResponse(full_path, media_type="image/jpeg", headers=cache_headers)

    # 1. Verifica se já existe a versão web pré-gerada
    web_path = full_path.with_name(f"{full_path.stem}_web{full_path.suffix}")
    if web_path.exists():
        return FileResponse(web_path, media_type="image/jpeg", headers=cache_headers)

    # 2. Se o arquivo original já for leve (< 280 KB), serve diretamente
    try:
        if full_path.stat().st_size <= 280 * 1024:
            return FileResponse(full_path, media_type="image/jpeg", headers=cache_headers)
    except OSError:
        pass

    # 3. Gera versão web otimizada em background para responder rápido e cachear no disco
    import asyncio
    def _create_web_preview():
        try:
            import cv2
            img = cv2.imread(str(full_path))
            if img is None:
                return full_path
            h, w = img.shape[:2]
            if w > 1280:
                scale = 1280.0 / w
                new_w = 1280
                new_h = int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img
            cv2.imwrite(str(web_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return web_path
        except Exception:
            return full_path

    target_file = await asyncio.to_thread(_create_web_preview)
    return FileResponse(target_file, media_type="image/jpeg", headers=cache_headers)


@router.get("/stats")
async def get_system_stats():
    """Retorna métricas de hardware, aceleração e resumo do sistema."""
    device = settings.resolve_device()
    gpu_info = {"device": device, "type": "CPU"}

    try:
        from app.detector.engine import detector_engine
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            mem_alloc = round(torch.cuda.memory_allocated(0) / 1024**2, 2)
            mem_res = round(torch.cuda.memory_reserved(0) / 1024**2, 2)
            total_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)
            backend_label = "NVIDIA CUDA (TensorRT FP16 ⚡)" if detector_engine.using_tensorrt else "NVIDIA CUDA"
            gpu_info = {
                "device": device,
                "type": backend_label,
                "name": dev_name,
                "tensorrt_active": detector_engine.using_tensorrt,
                "model_format": "TensorRT .engine" if detector_engine.using_tensorrt else "PyTorch .pt",
                "allocated_mb": mem_alloc,
                "reserved_mb": mem_res,
                "total_vram_mb": total_mem
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = {
                "device": "mps",
                "type": "Apple Silicon Metal (MPS)",
                "name": "Apple M4 / Metal",
                "tensorrt_active": False,
                "model_format": "PyTorch .pt"
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


@router.get("/stats/analytics")
async def get_analytics():
    """Retorna dados estatísticos consolidados para o dashboard analítico."""
    return await db.get_analytics_data()


@router.get("/faces")
async def get_known_faces():
    """Lista as pessoas atualmente cadastradas no módulo de reconhecimento facial."""
    from app.detector.recognizer import face_recognizer
    return {
        "is_ready": face_recognizer.is_ready,
        "known_persons": list(face_recognizer.known_embeddings.keys())
    }


@router.post("/faces/reload")
async def reload_known_entities():
    """Recarrega todas as fotos e placas de pessoas, veículos e pets sem reiniciar o servidor."""
    from app.detector.recognizer import face_recognizer
    from app.detector.visual_matcher import visual_matcher
    p_count = face_recognizer.load_known_faces()
    v_count = visual_matcher.load_vehicles()
    pet_count = visual_matcher.load_pets()
    return {
        "status": "success",
        "persons_loaded": p_count,
        "vehicles_loaded": v_count,
        "pets_loaded": pet_count,
        "persons": list(face_recognizer.known_embeddings.keys()),
        "vehicles": list(visual_matcher.known_vehicles.keys()),
        "pets": list(visual_matcher.known_pets.keys())
    }


@router.delete("/events/clear")
@router.post("/events/clear")
async def clear_events_database():
    """Limpa todo o histórico de eventos e evidências do banco de dados SQLite."""
    deleted = await db.clear_all_events(delete_files=True)
    return {
        "status": "success",
        "message": f"Histórico limpo com sucesso. {deleted} registros removidos.",
        "deleted_count": deleted
    }


