import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.stream.manager import stream_manager
from app.storage.database import db

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/whep/{camera_id}")
async def whep_proxy(camera_id: str, request: Request):
    """
    Proxy reverso WHEP para negociação WebRTC transparente com o gateway go2rtc.
    Recebe a SDP Offer do navegador e encaminha para o go2rtc local (porta 1984),
    retornando a SDP Answer sem necessidade de portas adicionais ou problemas de CORS.
    """
    sdp_offer = await request.body()
    go2rtc_url = f"http://127.0.0.1:1984/api/whep?src={camera_id}"

    try:
        req = urllib.request.Request(
            go2rtc_url,
            data=sdp_offer,
            headers={"Content-Type": "application/sdp"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            answer_sdp = resp.read()
            return Response(
                content=answer_sdp,
                status_code=resp.status,
                media_type="application/sdp"
            )
    except urllib.error.HTTPError as e:
        return Response(content=e.read(), status_code=e.code)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Gateway WebRTC go2rtc offline ou indisponível: {e}"
        )


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
        from app.detector.recognizer import face_recognizer
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            mem_alloc = round(torch.cuda.memory_allocated(0) / 1024**2, 2)
            mem_res = round(torch.cuda.memory_reserved(0) / 1024**2, 2)
            total_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)

            from app.stream.manager import stream_manager
            any_nvdec = any(getattr(cap, "is_nvdec_active", False) for cap in stream_manager.captures.values())

            trt_active = detector_engine.using_tensorrt or face_recognizer.using_tensorrt
            if detector_engine.using_tensorrt and face_recognizer.using_tensorrt and any_nvdec:
                backend_label = "NVIDIA CUDA (Full GPU: NVDEC + YOLO + FaceNet TRT ⚡)"
            elif detector_engine.using_tensorrt and face_recognizer.using_tensorrt:
                backend_label = "NVIDIA CUDA (YOLO + FaceNet TensorRT FP16 ⚡)"
            elif detector_engine.using_tensorrt:
                backend_label = "NVIDIA CUDA (YOLO TensorRT FP16 ⚡)"
            else:
                backend_label = "NVIDIA CUDA"

            gpu_info = {
                "device": device,
                "type": backend_label,
                "name": dev_name,
                "tensorrt_active": trt_active,
                "yolo_tensorrt": detector_engine.using_tensorrt,
                "facenet_tensorrt": face_recognizer.using_tensorrt,
                "nvdec_active": any_nvdec,
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
                "yolo_tensorrt": False,
                "facenet_tensorrt": False,
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


@router.get("/vlm/status")
async def get_edge_vlm_status():
    """Retorna o status do modelo multimodal generativo local (Edge VLM - Moondream2)."""
    from app.ai.edge_vlm import edge_vlm
    return {
        "enabled": settings.enable_edge_vlm,
        "is_ready": edge_vlm.is_ready,
        "model_id": edge_vlm.model_id,
        "revision": edge_vlm.revision,
        "device": edge_vlm.device
    }


@router.get("/vlm/describe/{event_id}")
async def describe_event_vlm(event_id: str):
    """Gera uma descrição visual detalhada da evidência usando o Moondream2 local."""
    from app.ai.edge_vlm import edge_vlm
    event = await db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado")

    crop_path = Path(event["crop_path"])
    target_img = crop_path if crop_path.exists() else Path(event.get("full_path", ""))
    if not target_img.exists():
        raise HTTPException(status_code=404, detail="Imagem de evidência não encontrada no disco")

    import asyncio
    description = await asyncio.to_thread(edge_vlm.caption, str(target_img), "normal")
    audit = await asyncio.to_thread(edge_vlm.audit_security_scene, str(target_img), event.get("entity_type", "person"))

    return {
        "event_id": event_id,
        "description": description,
        "audit": audit
    }


@router.post("/vlm/ask")
async def ask_event_vlm(payload: dict):
    """Responde a uma pergunta em linguagem natural sobre uma imagem de evento gravado."""
    from app.ai.edge_vlm import edge_vlm
    event_id = payload.get("event_id", "")
    question = payload.get("question", "").strip()
    if not event_id or not question:
        raise HTTPException(status_code=400, detail="Parâmetros 'event_id' e 'question' são obrigatórios")

    event = await db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado")

    crop_path = Path(event["crop_path"])
    target_img = crop_path if crop_path.exists() else Path(event.get("full_path", ""))
    if not target_img.exists():
        raise HTTPException(status_code=404, detail="Imagem de evidência não encontrada no disco")

    import asyncio
    answer = await asyncio.to_thread(edge_vlm.query, str(target_img), question)
    return {
        "event_id": event_id,
        "question": question,
        "answer": answer
    }



