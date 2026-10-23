import os
import re
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Request, Response, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.fusionsolar_client import fusionsolar_client
from app.stream.manager import stream_manager
from app.storage.database import db

router = APIRouter(prefix="/api", tags=["API"])

_WEATHER_CACHE = {"data": None, "timestamp": 0.0}


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


@router.get("/weather/solar")
async def get_weather_solar(lat: Optional[float] = None, lon: Optional[float] = None):
    """
    Retorna previsão do tempo e irradiação solar diária (kWh/m²) para Ji-Paraná, RO e os próximos 3 dias,
    enriquecido com a telemetria em tempo real da usina Huawei FusionSolar (potência em kW e geração do dia).
    Possui cache em memória de 30 minutos para o clima, com telemetria solar atualizada dinamicamente.
    """
    global _WEATHER_CACHE
    now = time.time()
    lat = lat if lat is not None else settings.weather_lat
    lon = lon if lon is not None else settings.weather_lon
    real_solar = await fusionsolar_client.get_kiosk_data()

    if _WEATHER_CACHE["data"] and (now - _WEATHER_CACHE["timestamp"] < 1800):
        cached_result = dict(_WEATHER_CACHE["data"])
        cached_result["real_solar"] = real_solar
        return cached_result

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,"
        f"shortwave_radiation_sum,weathercode&current_weather=true&timezone=auto"
    )

    weather_desc_map = {
        0: ("Ensolarado", "☀️"),
        1: ("Céu Aberto", "🌤️"),
        2: ("Parcialmente Nublado", "⛅"),
        3: ("Nublado", "☁️"),
        45: ("Nevoeiro", "🌫️"),
        48: ("Neblina", "🌫️"),
        51: ("Garoa Leve", "🌦️"),
        53: ("Garoa", "🌧️"),
        55: ("Garoa Densa", "🌧️"),
        61: ("Chuva Fraca", "🌧️"),
        63: ("Chuva Moderada", "🌧️"),
        65: ("Chuva Forte", "🌧️"),
        80: ("Pancadas Isoladas", "🌦️"),
        81: ("Pancadas de Chuva", "🌧️"),
        82: ("Chuva Torrencial", "⛈️"),
        95: ("Trovoadas", "⛈️"),
        96: ("Tempestade", "⛈️"),
    }

    day_labels = ["Hoje", "Amanhã", "Depois"]

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Cerberus-Solar-Forecast"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

            daily = raw.get("daily", {})
            times = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            radiation_mj = daily.get("shortwave_radiation_sum", [])
            codes = daily.get("weathercode", [])

            days_forecast = []
            for i in range(min(3, len(times))):
                mj = radiation_mj[i] if i < len(radiation_mj) and radiation_mj[i] is not None else 21.0
                # Conversão física: 1 MJ/m² = 0.27778 kWh/m² (MJ / 3.6)
                kwh_m2 = round(mj / 3.6, 2)
                # Estimativa de geração teórica por m² (~20% de eficiência fotovoltaica padrão)
                est_generation_kwh = round(kwh_m2 * 0.20, 2)

                if kwh_m2 >= 6.0:
                    potential = "Excelente"
                    solar_color = "#FFB300"
                elif kwh_m2 >= 4.5:
                    potential = "Bom"
                    solar_color = "#FFA000"
                elif kwh_m2 >= 3.0:
                    potential = "Moderado"
                    solar_color = "#FB8C00"
                else:
                    potential = "Baixo"
                    solar_color = "#90A4AE"

                wcode = codes[i] if i < len(codes) else 0
                desc, icon = weather_desc_map.get(wcode, ("Céu Aberto", "🌤️"))

                days_forecast.append({
                    "date": times[i],
                    "day_index": i,
                    "day_label": day_labels[i] if i < len(day_labels) else times[i],
                    "temp_max": round(max_temps[i]) if i < len(max_temps) else 32,
                    "temp_min": round(min_temps[i]) if i < len(min_temps) else 22,
                    "radiation_mj": round(mj, 1),
                    "solar_kwh_m2": kwh_m2,
                    "est_generation_kwh": est_generation_kwh,
                    "solar_potential": potential,
                    "solar_color": solar_color,
                    "weather_code": wcode,
                    "weather_desc": desc,
                    "weather_icon": icon,
                })

            current = raw.get("current_weather", {})
            current_code = current.get("weathercode", 0)
            c_desc, c_icon = weather_desc_map.get(current_code, ("Céu Aberto", "🌤️"))

            result = {
                "city": settings.weather_city_name,
                "current_temp": round(current.get("temperature", 28)),
                "current_code": current_code,
                "current_desc": c_desc,
                "current_icon": c_icon,
                "is_day": current.get("is_day", 1),
                "days": days_forecast,
                "real_solar": real_solar,
                "cached_at": now
            }
            _WEATHER_CACHE["data"] = result
            _WEATHER_CACHE["timestamp"] = now
            return result
    except Exception:
        if _WEATHER_CACHE["data"]:
            cached = dict(_WEATHER_CACHE["data"])
            cached["real_solar"] = real_solar
            return cached

        return {
            "current_temp": 31,
            "current_code": 0,
            "current_desc": "Ensolarado",
            "current_icon": "☀️",
            "is_day": 1,
            "days": [
                {"date": "2026-10-21", "day_index": 0, "day_label": "Hoje", "temp_max": 35, "temp_min": 24, "solar_kwh_m2": 6.4, "est_generation_kwh": 1.28, "solar_potential": "Excelente", "solar_color": "#FFB300", "weather_code": 0, "weather_desc": "Ensolarado", "weather_icon": "☀️"},
                {"date": "2026-10-22", "day_index": 1, "day_label": "Amanhã", "temp_max": 36, "temp_min": 24, "solar_kwh_m2": 6.1, "est_generation_kwh": 1.22, "solar_potential": "Excelente", "solar_color": "#FFB300", "weather_code": 1, "weather_desc": "Céu Aberto", "weather_icon": "🌤️"},
                {"date": "2026-10-23", "day_index": 2, "day_label": "Depois", "temp_max": 34, "temp_min": 23, "solar_kwh_m2": 5.2, "est_generation_kwh": 1.04, "solar_potential": "Bom", "solar_color": "#FFA000", "weather_code": 2, "weather_desc": "Parcialmente Nublado", "weather_icon": "⛅"}
            ],
            "real_solar": real_solar,
            "fallback": True
        }


@router.get("/solar/kiosk")
async def get_solar_kiosk():
    """Retorna a telemetria em tempo real da usina Huawei FusionSolar para o frontend."""
    return await fusionsolar_client.get_kiosk_data()


@router.get("/solar/debug")
async def get_solar_debug():
    """Endpoint de diagnóstico detalhado para verificar a comunicação com a Huawei."""
    debug_info = await fusionsolar_client.get_debug_info()
    return debug_info


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
    """Retorna a imagem recortada (crop) da pessoa detectada com cache HTTP e resolução resiliente."""
    event = await db.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado")

    crop_path = Path(event["crop_path"])
    if not crop_path.exists():
        # Tenta localizar diretamente dentro de settings.events_dir
        alt_crop = Path(settings.events_dir) / crop_path.name
        if alt_crop.exists():
            crop_path = alt_crop
        else:
            # Fallback gracioso para full_path se o crop não estiver disponível
            full_path = Path(event.get("full_path", ""))
            if full_path.exists():
                crop_path = full_path
            else:
                alt_full = Path(settings.events_dir) / full_path.name
                if alt_full.exists():
                    crop_path = alt_full
                else:
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
        alt_full = Path(settings.events_dir) / full_path.name
        if alt_full.exists():
            full_path = alt_full
        else:
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


def sanitize_entity_name(name: str) -> str:
    """Higieniza o nome da entidade para ser usado com segurança como nome de diretório."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    clean = re.sub(r'\s+', " ", clean).strip(". ")
    return clean


@router.get("/faces")
async def get_known_faces():
    """Lista as pessoas atualmente cadastradas no módulo de reconhecimento facial."""
    from app.detector.recognizer import face_recognizer
    return {
        "is_ready": face_recognizer.is_ready,
        "known_persons": list(face_recognizer.known_gallery.keys()) or list(face_recognizer.known_embeddings.keys())
    }


@router.get("/entities/known")
async def get_all_known_entities():
    """Retorna inventário completo consolidado de pessoas, veículos e pets cadastrados."""
    from app.detector.recognizer import face_recognizer
    from app.detector.visual_matcher import visual_matcher

    persons = list(face_recognizer.known_gallery.keys()) or list(face_recognizer.known_embeddings.keys())
    vehicles = list(visual_matcher.known_vehicles.keys())
    pets = list(visual_matcher.known_pets.keys())

    return {
        "persons": sorted(persons),
        "vehicles": sorted(vehicles),
        "pets": sorted(pets)
    }


class ClassifyEventRequest(BaseModel):
    event_id: str
    person_name: Optional[str] = None
    entity_name: Optional[str] = None
    entity_type: str = "person"  # "person", "vehicle", "pet"
    license_plate: Optional[str] = None


ClassifyEntityRequest = ClassifyEventRequest


@router.post("/entities/classify-event")
@router.post("/faces/classify-event")
async def classify_entity_event(payload: ClassifyEventRequest):
    """
    Cadastra uma evidência/crop diretamente no perfil de um morador, veículo ou pet,
    alimentando a galeria de referência com a foto real capturada pelas câmeras.
    """
    raw_name = (payload.entity_name or payload.person_name or "").strip()
    clean_name = sanitize_entity_name(raw_name)
    if not clean_name or clean_name.lower() in ["desconhecido", "unknown", "none", "null"]:
        raise HTTPException(status_code=400, detail="Nome de entidade inválido")

    ent_type = (payload.entity_type or "person").lower().strip()
    if ent_type not in ["person", "vehicle", "pet"]:
        ent_type = "person"

    event = await db.get_event_by_id(payload.event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Evento não encontrado no banco de dados")

    crop_path = Path(event.get("crop_path", ""))
    if not crop_path.exists():
        crop_path = Path(event.get("full_path", ""))
        if not crop_path.exists():
            raise HTTPException(status_code=404, detail="Arquivo de evidência não encontrado no disco")

    import shutil

    if ent_type == "vehicle":
        clean_plate = None
        if payload.license_plate:
            clean_plate = re.sub(r"[^A-Za-z0-9]", "", payload.license_plate).upper()
        elif event.get("license_plate"):
            clean_plate = re.sub(r"[^A-Za-z0-9]", "", event.get("license_plate")).upper()

        folder_name = f"{clean_name}_[{clean_plate}]" if clean_plate else clean_name
        target_dir = Path("data/known_vehicles") / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        dest_filename = f"cctv_veh_{payload.event_id[:8]}_{int(time.time())}.jpg"
        dest_path = target_dir / dest_filename
        shutil.copyfile(crop_path, dest_path)

        update_fields = {
            "person_name": clean_name,
            "is_known": 1
        }
        if clean_plate:
            update_fields["license_plate"] = clean_plate
        await db.update_event(payload.event_id, update_fields)

        from app.detector.visual_matcher import visual_matcher
        v_count = visual_matcher.load_vehicles()

        return {
            "status": "success",
            "message": f"Veículo '{clean_name}' cadastrado com sucesso!",
            "entity_type": "vehicle",
            "entity_name": clean_name,
            "license_plate": clean_plate,
            "saved_path": str(dest_path),
            "vehicles_loaded": v_count
        }

    elif ent_type == "pet":
        target_dir = Path("data/known_pets") / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)

        dest_filename = f"cctv_pet_{payload.event_id[:8]}_{int(time.time())}.jpg"
        dest_path = target_dir / dest_filename
        shutil.copyfile(crop_path, dest_path)

        await db.update_event(payload.event_id, {
            "person_name": clean_name,
            "is_known": 1
        })

        from app.detector.visual_matcher import visual_matcher
        pet_count = visual_matcher.load_pets()

        return {
            "status": "success",
            "message": f"Pet '{clean_name}' cadastrado com sucesso!",
            "entity_type": "pet",
            "entity_name": clean_name,
            "saved_path": str(dest_path),
            "pets_loaded": pet_count
        }

    else:  # person
        target_dir = Path("data/known_faces") / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)

        dest_filename = f"cctv_{payload.event_id[:8]}_{int(time.time())}.jpg"
        dest_path = target_dir / dest_filename
        shutil.copyfile(crop_path, dest_path)

        await db.update_event(payload.event_id, {
            "person_name": clean_name,
            "is_known": 1
        })

        from app.detector.recognizer import face_recognizer
        face_recognizer.load_known_faces()

        return {
            "status": "success",
            "message": f"Evidência cadastrada com sucesso para '{clean_name}'!",
            "entity_type": "person",
            "person_name": clean_name,
            "saved_path": str(dest_path),
            "total_gallery_vectors": face_recognizer.gallery_vector_count
        }


@router.get("/candidates")
@router.get("/faces/candidates")
async def get_training_candidates(entity_type: str = "person", limit: int = 12):
    """Retorna os eventos mais recentes não identificados para classificação rápida por categoria."""
    valid_type = entity_type.lower().strip()
    if valid_type not in ["person", "vehicle", "pet"]:
        valid_type = "person"
    events = await db.get_recent_events(limit=limit, is_known=False, entity_type=valid_type)
    return {"candidates": events, "entity_type": valid_type}


@router.post("/entities/upload")
async def upload_entity_photos(
    entity_type: str = Form(...),
    entity_name: str = Form(...),
    license_plate: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    """
    Cadastra uma nova pessoa, veículo ou pet diretamente via upload de fotos externas.
    Salva as fotos nas pastas correspondentes e recarrega os embeddings em GPU/RAM imediatamente.
    """
    raw_name = entity_name.strip()
    clean_name = sanitize_entity_name(raw_name)
    if not clean_name or clean_name.lower() in ["desconhecido", "unknown", "none", "null"]:
        raise HTTPException(status_code=400, detail="Nome de entidade inválido")

    ent_type = entity_type.lower().strip()
    if ent_type not in ["person", "vehicle", "pet"]:
        raise HTTPException(status_code=400, detail="Tipo de entidade inválido. Use 'person', 'vehicle' ou 'pet'")

    if not files:
        raise HTTPException(status_code=400, detail="Nenhuma foto enviada")

    clean_plate = None
    if ent_type == "vehicle":
        if license_plate:
            clean_plate = re.sub(r"[^A-Za-z0-9]", "", license_plate).upper()
        folder_name = f"{clean_name}_[{clean_plate}]" if clean_plate else clean_name
        target_dir = Path("data/known_vehicles") / folder_name
    elif ent_type == "pet":
        target_dir = Path("data/known_pets") / clean_name
    else:
        target_dir = Path("data/known_faces") / clean_name

    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    ts = int(time.time())
    for idx, f in enumerate(files):
        orig_ext = Path(f.filename or "image.jpg").suffix.lower()
        if orig_ext not in allowed_extensions:
            orig_ext = ".jpg"

        dest_name = f"upload_{ts}_{idx+1}{orig_ext}"
        dest_path = target_dir / dest_name

        content = await f.read()
        if len(content) == 0:
            continue

        with open(dest_path, "wb") as out_file:
            out_file.write(content)
        saved_files.append(str(dest_path))

    if not saved_files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo válido pôde ser salvo")

    if ent_type == "vehicle":
        from app.detector.visual_matcher import visual_matcher
        v_count = visual_matcher.load_vehicles()
        return {
            "status": "success",
            "message": f"Veículo '{clean_name}' cadastrado com {len(saved_files)} foto(s)!",
            "entity_type": "vehicle",
            "entity_name": clean_name,
            "license_plate": clean_plate,
            "saved_count": len(saved_files),
            "vehicles_loaded": v_count
        }
    elif ent_type == "pet":
        from app.detector.visual_matcher import visual_matcher
        pet_count = visual_matcher.load_pets()
        return {
            "status": "success",
            "message": f"Pet '{clean_name}' cadastrado com {len(saved_files)} foto(s)!",
            "entity_type": "pet",
            "entity_name": clean_name,
            "saved_count": len(saved_files),
            "pets_loaded": pet_count
        }
    else:
        from app.detector.recognizer import face_recognizer
        face_recognizer.load_known_faces()
        return {
            "status": "success",
            "message": f"Morador '{clean_name}' cadastrado com {len(saved_files)} foto(s)!",
            "entity_type": "person",
            "person_name": clean_name,
            "saved_count": len(saved_files),
            "total_gallery_vectors": face_recognizer.gallery_vector_count
        }


@router.post("/entities/reload")
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
        "persons": list(face_recognizer.known_gallery.keys()),
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



