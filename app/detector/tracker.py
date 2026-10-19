import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.storage.database import db
from app.detector.recognizer import face_recognizer
from app.detector.visual_matcher import visual_matcher
from app.detector.plate_reader import plate_reader

class TrackedEntity:
    def __init__(self, tracking_id: int, entity_type: str = "person", class_label: str = "Objeto"):
        self.tracking_id = tracking_id
        self.entity_type = entity_type
        self.class_label = class_label

        self.consecutive_frames: int = 0
        self.first_seen: float = time.time()
        self.last_seen: float = time.time()
        self.event_triggered: bool = False
        self.best_confidence: float = 0.0
        self.best_bbox: Optional[Tuple[int, int, int, int]] = None

        # Dados de Identificação (Pessoa, Veículo ou Pet)
        self.identity_name: str = f"{class_label} Desconhecido"
        self.is_known: bool = False
        self.match_confidence: float = 0.0
        self.license_plate: str = ""
        self.eval_count: int = 0

    def update(self, confidence: float, bbox: Tuple[int, int, int, int]) -> None:
        self.consecutive_frames += 1
        self.last_seen = time.time()
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_bbox = bbox


class CameraTrackerManager:
    """
    Gerencia o ciclo de vida dos rastros (ByteTrack) por câmera para pessoas,
    veículos (com placas) e animais de estimação (pets).
    Incorpora debounce temporal, herança de rastros e supressão de duplicidades.
    """
    def __init__(self, camera_id: str, camera_name: str, event_callback=None):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.event_callback = event_callback

        self.active_tracks: Dict[int, TrackedEntity] = {}
        self.timeout_seconds: float = 6.0

        # Histórico de rastros perdidos recentemente para Anti-ID-Switching
        self.recent_lost_tracks: List[Dict[str, Any]] = []

        # Cooldown temporal por chave de entidade (ex: cam-0_person_Desconhecido -> timestamp)
        self.last_alert_time: Dict[str, float] = {}

    def _calculate_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Calcula Intersection over Union (IoU) entre duas caixas delimitadoras."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def _cleanup_stale_tracks(self) -> None:
        now = time.time()
        stale_ids = [
            tid for tid, track in self.active_tracks.items()
            if (now - track.last_seen) > self.timeout_seconds
        ]
        for tid in stale_ids:
            track = self.active_tracks[tid]
            if track.best_bbox:
                # Salva no histórico recente para herança caso a pessoa volte a ser detectada segundos depois
                self.recent_lost_tracks.append({
                    "tracking_id": tid,
                    "bbox": track.best_bbox,
                    "last_seen": track.last_seen,
                    "event_triggered": track.event_triggered,
                    "identity_name": track.identity_name,
                    "entity_type": track.entity_type,
                    "is_known": track.is_known,
                    "license_plate": track.license_plate
                })
            del self.active_tracks[tid]

        # Mantém histórico de rastros perdidos por até 10 segundos
        self.recent_lost_tracks = [
            t for t in self.recent_lost_tracks if (now - t["last_seen"]) < 10.0
        ]

    def _check_track_inheritance(self, new_bbox: Tuple[int, int, int, int], entity_type: str) -> Optional[Dict[str, Any]]:
        """Verifica se um novo tracking ID sobrepõe espacialmente um rastro recém-perdido."""
        for lost in self.recent_lost_tracks:
            if lost["entity_type"] != entity_type:
                continue

            iou = self._calculate_iou(new_bbox, lost["bbox"])
            if iou >= 0.20:
                return lost

            # Checagem por proximidade de centroides (para pessoas em movimento)
            c1_x = (new_bbox[0] + new_bbox[2]) / 2
            c1_y = (new_bbox[1] + new_bbox[3]) / 2
            c2_x = (lost["bbox"][0] + lost["bbox"][2]) / 2
            c2_y = (lost["bbox"][1] + lost["bbox"][3]) / 2
            dist = np.hypot(c1_x - c2_x, c1_y - c2_y)

            box_diag = np.hypot(new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1])
            if dist < (box_diag * 0.75):
                return lost

        return None

    def _extract_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], margin_pct: float = 0.10) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        box_w = x2 - x1
        box_h = y2 - y1

        margin_x = int(box_w * margin_pct)
        margin_y = int(box_h * margin_pct)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(w, x2 + margin_x)
        crop_y2 = min(h, y2 + margin_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return frame
        return crop

    async def process_tracks(
        self,
        raw_frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processa as detecções do frame atual com anti-ID switching, debounce e cooldown.
        """
        from app.detector.engine import detector_engine

        self._cleanup_stale_tracks()
        now = time.time()
        triggered_events = []

        for det in detections:
            tid = det.get("tracking_id")
            if tid is None:
                continue

            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", (0, 0, 0, 0))
            entity_type = det.get("entity_type", "person")
            class_label = det.get("class_label", "Objeto")

            # --- Criação ou Atualização do Rastro ---
            if tid not in self.active_tracks:
                new_track = TrackedEntity(tid, entity_type, class_label)

                # Herança Espacial de Rastro (Anti-ID-Switching)
                inherited = self._check_track_inheritance(bbox, entity_type)
                if inherited:
                    new_track.identity_name = inherited["identity_name"]
                    new_track.is_known = inherited["is_known"]
                    new_track.license_plate = inherited["license_plate"]
                    new_track.event_triggered = inherited["event_triggered"]
                    new_track.consecutive_frames = settings.min_consecutive_frames # Já considera estável
                    logger.debug(
                        f"🔄 Herança de Rastro: ID #{tid} herdou estado de #{inherited['tracking_id']} "
                        f"({new_track.identity_name}) em {self.camera_name}"
                    )

                self.active_tracks[tid] = new_track

            track = self.active_tracks[tid]
            track.update(conf, bbox)

            # --- Identificação nos Primeiros Frames do Rastro ---
            if not track.is_known and track.eval_count < 5:
                track.eval_count += 1
                crop_obj = self._extract_crop(raw_frame, bbox, margin_pct=0.05)

                if entity_type == "person":
                    name, face_conf, is_known = face_recognizer.identify_person(raw_frame, bbox)
                    if is_known:
                        track.identity_name = name
                        track.is_known = True
                        track.match_confidence = face_conf
                        logger.info(f"👤 Rosto identificado em {self.camera_name}: {name} ({int(face_conf * 100)}%)")

                elif entity_type == "vehicle":
                    plate = plate_reader.read_plate(crop_obj)
                    if plate:
                        track.license_plate = plate

                    v_name, v_conf, v_known, matched_plate = visual_matcher.identify_vehicle(crop_obj, track.license_plate)
                    if v_known:
                        track.identity_name = v_name
                        track.is_known = True
                        track.match_confidence = v_conf
                        track.license_plate = matched_plate
                        logger.info(f"🚗 Veículo reconhecido em {self.camera_name}: {v_name}")
                    elif track.license_plate:
                        track.identity_name = f"{class_label} [{track.license_plate}]"

                elif entity_type == "pet":
                    pet_name, pet_conf, pet_known = visual_matcher.identify_pet(crop_obj, class_label)
                    if pet_known:
                        track.identity_name = pet_name
                        track.is_known = True
                        track.match_confidence = pet_conf
                        logger.info(f"🐾 Pet reconhecido em {self.camera_name}: {pet_name} ({int(pet_conf * 100)}%)")

            # Atualiza informações de identidade no objeto de detecção
            det["person_name"] = track.identity_name
            det["is_known"] = track.is_known
            det["face_confidence"] = track.match_confidence
            det["license_plate"] = track.license_plate

        # Gera o frame anotado com as cores e nomes correspondentes
        annotated_frame = detector_engine.draw_annotations(raw_frame, detections)

        # --- Verificação de Disparo de Alertas e Registro no SQLite ---
        for det in detections:
            tid = det.get("tracking_id")
            if tid is None:
                continue

            track = self.active_tracks.get(tid)
            if not track:
                continue

            # Chave de Cooldown por câmera + tipo + identidade
            cooldown_key = f"{self.camera_id}_{track.entity_type}_{track.identity_name}"
            last_alert = self.last_alert_time.get(cooldown_key, 0.0)
            elapsed_since_last_alert = now - last_alert

            # Critérios estritos para evitar repetições:
            # 1. Deve ser estável (>= settings.min_consecutive_frames)
            # 2. Não deve ter disparado ainda neste rastro
            if track.consecutive_frames >= settings.min_consecutive_frames and not track.event_triggered:
                track.event_triggered = True

                # Se já alertou recentemente essa mesma pessoa/entidade nesta câmera, suprime duplicidade
                if elapsed_since_last_alert < settings.alert_cooldown_seconds:
                    logger.debug(
                        f"⏳ Alerta suprimido por cooldown ({int(elapsed_since_last_alert)}s / {settings.alert_cooldown_seconds}s): "
                        f"{self.camera_name} - {track.identity_name}"
                    )
                    continue

                # Registra o novo alerta e atualiza timestamp de cooldown
                self.last_alert_time[cooldown_key] = now

                event_id = f"evt_{uuid.uuid4().hex[:12]}"
                timestamp_str = datetime.now().isoformat()
                date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

                events_dir = Path(settings.events_dir)
                events_dir.mkdir(parents=True, exist_ok=True)

                tag_clean = track.identity_name.replace(" ", "_").replace("[", "").replace("]", "")
                crop_filename = f"{date_prefix}_{self.camera_id}_{track.entity_type}_{tag_clean}_id{tid}_crop.jpg"
                full_filename = f"{date_prefix}_{self.camera_id}_{track.entity_type}_{tag_clean}_id{tid}_full.jpg"

                crop_path = str(events_dir / crop_filename)
                full_path = str(events_dir / full_filename)

                # Salva recortes de evidência
                crop_img = self._extract_crop(raw_frame, track.best_bbox or det["bbox"], margin_pct=0.10)
                cv2.imwrite(crop_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(full_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                event_record = {
                    "event_id": event_id,
                    "camera_id": self.camera_id,
                    "camera_name": self.camera_name,
                    "timestamp": timestamp_str,
                    "tracking_id": tid,
                    "confidence": round(det["confidence"], 3),
                    "crop_path": crop_path,
                    "full_path": full_path,
                    "person_name": track.identity_name,
                    "is_known": track.is_known,
                    "face_confidence": track.match_confidence,
                    "entity_type": track.entity_type,
                    "license_plate": track.license_plate
                }

                # Persistência assíncrona no SQLite (24/7 autônomo)
                await db.save_event(event_record)

                category_icons = {"person": "👤", "vehicle": "🚗", "pet": "🐾"}
                icon = category_icons.get(track.entity_type, "🎯")
                logger.info(
                    f"{icon} Alerta registrado: {self.camera_name} - {track.identity_name} "
                    f"(ID #{tid}) -> {crop_filename}"
                )

                triggered_events.append(event_record)

                if self.event_callback:
                    try:
                        await self.event_callback(event_record)
                    except Exception as err:
                        logger.error(f"Erro no callback de evento: {err}")

        return annotated_frame, triggered_events
