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
    """
    def __init__(self, camera_id: str, camera_name: str, event_callback=None):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.event_callback = event_callback
        self.active_tracks: Dict[int, TrackedEntity] = {}
        self.timeout_seconds: float = 6.0

    def _cleanup_stale_tracks(self) -> None:
        now = time.time()
        stale_ids = [
            tid for tid, track in self.active_tracks.items()
            if (now - track.last_seen) > self.timeout_seconds
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]

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
        Processa as detecções do frame atual.
        Executa reconhecimento facial para pessoas, leitura de placa e visual match para veículos,
        e reconhecimento visual para animais.
        """
        from app.detector.engine import detector_engine

        self._cleanup_stale_tracks()
        triggered_events = []

        for det in detections:
            tid = det.get("tracking_id")
            if tid is None:
                continue

            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", (0, 0, 0, 0))
            entity_type = det.get("entity_type", "person")
            class_label = det.get("class_label", "Objeto")

            if tid not in self.active_tracks:
                self.active_tracks[tid] = TrackedEntity(tid, entity_type, class_label)

            track = self.active_tracks[tid]
            track.update(conf, bbox)

            # --- Identificação nos Primeiros Frames do Rastro ---
            if not track.is_known and track.eval_count < 5:
                track.eval_count += 1
                crop_obj = self._extract_crop(raw_frame, bbox, margin_pct=0.05)

                if entity_type == "person":
                    # Reconhecimento Facial (FaceNet)
                    name, face_conf, is_known = face_recognizer.identify_person(raw_frame, bbox)
                    if is_known:
                        track.identity_name = name
                        track.is_known = True
                        track.match_confidence = face_conf
                        logger.info(f"👤 Rosto identificado em {self.camera_name}: {name} ({int(face_conf * 100)}%)")

                elif entity_type == "vehicle":
                    # 1. Tenta ler placa do veículo (ANPR/LPR)
                    plate = plate_reader.read_plate(crop_obj)
                    if plate:
                        track.license_plate = plate

                    # 2. Identificação de veículo (por placa ou por foto)
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
                    # Reconhecimento visual de gato ou cachorro
                    pet_name, pet_conf, pet_known = visual_matcher.identify_pet(crop_obj, class_label)
                    if pet_known:
                        track.identity_name = pet_name
                        track.is_known = True
                        track.match_confidence = pet_conf
                        logger.info(f"🐾 Pet reconhecido em {self.camera_name}: {pet_name} ({int(pet_conf * 100)}%)")

            # Atualiza informações de identidade no objeto de detecção para exibição no vídeo
            det["person_name"] = track.identity_name
            det["is_known"] = track.is_known
            det["face_confidence"] = track.match_confidence
            det["license_plate"] = track.license_plate

        # Gera o frame anotado com as cores e nomes correspondentes
        annotated_frame = detector_engine.draw_annotations(raw_frame, detections)

        # Registro de Eventos e Evidências
        for det in detections:
            tid = det.get("tracking_id")
            if tid is None:
                continue

            track = self.active_tracks.get(tid)
            if not track:
                continue

            # Confirmar presença por mais de 2 frames consecutivos antes de registrar
            if track.consecutive_frames >= 2 and not track.event_triggered:
                track.event_triggered = True

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

                # Salvar imagens de evidência
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

                # Persistência assíncrona no SQLite (funciona mesmo com navegador fechado)
                await db.save_event(event_record)

                category_icons = {"person": "👤", "vehicle": "🚗", "pet": "🐾"}
                icon = category_icons.get(track.entity_type, "🎯")
                logger.info(
                    f"{icon} Evento registrado: {self.camera_name} - {track.identity_name} "
                    f"(ID #{tid}) -> {crop_filename}"
                )

                triggered_events.append(event_record)

                if self.event_callback:
                    try:
                        await self.event_callback(event_record)
                    except Exception as err:
                        logger.error(f"Erro no callback de evento: {err}")

        return annotated_frame, triggered_events
