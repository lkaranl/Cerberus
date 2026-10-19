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

class TrackedPerson:
    def __init__(self, tracking_id: int):
        self.tracking_id = tracking_id
        self.consecutive_frames: int = 0
        self.first_seen: float = time.time()
        self.last_seen: float = time.time()
        self.event_triggered: bool = False
        self.best_confidence: float = 0.0
        self.best_bbox: Optional[Tuple[int, int, int, int]] = None

    def update(self, confidence: float, bbox: Tuple[int, int, int, int]) -> None:
        self.consecutive_frames += 1
        self.last_seen = time.time()
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_bbox = bbox


class CameraTrackerManager:
    """
    Gerencia o ciclo de vida dos identificadores rastreados (ByteTrack) por câmera,
    controlando disparo único de alertas com evidências recortadas (crop com 10% de margem)
    e frame completo anotado.
    """
    def __init__(self, camera_id: str, camera_name: str, event_callback=None):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.event_callback = event_callback
        self.active_tracks: Dict[int, TrackedPerson] = {}
        self.timeout_seconds: float = 6.0  # Tempo para considerar que a pessoa saiu da cena

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
        annotated_frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Processa as detecções do frame atual.
        Se uma pessoa for confirmada por > 2 frames consecutivos e ainda não tiver gerado
        evento na sessão de presença atual, salva o recorte, frame completo e insere no DB.
        """
        self._cleanup_stale_tracks()
        current_seen_ids = set()
        triggered_events = []

        for det in detections:
            tid = det.get("tracking_id")
            if tid is None:
                continue

            current_seen_ids.add(tid)
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", (0, 0, 0, 0))

            if tid not in self.active_tracks:
                self.active_tracks[tid] = TrackedPerson(tid)

            track = self.active_tracks[tid]
            track.update(conf, bbox)

            # Requisito do Spec: Confirmar pessoa por mais de 2 frames consecutivos
            if track.consecutive_frames >= 2 and not track.event_triggered:
                track.event_triggered = True

                event_id = f"evt_{uuid.uuid4().hex[:12]}"
                timestamp_str = datetime.now().isoformat()
                date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

                events_dir = Path(settings.events_dir)
                events_dir.mkdir(parents=True, exist_ok=True)

                crop_filename = f"{date_prefix}_{self.camera_id}_id{tid}_crop.jpg"
                full_filename = f"{date_prefix}_{self.camera_id}_id{tid}_full.jpg"

                crop_path = str(events_dir / crop_filename)
                full_path = str(events_dir / full_filename)

                # Salvar imagens de evidência
                crop_img = self._extract_crop(raw_frame, bbox, margin_pct=0.10)
                cv2.imwrite(crop_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(full_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                event_record = {
                    "event_id": event_id,
                    "camera_id": self.camera_id,
                    "camera_name": self.camera_name,
                    "timestamp": timestamp_str,
                    "tracking_id": tid,
                    "confidence": round(conf, 3),
                    "crop_path": crop_path,
                    "full_path": full_path
                }

                # Persistência assíncrona no SQLite
                await db.save_event(event_record)
                logger.info(
                    f"🔔 Evento de presença registrado: {self.camera_name} - Pessoa #{tid} "
                    f"({int(conf * 100)}%) -> {crop_filename}"
                )

                triggered_events.append(event_record)

                if self.event_callback:
                    try:
                        await self.event_callback(event_record)
                    except Exception as err:
                        logger.error(f"Erro no callback de evento: {err}")

        return triggered_events
