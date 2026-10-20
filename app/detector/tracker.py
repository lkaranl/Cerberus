import time
import uuid
import asyncio
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

def _save_evidence_disk(crop_path: str, crop_img: np.ndarray, full_path: str, annotated_frame: np.ndarray, web_path: str) -> None:
    """Grava as imagens de evidência em disco em thread separada sem bloquear a inferência."""
    try:
        cv2.imwrite(crop_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(full_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        h_full, w_full = annotated_frame.shape[:2]
        if w_full > 1280:
            scale = 1280.0 / w_full
            web_frame = cv2.resize(annotated_frame, (1280, int(h_full * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            web_frame = annotated_frame
        cv2.imwrite(web_path, web_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    except Exception as err:
        logger.debug(f"Falha ao salvar evidências em disco: {err}")


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

        # Rastreamento Facial & Gestão Inteligente de Evidências
        self.has_face_captured: bool = False
        self.latest_face_bbox: Optional[Tuple[int, int, int, int]] = None
        self.active_event_id: Optional[str] = None
        self.event_crop_path: Optional[str] = None
        self.event_full_path: Optional[str] = None
        self.event_web_path: Optional[str] = None
        self.needs_face_upgrade: bool = False

        # Análise de Movimento & Detecção de Estacionamento
        self.centroid_history: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
        self.is_stationary: bool = False
        self.stationary_anchor_pos: Optional[Tuple[float, float]] = None
        self.stationary_since: Optional[float] = None

    def update(self, confidence: float, bbox: Tuple[int, int, int, int]) -> None:
        now = time.time()
        self.consecutive_frames += 1
        self.last_seen = now
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_bbox = bbox

        # Atualiza centróide para análise de movimento
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.centroid_history.append((cx, cy, now))

        # Mantém histórico recente (últimos 8 segundos)
        self.centroid_history = [pt for pt in self.centroid_history if (now - pt[2]) <= 8.0]

        # Calcula dispersão de movimento para identificar se está estacionado
        box_w = max(1, bbox[2] - bbox[0])
        box_h = max(1, bbox[3] - bbox[1])
        diag = np.hypot(box_w, box_h)

        if len(self.centroid_history) >= 6:
            xs = [pt[0] for pt in self.centroid_history]
            ys = [pt[1] for pt in self.centroid_history]
            motion_spread = np.hypot(max(xs) - min(xs), max(ys) - min(ys))

            # Limiar de imobilidade: jitter típico de câmera/detector é de 10-20px
            stationary_thresh = max(18.0, diag * 0.12)

            if motion_spread < stationary_thresh:
                if not self.is_stationary:
                    self.is_stationary = True
                    self.stationary_anchor_pos = (cx, cy)
                    self.stationary_since = now
            else:
                # Se houver deslocamento significativo em relação à âncora estacionária (> 35% da diagonal ou > 45px)
                if self.stationary_anchor_pos:
                    anchor_dist = np.hypot(cx - self.stationary_anchor_pos[0], cy - self.stationary_anchor_pos[1])
                    if anchor_dist > max(45.0, diag * 0.35):
                        self.is_stationary = False
                        self.stationary_anchor_pos = None
                        self.stationary_since = None


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

        # Âncoras espaciais de objetos estacionados (veículos em vagas, pets deitados)
        # Permite persistir a supressão de alertas mesmo com perdas de tracking por minutos
        self.parked_anchors: List[Dict[str, Any]] = []

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

    def _update_parked_anchor(self, track: TrackedEntity, bbox: Tuple[int, int, int, int]) -> None:
        """Mantém ou atualiza uma âncora espacial para um veículo ou pet estacionado."""
        now = time.time()
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        for anchor in self.parked_anchors:
            if anchor["entity_type"] == track.entity_type:
                iou = self._calculate_iou(bbox, anchor["bbox"])
                dist = np.hypot(cx - anchor["centroid"][0], cy - anchor["centroid"][1])
                if iou >= 0.25 or dist < 45.0:
                    anchor["bbox"] = bbox
                    anchor["centroid"] = (cx, cy)
                    anchor["last_seen"] = now
                    anchor["identity_name"] = track.identity_name
                    anchor["license_plate"] = track.license_plate
                    anchor["is_known"] = track.is_known
                    anchor["event_triggered"] = track.event_triggered
                    return

        self.parked_anchors.append({
            "bbox": bbox,
            "centroid": (cx, cy),
            "entity_type": track.entity_type,
            "identity_name": track.identity_name,
            "license_plate": track.license_plate,
            "is_known": track.is_known,
            "event_triggered": track.event_triggered,
            "first_parked": now,
            "last_seen": now
        })

    def _cleanup_stale_tracks(self) -> None:
        now = time.time()
        stale_ids = [
            tid for tid, track in self.active_tracks.items()
            if (now - track.last_seen) > self.timeout_seconds
        ]
        for tid in stale_ids:
            track = self.active_tracks[tid]
            if track.best_bbox:
                # Salva no histórico recente para herança caso a entidade volte a ser detectada segundos depois
                self.recent_lost_tracks.append({
                    "tracking_id": tid,
                    "bbox": track.best_bbox,
                    "last_seen": track.last_seen,
                    "event_triggered": track.event_triggered,
                    "identity_name": track.identity_name,
                    "entity_type": track.entity_type,
                    "is_known": track.is_known,
                    "license_plate": track.license_plate,
                    "is_stationary": track.is_stationary
                })
                # Se estava estacionado, preserva nas âncoras da vaga
                if track.is_stationary:
                    self._update_parked_anchor(track, track.best_bbox)

            del self.active_tracks[tid]

        # Mantém histórico de rastros em movimento perdidos por até 10 segundos
        self.recent_lost_tracks = [
            t for t in self.recent_lost_tracks if (now - t["last_seen"]) < 10.0
        ]

        # Mantém âncoras de vagas ocupadas/estacionadas por até 10 minutos sem detecção
        self.parked_anchors = [
            a for a in self.parked_anchors if (now - a["last_seen"]) < 600.0
        ]

    def _check_track_inheritance(self, new_bbox: Tuple[int, int, int, int], entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Verifica se um novo tracking ID sobrepõe:
        1. Uma vaga/âncora estacionada (veículo estacionado).
        2. Um rastro em movimento recém-perdido (anti-ID-switching).
        """
        now = time.time()
        c1_x = (new_bbox[0] + new_bbox[2]) / 2.0
        c1_y = (new_bbox[1] + new_bbox[3]) / 2.0

        # 1. Prioridade: Verifica se pertence a uma vaga estacionada já conhecida
        for anchor in self.parked_anchors:
            if anchor["entity_type"] != entity_type:
                continue
            iou = self._calculate_iou(new_bbox, anchor["bbox"])
            dist = np.hypot(c1_x - anchor["centroid"][0], c1_y - anchor["centroid"][1])
            if iou >= 0.25 or dist < 45.0:
                anchor["last_seen"] = now
                return {
                    "tracking_id": -1,
                    "bbox": anchor["bbox"],
                    "last_seen": anchor["last_seen"],
                    "event_triggered": True,  # Já alertou no passado, não repete alerta
                    "identity_name": anchor["identity_name"],
                    "entity_type": anchor["entity_type"],
                    "is_known": anchor["is_known"],
                    "license_plate": anchor["license_plate"],
                    "is_stationary": True
                }

        # 2. Verifica no histórico de rastros dinâmicos recém-perdidos
        for lost in self.recent_lost_tracks:
            if lost["entity_type"] != entity_type:
                continue

            iou = self._calculate_iou(new_bbox, lost["bbox"])
            if iou >= 0.20:
                return lost

            # Checagem por proximidade de centroides (para pessoas/veículos em movimento)
            c2_x = (lost["bbox"][0] + lost["bbox"][2]) / 2.0
            c2_y = (lost["bbox"][1] + lost["bbox"][3]) / 2.0
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

    def _extract_person_crop(
        self,
        frame: np.ndarray,
        body_bbox: Tuple[int, int, int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Recorta com máxima prioridade o ROSTO da pessoa com margem anatômica completa (cabelo, queixo e orelhas).
        Se a pessoa estiver de costas ou o rosto não estiver visível, recorta a porção superior do corpo como fallback.
        """
        h, w = frame.shape[:2]

        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            fw = fx2 - fx1
            fh = fy2 - fy1

            if fw > 10 and fh > 10:
                # Margem confortável para enquadrar cabeça inteira em altíssima definição
                pad_w = int(fw * 0.38)
                pad_h_top = int(fh * 0.45)
                pad_h_bottom = int(fh * 0.35)

                cx1 = max(0, fx1 - pad_w)
                cy1 = max(0, fy1 - pad_h_top)
                cx2 = min(w, fx2 + pad_w)
                cy2 = min(h, fy2 + pad_h_bottom)

                face_crop = frame[cy1:cy2, cx1:cx2]
                if face_crop.size > 0 and face_crop.shape[0] > 15 and face_crop.shape[1] > 15:
                    return face_crop

        # Fallback (pessoa de costas): recorta os 60% superiores do corpo
        bx1, by1, bx2, by2 = body_bbox
        bw = bx2 - bx1
        bh = by2 - by1
        pad_x = int(bw * 0.12)
        top_y2 = min(h, by1 + int(bh * 0.65))

        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - 10)
        cx2 = min(w, bx2 + pad_x)
        cy2 = top_y2

        crop = frame[cy1:cy2, cx1:cx2]
        return crop if crop.size > 0 else frame[by1:by2, bx1:bx2]

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
                    new_track.is_stationary = inherited.get("is_stationary", False)
                    if new_track.is_stationary:
                        new_track.stationary_anchor_pos = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                    new_track.consecutive_frames = settings.min_consecutive_frames  # Já considera estável
                    logger.debug(
                        f"🔄 Herança de Rastro: ID #{tid} herdou estado de #{inherited['tracking_id']} "
                        f"({new_track.identity_name}, estacionado={new_track.is_stationary}) em {self.camera_name}"
                    )

                self.active_tracks[tid] = new_track

            track = self.active_tracks[tid]
            track.update(conf, bbox)

            # Se o objeto estiver estacionado/parado, preserva/atualiza na âncora da vaga
            if track.is_stationary:
                self._update_parked_anchor(track, bbox)

            # --- Identificação Contínua (não desiste nos primeiros frames se o alvo estiver de costas) ---
            should_eval = not track.is_known and (track.eval_count < 3 or track.consecutive_frames % 2 == 0)
            if should_eval:
                track.eval_count += 1
                crop_obj = self._extract_crop(raw_frame, bbox, margin_pct=0.05)

                if entity_type == "person":
                    name, face_conf, is_known, face_bbox = face_recognizer.detect_and_identify_person(raw_frame, bbox)
                    if face_bbox is not None:
                        track.latest_face_bbox = face_bbox
                        # Se já disparou o alerta inicial de costas e o rosto ainda não tinha sido capturado:
                        if track.event_triggered and not track.has_face_captured:
                            track.needs_face_upgrade = True

                    if is_known:
                        was_unknown = not track.is_known
                        had_alerted_unknown = track.event_triggered and "Desconhecido" in track.identity_name
                        track.identity_name = name
                        track.is_known = True
                        track.match_confidence = face_conf
                        logger.info(f"👤 Rosto identificado em {self.camera_name}: {name} ({int(face_conf * 100)}%)")
                        if had_alerted_unknown:
                            track.needs_face_upgrade = True

                elif entity_type == "vehicle":
                    plate = plate_reader.read_plate(crop_obj)
                    if plate:
                        track.license_plate = plate

                    v_name, v_conf, v_known, matched_plate = visual_matcher.identify_vehicle(crop_obj, track.license_plate)
                    if v_known:
                        was_unknown = not track.is_known
                        had_alerted_unknown = track.event_triggered and "Desconhecido" in track.identity_name
                        track.identity_name = v_name
                        track.is_known = True
                        track.match_confidence = v_conf
                        track.license_plate = matched_plate
                        logger.info(f"🚗 Veículo reconhecido em {self.camera_name}: {v_name}")
                        if had_alerted_unknown:
                            track.event_triggered = False
                            track.needs_identity_upgrade = True
                    elif track.license_plate:
                        track.identity_name = f"{class_label} [{track.license_plate}]"

                elif entity_type == "pet":
                    pet_name, pet_conf, pet_known = visual_matcher.identify_pet(crop_obj, class_label)
                    if pet_known:
                        was_unknown = not track.is_known
                        had_alerted_unknown = track.event_triggered and "Desconhecido" in track.identity_name
                        track.identity_name = pet_name
                        track.is_known = True
                        track.match_confidence = pet_conf
                        logger.info(f"🐾 Pet reconhecido em {self.camera_name}: {pet_name} ({int(pet_conf * 100)}%)")
                        if had_alerted_unknown:
                            track.event_triggered = False
                            track.needs_identity_upgrade = True

            # Atualiza informações de identidade no objeto de detecção
            det["person_name"] = track.identity_name
            det["is_known"] = track.is_known
            det["face_confidence"] = track.match_confidence
            det["license_plate"] = track.license_plate
            det["is_stationary"] = track.is_stationary

        # --- Resolução de Conflitos de Identidade Exclusiva em Cena (Anti-Clone) ---
        # Impede que dois veículos ou pessoas físicas distintas recebam o mesmo nome ao mesmo tempo
        claimed_identities: Dict[str, int] = {}
        for det in detections:
            tid = det.get("tracking_id")
            if tid is None:
                continue
            track = self.active_tracks.get(tid)
            if not track or not track.is_known:
                continue

            name = track.identity_name
            if name in claimed_identities:
                other_tid = claimed_identities[name]
                other_track = self.active_tracks[other_tid]

                # Placa confirmada tem prioridade máxima
                if other_track.license_plate and not track.license_plate:
                    winner_tid, loser_track = other_tid, track
                elif track.license_plate and not other_track.license_plate:
                    winner_tid, loser_track = tid, other_track
                    claimed_identities[name] = tid
                elif other_track.match_confidence >= track.match_confidence:
                    winner_tid, loser_track = other_tid, track
                else:
                    winner_tid, loser_track = tid, other_track
                    claimed_identities[name] = tid

                # Reverte o rastro perdedor para Desconhecido
                loser_track.is_known = False
                plate_tag = f" [{loser_track.license_plate}]" if loser_track.license_plate else ""
                loser_track.identity_name = f"{loser_track.class_label} Desconhecido{plate_tag}"
                logger.debug(
                    f"🛡️ Anti-Clone: Conflito resolvido em {self.camera_name}. "
                    f"'{name}' atribuído exclusivamente ao ID #{winner_tid}. ID #{loser_track.tracking_id} revertido para desconhecido."
                )
            else:
                claimed_identities[name] = tid

        # Sincroniza as informações definitivas nas detecções para anotação e eventos
        for det in detections:
            tid = det.get("tracking_id")
            track = self.active_tracks.get(tid)
            if track:
                det["person_name"] = track.identity_name
                det["is_known"] = track.is_known
                det["face_confidence"] = track.match_confidence
                det["license_plate"] = track.license_plate
                det["is_stationary"] = track.is_stationary

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

            # Se o objeto estiver estacionado/parado e já tiver alertado,
            # suprime QUALQUER novo alerta enquanto continuar na vaga!
            if track.is_stationary and track.event_triggered:
                continue

            # Chave de Cooldown por câmera + tipo + identidade
            cooldown_key = f"{self.camera_id}_{track.entity_type}_{track.identity_name}"
            last_alert = self.last_alert_time.get(cooldown_key, 0.0)
            elapsed_since_last_alert = now - last_alert

            is_upgrade = getattr(track, "needs_identity_upgrade", False)

            # Critérios para evitar repetições sem perder alvos rápidos:
            # 1. Pessoas ou Veículos: >= settings.min_consecutive_frames (2 frames)
            # 2. Pets rápidos (cães/gatos passando em corrida): se confiança >= 0.45, dispara imediatamente
            is_fast_pet = (track.entity_type == "pet" and det["confidence"] >= 0.45)
            is_stable = (track.consecutive_frames >= settings.min_consecutive_frames) or is_fast_pet

            # Caso 1: Novo Alerta Inicial
            if is_stable and not track.event_triggered:
                track.event_triggered = True
                track.needs_face_upgrade = False
                if track.is_stationary:
                    self._update_parked_anchor(track, det["bbox"])

                if not is_upgrade and elapsed_since_last_alert < settings.alert_cooldown_seconds:
                    logger.debug(
                        f"⏳ Alerta suprimido por cooldown ({int(elapsed_since_last_alert)}s / {settings.alert_cooldown_seconds}s): "
                        f"{self.camera_name} - {track.identity_name}"
                    )
                    continue

                # Registra o novo alerta e atualiza timestamp de cooldown
                self.last_alert_time[cooldown_key] = now

                event_id = f"evt_{uuid.uuid4().hex[:12]}"
                track.active_event_id = event_id
                timestamp_str = datetime.now().isoformat()
                date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

                events_dir = Path(settings.events_dir)
                events_dir.mkdir(parents=True, exist_ok=True)

                tag_clean = track.identity_name.replace(" ", "_").replace("[", "").replace("]", "")
                crop_filename = f"{date_prefix}_{self.camera_id}_{track.entity_type}_{tag_clean}_id{tid}_crop.jpg"
                full_filename = f"{date_prefix}_{self.camera_id}_{track.entity_type}_{tag_clean}_id{tid}_full.jpg"
                web_filename = f"{date_prefix}_{self.camera_id}_{track.entity_type}_{tag_clean}_id{tid}_full_web.jpg"

                crop_path = str(events_dir / crop_filename)
                full_path = str(events_dir / full_filename)
                web_path = str(events_dir / web_filename)

                track.event_crop_path = crop_path
                track.event_full_path = full_path
                track.event_web_path = web_path

                # Estratégia de prioridade máxima do print:
                # Para pessoas: se houver rosto detectado, o CROP É O ROSTO!
                if track.entity_type == "person":
                    crop_img = self._extract_person_crop(raw_frame, det["bbox"], face_bbox=track.latest_face_bbox)
                    track.has_face_captured = (track.latest_face_bbox is not None)
                else:
                    crop_img = self._extract_crop(raw_frame, track.best_bbox or det["bbox"], margin_pct=0.10)

                # Salva recortes de evidência em background assíncrono (zero impacto no frame loop)
                asyncio.create_task(
                    asyncio.to_thread(_save_evidence_disk, crop_path, crop_img, full_path, annotated_frame.copy(), web_path)
                )

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
                face_status = "com foto nítida do rosto" if track.has_face_captured else "visto de costas (aguardando rosto)"
                logger.info(
                    f"{icon} Alerta registrado: {self.camera_name} - {track.identity_name} "
                    f"(ID #{tid}) [{face_status}] -> {crop_filename}"
                )

                triggered_events.append(event_record)

                if self.event_callback:
                    try:
                        await self.event_callback(event_record)
                    except Exception as err:
                        logger.error(f"Erro no callback de evento: {err}")

            # Caso 2: Upgrade de Evidência com Rosto (pessoa que foi vista inicialmente de costas e agora mostrou o rosto)
            elif (
                track.entity_type == "person"
                and track.event_triggered
                and track.active_event_id
                and track.latest_face_bbox is not None
                and (track.needs_face_upgrade or not track.has_face_captured)
            ):
                track.needs_face_upgrade = False
                track.has_face_captured = True

                # Recorta o rosto nítido em close
                face_crop_img = self._extract_person_crop(raw_frame, det["bbox"], face_bbox=track.latest_face_bbox)

                # Sobrescreve as imagens em disco com o momento em que o rosto está de frente
                if track.event_crop_path and track.event_full_path and track.event_web_path:
                    asyncio.create_task(
                        asyncio.to_thread(
                            _save_evidence_disk,
                            track.event_crop_path,
                            face_crop_img,
                            track.event_full_path,
                            annotated_frame.copy(),
                            track.event_web_path
                        )
                    )

                # Atualiza registro correspondente no SQLite
                await db.update_event(track.active_event_id, {
                    "confidence": round(det["confidence"], 3),
                    "person_name": track.identity_name,
                    "is_known": 1 if track.is_known else 0,
                    "face_confidence": track.match_confidence
                })

                logger.info(
                    f"📸 [Upgrade de Evidência] Rosto capturado para ID #{tid} ({track.identity_name}) em {self.camera_name}! "
                    f"Recorte de rosto e cena atualizados."
                )

                # Notifica a interface web para trocar o card de costas pela foto do rosto em tempo real
                updated_record = {
                    "_msg_type": "EVENT_UPDATED",
                    "event_id": track.active_event_id,
                    "camera_id": self.camera_id,
                    "camera_name": self.camera_name,
                    "tracking_id": tid,
                    "confidence": round(det["confidence"], 3),
                    "person_name": track.identity_name,
                    "is_known": track.is_known,
                    "face_confidence": track.match_confidence,
                    "entity_type": track.entity_type
                }
                if self.event_callback:
                    try:
                        await self.event_callback(updated_record)
                    except Exception as err:
                        logger.debug(f"Erro no callback de upgrade: {err}")

        return annotated_frame, triggered_events
