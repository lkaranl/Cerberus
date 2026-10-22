import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.ai.face_super_res import face_super_resolver

class FaceRecognizer:
    """
    Módulo Avançado de Reconhecimento Facial de Alta Precisão (SOTA).
    
    Arquitetura de Elite:
    - Motor Primário: InsightFace (SCRFD para detecção 640x640 + ArcFace ResNet-50 512-d)
      com aceleração nativa via ONNX Runtime GPU (CUDA/TensorRT).
    - Galeria Multi-Embedding (Multi-View Vector Gallery): armazena todos os vetores biométricos
      válidos de cada morador, eliminando a diluição da média simples.
    - Face Quality Assessment (FQA): filtros estritos de resolução (>=48px) e nitidez Laplaciana.
    - Motor de Retaguarda: FaceNet como fallback automático garantindo disponibilidade contínua.
    """
    def __init__(self, known_faces_dir: str = "data/known_faces"):
        self.known_faces_dir = Path(known_faces_dir)
        self.device = settings.resolve_device()
        self.known_gallery: Dict[str, List[np.ndarray]] = {}
        self.known_embeddings: Dict[str, Any] = {}
        self.is_ready = False
        self.engine_name = "Nenhum"

        # InsightFace
        self.insight_app = None
        self.using_insightface = False

        # FaceNet (Fallback)
        self.mtcnn = None
        self.resnet = None
        self.torch = None
        self.using_tensorrt = False
        self.trt_context = None
        self.cuda_stream = None

    def initialize(self) -> bool:
        """Inicializa o InsightFace na GPU ou faz fallback gracioso para FaceNet."""
        try:
            from insightface.app import FaceAnalysis
            logger.info("⚡ Inicializando InsightFace SOTA (ArcFace ResNet / SCRFD 640x640)...")
            
            # InsightFace opera com máxima estabilidade e performance em CUDA nativo
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            ctx_id = 0 if self.device.startswith("cuda") else -1
            app = None

            # 1. Tenta modelo de alta capacidade (buffalo_l: ArcFace ResNet-50 w600k_r50)
            target_model = settings.face_model_name or "buffalo_l"
            try:
                logger.info(f"Carregando pacote biométrico [{target_model}] com det_size=(640, 640)...")
                app = FaceAnalysis(name=target_model, providers=providers)
                app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.35)
                self.engine_name = f"InsightFace ArcFace [{target_model}] (CUDA/SCRFD 640x640) 🚀"
            except Exception as load_err:
                logger.warning(f"Pacote {target_model} indisponível ({load_err}). Tentando buffalo_s...")
                app = FaceAnalysis(name="buffalo_s", providers=providers)
                app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.35)
                self.engine_name = "InsightFace ArcFace [buffalo_s] (CUDA/SCRFD 640x640) 🚀"

            self.insight_app = app
            self.using_insightface = True
            self.is_ready = True
            logger.info(f"🚀 {self.engine_name} inicializado com sucesso!")
        except Exception as err:
            logger.warning(f"InsightFace indisponível ({err}). Carregando FaceNet como fallback...")
            self.using_insightface = False

        # 2. Se o InsightFace não estiver disponível, inicializa o FaceNet
        if not self.using_insightface:
            self._init_facenet_fallback()

        self.load_known_faces()
        return self.is_ready

    def _init_facenet_fallback(self) -> None:
        """Inicializa o modelo clássico FaceNet caso o InsightFace não esteja instalado."""
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.torch = torch

            logger.info(f"Carregando FaceNet (InceptionResnetV1) no dispositivo: {self.device}...")
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=28,
                thresholds=[0.65, 0.75, 0.80],
                keep_all=False,
                select_largest=True,
                device=self.device
            )
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.is_ready = True
            self.engine_name = "FaceNet PyTorch"
            self.cuda_stream = self.torch.cuda.Stream(device=self.device) if self.device.startswith("cuda") else None

            # Tenta TensorRT para FaceNet
            engine_path = Path(settings.facenet_engine_path)
            if settings.use_tensorrt and engine_path.exists() and self.device.startswith("cuda"):
                try:
                    import tensorrt as trt
                    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                    with open(engine_path, "rb") as f:
                        trt_engine = trt_runtime.deserialize_cuda_engine(f.read())
                    self.trt_context = trt_engine.create_execution_context()
                    self.using_tensorrt = True
                    self.engine_name = "FaceNet TensorRT FP16 ⚡"
                except Exception:
                    self.using_tensorrt = False
        except Exception as e:
            logger.error(f"Falha ao carregar FaceNet fallback: {e}")
            self.is_ready = False

    @property
    def app(self):
        """Atalho de conveniência para acessar a instância FaceAnalysis."""
        return self.insight_app

    @property
    def gallery_vector_count(self) -> int:
        """Número total de vetores biométricos registrados em memória."""
        return sum(len(vecs) for vecs in self.known_gallery.values())

    def load_known_faces(self) -> int:
        """
        Escaneia as pastas em data/known_faces/ e gera a Galeria Multi-Embedding (Multi-View).
        Cada foto válida do morador é indexada individualmente para máxima fidelidade métrica.
        """
        self.known_gallery.clear()
        self.known_embeddings.clear()
        if not self.is_ready:
            return 0

        target_dir = self.known_faces_dir if self.known_faces_dir.is_absolute() else self.known_faces_dir.resolve()
        if not target_dir.exists():
            logger.warning(f"Diretório de moradores não encontrado: {target_dir}")
            return 0

        logger.info(f"Escaneando rostos conhecidos via {self.engine_name} em: {target_dir}")
        person_count = 0
        total_photos = 0

        for person_dir in sorted(target_dir.iterdir(), key=lambda p: p.name.lower()):
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue

            person_name = person_dir.name.replace("_", " ").strip()
            embeddings_list = []

            for img_path in sorted(person_dir.iterdir(), key=lambda p: p.name.lower()):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                raw_bgr = cv2.imread(str(img_path))
                if raw_bgr is None:
                    logger.warning(f"Não foi possível decodificar foto: {img_path.name}")
                    continue

                # Redimensiona fotos gigantes de celular para ~1280px para otimizar o detector SCRFD
                max_dim = max(raw_bgr.shape[:2])
                if max_dim > 1600:
                    scale = 1280.0 / max_dim
                    raw_bgr = cv2.resize(raw_bgr, (int(raw_bgr.shape[1] * scale), int(raw_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                emb = self._extract_probe_embedding(raw_bgr)
                if emb is not None:
                    embeddings_list.append(emb)
                    total_photos += 1
                else:
                    logger.warning(f"Face não detectada na foto de cadastro [{person_name}]: {img_path.name}")

            if embeddings_list:
                # Galeria Multi-Embedding pura (SOTA): preserva todos os vetores da pessoa
                self.known_gallery[person_name] = embeddings_list
                # Centróide mantido como espelho para rotas informativas da API
                mean_emb = np.mean(embeddings_list, axis=0)
                norm = np.linalg.norm(mean_emb)
                self.known_embeddings[person_name] = (mean_emb / norm) if norm > 0 else mean_emb

                person_count += 1
                logger.info(f"✅ Morador cadastrado: {person_name} ({len(embeddings_list)} vetores biométricos ativos)")

        logger.info(f"Carregamento biométrico concluído: {person_count} pessoas ({total_photos} fotos processadas).")
        return person_count

    @staticmethod
    def enhance_shadow_face(img_bgr: np.ndarray, clip_limit: float = 3.0, gamma: float = 0.60) -> np.ndarray:
        """
        Módulo 1 SOTA: Realce Fotométrico Adaptativo (P&B + CLAHE & Shadow Lifting)
        - Isola o canal de luminância pura (LAB L) e elimina artefatos cromáticos de compressão H.264/H.265.
        - Aplica CLAHE adaptativo para redistribuir iluminação sem estourar highlights.
        - Aplica Gamma Correction adaptativo para elevar sombras profundas sob marquises.
        - Retorna imagem de 3 canais cinza calibrada para detecção e reconhecimento biométrico.
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        try:
            if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
                work = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            elif img_bgr.ndim == 2:
                work = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            else:
                work = img_bgr.copy()

            # Espaço LAB para isolar Luminância de Croma
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)

            # CLAHE adaptativo no canal L
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l_chan)

            # Gamma Lifting para sombras profundas
            inv_gamma = 1.0 / max(0.1, gamma)
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            cl_gamma = cv2.LUT(cl, table)

            # Recompõe em 3 canais P&B limpos e de alto contraste
            enhanced_bgr = cv2.cvtColor(cl_gamma, cv2.COLOR_GRAY2BGR)
            return enhanced_bgr
        except Exception:
            return img_bgr

    def _check_face_quality(self, face_crop: np.ndarray, fw: int, fh: int) -> Tuple[bool, str]:
        """
        Face Quality Assessment (FQA):
        Rejeita alucinações biométricas baseadas em ruídos extremos.
        """
        # 1. Filtro de Resolução Real (>= settings.face_min_size pixels)
        min_sz = getattr(settings, "face_min_size", 20)
        if fw < min_sz or fh < min_sz:
            return False, f"Resolução insuficiente ({fw}x{fh} < {min_sz}px)"

        # 2. Proporção Anatômica Humana (flexível para ângulos inclinados e poses de CFTV)
        aspect = fw / float(fh)
        if aspect < 0.40 or aspect > 2.20:
            return False, f"Proporção anatômica inválida ({aspect:.2f})"

        # 3. Motion Blur Gate (Filtro de Nitidez Laplaciana suave)
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            min_blur = getattr(settings, "face_quality_min_blur", 6.0)
            if lap_var < min_blur:
                return False, f"Borrão de movimento (nitidez={lap_var:.1f} < {min_blur})"
        except Exception:
            return False, "Erro ao calcular nitidez"

        return True, "Aprovado"

    def _extract_probe_embedding(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extrai o embedding normalizado de uma imagem (via InsightFace ou FaceNet)."""
        if img_bgr is None or img_bgr.size == 0:
            return None

        # Garante 3 canais BGR (converte RGBA de PNGs ou monocromático)
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        elif img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        if self.using_insightface and self.insight_app is not None:
            try:
                faces = self.insight_app.get(img_bgr)
                # Se não detectou faces na imagem original (típico de recortes muito fechados sem contexto de ombros),
                # adiciona borda reflexiva progressiva e tenta ângulos de rotação (TTA) para selfies inclinadas
                if not faces or len(faces) == 0:
                    h, w = img_bgr.shape[:2]
                    for pad_factor in [0.25, 0.45]:
                        pad_y = max(32, int(h * pad_factor))
                        pad_x = max(32, int(w * pad_factor))
                        padded_bgr = cv2.copyMakeBorder(
                            img_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101
                        )
                        faces = self.insight_app.get(padded_bgr)
                        if faces and len(faces) > 0:
                            break

                        # Tenta compensar inclinação da cabeça (roll de selfies)
                        center = (padded_bgr.shape[1] // 2, padded_bgr.shape[0] // 2)
                        for angle in [-20.0, 20.0, -12.0, 12.0, -30.0, 30.0]:
                            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                            rotated = cv2.warpAffine(
                                padded_bgr, rot_mat, (padded_bgr.shape[1], padded_bgr.shape[0]),
                                borderMode=cv2.BORDER_REFLECT_101
                            )
                            faces = self.insight_app.get(rotated)
                            if faces and len(faces) > 0:
                                break
                        if faces and len(faces) > 0:
                            break

                # Se ainda não detectou (ex: foto já recortada somente da face pelo usuário),
                # extrai embedding direto via recognition model (ArcFace) em 112x112
                if not faces or len(faces) == 0:
                    rec_model = self.insight_app.models.get('recognition') if self.insight_app else None
                    if rec_model is not None:
                        try:
                            aligned_112 = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_LANCZOS4)
                            emb = rec_model.get_feat(aligned_112).flatten().astype(np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                return emb / norm
                        except Exception:
                            pass

                if faces and len(faces) > 0:
                    best_face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
                    if getattr(best_face, "embedding", None) is not None:
                        emb = best_face.embedding.astype(np.float32)
                        norm = np.linalg.norm(emb)
                        return (emb / norm) if norm > 0 else emb
            except Exception as e:
                logger.debug(f"Erro na extração InsightFace: {e}")

        # Fallback FaceNet
        if self.mtcnn is not None and self.resnet is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                face_tensor = self.mtcnn(img_rgb)
                if face_tensor is not None:
                    with self.torch.no_grad():
                        face_batch = face_tensor.unsqueeze(0).to(self.device)
                        emb = self.resnet(face_batch)
                        emb_norm = self.torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0)
                        return emb_norm.cpu().numpy().astype(np.float32)
            except Exception:
                pass
        return None

    def detect_and_identify_person(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, float, bool, Optional[Tuple[int, int, int, int]]]:
        """
        Localiza o rosto com alta tolerância a ângulos difíceis (SCRFD 640x640),
        valida qualidade biométrica (FQA) e identifica via Galeria Multi-Embedding ArcFace.
        Retorna: (nome_da_pessoa, confianca_similaridade, is_conhecido, face_bbox_global)
        """
        if not self.is_ready:
            return "Desconhecido", 0.0, False, None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        box_h = y2 - y1
        box_w = x2 - x1

        # Região da cabeça/busto (70% superiores da pessoa)
        head_y2 = min(y2, y1 + int(box_h * 0.70))
        pad_x = int(box_w * 0.20)
        crop_x1 = max(0, x1 - pad_x)
        crop_x2 = min(w, x2 + pad_x)
        crop_y1 = max(0, y1 - 15)
        crop_y2 = min(h, head_y2 + 15)

        head_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if head_crop.size == 0 or head_crop.shape[0] < 16 or head_crop.shape[1] < 16:
            return "Desconhecido", 0.0, False, None

        # =========================================================================
        # 1. MOTOR INSIGHTFACE (ARCFACE SOTA + SCRFD 640x640)
        # =========================================================================
        if self.using_insightface and self.insight_app is not None:
            try:
                faces = self.insight_app.get(head_crop)
                if not faces:
                    # Se não detectou no corte restrito, tenta na bounding box completa da pessoa
                    body_crop = frame_bgr[y1:y2, x1:x2]
                    if body_crop.size > 0:
                        faces = self.insight_app.get(body_crop)
                        crop_x1, crop_y1 = x1, y1

                # [MÓDULO 1 SOTA] DUAL-PASS SHADOW LIFTING:
                # Se não detectou face devido a sombras profundas ou contraluz severa,
                # aplica realce fotométrico adaptativo (P&B + CLAHE + Gamma) e repete a busca!
                if not faces and settings.face_shadow_lifting_enabled:
                    enhanced_head = self.enhance_shadow_face(
                        head_crop,
                        clip_limit=settings.face_shadow_clip_limit,
                        gamma=settings.face_shadow_gamma
                    )
                    faces = self.insight_app.get(enhanced_head)
                    if not faces:
                        body_crop = frame_bgr[y1:y2, x1:x2]
                        if body_crop.size > 0:
                            enhanced_body = self.enhance_shadow_face(
                                body_crop,
                                clip_limit=settings.face_shadow_clip_limit,
                                gamma=settings.face_shadow_gamma
                            )
                            faces = self.insight_app.get(enhanced_body)
                            crop_x1, crop_y1 = x1, y1

                if not faces:
                    return "Desconhecido", 0.0, False, None

                # Avalia cada face com Face Quality Assessment (FQA)
                best_face = None
                face_bbox_global = None

                for face in faces:
                    if getattr(face, "det_score", 0.0) < 0.30:
                        continue
                    bx1, by1, bx2, by2 = face.bbox
                    fx1 = max(0, int(crop_x1 + bx1))
                    fy1 = max(0, int(crop_y1 + by1))
                    fx2 = min(w, int(crop_x1 + bx2))
                    fy2 = min(h, int(crop_y1 + by2))

                    fw = fx2 - fx1
                    fh = fy2 - fy1

                    face_crop_img = frame_bgr[fy1:fy2, fx1:fx2]
                    if face_crop_img.size == 0:
                        continue

                    # Filtro FQA de Resolução e Nitidez
                    is_valid, reason = self._check_face_quality(face_crop_img, fw, fh)
                    if not is_valid:
                        logger.debug(f"Face descartada pelo FQA: {reason}")
                        continue

                    face_center_y = (fy1 + fy2) / 2.0
                    body_top_limit = y1 + (box_h * 0.75)
                    if face_center_y <= body_top_limit:
                        best_face = face
                        face_bbox_global = (fx1, fy1, fx2, fy2)
                        break

                if best_face is None or face_bbox_global is None:
                    return "Desconhecido", 0.0, False, None

                if not self.known_gallery:
                    return "Desconhecido", 0.0, False, face_bbox_global

                # Vetor biométrico probe 512-d L2-normalizado
                probe_emb = best_face.embedding.astype(np.float32)

                # [MÓDULO 2 SOTA] SUPER-RESOLUÇÃO FACIAL CONDICIONAL:
                # Se a face for distante/pequena (< settings.face_super_res_threshold px),
                # reconstrói a alta frequência anatômica e re-extrai os 512 números via ArcFace
                if settings.face_super_res_enabled and (fw < settings.face_super_res_threshold or fh < settings.face_super_res_threshold):
                    rec_model = self.insight_app.models.get('recognition') if self.insight_app else None
                    if rec_model is not None:
                        try:
                            enhanced_face = face_super_resolver.enhance(face_crop_img, target_size=(112, 112))
                            sr_emb = rec_model.get_feat(enhanced_face).flatten().astype(np.float32)
                            sr_norm = np.linalg.norm(sr_emb)
                            if sr_norm > 0:
                                probe_emb = sr_emb / sr_norm
                                logger.debug(f"🔍 Face distante ({fw}x{fh}px) enriquecida via Super-Resolução")
                        except Exception as sr_err:
                            logger.debug(f"Falha ao re-extrair com Super-Resolução: {sr_err}")

                p_norm = np.linalg.norm(probe_emb)
                if p_norm > 0:
                    probe_emb = probe_emb / p_norm

                best_name = "Desconhecido"
                best_sim = 0.0

                # Busca por similaridade máxima k-NN contra todos os vetores cadastrados
                for name, gallery in self.known_gallery.items():
                    if not gallery:
                        continue
                    sims = [float(np.dot(probe_emb, ref_emb)) for ref_emb in gallery]
                    max_sim = max(sims)
                    if max_sim > best_sim:
                        best_sim = max_sim
                        best_name = name

                # Decisão calibrada contra limiar rigoroso (padrão 0.58)
                is_known = (best_sim >= settings.face_confidence_threshold)
                final_name = best_name if is_known else "Desconhecido"
                return final_name, round(best_sim, 3), is_known, face_bbox_global

            except Exception as e:
                logger.debug(f"Falha na inferência InsightFace: {e}")

        # =========================================================================
        # 2. MOTOR FACENET (FALLBACK)
        # =========================================================================
        return self._facenet_fallback_detect(frame_bgr, head_crop, crop_x1, crop_y1, y1, box_h, w, h)

    def _facenet_fallback_detect(
        self,
        frame_bgr: np.ndarray,
        head_crop_bgr: np.ndarray,
        crop_x1: int,
        crop_y1: int,
        y1: int,
        box_h: int,
        w: int,
        h: int
    ) -> Tuple[str, float, bool, Optional[Tuple[int, int, int, int]]]:
        if not self.mtcnn or not self.resnet:
            return "Desconhecido", 0.0, False, None

        try:
            head_crop_rgb = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(head_crop_rgb)
            if boxes is None or len(boxes) == 0 or probs[0] is None or probs[0] < 0.88:
                return "Desconhecido", 0.0, False, None

            bx1, by1, bx2, by2 = boxes[0]
            fx1 = max(0, int(crop_x1 + bx1))
            fy1 = max(0, int(crop_y1 + by1))
            fx2 = min(w, int(crop_x1 + bx2))
            fy2 = min(h, int(crop_y1 + by2))

            face_bbox_global = (fx1, fy1, fx2, fy2)
            if not self.known_embeddings:
                return "Desconhecido", 0.0, False, face_bbox_global

            face_tensor = self.mtcnn(head_crop_rgb)
            if face_tensor is None:
                return "Desconhecido", 0.0, False, face_bbox_global

            with self.torch.no_grad():
                face_batch = face_tensor.unsqueeze(0).to(self.device)
                emb = self.resnet(face_batch)
                emb_norm = self.torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()

                best_name = "Desconhecido"
                best_sim = 0.0
                for name, gallery in self.known_gallery.items():
                    if not gallery:
                        continue
                    sims = [float(np.dot(emb_norm, ref_emb)) for ref_emb in gallery]
                    max_sim = max(sims)
                    if max_sim > best_sim:
                        best_sim = max_sim
                        best_name = name

                if best_sim >= settings.face_confidence_threshold:
                    return best_name, round(best_sim, 3), True, face_bbox_global
                return "Desconhecido", round(best_sim, 3), False, face_bbox_global
        except Exception:
            return "Desconhecido", 0.0, False, None

    def identify_person(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float, bool]:
        name, conf, is_known, _ = self.detect_and_identify_person(frame_bgr, bbox)
        return name, conf, is_known

face_recognizer = FaceRecognizer()
