import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

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
            
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if settings.use_tensorrt:
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

            ctx_id = 0 if self.device.startswith("cuda") else -1
            app = None

            # 1. Tenta modelo de alta capacidade (buffalo_l: ArcFace ResNet-50 w600k_r50)
            target_model = settings.face_model_name or "buffalo_l"
            try:
                logger.info(f"Carregando pacote biométrico [{target_model}] com det_size=(640, 640)...")
                app = FaceAnalysis(name=target_model, providers=providers)
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self.engine_name = f"InsightFace ArcFace [{target_model}] (CUDA/SCRFD 640x640) 🚀"
            except Exception as load_err:
                logger.warning(f"Pacote {target_model} indisponível ({load_err}). Tentando buffalo_s...")
                app = FaceAnalysis(name="buffalo_s", providers=providers)
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
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

    def load_known_faces(self) -> int:
        """
        Escaneia as pastas em data/known_faces/ e gera a Galeria Multi-Embedding (Multi-View).
        Cada foto válida do morador é indexada individualmente para máxima fidelidade métrica.
        """
        self.known_gallery.clear()
        self.known_embeddings.clear()
        if not self.is_ready or not self.known_faces_dir.exists():
            return 0

        logger.info(f"Escaneando rostos conhecidos via {self.engine_name} em: {self.known_faces_dir}")
        person_count = 0
        total_photos = 0

        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue

            person_name = person_dir.name.replace("_", " ").strip()
            embeddings_list = []

            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                raw_bgr = cv2.imread(str(img_path))
                if raw_bgr is None:
                    continue

                emb = self._extract_probe_embedding(raw_bgr)
                if emb is not None:
                    embeddings_list.append(emb)
                    total_photos += 1

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

    def _check_face_quality(self, face_crop: np.ndarray, fw: int, fh: int) -> Tuple[bool, str]:
        """
        Face Quality Assessment (FQA):
        Rejeita alucinações biométricas baseadas em rostos minúsculos ou borrados.
        """
        # 1. Filtro de Resolução Real (>= 48x48 pixels)
        if fw < settings.face_min_size or fh < settings.face_min_size:
            return False, f"Resolução insuficiente ({fw}x{fh} < {settings.face_min_size}px)"

        # 2. Proporção Anatômica Humana
        aspect = fw / float(fh)
        if aspect < 0.55 or aspect > 1.55:
            return False, f"Proporção anatômica inválida ({aspect:.2f})"

        # 3. Motion Blur Gate (Filtro de Nitidez Laplaciana)
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap_var < settings.face_quality_min_blur:
                return False, f"Borrão de movimento (nitidez={lap_var:.1f} < {settings.face_quality_min_blur})"
        except Exception:
            return False, "Erro ao calcular nitidez"

        return True, "Aprovado"

    def _extract_probe_embedding(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extrai o embedding normalizado de uma imagem (via InsightFace ou FaceNet)."""
        if self.using_insightface and self.insight_app is not None:
            faces = self.insight_app.get(img_bgr)
            if faces and len(faces) > 0:
                best_face = max(faces, key=lambda f: f.det_score)
                emb = best_face.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                return (emb / norm) if norm > 0 else emb
            return None

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

        # Região da cabeça/busto (65% superiores da pessoa)
        head_y2 = min(y2, y1 + int(box_h * 0.65))
        pad_x = int(box_w * 0.15)
        crop_x1 = max(0, x1 - pad_x)
        crop_x2 = min(w, x2 + pad_x)
        crop_y1 = max(0, y1 - 10)
        crop_y2 = min(h, head_y2 + 10)

        head_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if head_crop.size == 0 or head_crop.shape[0] < settings.face_min_size or head_crop.shape[1] < settings.face_min_size:
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

                if not faces:
                    return "Desconhecido", 0.0, False, None

                # Avalia cada face com Face Quality Assessment (FQA)
                best_face = None
                face_bbox_global = None

                for face in faces:
                    if face.det_score < 0.55:
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
