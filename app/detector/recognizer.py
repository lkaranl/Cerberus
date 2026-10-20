import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

class FaceRecognizer:
    """
    Módulo de Reconhecimento Facial baseado em Embeddings de Redes Neurais Profundas.
    Utiliza FaceNet (Inception-ResNet) pré-treinada para extrair vetores faciais (512-d)
    e comparar via similaridade de cosseno com as fotos cadastradas em data/known_faces/.
    """
    def __init__(self, known_faces_dir: str = "data/known_faces", similarity_threshold: float = 0.65):
        self.known_faces_dir = Path(known_faces_dir)
        self.similarity_threshold = similarity_threshold
        self.device = settings.resolve_device()
        self.known_embeddings: Dict[str, Any] = {} # nome -> tensor de embedding médio
        self.is_ready = False
        self.using_tensorrt = False

        self.mtcnn = None
        self.resnet = None
        self.torch = None
        self.trt_runtime = None
        self.trt_engine = None
        self.trt_context = None

    def initialize(self) -> bool:
        """Inicializa os modelos de detecção e extração facial em GPU/CPU com suporte a TensorRT."""
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.torch = torch

            logger.info(f"Carregando FaceNet (InceptionResnetV1) no dispositivo: {self.device}...")
            # MTCNN com limiares rigorosos para garantir detecção exclusiva de rostos reais
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=28,
                thresholds=[0.65, 0.75, 0.80],
                keep_all=False,
                select_largest=True,
                device=self.device
            )
            # Modelo PyTorch de extração de embeddings (512 dimensões)
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.is_ready = True
            self.cuda_stream = self.torch.cuda.Stream(device=self.device) if self.device.startswith("cuda") else None

            # Tenta carregar a engine TensorRT compilada (FP16 nativa)
            engine_path = Path(settings.facenet_engine_path)
            if settings.use_tensorrt and engine_path.exists() and self.device.startswith("cuda"):
                try:
                    import tensorrt as trt
                    logger.info(f"⚡ Carregando FaceNet com aceleração TensorRT FP16 [{engine_path}]...")
                    self.trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                    with open(engine_path, "rb") as f:
                        self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
                    self.trt_context = self.trt_engine.create_execution_context()
                    self.using_tensorrt = True
                    logger.info("🚀 FaceNet TensorRT FP16 carregado com sucesso na GPU GTX 1660.")
                except Exception as trt_err:
                    logger.warning(f"Falha ao inicializar FaceNet TensorRT ({trt_err}). Revertendo para PyTorch CUDA.")
                    self.using_tensorrt = False
            elif self.device.startswith("cuda") and settings.use_tensorrt and not engine_path.exists():
                logger.info(
                    "💡 Dica de Desempenho FaceNet: Para acelerar o reconhecimento facial para ~2ms, "
                    "execute no servidor: python scripts/export_facenet_tensorrt.py"
                )

            self.load_known_faces()
            return True
        except ImportError:
            logger.warning(
                "facenet-pytorch não instalado. Reconhecimento facial operando em modo fallback "
                "(identificará todas as detecções como 'Desconhecido')."
            )
            self.is_ready = False
            return False
        except Exception as e:
            logger.error(f"Erro ao inicializar FaceRecognizer: {e}")
            self.is_ready = False
            return False

    def load_known_faces(self) -> int:
        """
        Varre as subpastas de data/known_faces/<Nome_da_Pessoa>/,
        extrai os embeddings faciais e armazena a média vetorial por pessoa.
        """
        if not self.is_ready or not self.known_faces_dir.exists():
            return 0

        self.known_embeddings.clear()
        person_count = 0
        total_photos = 0

        logger.info(f"Escaneando rostos conhecidos em: {self.known_faces_dir}")

        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue

            person_name = person_dir.name
            embeddings_list = []

            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    face_tensor = self.mtcnn(img_rgb)

                    if face_tensor is not None:
                        with self.torch.no_grad():
                            face_batch = face_tensor.unsqueeze(0).to(self.device)
                            emb = self._extract_embedding(face_batch)
                            # Normaliza o vetor para comparação de cosseno
                            emb_norm = self.torch.nn.functional.normalize(emb, p=2, dim=1)
                            embeddings_list.append(emb_norm.squeeze(0))
                            total_photos += 1
                except Exception as err:
                    logger.warning(f"Erro ao processar imagem {img_path}: {err}")

            if embeddings_list:
                # Calcula o vetor médio representativo dessa pessoa
                mean_emb = self.torch.stack(embeddings_list).mean(dim=0)
                mean_emb_norm = self.torch.nn.functional.normalize(mean_emb, p=2, dim=0)
                self.known_embeddings[person_name] = mean_emb_norm
                person_count += 1
                logger.info(f"✅ Pessoa cadastrada: {person_name} ({len(embeddings_list)} fotos processadas)")

        logger.info(f"Carregamento concluído: {person_count} pessoas ({total_photos} fotos).")
        return person_count

    def _extract_embedding(self, face_batch: Any) -> Any:
        """
        Extrai o embedding facial de 512 dimensões.
        Utiliza TensorRT FP16 com ponteiros diretos de VRAM (zero-copy) quando disponível,
        ou PyTorch InceptionResnetV1 como fallback transparente.
        """
        if self.using_tensorrt and self.trt_context is not None:
            try:
                batch_size = face_batch.shape[0]
                output_tensor = self.torch.empty((batch_size, 512), dtype=self.torch.float32, device=self.device)
                self.trt_context.set_input_shape("input", face_batch.shape)
                self.trt_context.set_tensor_address("input", face_batch.data_ptr())
                self.trt_context.set_tensor_address("embedding", output_tensor.data_ptr())
                stream_ptr = self.cuda_stream.cuda_stream if self.cuda_stream else self.torch.cuda.current_stream().cuda_stream
                self.trt_context.execute_async_v3(stream_ptr)
                if self.cuda_stream:
                    self.cuda_stream.synchronize()
                return output_tensor
            except Exception as trt_err:
                logger.debug(f"Falha na inferência FaceNet TensorRT ({trt_err}), usando PyTorch fallback.")

        return self.resnet(face_batch)

    def detect_and_identify_person(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, float, bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detecta a localização exata do rosto (face_bbox) e identifica a pessoa via FaceNet.
        Retorna: (nome_da_pessoa, confianca_similaridade, is_conhecido, face_bbox_global)
        Se a pessoa estiver de costas ou o rosto não estiver visível, face_bbox_global será None.
        """
        if not self.is_ready:
            return "Desconhecido", 0.0, False, None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        # Foca estritamente nos 42% superiores da bounding box (onde fica a cabeça/pescoço)
        box_h = y2 - y1
        head_y2 = min(y2, y1 + int(box_h * 0.42))

        # Margem lateral para compensar inclinação natural ao caminhar
        box_w = x2 - x1
        pad_x = int(box_w * 0.12)
        crop_x1 = max(0, x1 - pad_x)
        crop_x2 = min(w, x2 + pad_x)
        crop_y1 = max(0, y1 - 10)
        crop_y2 = min(h, head_y2 + 10)

        head_crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if head_crop_bgr.size == 0 or head_crop_bgr.shape[0] < 20 or head_crop_bgr.shape[1] < 20:
            return "Desconhecido", 0.0, False, None

        # Pré-downscale adaptativo do recorte (max 320px) preservando nitidez
        ch, cw = head_crop_bgr.shape[:2]
        max_dim = 320
        scale = 1.0
        if max(ch, cw) > max_dim:
            scale = max_dim / max(ch, cw)
            head_crop_bgr = cv2.resize(head_crop_bgr, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_LINEAR)

        try:
            head_crop_rgb = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2RGB)
            face_bbox_global: Optional[Tuple[int, int, int, int]] = None

            # Detecta coordenadas precisas do rosto com alta taxa de certeza
            boxes, probs = self.mtcnn.detect(head_crop_rgb)
            if boxes is not None and len(boxes) > 0 and probs[0] is not None:
                prob = float(probs[0])
                # Limiar rigoroso: probabilidade mínima de 88% de ser um rosto humano real
                if prob >= 0.88:
                    bx1, by1, bx2, by2 = boxes[0]
                    inv_scale = 1.0 / scale
                    fx1 = max(0, int(crop_x1 + (bx1 * inv_scale)))
                    fy1 = max(0, int(crop_y1 + (by1 * inv_scale)))
                    fx2 = min(w, int(crop_x1 + (bx2 * inv_scale)))
                    fy2 = min(h, int(crop_y1 + (by2 * inv_scale)))

                    fw = fx2 - fx1
                    fh = fy2 - fy1

                    # Filtro Anatômico Rígido Anti-Braço / Anti-Corpo:
                    # 1. Dimensões mínimas: rosto não pode ser minúsculo (< 22px)
                    # 2. Aspect Ratio: proporção largura/altura deve ser compatível com rosto humano (0.65 a 1.40)
                    # 3. Posição corporal: o centro do rosto DEVE estar no terço superior do corpo
                    face_center_y = (fy1 + fy2) / 2.0
                    body_top_limit = y1 + (box_h * 0.40)

                    if (
                        fw >= 22
                        and fh >= 22
                        and 0.65 <= (fw / float(fh)) <= 1.40
                        and face_center_y <= body_top_limit
                    ):
                        face_bbox_global = (fx1, fy1, fx2, fy2)

            # Se nenhum rosto autêntico foi detectado (pessoa de costas, nuca, braço ou corpo)
            if face_bbox_global is None:
                return "Desconhecido", 0.0, False, None

            # 2. Se não houver rostos conhecidos cadastrados, retorna que o rosto foi detectado como Desconhecido
            if not self.known_embeddings:
                return "Desconhecido", 0.0, False, face_bbox_global

            # 3. Extrai embedding e compara com a base conhecida
            face_tensor = self.mtcnn(head_crop_rgb)
            if face_tensor is None:
                # Tenta recortar diretamente a partir da face_bbox encontrada
                fx1, fy1, fx2, fy2 = face_bbox_global
                direct_face_rgb = cv2.cvtColor(frame_bgr[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2RGB)
                if direct_face_rgb.size > 0:
                    face_tensor = self.mtcnn(direct_face_rgb)

            if face_tensor is None:
                return "Desconhecido", 0.0, False, face_bbox_global

            with self.torch.no_grad():
                face_batch = face_tensor.unsqueeze(0).to(self.device)
                probe_emb = self._extract_embedding(face_batch)
                probe_norm = self.torch.nn.functional.normalize(probe_emb, p=2, dim=1).squeeze(0)

                best_name = "Desconhecido"
                best_sim = 0.0

                for name, known_emb in self.known_embeddings.items():
                    sim = self.torch.dot(probe_norm, known_emb).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

                if best_sim >= self.similarity_threshold:
                    return best_name, round(best_sim, 3), True, face_bbox_global
                else:
                    return "Desconhecido", round(best_sim, 3), False, face_bbox_global

        except Exception as e:
            logger.debug(f"Falha ao identificar rosto: {e}")
            return "Desconhecido", 0.0, False, None

    def identify_person(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float, bool]:
        """Wrapper de compatibilidade retroativa que retorna apenas (nome, confiança, is_conhecido)."""
        name, conf, is_known, _ = self.detect_and_identify_person(frame_bgr, bbox)
        return name, conf, is_known


face_recognizer = FaceRecognizer()
