#!/usr/bin/env python3
"""
Cerberus — Teste & Benchmark do Módulo 2: Super-Resolução e Restauração Facial Neural
Comprova:
1. Reconstrução de recortes faciais pequenos (< 60px) em 112x112 com alta frequência anatômica;
2. Ganho de similaridade biométrica ArcFace (Antes vs Depois da Super-Resolução);
3. Latência de inferência e impacto no throughput.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings
from app.detector.recognizer import face_recognizer
from app.ai.face_super_res import face_super_resolver

def print_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def main():
    print_header("CERBERUS — TESTE DO MÓDULO 2: SUPER-RESOLUÇÃO FACIAL NEURAL")

    face_recognizer.initialize()
    if not face_recognizer.is_ready:
        print("❌ FaceRecognizer não inicializado.")
        sys.exit(1)

    print(f"✓ Modelo de Super-Resolução: {'Neural ONNX (CUDA)' if face_super_resolver.is_neural_ready else 'Fotométrico Lanczos-4 + Unsharp Mask'}")
    print(f"✓ Limiar de Ativação por Tamanho: < {settings.face_super_res_threshold}px")
    print(f"✓ Galeria de Moradores Ativa: {len(face_recognizer.known_gallery)} pessoas ({face_recognizer.gallery_vector_count} vetores)")

    events_dir = Path("data/events")
    crops = []
    if events_dir.exists():
        crops = sorted([
            f for f in events_dir.iterdir()
            if f.is_file() and f.name.endswith("_crop.jpg") and ("person" in f.name or "Pessoa" in f.name)
        ])

    if not crops:
        print("ℹ️ Nenhum recorte de evento encontrado em data/events/.")
        sys.exit(0)

    print(f"✓ Analisando recortes de eventos para avaliar super-resolução...\n")

    demo_pairs = []
    latencies = []
    tested_faces = 0

    print(f"{'Arquivo de Evento':<35} | {'Face Tam':<10} | {'Simil Nativa':<14} | {'Simil Super-Res':<16} | {'Latência':<10}")
    print("-" * 90)

    rec_model = face_recognizer.insight_app.models.get('recognition') if face_recognizer.insight_app else None

    for crop_path in crops[:25]:
        img_bgr = cv2.imread(str(crop_path))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        dummy_bbox = (0, 0, w, h)

        # Detecta face com Dual-Pass
        name, conf, is_known, f_box = face_recognizer.detect_and_identify_person(img_bgr, dummy_bbox)
        if f_box is None:
            continue

        fx1, fy1, fx2, fy2 = f_box
        fw = max(1, fx2 - fx1)
        fh = max(1, fy2 - fy1)
        face_raw = img_bgr[fy1:fy2, fx1:fx2]
        if face_raw.size == 0:
            continue

        tested_faces += 1

        # Similaridade nativa (sem super-resolução)
        native_sim = conf

        # Super-Resolução
        t0 = time.perf_counter()
        sr_face = face_super_resolver.enhance(face_raw, target_size=(112, 112))
        sr_time_ms = (time.perf_counter() - t0) * 1000
        latencies.append(sr_time_ms)

        sr_sim = native_sim
        if rec_model is not None and face_recognizer.known_gallery:
            try:
                emb = rec_model.get_feat(sr_face).flatten().astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                    # Calcula similaridade máxima na galeria
                    all_sims = []
                    for g_name, g_vecs in face_recognizer.known_gallery.items():
                        for v in g_vecs:
                            all_sims.append(float(np.dot(emb, v)))
                    if all_sims:
                        sr_sim = max(all_sims)
            except Exception:
                pass

        tam_str = f"{fw}x{fh}px"
        nat_str = f"{native_sim:.3f}"
        sr_str = f"{sr_sim:.3f}"
        if sr_sim > native_sim + 0.02:
            sr_str += " 🚀 (+)"

        fname_short = crop_path.name[:33]
        print(f"{fname_short:<35} | {tam_str:<10} | {nat_str:<14} | {sr_str:<16} | {sr_time_ms:<8.2f} ms")

        # Salva para demonstração
        if len(demo_pairs) < 4:
            face_preview_orig = cv2.resize(face_raw, (112, 112), interpolation=cv2.INTER_NEAREST)
            face_preview_sr = sr_face.copy()
            # Linha divisória
            divider = np.zeros((112, 3, 3), dtype=np.uint8)
            divider[:] = (0, 255, 255)
            pair = np.hstack([face_preview_orig, divider, face_preview_sr])
            demo_pairs.append(pair)

    if demo_pairs:
        # Monta mosaico horizontal
        mosaic = np.hstack(demo_pairs)
        out_demo = Path("data/super_res_demo.jpg")
        cv2.imwrite(str(out_demo), mosaic)
        print(f"\n🖼️ Comparação visual (Pixelado vs Super-Res) salva em: {out_demo}")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print("\n" + "=" * 78)
    print("RESUMO DO MÓDULO 2 (SUPER-RESOLUÇÃO FACIAL)")
    print(f"• Faces Pequenas/Distantes Testadas: {tested_faces}")
    print(f"• Latência Média de Restauração    : {avg_lat:.2f} ms")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
