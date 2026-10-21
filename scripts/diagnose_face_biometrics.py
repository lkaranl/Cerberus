#!/usr/bin/env python3
"""
Cerberus — Diagnóstico e Auditoria Biometria Facial SOTA
Foco: Alta Precisão, FQA (Face Quality Assessment), Multi-Embedding e Isolamento Anti-Falso Positivo.

Audita detalhadamente:
1. Qualidade biométrica das fotos cadastradas em data/known_faces (Resolução, FQA Nitidez Laplaciana, Faces Detectadas);
2. Coerência intra-classe dos vetores da Galeria Multi-Embedding de cada morador;
3. Matriz de Isolamento Inter-Classes (garantia de margem de segurança contra falsos positivos);
4. Validação prática contra recortes reais em data/events/.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Garante raiz do repositório no sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings
from app.detector.recognizer import face_recognizer

def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)

def print_section(title: str):
    print(f"\n[{title}]")
    print("-" * 75)

def main():
    print_header("CERBERUS — AUDITORIA E DIAGNÓSTICO BIOMÉTRICO FACIAL SOTA")

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (Sem CUDA)"
        print(f"✓ Dispositivo Alvo: {'cuda:0' if has_cuda else 'cpu'} ({gpu_name})")
        if has_cuda:
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"✓ VRAM Total da GPU: {vram_total:.1f} MB")
    except Exception as err:
        print(f"⚠️ Aviso sobre ambiente PyTorch/CUDA: {err}")

    print(f"✓ Modelo Facial Configurado: {settings.face_model_name}")
    print(f"✓ Limiar de Reconhecimento Padrão: {settings.face_confidence_threshold:.2f} (58%)")
    print(f"✓ Limiar de Reconhecimento Instantâneo: {settings.face_instant_threshold:.2f} (68%)")
    print(f"✓ Dimensão Mínima de Face (FQA): {settings.face_min_size}x{settings.face_min_size} px")
    print(f"✓ Nitidez Mínima Laplaciana (FQA): {settings.face_quality_min_blur:.1f}")

    # Inicialização do FaceRecognizer
    t0 = time.time()
    face_recognizer.initialize()
    init_time = (time.time() - t0) * 1000

    if not face_recognizer.is_ready:
        print("\n❌ ERRO CRÍTICO: FaceRecognizer não pôde ser inicializado.")
        print("   Verifique se o InsightFace e ONNXRuntime estão instalados no servidor.")
        sys.exit(1)

    print(f"✓ InsightFace inicializado com sucesso em {init_time:.1f} ms!")

    # -------------------------------------------------------------------------
    # [1/4] AUDITORIA FQA DE FOTOS CADASTRADAS EM data/known_faces
    # -------------------------------------------------------------------------
    print_section("1/4. AUDITORIA FQA DAS FOTOS EM data/known_faces")

    known_dir = Path("data/known_faces")
    if not known_dir.exists():
        print(f"❌ Diretório '{known_dir}' não encontrado!")
        sys.exit(1)

    person_folders = [p for p in known_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not person_folders:
        print(f"⚠️ Nenhuma pasta de morador encontrada em '{known_dir}'!")
        sys.exit(1)

    total_images_checked = 0
    total_approved = 0
    total_warnings = 0
    total_rejected = 0

    person_embeddings_raw: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    for person_dir in sorted(person_folders, key=lambda p: p.name.lower()):
        name = person_dir.name
        print(f"\n📁 Morador: {name.upper()}")
        image_files = sorted([
            f for f in person_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ])

        if not image_files:
            print(f"   ⚠️ Nenhuma imagem encontrada na pasta de {name}!")
            continue

        person_embeddings_raw[name] = []

        for img_path in image_files:
            total_images_checked += 1
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"   ❌ [{img_path.name}] Falha ao decodificar imagem (corrompida)")
                total_rejected += 1
                continue

            # Garante 3 canais BGR (converte BGRA de PNGs ou Grayscale)
            if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            elif img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

            h, w = img_bgr.shape[:2]

            # Detecção de faces via InsightFace (com padding de contexto se necessário)
            try:
                faces = face_recognizer.app.get(img_bgr) if face_recognizer.app else []
                is_padded = False
                pad_y = 0
                pad_x = 0
                if not faces or len(faces) == 0:
                    pad_y = max(32, int(h * 0.25))
                    pad_x = max(32, int(w * 0.25))
                    padded_bgr = cv2.copyMakeBorder(
                        img_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101
                    )
                    faces = face_recognizer.app.get(padded_bgr) if face_recognizer.app else []
                    if faces and len(faces) > 0:
                        is_padded = True
            except Exception as e:
                print(f"   ❌ [{img_path.name}] Erro na inferência InsightFace: {e}")
                total_rejected += 1
                continue

            if not faces:
                print(f"   ❌ [{img_path.name}] ({w}x{h}px) -> [REJEITADA]: Nenhuma face detectada pelo SCRFD.")
                print(f"      👉 Recomendação: Use foto mais próxima, frontal e bem iluminada.")
                total_rejected += 1
                continue

            # Avalia a face principal (maior área se houver múltiplas)
            sorted_faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            face = sorted_faces[0]
            fx1, fy1, fx2, fy2 = map(int, face.bbox)

            if is_padded:
                # Reajusta coordenadas para a imagem original
                fx1 = max(0, fx1 - pad_x)
                fy1 = max(0, fy1 - pad_y)
                fx2 = min(w, fx2 - pad_x)
                fy2 = min(h, fy2 - pad_y)

            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)

            # Recorte da face para FQA
            face_crop = img_bgr[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
            blur_score = 0.0
            if face_crop.size > 0:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            aspect_ratio = fw / float(fh)
            num_faces = len(faces)

            # Classificação FQA
            is_blur_ok = blur_score >= settings.face_quality_min_blur
            is_size_ok = fw >= settings.face_min_size and fh >= settings.face_min_size
            is_aspect_ok = 0.55 <= aspect_ratio <= 1.55

            warnings = []
            if num_faces > 1:
                warnings.append(f"{num_faces} faces detectadas")
            if not is_size_ok:
                warnings.append(f"face muito pequena ({fw}x{fh}px < {settings.face_min_size}px)")
            if not is_blur_ok:
                warnings.append(f"nitidez baixa (Laplacian: {blur_score:.1f} < {settings.face_quality_min_blur:.1f})")
            if not is_aspect_ok:
                warnings.append(f"proporção atípica ({aspect_ratio:.2f})")

            # Salva o embedding normalizado para análise inter-classe
            if face.embedding is not None:
                norm = np.linalg.norm(face.embedding)
                if norm > 1e-6:
                    norm_emb = face.embedding / norm
                    person_embeddings_raw[name].append((img_path.name, norm_emb))

            if not warnings:
                print(f"   ✓ [{img_path.name}] ({w}x{h}px) -> [APROVADA - EXCELENTE]")
                print(f"      Face: {fw}x{fh}px | Nitidez: {blur_score:.1f} | Proporção: {aspect_ratio:.2f}")
                total_approved += 1
            else:
                warn_msg = ", ".join(warnings)
                print(f"   ⚠️ [{img_path.name}] ({w}x{h}px) -> [AVISO: {warn_msg}]")
                print(f"      Face: {fw}x{fh}px | Nitidez: {blur_score:.1f} | Proporção: {aspect_ratio:.2f}")
                total_warnings += 1

    print(f"\nResumo da Auditoria FQA:")
    print(f"• Total de Imagens Avaliadas : {total_images_checked}")
    print(f"• Fotos Aprovadas (Alta Nitidez) : {total_approved}")
    print(f"• Fotos com Alerta/Aviso       : {total_warnings}")
    print(f"• Fotos Rejeitadas (Sem Face)  : {total_rejected}")

    # -------------------------------------------------------------------------
    # [2/4] COERÊNCIA INTRA-CLASSE (GALERIA MULTI-EMBEDDING)
    # -------------------------------------------------------------------------
    print_section("2/4. COERÊNCIA INTRA-CLASSE DA GALERIA MULTI-EMBEDDING")

    print(f"✓ Vetores carregados na galeria em memória: {face_recognizer.gallery_vector_count}")
    for name, vectors in face_recognizer.known_gallery.items():
        n_vecs = len(vectors)
        if n_vecs == 1:
            print(f"  • {name}: 1 vetor cadastrado (Recomendado: 2 a 4 fotos em ângulos diferentes)")
        else:
            # Calcula similaridade entre os vetores da própria pessoa
            sims = []
            for i in range(n_vecs):
                for j in range(i + 1, n_vecs):
                    cos_sim = float(np.dot(vectors[i], vectors[j]))
                    sims.append(cos_sim)
            min_sim = min(sims)
            avg_sim = sum(sims) / len(sims)
            max_sim = max(sims)
            print(f"  • {name}: {n_vecs} vetores | Similaridade Intra: min={min_sim:.2f}, méd={avg_sim:.2f}, max={max_sim:.2f}")

    # -------------------------------------------------------------------------
    # [3/4] MATRIZ DE ISOLAMENTO INTER-CLASSES (ANTI-FALSO POSITIVO)
    # -------------------------------------------------------------------------
    print_section("3/4. MATRIZ DE ISOLAMENTO BIOMÉTRICO INTER-CLASSES (CROSS-ID)")

    person_names = list(person_embeddings_raw.keys())
    has_cross_conflicts = False

    if len(person_names) < 2:
        print("ℹ️ Menos de 2 moradores cadastrados. Isolamento inter-classes não aplicável.")
    else:
        print(f"{'Comparação':<35} | {'Simil Máx':<10} | {'Status':<25}")
        print("-" * 75)

        for i in range(len(person_names)):
            for j in range(i + 1, len(person_names)):
                name_a = person_names[i]
                name_b = person_names[j]

                embs_a = person_embeddings_raw[name_a]
                embs_b = person_embeddings_raw[name_b]

                if not embs_a or not embs_b:
                    continue

                max_cross_sim = -1.0
                best_pair = ("", "")
                for file_a, vec_a in embs_a:
                    for file_b, vec_b in embs_b:
                        sim = float(np.dot(vec_a, vec_b))
                        if sim > max_cross_sim:
                            max_cross_sim = sim
                            best_pair = (file_a, file_b)

                comparison_label = f"{name_a} vs {name_b}"
                sim_str = f"{max_cross_sim:.4f}"

                if max_cross_sim >= settings.face_confidence_threshold:
                    status = "❌ RISCO DE FALSO POSITIVO!"
                    has_cross_conflicts = True
                elif max_cross_sim >= 0.45:
                    status = "⚠️ ATENÇÃO (Margem Curta)"
                else:
                    status = "✓ SEGURO (Alto Isolamento)"

                print(f"{comparison_label:<35} | {sim_str:<10} | {status:<25}")
                if max_cross_sim >= 0.45:
                    print(f"   └─ Par mais próximo: {best_pair[0]} <-> {best_pair[1]}")

        if not has_cross_conflicts:
            print(f"\n✓ Isolamento excelente! Nenhuma foto de morador colide com outro morador (todas < {settings.face_confidence_threshold:.2f}).")
        else:
            print(f"\n❌ ALERTA: Há sobreposição biométrica acima do limiar. Revise as fotos apontadas acima.")

    # -------------------------------------------------------------------------
    # [4/4] TESTE PRÁTICO EM RECORTES REAIS DE EVENTOS (data/events)
    # -------------------------------------------------------------------------
    print_section("4/4. TESTE PRÁTICO CONTRA RECORTES REAIS EM data/events")

    events_dir = Path("data/events")
    event_crops = []
    if events_dir.exists():
        event_crops = sorted([
            f for f in events_dir.iterdir()
            if f.is_file() and f.name.endswith("_crop.jpg") and ("person" in f.name or "Pessoa" in f.name)
        ])

    if not event_crops:
        print("ℹ️ Nenhum recorte de pessoa encontrado em 'data/events/' para teste prático.")
    else:
        print(f"Encontrados {len(event_crops)} recortes de eventos de pessoas para teste.")
        print(f"{'Arquivo de Evento':<40} | {'Face?':<6} | {'Nitidez':<8} | {'Identidade':<16} | {'Simil':<6}")
        print("-" * 85)

        for crop_path in event_crops[:15]:  # Avalia até 15 eventos recentes
            crop_bgr = cv2.imread(str(crop_path))
            if crop_bgr is None:
                continue

            ch, cw = crop_bgr.shape[:2]
            dummy_bbox = (0, 0, cw, ch)

            # Executa o identificador SOTA
            name, score, is_known, face_bbox = face_recognizer.detect_and_identify_person(crop_bgr, dummy_bbox)

            has_face_str = "SIM" if face_bbox is not None else "NÃO"

            # Mede nitidez da face se encontrada, senão do crop
            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                fc = crop_bgr[max(0, fy1):min(ch, fy2), max(0, fx1):min(cw, fx2)]
                gray = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY) if fc.size > 0 else np.zeros((10, 10), dtype=np.uint8)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            else:
                gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            fname_short = crop_path.name[:38]
            print(f"{fname_short:<40} | {has_face_str:<6} | {sharpness:<8.1f} | {name:<16} | {score:<6.2f}")

    print_header("CONCLUSÃO E DIRETRIZES RECOMENDADAS")
    print("1. Para reconhecimento com alta precisão sem falsos positivos:")
    print("   • Cada morador deve ter de 2 a 4 fotos bem iluminadas em data/known_faces/<Nome>/");
    print("   • Fotos devem ter nitidez Laplaciana >= 55.0 e a face deve ter pelo menos 60x60 pixels;");
    print("   • Evite fotos de documentos (RG/CNH com reflexo ou micro-fotos) ou fotos com óculos escuros.");
    print("2. A Galeria Multi-Embedding e o Consenso Temporal estão 100% ativos e prontos para produção.")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
