#!/usr/bin/env python3
"""
Script de validação e benchmark do motor neural de Person Re-ID (OSNet x0.25).
Avalia extração de embeddings de corpo inteiro, latência em GPU e similaridade cruzada.
"""

import os
import sys
import glob
import time
import cv2
import numpy as np

# Assegura que o diretório raiz esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.detector.person_reid import person_reid
from app.detector.recognizer import face_recognizer

def main():
    print("=" * 65)
    print("  CERBERUS — BENCHMARK DE BIOMETRIA CORPORAL SOTA (OSNet Re-ID)")
    print("=" * 65)

    print("\n[1/3] Inicializando modelo OSNet x0.25...")
    t0 = time.perf_counter()
    ok = person_reid.initialize()
    init_ms = (time.perf_counter() - t0) * 1000

    if not ok:
        print("❌ Falha ao inicializar o OSNet. Verifique conexão para download ou drivers CUDA.")
        return

    provider = person_reid.session.get_providers()[0]
    print(f"  -> Status: OSNet Ativo via {provider}")
    print(f"  -> Tempo de Carga do Modelo: {init_ms:.1f} ms")

    # [2/3] Teste de latência de inferência com imagem sintética
    print("\n[2/3] Medindo latência de inferência pura na GPU...")
    dummy_crop = np.zeros((300, 150, 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        person_reid.extract_embedding(dummy_crop)

    times = []
    emb = None
    for _ in range(30):
        t_start = time.perf_counter()
        res = person_reid.extract_embedding(dummy_crop)
        if res is not None:
            emb = res
            times.append((time.perf_counter() - t_start) * 1000)

    if emb is None:
        print("❌ extract_embedding retornou None. Inspecione o erro detalhado acima.")
        return

    avg_latency = float(np.mean(times))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    print(f"  -> Dimensão do Embedding: {len(emb)} floats (L2-normalizado)")
    print(f"  -> Latência Média por Pessoa: {avg_latency:.2f} ms (~{fps:.0f} pessoas/segundo)")

    # [3/3] Teste em recortes reais de pessoas gravadas pelas câmeras
    print("\n[3/3] Avaliando recortes reais de pessoas gravados em data/events/...")
    person_crops = sorted(glob.glob("data/events/*person*crop.jpg"))
    if not person_crops:
        print("  (Nenhum recorte em data/events/*person*crop.jpg encontrado para teste de eventos)")
    else:
        print(f"  Encontrados {len(person_crops)} recortes de pessoas para análise.")
        sample_crops = person_crops[-6:]
        embeddings = []
        labels = []

        for p_path in sample_crops:
            img = cv2.imread(p_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            t_extract = time.perf_counter()
            emb = person_reid.extract_embedding(img)
            cost_ms = (time.perf_counter() - t_extract) * 1000

            if emb is not None:
                embeddings.append(emb)
                labels.append(os.path.basename(p_path))
                print(f"  📸 {os.path.basename(p_path)} ({w}x{h}) -> Extraído em {cost_ms:.1f} ms")

        # Simulação do fluxo de Re-ID Dinâmico de Morador
        if len(embeddings) >= 2:
            print("\n" + "-" * 65)
            print("  SIMULAÇÃO DE RASTREAMENTO CONTÍNUO (FUSÃO FACIAL + RE-ID CORPORAL)")
            print("-" * 65)
            
            # Suponha que na primeira foto o ArcFace identificou 'Morador'
            ref_label = labels[0]
            ref_emb = embeddings[0]
            person_reid.active_residents["Morador Teste"] = {
                "embedding": ref_emb,
                "last_seen": time.time(),
                "camera": "cam-0"
            }
            print(f"  1. Morador identificado via face na foto 1 ({ref_label})")
            print(f"     -> Assinatura corporal de 512-d registrada no pool dinâmico.")

            print(f"\n  2. Testando Re-ID com as fotos subsequentes (simulando pessoa de costas ou em outra câmera):")
            for i in range(1, len(embeddings)):
                probe_crop = cv2.imread(sample_crops[i])
                matched_name, sim = person_reid.match_resident(probe_crop, threshold=0.65)
                status = "✅ RECONHECIDO COMO MORADOR" if matched_name else "👤 PESSOA DIFERENTE / DESCONHECIDO"
                print(f"     - {labels[i]} -> Similaridade: {sim:.3f} | {status}")

    print("\n" + "=" * 65)
    print("  FIM DO BENCHMARK DA FASE 2")
    print("=" * 65)

if __name__ == "__main__":
    main()
