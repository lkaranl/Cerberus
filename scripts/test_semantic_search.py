#!/usr/bin/env python3
"""
Cerberus — Script de Teste e Benchmark de Busca Semântica Multimodal (Fase 4)
Testa a indexação de imagens de eventos reais e a busca vetorial cruzada
usando CLIP / SigLIP Multilíngue em Português do Brasil.
"""
import sys
import os
import time
import asyncio
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from app.core.config import settings
from app.core.logging import logger
from app.storage.database import db
from app.ai.semantic_search import semantic_search

async def run_benchmark():
    print("=" * 65)
    print("  CERBERUS — BENCHMARK DE BUSCA SEMÂNTICA MULTIMODAL (FASE 4)")
    print("=" * 65)

    device = settings.resolve_device()
    print(f"✓ Dispositivo Alvo: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    # 1. Inicialização do modelo multimodal
    print("\n[1/4] Inicializando Modelo Multimodal CLIP Multilíngue...")
    t0 = time.time()
    ok = semantic_search.initialize()
    if not ok:
        print("❌ Falha ao inicializar o modelo CLIP. Verifique se 'sentence-transformers' está instalado.")
        return
    t_load = (time.time() - t0) * 1000
    print(f"✓ Modelo carregado com sucesso na GPU em {t_load:.1f} ms!")

    # 2. Teste de latência de codificação textual em Português
    print("\n[2/4] Medindo latência de codificação de texto (Português do Brasil)...")
    sample_queries = [
        "homem de casaco vermelho",
        "carro prata estacionado",
        "entregador de moto com capacete",
        "cachorro andando na calçada",
        "pessoa com camiseta preta e bermuda"
    ]

    latencies = []
    for q in sample_queries:
        t_start = time.perf_counter()
        emb = semantic_search.encode_text_query(q)
        lat = (time.perf_counter() - t_start) * 1000
        latencies.append(lat)
        print(f"  • '{q}' -> dim: {emb.shape if emb is not None else 'None'} ({lat:.2f} ms)")

    avg_txt_lat = sum(latencies) / len(latencies)
    print(f"  -> Latência Média de Codificação de Texto: {avg_txt_lat:.2f} ms (~{1000/avg_txt_lat:.0f} consultas/seg)")

    # 3. Indexação de recortes de eventos reais salvos em disco
    print("\n[3/4] Indexando imagens de eventos existentes em 'data/events/'...")
    events_dir = Path("data/events")
    crops = list(events_dir.glob("*_crop.jpg"))
    if not crops:
        crops = list(events_dir.glob("*.jpg"))
    if not crops:
        crops = list(events_dir.rglob("*.jpg"))

    print(f"  Encontradas {len(crops)} imagens de evidência para teste.")
    
    indexed_data = []
    img_latencies = []

    # Testa até 25 recortes para medição precisa de throughput
    sample_crops = crops[:25] if len(crops) > 25 else crops
    for idx, img_path in enumerate(sample_crops, 1):
        t_start = time.perf_counter()
        emb = semantic_search.extract_image_embedding(str(img_path))
        lat = (time.perf_counter() - t_start) * 1000
        if emb is not None:
            img_latencies.append(lat)
            indexed_data.append((img_path.name, emb))
            print(f"  • [{idx}/{len(sample_crops)}] {img_path.name} ({lat:.2f} ms)")
        else:
            print(f"  ⚠️ [{idx}/{len(sample_crops)}] Falha ao extrair embedding de {img_path.name}")

    if img_latencies:
        avg_img_lat = sum(img_latencies) / len(img_latencies)
        print(f"  ✓ {len(indexed_data)} imagens codificadas com sucesso!")
        print(f"  -> Latência Média de Extração Visual: {avg_img_lat:.2f} ms (~{1000/avg_img_lat:.0f} crops/seg)")

    # 4. Demonstração de Busca Semântica Cruzada
    print("\n[4/4] Executando consultas de busca semântica em linguagem natural...")
    if not indexed_data:
        print("  ⚠️ Nenhuma imagem disponível para demonstrar a busca.")
        return

    test_searches = [
        "pessoa ou pedestre",
        "veículo ou automóvel",
        "roupa escura ou preta"
    ]

    for q in test_searches:
        q_emb = semantic_search.encode_text_query(q)
        if q_emb is None:
            continue

        scores = []
        for name, img_emb in indexed_data:
            # Similaridade de cosseno (já são normalizados L2)
            sim = float(np.dot(q_emb, img_emb))
            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  🔎 Consulta: \"{q}\"")
        for rank, (name, sim) in enumerate(scores[:3], 1):
            pct = sim * 100
            print(f"     #{rank} {name:<35} -> Similaridade: {pct:.1f}% ({sim:.3f})")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / (1024 * 1024)
        print(f"\n💾 VRAM alocada pelo modelo na GPU: {vram_used:.1f} MB")

    print("\n" + "=" * 65)
    print("✅ TESTE DA FASE 4 (BUSCA SEMÂNTICA MULTIMODAL) CONCLUÍDO COM SUCESSO!")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
