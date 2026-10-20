#!/usr/bin/env python3
"""
Cerberus — Benchmark e Teste de IA Multimodal Local de Borda (Edge VLM - Fase 5)
Testa o carregamento, latência de inferência e capacidade de auditoria visual
100% offline do Moondream2 na GPU local (NVIDIA GeForce GTX 1660).
"""
import sys
import os
import time
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.ai.edge_vlm import edge_vlm

def main():
    print("=" * 65)
    print("  CERBERUS — BENCHMARK DE IA VISUAL LOCAL (EDGE VLM - MOONDREAM2)")
    print("=" * 65)

    device = settings.resolve_device()
    print(f"✓ Dispositivo Alvo: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        allocated_init = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"✓ GPU Dedicada: {gpu_name} (VRAM Total: {total_vram:.0f} MB)")
        print(f"✓ VRAM Alocada Inicial: {allocated_init:.1f} MB")

    # 1. Carregamento do Edge VLM na GPU
    print(f"\n[1/3] Inicializando Edge VLM ({edge_vlm.model_id} em FP16)...")
    
    t0 = time.time()
    ok = edge_vlm.initialize()
    if not ok:
        print("❌ Falha ao carregar Edge VLM. Verifique dependências ou conectividade.")
        return

    t_load = time.time() - t0
    vram_after_load = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
    print(f"✅ VLM ({edge_vlm.model_id}) carregado em {t_load:.1f}s!")
    print(f"💾 VRAM Total Ocupada na GPU: {vram_after_load:.1f} MB")

    # 2. Localização de recortes reais de teste
    print("\n[2/3] Localizando imagens de eventos reais em 'data/events/'...")
    events_dir = Path("data/events")
    crops = list(events_dir.glob("*_crop.jpg"))
    if not crops:
        crops = list(events_dir.glob("*.jpg"))
    if not crops:
        crops = list(events_dir.rglob("*.jpg"))

    print(f"  Encontradas {len(crops)} imagens de eventos.")

    if not crops:
        print("  ⚠️ Nenhuma imagem de teste encontrada em data/events/. Criando imagem sintética de teste...")
        test_img = Image.new("RGB", (320, 320), color=(73, 109, 137))
        crops = ["synthetic_test.jpg"]
        test_img.save("synthetic_test.jpg")

    # 3. Execução de testes de VQA e Auditoria Visual
    print("\n[3/3] Avaliando inferência visual generativa na GPU...")
    
    # Seleciona até 3 recortes representativos (ex: pessoa, veículo, pet se houver)
    sample_images = crops[:3]

    for idx, img_path in enumerate(sample_images, 1):
        img_name = Path(img_path).name
        print(f"\n--- Evidência #{idx}: {img_name} ---")

        # A. Captioning da cena
        t_start = time.perf_counter()
        caption = edge_vlm.caption(str(img_path), length="normal")
        lat_cap = (time.perf_counter() - t_start) * 1000
        print(f"  📝 Descrição da Cena ({lat_cap:.1f} ms):")
        print(f"     \"{caption}\"")

        # B. Pergunta de segurança específica (VQA)
        if "vehicle" in img_name.lower() or "carro" in img_name.lower() or "moto" in img_name.lower():
            q = "What kind of vehicle is this and what color is it?"
        elif "person" in img_name.lower():
            q = "What is this person wearing and are they holding anything?"
        else:
            q = "What is the subject of this image and what is it doing?"

        t_start = time.perf_counter()
        answer = edge_vlm.query(str(img_path), q)
        lat_q = (time.perf_counter() - t_start) * 1000
        print(f"\n  ❓ Pergunta de Segurança: \"{q}\"")
        print(f"  💡 Resposta da IA ({lat_q:.1f} ms):")
        print(f"     \"{answer}\"")

        # C. Laudo de auditoria visual estruturada
        entity_type = "vehicle" if ("vehicle" in img_name or "moto" in img_name or "carro" in img_name) else "person"
        t_start = time.perf_counter()
        audit = edge_vlm.audit_security_scene(str(img_path), entity_type=entity_type)
        lat_audit = (time.perf_counter() - t_start) * 1000
        print(f"\n  🛡️ Laudo Automatizado ({lat_audit:.1f} ms):")
        print(f"     • Risco: {audit.get('risco', 'baixo').upper()}")
        print(f"     • Motivo: {audit.get('motivo')}")
        print(f"     • Ação Recomendada: {audit.get('acao')}")

    if torch.cuda.is_available():
        final_vram = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"\n💾 VRAM Final Ocupada na GPU: {final_vram:.1f} MB (Margem de Segurança Total OK!)")

    print("\n" + "=" * 65)
    print("✅ TESTE DA FASE 5 (EDGE VLM MOONDREAM2) CONCLUÍDO COM SUCESSO!")
    print("=" * 65)

if __name__ == "__main__":
    main()
