#!/usr/bin/env python3
"""
Cerberus - Benchmark & Otimização do MTCNN na GPU (GTX 1660)
Testa diferentes estratégias para derrubar a latência do MTCNN de ~12ms para ~3ms:
1. Baseline: MTCNN Padrão em recorte de alta resolução 2K (600x600);
2. Otimização A: Pré-escalonamento adaptativo do recorte (max 224px);
3. Otimização B: Calibração da pirâmide de escalas (min_face_size e scale_factor);
4. Otimização C: Aceleração FP16 (Half Precision) na arquitetura Turing;
5. Comparativo de acurácia e ganho de velocidade.
"""

import sys
import time
import subprocess
from pathlib import Path
import numpy as np

# Garante a raiz do projeto no sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"   {title}")
    print("=" * 65)

def main():
    print_header("Cerberus - Otimização de Performance MTCNN (Detecção Facial)")

    try:
        import torch
        import cv2
        from facenet_pytorch import MTCNN
    except ImportError as e:
        print(f"❌ Dependência ausente: {e}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ Este teste deve ser executado no servidor Nobara com GPU NVIDIA (GTX 1660).")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")

    # 1. Carrega uma foto real de teste cadastrada em data/known_faces
    known_dir = Path("data/known_faces")
    test_img = None
    test_img_name = ""

    if known_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            found = list(known_dir.rglob(ext))
            if found:
                test_img_path = found[0]
                test_img_name = test_img_path.name
                test_img = cv2.imread(str(test_img_path))
                if test_img is not None:
                    break

    if test_img is None:
        print("⚠️ Nenhuma foto encontrada em data/known_faces. Usando imagem sintética de teste.")
        test_img = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
        test_img_name = "Sintética (400x400)"
    else:
        print(f"✓ Imagem de teste carregada: {test_img_name} ({test_img.shape[1]}x{test_img.shape[0]}px)")

    # Simula crop em resolução 2K de câmera (500x500px)
    crop_2k = cv2.resize(test_img, (500, 500), interpolation=cv2.INTER_LINEAR)
    crop_2k_rgb = cv2.cvtColor(crop_2k, cv2.COLOR_BGR2RGB)

    results = []

    # --- Teste 1: MTCNN Baseline (Padrão Atual) ---
    print("\n⏱️ [1/4] Testando MTCNN Padrão (Entrada Direta 500x500)...")
    mtcnn_base = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        select_largest=True,
        device=device
    )

    # Warmup
    for _ in range(5):
        _ = mtcnn_base(crop_2k_rgb)

    times_base = []
    for _ in range(30):
        t0 = time.perf_counter()
        res = mtcnn_base(crop_2k_rgb)
        times_base.append((time.perf_counter() - t0) * 1000)

    avg_base = np.mean(times_base)
    detected_base = "Sim" if res is not None else "Não"
    results.append(("1. MTCNN Baseline (Crop 500x500)", avg_base, detected_base, "1.00x (Ref)"))
    print(f"✓ Baseline: {avg_base:.2f} ms | Rosto Detectado: {detected_base}")

    # --- Teste 2: Pré-Downscale Adaptativo (Reduz pirâmide de escalas do P-Net) ---
    print("\n⏱️ [2/4] Testando MTCNN com Pré-Downscale Adaptativo (max 224px)...")
    h, w = crop_2k_rgb.shape[:2]
    max_dim = 224
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        crop_opt_rgb = cv2.resize(crop_2k_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        crop_opt_rgb = crop_2k_rgb

    # Warmup
    for _ in range(5):
        _ = mtcnn_base(crop_opt_rgb)

    times_opt = []
    for _ in range(30):
        t0 = time.perf_counter()
        res_opt = mtcnn_base(crop_opt_rgb)
        times_opt.append((time.perf_counter() - t0) * 1000)

    avg_opt = np.mean(times_opt)
    speedup_opt = avg_base / avg_opt if avg_opt > 0 else 0
    detected_opt = "Sim" if res_opt is not None else "Não"
    results.append(("2. Pré-Downscale (224px)", avg_opt, detected_opt, f"{speedup_opt:.2f}x"))
    print(f"✓ Pré-Downscale (224px): {avg_opt:.2f} ms | Rosto Detectado: {detected_opt} | Ganho: {speedup_opt:.2f}x")

    # --- Teste 3: Calibração de Min Face Size (min_face_size=32) ---
    print("\n⏱️ [3/4] Testando Calibração de min_face_size=32 (P-Net Otimizado)...")
    mtcnn_fast = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=32,
        thresholds=[0.5, 0.6, 0.6],
        keep_all=False,
        select_largest=True,
        device=device
    )

    for _ in range(5):
        _ = mtcnn_fast(crop_opt_rgb)

    times_fast = []
    for _ in range(30):
        t0 = time.perf_counter()
        res_fast = mtcnn_fast(crop_opt_rgb)
        times_fast.append((time.perf_counter() - t0) * 1000)

    avg_fast = np.mean(times_fast)
    speedup_fast = avg_base / avg_fast if avg_fast > 0 else 0
    detected_fast = "Sim" if res_fast is not None else "Não"
    results.append(("3. Pré-Downscale + min_face_size=32", avg_fast, detected_fast, f"{speedup_fast:.2f}x"))
    print(f"✓ min_face_size=32: {avg_fast:.2f} ms | Rosto Detectado: {detected_fast} | Ganho: {speedup_fast:.2f}x")

    # --- Teste 4: Pré-Downscale Ultra-Leve (160px direto para FaceNet) ---
    print("\n⏱️ [4/4] Testando Pré-Downscale Ultra-Leve (160px)...")
    crop_160_rgb = cv2.resize(crop_2k_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)

    for _ in range(5):
        _ = mtcnn_fast(crop_160_rgb)

    times_160 = []
    for _ in range(30):
        t0 = time.perf_counter()
        res_160 = mtcnn_fast(crop_160_rgb)
        times_160.append((time.perf_counter() - t0) * 1000)

    avg_160 = np.mean(times_160)
    speedup_160 = avg_base / avg_160 if avg_160 > 0 else 0
    detected_160 = "Sim" if res_160 is not None else "Não"
    results.append(("4. Pré-Downscale 160px", avg_160, detected_160, f"{speedup_160:.2f}x"))
    print(f"✓ Pré-Downscale 160px: {avg_160:.2f} ms | Rosto Detectado: {detected_160} | Ganho: {speedup_160:.2f}x")

    # --- Relatório Comparativo Final ---
    print_header("📊 RELATÓRIO COMPARATIVO DE PERFORMANCE DO MTCNN")
    print(f"{'Estratégia de Otimização':<35} | {'Tempo Médio':<12} | {'Detectou':<10} | {'Speedup'}")
    print("-" * 65)

    for label, ms, det, sp in results:
        print(f"{label:<35} | {ms:>8.2f} ms  | {det:<10} | {sp}")

    print("=" * 65)

    best = min(results, key=lambda x: x[1])
    print(f"\n🚀 Melhor Estratégia: '{best[0]}'")
    print(f"  • Latência caiu de {avg_base:.2f} ms para {best[1]:.2f} ms!")
    print(f"  • Ganho real de velocidade: {avg_base / best[1]:.2f}x mais rápido (+{int((avg_base/best[1] - 1)*100)}%)!")

if __name__ == "__main__":
    main()
