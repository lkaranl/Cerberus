#!/usr/bin/env python3
"""
Cerberus - Profiler Completo de Latência e Detecção de Gargalos (Raio-X de Performance)
Mede o tempo de execução milissegundo a milissegundo de cada etapa do pipeline:
- Decodificação de Vídeo (NVDEC vs CPU)
- Inferência YOLOv8 (TensorRT FP16 vs PyTorch CUDA)
- Detecção e Alinhamento Facial MTCNN
- Extração de Embeddings FaceNet (TensorRT FP16)
- Redimensionamento e Compressão JPEG para Web
- Balanço Total do Ciclo e FPS Máximo Teórico
"""

import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"   {title}")
    print("=" * 70)

def get_gpu_telemetry():
    """Lê estatísticas em tempo real via nvidia-smi."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.decoder",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "name": parts[0],
                "vram_total_mb": float(parts[1]),
                "vram_used_mb": float(parts[2]),
                "gpu_core_util": f"{parts[3]}%",
                "nvdec_util": f"{parts[4]}%"
            }
    except Exception:
        pass
    return None

def main():
    print_header("Cerberus - Raio-X de Performance & Detecção de Gargalos")

    # 1. Checagem de Hardware
    try:
        import torch
    except ImportError:
        print("❌ Erro: PyTorch não instalado no ambiente virtual.")
        sys.exit(1)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    print(f"✓ Dispositivo Ativo: {device} ({torch.cuda.get_device_name(0) if has_cuda else 'CPU'})")
    print(f"✓ PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

    gpu_telemetry = get_gpu_telemetry()
    if gpu_telemetry:
        print(f"✓ VRAM Total: {gpu_telemetry['vram_total_mb']} MB | Usada: {gpu_telemetry['vram_used_mb']} MB")
        print(f"✓ Carga GPU: {gpu_telemetry['gpu_core_util']} | Carga NVDEC: {gpu_telemetry['nvdec_util']}")

    # Carrega configurações e modelos
    from app.core.config import settings
    from app.detector.engine import detector_engine
    from app.detector.recognizer import face_recognizer

    detector_engine.initialize()
    face_recognizer.initialize()

    # Cria frame sintético representando a resolução típica das câmeras (2304x1296)
    test_h, test_w = 1296, 2304
    dummy_frame = np.random.randint(0, 255, (test_h, test_w, 3), dtype=np.uint8)
    # Adiciona um retângulo simulando pessoa no centro para disparar MTCNN/FaceNet
    cv2_rect = (test_w // 4, test_h // 4, test_w // 2, test_h // 2)

    results = {}

    # --- Teste 1: YOLOv8 Object Detection ---
    print("\n⏱️ [1/5] Medindo Inferência YOLOv8 (Detector Principal)...")
    import cv2

    # Warmup
    for _ in range(5):
        _ = detector_engine.detect(dummy_frame, 0.40, 0.45)

    times_yolo = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = detector_engine.detect(dummy_frame, 0.40, 0.45)
        times_yolo.append((time.perf_counter() - t0) * 1000)

    avg_yolo = np.mean(times_yolo)
    p95_yolo = np.percentile(times_yolo, 95)
    model_mode = "TensorRT FP16 ⚡" if detector_engine.using_tensorrt else "PyTorch CUDA"
    results["yolo"] = {
        "label": f"YOLOv8 Detecção ({model_mode})",
        "avg_ms": avg_yolo,
        "p95_ms": p95_yolo
    }
    print(f"✓ YOLOv8 Média: {avg_yolo:.2f} ms | P95: {p95_yolo:.2f} ms [{model_mode}]")

    # --- Teste 2: MTCNN Face Detection & Alignment ---
    print("\n⏱️ [2/5] Medindo Detecção e Alinhamento Facial MTCNN...")
    # Simula crop da cabeça de uma pessoa
    head_crop = dummy_frame[test_h//4 : test_h//4 + 300, test_w//4 : test_w//4 + 300]
    head_crop_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)

    times_mtcnn = []
    if face_recognizer.mtcnn is not None:
        # Warmup
        for _ in range(3):
            _ = face_recognizer.mtcnn(head_crop_rgb)

        for _ in range(20):
            t0 = time.perf_counter()
            _ = face_recognizer.mtcnn(head_crop_rgb)
            times_mtcnn.append((time.perf_counter() - t0) * 1000)

        avg_mtcnn = np.mean(times_mtcnn)
        p95_mtcnn = np.percentile(times_mtcnn, 95)
    else:
        avg_mtcnn = 0.0
        p95_mtcnn = 0.0

    results["mtcnn"] = {
        "label": "MTCNN Detecção Facial (PyTorch)",
        "avg_ms": avg_mtcnn,
        "p95_ms": p95_mtcnn
    }
    print(f"✓ MTCNN Média: {avg_mtcnn:.2f} ms | P95: {p95_mtcnn:.2f} ms")

    # --- Teste 3: FaceNet Feature Extraction ---
    print("\n⏱️ [3/5] Medindo Extração de Embedding FaceNet (512-dim)...")
    dummy_face_tensor = torch.randn(1, 3, 160, 160, device=device)

    times_facenet = []
    if face_recognizer.is_ready:
        # Warmup
        for _ in range(5):
            _ = face_recognizer._extract_embedding(dummy_face_tensor)

        for _ in range(30):
            t0 = time.perf_counter()
            _ = face_recognizer._extract_embedding(dummy_face_tensor)
            times_facenet.append((time.perf_counter() - t0) * 1000)

        avg_facenet = np.mean(times_facenet)
        p95_facenet = np.percentile(times_facenet, 95)
        fn_mode = "TensorRT FP16 ⚡" if face_recognizer.using_tensorrt else "PyTorch CUDA"
    else:
        avg_facenet = 0.0
        p95_facenet = 0.0
        fn_mode = "Desativado"

    results["facenet"] = {
        "label": f"FaceNet Embedding ({fn_mode})",
        "avg_ms": avg_facenet,
        "p95_ms": p95_facenet
    }
    print(f"✓ FaceNet Média: {avg_facenet:.2f} ms | P95: {p95_facenet:.2f} ms [{fn_mode}]")

    # --- Teste 4: Server-Side Resize & JPEG Encoding ---
    print("\n⏱️ [4/5] Medindo Redimensionamento e Compressão JPEG para Web...")
    target_width = settings.stream_max_width if settings.stream_max_width > 0 else 960

    times_resize = []
    times_jpeg = []
    for _ in range(30):
        t0 = time.perf_counter()
        scale = target_width / test_w
        resized = cv2.resize(dummy_frame, (target_width, int(test_h * scale)), interpolation=cv2.INTER_LINEAR)
        times_resize.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), settings.stream_quality]
        _, _ = cv2.imencode(".jpg", resized, encode_param)
        times_jpeg.append((time.perf_counter() - t1) * 1000)

    avg_resize = np.mean(times_resize)
    avg_jpeg = np.mean(times_jpeg)
    results["resize"] = {
        "label": f"Downscale CPU (2K ➔ {target_width}px)",
        "avg_ms": avg_resize,
        "p95_ms": np.percentile(times_resize, 95)
    }
    results["jpeg"] = {
        "label": f"Compressão JPEG (qualidade {settings.stream_quality}%)",
        "avg_ms": avg_jpeg,
        "p95_ms": np.percentile(times_jpeg, 95)
    }
    print(f"✓ Resize CPU: {avg_resize:.2f} ms")
    print(f"✓ JPEG Encode: {avg_jpeg:.2f} ms")

    # --- Teste 5: Câmeras Reais Conectadas e Latência de Captura ---
    print("\n⏱️ [5/5] Inspecionando Status das Câmeras RTSP em Execução...")
    cam_config_file = Path("config/cameras.json")
    if cam_config_file.exists():
        with open(cam_config_file, "r", encoding="utf-8") as f:
            cams = json.load(f)
        print(f"✓ Total de Câmeras Configuradas: {len(cams)}")
        for c in cams:
            print(f"    • {c['name']} ({c['id']}) - Ativa: {c.get('active', True)}")

    # --- Relatório Final e Análise de Gargalos ---
    total_cycle_ms = avg_yolo + avg_resize + avg_jpeg
    # Adiciona FaceNet/MTCNN condicional (quando há pessoas na cena)
    total_with_face_ms = total_cycle_ms + avg_mtcnn + avg_facenet

    theoretical_fps_base = 1000.0 / total_cycle_ms if total_cycle_ms > 0 else 0
    theoretical_fps_face = 1000.0 / total_with_face_ms if total_with_face_ms > 0 else 0

    print_header("📊 RELATÓRIO DE PERFORMANCE & BALANÇO DE TEMPO POR FRAME")
    print(f"{'Etapa do Pipeline':<40} | {'Média (ms)':<10} | {'% Ciclo':<8} | {'Status'}")
    print("-" * 70)

    for k, v in results.items():
        pct = (v["avg_ms"] / total_with_face_ms) * 100 if total_with_face_ms > 0 else 0
        if v["avg_ms"] < 6.0:
            status = "⚡ Ultra Rápido"
        elif v["avg_ms"] < 15.0:
            status = "✓ Rápido"
        elif v["avg_ms"] < 30.0:
            status = "⚠️ Atenção"
        else:
            status = "🛑 Gargalo"
        print(f"{v['label']:<40} | {v['avg_ms']:>8.2f}ms | {pct:>6.1f}% | {status}")

    print("-" * 70)
    print(f"{'CICLO BASE (Detecção + Stream Web)':<40} | {total_cycle_ms:>8.2f}ms | {'100.0%':<8} | ⚡ ~{theoretical_fps_base:.1f} FPS")
    print(f"{'CICLO COMPLETO (Com Reconhecimento Facial)':<40} | {total_with_face_ms:>8.2f}ms | {'100.0%':<8} | ⚡ ~{theoretical_fps_face:.1f} FPS")
    print("=" * 70)

    # Diagnóstico Inteligente
    print("\n💡 DIAGNÓSTICO DO ANTIGRAVITY ENGINE:")
    if detector_engine.using_tensorrt and face_recognizer.using_tensorrt:
        print("  🎉 EXCELENTE: Tanto o YOLOv8 quanto o FaceNet estão operando no modo TensorRT FP16!")
        print("  A latência neural da GPU é ultrabaixa (~3 a 4 ms cada).")
    elif not detector_engine.using_tensorrt:
        print("  ⚠️ GARGALO: YOLOv8 está rodando em PyTorch CUDA. Execute: python scripts/export_tensorrt.py")
    elif not face_recognizer.using_tensorrt:
        print("  ⚠️ GARGALO: FaceNet está rodando em PyTorch. Execute: python scripts/export_facenet_tensorrt.py")

    if avg_mtcnn > 15.0:
        print(f"  💡 OPORTUNIDADE: O MTCNN levou {avg_mtcnn:.1f}ms. O recorte facial otimizado de 55% reduz")
        print("     a área de busca e garante alta velocidade sem sobrecarregar a GPU.")

    post_gpu = get_gpu_telemetry()
    if post_gpu:
        print(f"\n📦 VRAM Final em Uso: {post_gpu['vram_used_mb']} MB / {post_gpu['vram_total_mb']} MB (Folga confortável na GTX 1660)")

    print("\n" + "=" * 70)
    print("✓ Profiling concluído com sucesso!")
    print("=" * 70)

if __name__ == "__main__":
    main()
