#!/usr/bin/env python3
"""
Script de Exportação e Benchmark de Aceleração Máxima TensorRT (FP16).
Otimizado para arquitetura NVIDIA Turing (GTX 1660) no servidor Nobara Linux.
"""

import sys
import time
from pathlib import Path
import numpy as np

def main():
    print("=" * 60)
    print("   Cerberus - Otimização com NVIDIA TensorRT (FP16)")
    print("=" * 60)

    # 1. Checagem de Hardware e CUDA
    try:
        import torch
    except ImportError:
        print("❌ Erro: PyTorch não encontrado no ambiente atual.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ Erro: CUDA não disponível. Este script deve ser executado no servidor")
        print("   com GPU dedicada NVIDIA (GTX 1660).")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU Detectada: {gpu_name}")
    print(f"✓ PyTorch Version: {torch.__version__} (CUDA {torch.version.cuda})")

    # 2. Checagem do pacote TensorRT
    try:
        import tensorrt as trt
        print(f"✓ TensorRT Version: {trt.__version__}")
    except ImportError:
        print("\n⚠️ O pacote Python 'tensorrt' não está instalado no ambiente virtual.")
        print("Para instalar, execute:")
        print("   pip install tensorrt\n")
        print("Tentando continuar com a exportação via Ultralytics...")

    # 3. Carregamento do Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Erro: Ultralytics não instalado. Execute: pip install ultralytics")
        sys.exit(1)

    model_pt_path = Path("models/yolov8n.pt")
    if not model_pt_path.exists():
        print(f"Baixando modelo base YOLOv8n para {model_pt_path}...")
        model = YOLO("yolov8n.pt")
        model_pt_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        print(f"✓ Carregando modelo base: {model_pt_path}")
        model = YOLO(str(model_pt_path))

    # 4. Exportação para TensorRT FP16
    engine_dest = Path("models/yolov8n.engine")
    print("\n⚡ Iniciando compilação da engine TensorRT FP16...")
    print("Isso pode levar de 1 a 3 minutos na primeira vez (calibração de kernels CUDA)...")

    start_export = time.time()
    try:
        exported_path = model.export(
            format="engine",
            half=True,        # Meia precisão (FP16) nativa para acelerar os Tensor Cores da Turing
            device=0,         # GPU NVIDIA primária
            workspace=4,      # 4GB de workspace máximo para calibração
            imgsz=640,
            dynamic=False,
            verbose=False
        )
        export_duration = time.time() - start_export
        print(f"✓ Compilação concluída com sucesso em {export_duration:.1f}s!")
        print(f"✓ Arquivo gerado: {exported_path}")
    except Exception as e:
        print(f"\n❌ Falha na compilação do TensorRT: {e}")
        print("\n💡 Dica de resolução no Nobara/Fedora Linux:")
        print("1. Certifique-se de que o pacote tensorrt está instalado no venv: pip install tensorrt")
        print("2. Verifique se o driver NVIDIA e o CUDA Toolkit estão atualizados.")
        sys.exit(1)

    # 5. Benchmark de Velocidade Real: PyTorch (.pt) vs TensorRT (.engine)
    print("\n" + "=" * 60)
    print("   Benchmarking de Velocidade: PyTorch CUDA vs TensorRT")
    print("=" * 60)

    # Frame dummy de teste 640x640 BGR
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    iterations = 25

    # Teste 1: PyTorch .pt
    print("Medindo PyTorch (.pt)...")
    model_pt = YOLO(str(model_pt_path))
    # Warmup
    for _ in range(5):
        _ = model_pt(dummy_frame, device=0, verbose=False)
    
    t0 = time.time()
    for _ in range(iterations):
        _ = model_pt(dummy_frame, device=0, verbose=False)
    torch.cuda.synchronize()
    duration_pt = (time.time() - t0) / iterations
    fps_pt = 1.0 / duration_pt
    ms_pt = duration_pt * 1000

    # Teste 2: TensorRT .engine
    print("Medindo TensorRT FP16 (.engine)...")
    model_trt = YOLO(str(engine_dest), task="detect")
    # Warmup
    for _ in range(5):
        _ = model_trt(dummy_frame, verbose=False)

    t0 = time.time()
    for _ in range(iterations):
        _ = model_trt(dummy_frame, verbose=False)
    torch.cuda.synchronize()
    duration_trt = (time.time() - t0) / iterations
    fps_trt = 1.0 / duration_trt
    ms_trt = duration_trt * 1000

    speedup = (ms_pt / ms_trt) if ms_trt > 0 else 1.0

    print("\n📊 Resultados do Benchmark na GPU GTX 1660:")
    print(f"  • PyTorch (.pt):       {ms_pt:.2f} ms por frame  (~{fps_pt:.1f} FPS)")
    print(f"  • TensorRT FP16 (.engine): {ms_trt:.2f} ms por frame  (~{fps_trt:.1f} FPS)")
    print(f"  🚀 Ganho de Velocidade: {speedup:.2f}x mais rápido (+{int((speedup - 1) * 100)}%)!")

    print("\n" + "=" * 60)
    print("✓ Concluído! O Cerberus usará automaticamente o modelo TensorRT.")
    print("Para aplicar imediatamente no serviço em execução:")
    print("   sudo systemctl restart cerberus")
    print("=" * 60)

if __name__ == "__main__":
    main()
