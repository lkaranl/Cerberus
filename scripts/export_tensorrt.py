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
    # 4. Exportação Nativa Direta para TensorRT FP16 (sem dependência de modelopt)
    engine_dest = Path("models/yolov8n.engine")
    onnx_dest = Path("models/yolov8n.onnx")
    print("\n⚡ Iniciando compilação da engine TensorRT FP16...")
    print("Isso pode levar de 1 a 3 minutos na primeira vez (calibração de kernels CUDA)...")

    start_export = time.time()

    # 1. Garante que temos o modelo ONNX
    if not onnx_dest.exists():
        print("Exportando modelo base para ONNX...")
        model.export(format="onnx", imgsz=640, dynamic=False, opset=17, verbose=False)

    print("🔨 Compilando ONNX diretamente para engine TensorRT FP16 via NVIDIA TensorRT 11.4...")
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        # No TensorRT 10/11+, create_network não requer mais a flag EXPLICIT_BATCH
        if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        else:
            network = builder.create_network()

        parser = trt.OnnxParser(network, logger)

        with open(onnx_dest, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("Erro do parser TRT:", parser.get_error(error))
                raise RuntimeError("Falha ao analisar o arquivo ONNX com TensorRT.")

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = 4 * 1024 * 1024 * 1024

        # Habilita FP16 nativo (suportado por padrão nas arquiteturas Turing GTX 1660 e superiores)
        try:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ Modo de meia-precisão FP16 habilitado na GPU GTX 1660.")
        except Exception as fp16_warn:
            print(f"⚠️ Aviso ao ativar flag FP16: {fp16_warn}")

        print("⚡ Otimizando e calibrando kernels CUDA Turing (GTX 1660)...")
        if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_cuda_engine(network)
            serialized_engine = engine.serialize()

        if serialized_engine is None:
            raise RuntimeError("O construtor do TensorRT retornou None ao gerar a engine.")

        with open(engine_dest, "wb") as f:
            f.write(serialized_engine)

        print(f"✓ Engine TensorRT FP16 gerada com sucesso pela API nativa da NVIDIA!")
    except Exception as trt_err:
        print(f"\n❌ Falha na compilação nativa do TensorRT: {trt_err}")
        sys.exit(1)

    export_duration = time.time() - start_export
    print(f"✓ Concluído em {export_duration:.1f}s! Arquivo: {engine_dest}")

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
