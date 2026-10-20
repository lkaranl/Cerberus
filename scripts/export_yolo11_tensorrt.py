#!/usr/bin/env python3
"""
Script de Exportação, Otimização e Benchmark do YOLO11 com TensorRT (FP16).
Compila o modelo YOLO11n para a arquitetura NVIDIA Turing (GTX 1660) no Nobara Linux.
"""

import sys
import time
from pathlib import Path
import numpy as np

def main():
    print("=" * 65)
    print("   CERBERUS — UPGRADE DO DETECTOR: YOLO11 TENSORRT FP16")
    print("=" * 65)

    # 1. Checagem de Hardware e CUDA
    try:
        import torch
    except ImportError:
        print("❌ Erro: PyTorch não encontrado no ambiente atual.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ Erro: CUDA não disponível. Este script deve rodar na máquina Nobara com a GTX 1660.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU Dedicada: {gpu_name}")
    print(f"✓ PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")

    # 2. Checagem do TensorRT
    try:
        import tensorrt as trt
        print(f"✓ TensorRT Version: {trt.__version__}")
    except ImportError:
        print("\n❌ Erro: Pacote 'tensorrt' não encontrado. Execute: pip install tensorrt")
        sys.exit(1)

    # 3. Carregamento do Ultralytics YOLO11
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Erro: Ultralytics não instalado. Execute: pip install ultralytics")
        sys.exit(1)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_pt_path = models_dir / "yolo11n.pt"
    onnx_dest = models_dir / "yolo11n.onnx"
    engine_dest = models_dir / "yolo11n.engine"

    if not model_pt_path.exists():
        print(f"\n📥 Baixando pesos oficiais do YOLO11 Nano (yolo11n.pt)...")
        model = YOLO("yolo11n.pt")
        # Salva ou copia para a pasta models/
        try:
            import shutil
            if Path("yolo11n.pt").exists():
                shutil.move("yolo11n.pt", str(model_pt_path))
        except Exception:
            pass
    else:
        print(f"✓ Modelo base localizado: {model_pt_path}")
        model = YOLO(str(model_pt_path))

    # 4. Exportação para ONNX (se necessário)
    if not onnx_dest.exists():
        print("\n⚙️  Exportando YOLO11 para ONNX intermediário (imgsz=640, opset=17)...")
        exported_path = model.export(format="onnx", imgsz=640, dynamic=False, opset=17, verbose=False)
        if Path(exported_path) != onnx_dest and Path(exported_path).exists():
            import shutil
            shutil.move(exported_path, str(onnx_dest))
        print(f"✓ Arquivo ONNX gerado: {onnx_dest}")
    else:
        print(f"✓ Arquivo ONNX existente: {onnx_dest}")

    # 5. Compilação da Engine TensorRT FP16
    print("\n🔨 Compilando Engine TensorRT FP16 nativa para a GTX 1660 (Turing TU116)...")
    print("   Calibrando kernels de convolução e camadas de atenção espacial C2PSA...")
    t0_comp = time.time()

    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        else:
            network = builder.create_network()

        parser = trt.OnnxParser(network, logger)

        with open(onnx_dest, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("Erro do parser TRT:", parser.get_error(error))
                raise RuntimeError("Falha ao analisar o arquivo ONNX do YOLO11.")

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = 4 * 1024 * 1024 * 1024

        # Habilita FP16 nativo (Half Precision na arquitetura Turing)
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ Otimização FP16 ativada na GPU.")

        if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_cuda_engine(network)
            serialized_engine = engine.serialize()

        if serialized_engine is None:
            raise RuntimeError("O builder do TensorRT retornou None.")

        with open(engine_dest, "wb") as f:
            f.write(serialized_engine)

        comp_duration = time.time() - t0_comp
        print(f"✅ Engine compilada em {comp_duration:.1f}s com sucesso: {engine_dest}")

    except Exception as e:
        print(f"❌ Falha ao compilar engine TensorRT nativa: {e}")
        sys.exit(1)

    # 6. Benchmark de Desempenho Real (FPS & Latência)
    print("\n" + "=" * 65)
    print("   BENCHMARK DE PERFORMANCE: YOLO11 TENSORRT FP16")
    print("=" * 65)

    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    iterations = 35

    print("Carregando modelo compilado YOLO11 TensorRT...")
    model_y11 = YOLO(str(engine_dest), task="detect")
    for _ in range(10):
        _ = model_y11(dummy_frame, verbose=False)

    t0 = time.time()
    for _ in range(iterations):
        _ = model_y11(dummy_frame, verbose=False)
    torch.cuda.synchronize()
    duration_y11 = (time.time() - t0) / iterations
    fps_y11 = 1.0 / duration_y11
    ms_y11 = duration_y11 * 1000

    print(f"\n📊 Resultados na GPU NVIDIA GeForce GTX 1660:")
    print(f"  • YOLO11n TensorRT FP16: {ms_y11:.2f} ms por frame  (~{fps_y11:.1f} FPS)")

    # Se o yolov8n.engine anterior existir, compara
    old_engine = models_dir / "yolov8n.engine"
    if old_engine.exists():
        print("\nComparando com detector anterior (YOLOv8n TensorRT):")
        try:
            model_v8 = YOLO(str(old_engine), task="detect")
            for _ in range(5):
                _ = model_v8(dummy_frame, verbose=False)
            t0 = time.time()
            for _ in range(iterations):
                _ = model_v8(dummy_frame, verbose=False)
            torch.cuda.synchronize()
            duration_v8 = (time.time() - t0) / iterations
            fps_v8 = 1.0 / duration_v8
            ms_v8 = duration_v8 * 1000
            print(f"  • YOLOv8n TensorRT FP16: {ms_v8:.2f} ms por frame  (~{fps_v8:.1f} FPS)")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("✓ FASE 3 PRONTA! O Cerberus usará o YOLO11n nativamente.")
    print("=" * 65)

if __name__ == "__main__":
    main()
