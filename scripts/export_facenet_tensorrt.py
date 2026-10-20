#!/usr/bin/env python3
"""
Script de Exportação e Benchmark de Aceleração Máxima do FaceNet (Inception-ResNet)
para NVIDIA TensorRT FP16 na GPU GTX 1660.
"""

import sys
import time
from pathlib import Path
import numpy as np

def main():
    print("=" * 65)
    print("   Cerberus - Otimização FaceNet (Inception-ResNet) TensorRT FP16")
    print("=" * 65)

    # 1. Checagem de Hardware e CUDA
    try:
        import torch
    except ImportError:
        print("❌ Erro: PyTorch não encontrado no ambiente virtual.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ Erro: CUDA não disponível. Este script deve ser executado no servidor")
        print("   com GPU dedicada NVIDIA (GTX 1660).")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU Detectada: {gpu_name}")
    print(f"✓ PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")

    # 2. Checagem do pacote TensorRT
    try:
        import tensorrt as trt
        print(f"✓ TensorRT Version: {trt.__version__}")
    except ImportError:
        print("❌ Erro: Pacote 'tensorrt' não encontrado no venv.")
        print("Execute: pip install tensorrt")
        sys.exit(1)

    # 3. Carregamento do InceptionResnetV1
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print("❌ Erro: facenet-pytorch não instalado. Execute: pip install --no-deps facenet-pytorch")
        sys.exit(1)

    device = torch.device("cuda:0")
    print("\n📦 Carregando modelo base InceptionResnetV1 (vggface2)...")
    try:
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        print("✓ Modelo base carregado na GPU com sucesso.")
    except Exception as e:
        print(f"❌ Falha ao carregar InceptionResnetV1: {e}")
        sys.exit(1)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_dest = models_dir / "facenet.onnx"
    engine_dest = models_dir / "facenet.engine"

    # 4. Exportação para ONNX (160x160 -> 512-dim embedding)
    print("\n⚡ [1/2] Exportando InceptionResnetV1 para ONNX...")
    dummy_input = torch.randn(1, 3, 160, 160, device=device)

    try:
        export_kwargs = {
            "export_params": True,
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["embedding"],
            "dynamic_axes": {
                "input": {0: "batch_size"},
                "embedding": {0: "batch_size"}
            }
        }
        # No PyTorch 2.x, forçar dynamo=False utiliza o exportador TorchScript (dispensando onnxscript)
        try:
            torch.onnx.export(resnet, dummy_input, str(onnx_dest), dynamo=False, **export_kwargs)
        except (TypeError, ValueError):
            torch.onnx.export(resnet, dummy_input, str(onnx_dest), **export_kwargs)

        print(f"✓ Modelo ONNX salvo com sucesso em: {onnx_dest} ({onnx_dest.stat().st_size / 1024**2:.1f} MB)")
    except Exception as onnx_err:
        print(f"❌ Erro ao exportar para ONNX: {onnx_err}")
        if "onnxscript" in str(onnx_err):
            print("\n💡 Para resolver, instale o pacote onnxscript no ambiente virtual:")
            print("   pip install onnxscript\n")
        sys.exit(1)

    # 5. Compilação Nativa para TensorRT FP16 via NVIDIA TensorRT 11.4
    print("\n⚡ [2/2] Compilando ONNX para Engine TensorRT FP16 (GTX 1660)...")
    print("Isso levará de 30 a 60 segundos (otimizando kernels CUDA da Turing)...")

    start_build = time.time()
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
                raise RuntimeError("Falha ao analisar o arquivo ONNX do FaceNet.")

        config = builder.create_builder_config()

        # Configura memória de calibração
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = 2 * 1024 * 1024 * 1024

        # Habilita FP16
        try:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ Modo de meia-precisão FP16 habilitado com sucesso.")
        except Exception:
            pass

        # Configura profile dinâmico (batch 1 a 4)
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 160, 160), (1, 3, 160, 160), (4, 3, 160, 160))
        config.add_optimization_profile(profile)

        print("⚡ Compilando serialized engine...")
        if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_cuda_engine(network)
            serialized_engine = engine.serialize()

        if serialized_engine is None:
            raise RuntimeError("O construtor retornou None.")

        with open(engine_dest, "wb") as f:
            f.write(serialized_engine)

        build_time = time.time() - start_build
        print(f"✓ Engine FaceNet TensorRT compilada em {build_time:.1f}s!")
        print(f"✓ Arquivo gerado: {engine_dest} ({engine_dest.stat().st_size / 1024**2:.1f} MB)")
    except Exception as trt_err:
        print(f"\n❌ Falha na compilação do FaceNet TensorRT: {trt_err}")
        sys.exit(1)

    # 6. Benchmark Comparativo Real: PyTorch CUDA vs TensorRT FP16
    print("\n" + "=" * 65)
    print("   Benchmarking de Velocidade: PyTorch CUDA vs TensorRT FP16")
    print("=" * 65)

    iterations = 50
    dummy_face = torch.randn(1, 3, 160, 160, device=device)

    # Teste 1: PyTorch
    print("Medindo PyTorch CUDA (InceptionResnetV1)...")
    with torch.no_grad():
        for _ in range(10):  # Warmup
            _ = resnet(dummy_face)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iterations):
            _ = resnet(dummy_face)
        torch.cuda.synchronize()
        duration_pt = (time.time() - t0) / iterations
        ms_pt = duration_pt * 1000
        fps_pt = 1.0 / duration_pt

    # Teste 2: TensorRT
    print("Medindo TensorRT FP16 (facenet.engine)...")
    runtime = trt.Runtime(logger)
    with open(engine_dest, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output_tensor = torch.empty((1, 512), dtype=torch.float32, device=device)
    context.set_input_shape("input", (1, 3, 160, 160))
    context.set_tensor_address("input", dummy_face.data_ptr())
    context.set_tensor_address("embedding", output_tensor.data_ptr())

    stream_ptr = torch.cuda.current_stream().cuda_stream
    for _ in range(10):  # Warmup
        context.execute_async_v3(stream_ptr)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iterations):
        context.execute_async_v3(stream_ptr)
    torch.cuda.synchronize()
    duration_trt = (time.time() - t0) / iterations
    ms_trt = duration_trt * 1000
    fps_trt = 1.0 / duration_trt

    speedup = (ms_pt / ms_trt) if ms_trt > 0 else 1.0

    print("\n📊 Resultados do Benchmark FaceNet na GPU GTX 1660:")
    print(f"  • PyTorch CUDA:         {ms_pt:.2f} ms por rosto  (~{fps_pt:.1f} rostos/segundo)")
    print(f"  • TensorRT FP16:        {ms_trt:.2f} ms por rosto  (~{fps_trt:.1f} rostos/segundo)")
    print(f"  🚀 Ganho de Velocidade: {speedup:.2f}x mais rápido (+{int((speedup - 1) * 100)}%)!")

    print("\n" + "=" * 65)
    print("✓ Concluído com sucesso! O Cerberus usará o FaceNet TensorRT.")
    print("Para aplicar imediatamente no serviço:")
    print("   sudo systemctl restart cerberus")
    print("=" * 65)

if __name__ == "__main__":
    main()
