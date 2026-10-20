#!/usr/bin/env python3
"""
Cerberus - Teste de Latência e Estabilidade de Múltiplos Streams RTSP
Conecta a todas as câmeras ativas em paralelo por 10 segundos, medindo:
- FPS real de entrega de cada câmera;
- Jitter e variação de tempo entre frames;
- Taxa de descarte de quadros (frame drops);
- Carga de NVDEC e GPU em tempo real.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np

# Garante que a raiz do projeto esteja no sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"   {title}")
    print("=" * 65)

def get_gpu_stats():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.decoder,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return f"NVDEC: {parts[0]}% | GPU Core: {parts[1]}% | VRAM: {parts[2]} MB"
    except Exception:
        pass
    return "N/A"

def main():
    print_header("Cerberus - Teste de Múltiplos Streams & Detecção de Jitter")

    cam_config_file = Path("config/cameras.json")
    if not cam_config_file.exists():
        print("❌ config/cameras.json não encontrado.")
        sys.exit(1)

    with open(cam_config_file, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    active_cams = [c for c in cameras if c.get("active", True)]
    if not active_cams:
        print("❌ Nenhuma câmera ativa encontrada em config/cameras.json.")
        sys.exit(1)

    print(f"✓ Câmeras Ativas para Teste Simultâneo: {len(active_cams)}")
    for c in active_cams:
        print(f"    • [{c['id']}] {c['name']}")

    from app.stream.capture import RTSPCapture

    captures = []
    for c in active_cams:
        cap = RTSPCapture(c["id"], c["name"], c["url"])
        cap.start()
        captures.append((c, cap))

    print("\n⏳ Aguardando conexão dos streams (3 segundos)...")
    time.sleep(3)

    print("\n🚀 Medindo fluxo contínuo durante 10 segundos...")
    start_time = time.time()
    measure_duration = 10.0

    frame_intervals = {c["id"]: [] for c, _ in captures}
    last_frame_timestamps = {c["id"]: time.time() for c, _ in captures}
    total_frames = {c["id"]: 0 for c, _ in captures}

    while (time.time() - start_time) < measure_duration:
        now = time.time()
        for c, cap in captures:
            frame = cap.get_latest_frame()
            if frame is not None:
                interval = (now - last_frame_timestamps[c["id"]]) * 1000
                frame_intervals[c["id"]].append(interval)
                last_frame_timestamps[c["id"]] = now
                total_frames[c["id"]] += 1
        time.sleep(0.005)

    gpu_telemetry = get_gpu_stats()

    # Encerra capturas
    for _, cap in captures:
        cap.stop()

    print_header("📊 RESULTADOS DO TESTE DE FLUXO RTSP SIMULTÂNEO")
    print(f"Telemetria da GPU durante teste: [{gpu_telemetry}]\n")
    print(f"{'Câmera':<25} | {'Frames':<8} | {'FPS Real':<10} | {'Jitter Médio':<12} | {'Status'}")
    print("-" * 65)

    for c, cap in captures:
        cid = c["id"]
        count = total_frames[cid]
        real_fps = count / measure_duration
        intervals = frame_intervals[cid]
        # Jitter = desvio padrão do intervalo entre frames (ms)
        jitter = np.std(intervals) if len(intervals) > 2 else 0.0

        if real_fps >= 15.0 and jitter < 25.0:
            status = "⚡ Perfeito (Zero Lag)"
        elif real_fps >= 10.0:
            status = "✓ Estável"
        else:
            status = "⚠️ Instável / Perda de Frames"

        backend_tag = "NVDEC" if getattr(cap, "is_nvdec_active", False) else "OpenCV"
        cam_label = f"{c['name']} [{backend_tag}]"
        print(f"{cam_label:<25} | {count:<8} | {real_fps:>6.1f} FPS  | {jitter:>8.1f} ms   | {status}")

    print("=" * 65)
    print("✓ Teste de múltiplos streams concluído!")
    print("=" * 65)

if __name__ == "__main__":
    main()
