#!/usr/bin/env python3
"""
Cerberus - Teste de Carga de Streaming Web (WebSocket Multi-Cliente)
Simula múltiplos clientes simultâneos (navegadores/celulares) conectados
ao endpoint WebSocket de vídeo do Cerberus para medir:
- Taxa de quadros entregue a cada cliente (FPS);
- Consumo de largura de banda por cliente (KB/s);
- Jitter e estabilidade da transmissão;
- Impacto de escalabilidade (1 vs 3 vs 6 clientes concorrentes).
"""

import sys
import time
import asyncio
from pathlib import Path
import numpy as np

# Garante raiz do repositório no sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"   {title}")
    print("=" * 65)

async def run_client_worker(client_id: int, camera_id: str, port: int, duration: float, results_dict: dict):
    """Simula o comportamento de um navegador real consumindo stream WebSocket."""
    try:
        import websockets
    except ImportError:
        print("❌ Pacote 'websockets' não encontrado no venv. Execute: pip install websockets")
        return

    uri = f"ws://127.0.0.1:{port}/ws/stream/{camera_id}"
    frame_count = 0
    total_bytes = 0
    intervals = []
    last_time = time.time()

    try:
        async with websockets.connect(uri, max_size=10**7, ping_interval=None) as ws:
            start_t = time.time()
            while (time.time() - start_t) < duration:
                # Solicita o próximo frame (mesmo protocolo do navegador frontend)
                await ws.send("next")
                msg = await ws.recv()

                now = time.time()
                intervals.append((now - last_time) * 1000)
                last_time = now

                frame_count += 1
                total_bytes += len(msg)

        results_dict[client_id] = {
            "success": True,
            "frames": frame_count,
            "bytes": total_bytes,
            "fps": frame_count / duration if duration > 0 else 0,
            "bandwidth_kbs": (total_bytes / 1024) / duration if duration > 0 else 0,
            "jitter": np.std(intervals) if len(intervals) > 2 else 0.0
        }
    except Exception as e:
        results_dict[client_id] = {
            "success": False,
            "error": str(e)
        }

async def benchmark_scenario(client_count: int, camera_id: str, port: int, duration: float):
    print(f"\n🚀 Testando cenário com {client_count} cliente(s) simultâneo(s)...")
    results = {}
    tasks = [
        run_client_worker(i, camera_id, port, duration, results)
        for i in range(client_count)
    ]
    await asyncio.gather(*tasks)
    return results

def main():
    print_header("Cerberus - Teste de Carga de Streaming WebSocket")

    port = settings.port
    cam_id = "cam-0"
    duration = 6.0

    print(f"✓ Alvo: ws://127.0.0.1:{port}/ws/stream/{cam_id}")
    print(f"✓ Duração de cada bateria: {duration}s")
    print("✓ Certifique-se de que o Cerberus está ativo no servidor.")

    scenarios = [1, 3, 5]
    summary = []

    for num_clients in scenarios:
        res = asyncio.run(benchmark_scenario(num_clients, cam_id, port, duration))

        successful = [v for v in res.values() if v.get("success")]
        if not successful:
            print(f"❌ Falha de conexão: O serviço Cerberus está ativo na porta {port}?")
            print("   Execute: sudo systemctl status cerberus")
            sys.exit(1)

        avg_fps = np.mean([v["fps"] for v in successful])
        total_bandwidth = sum([v["bandwidth_kbs"] for v in successful])
        avg_jitter = np.mean([v["jitter"] for v in successful])

        summary.append({
            "clients": num_clients,
            "avg_fps": avg_fps,
            "bandwidth_kbs": total_bandwidth,
            "jitter": avg_jitter
        })
        time.sleep(1)

    print_header("📊 RELATÓRIO DE ESCALABILIDADE DE CLIENTES WEB")
    print(f"{'Clientes Conectados':<22} | {'FPS Médio / Cliente':<20} | {'Banda Total (KB/s)':<20} | {'Jitter'}")
    print("-" * 75)

    for s in summary:
        print(
            f"{s['clients']:<22} | "
            f"{s['avg_fps']:>14.1f} FPS  | "
            f"{s['bandwidth_kbs']:>14.1f} KB/s  | "
            f"{s['jitter']:>6.1f} ms"
        )

    print("=" * 75)
    print("✓ Diagnóstico de streaming web concluído com sucesso!")
    print("=" * 75)

if __name__ == "__main__":
    main()
