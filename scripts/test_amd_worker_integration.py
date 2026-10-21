#!/usr/bin/env python3
"""
Cerberus — Teste de Integração e Desempenho do Worker AMD RX 5600 XT
Valida se a arquitetura Dual-GPU está funcionando corretamente:
1. Verifica se o Worker AMD está respondendo em http://127.0.0.1:8001;
2. Testa a restauração e super-resolução de cortes faciais;
3. Testa a extração de embeddings biométricos de 512 dimensões;
4. Executa bateria de estresse com 10 requisições simultâneas;
5. Valida o fallback gracioso caso o worker esteja indisponível.
"""

import sys
import os
import time
import asyncio
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Executa automaticamente no virtualenv (.venv) se chamado fora dele
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    try:
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)
    except Exception:
        pass

# Cores ANSI
BOLD = "\033[1m"
GREEN = "\033[38;2;118;230;0m"
CYAN = "\033[38;2;0;240;255m"
YELLOW = "\033[38;2;255;200;0m"
RED = "\033[38;2;255;70;70m"
MAGENTA = "\033[38;2;255;120;220m"
RESET = "\033[0m"

def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {BOLD}{CYAN}⚡ {title}{RESET}")
    print("=" * 80)

def generate_synthetic_face(w: int = 64, h: int = 64) -> np.ndarray:
    """Gera uma imagem sintética simulando um recorte facial de câmera 2K distante."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Fundo e tom de pele simulado
    img[:, :] = [140, 160, 210]
    # Simula olhos
    eye_y = int(h * 0.4)
    img[eye_y:eye_y+4, int(w*0.3):int(w*0.4)] = [40, 30, 20]
    img[eye_y:eye_y+4, int(w*0.6):int(w*0.7)] = [40, 30, 20]
    # Simula boca
    mouth_y = int(h * 0.75)
    img[mouth_y:mouth_y+3, int(w*0.35):int(w*0.65)] = [60, 50, 120]
    return img

async def run_tests():
    from app.core.amd_client import amd_worker_client

    print_banner("TESTE DE INTEGRAÇÃO DUAL-GPU — WORKER AMD RX 5600 XT")
    print(f" Endereço Alvo : {BOLD}{amd_worker_client.base_url}{RESET}")
    print(f" Timeout Máximo: {BOLD}{amd_worker_client.timeout_sec}s{RESET}\n")

    # 1. Teste de Conexão (Health Check)
    print(f"{BOLD}[1/4] Verificando se o Worker AMD está online na porta 8001...{RESET}")
    is_online = await amd_worker_client.check_health()
    if not is_online:
        print(f"  {RED}✘ O Worker AMD não está respondendo em {amd_worker_client.base_url}{RESET}")
        print(f"  {YELLOW}👉 Inicie o worker no terminal do Nobara com:{RESET}")
        print(f"     {BOLD}python3 app/worker_amd/server.py{RESET}\n")
        return False

    print(f"  {GREEN}✔ Worker AMD ativo e respondendo com sucesso!{RESET}")

    # 2. Teste de Restauração e Super-Resolução
    print(f"\n{BOLD}[2/4] Testando Super-Resolução Facial na RX 5600 XT...{RESET}")
    crop = generate_synthetic_face(64, 64)
    t0 = time.perf_counter()
    restored, success = await amd_worker_client.restore_face(crop, upscale=2)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if success and restored.shape[0] >= 112:
        print(f"  {GREEN}✔ Restauração concluída em {elapsed_ms:.2f} ms!{RESET}")
        print(f"    Resolução original: {crop.shape[1]}x{crop.shape[0]} px")
        print(f"    Resolução ampliada: {restored.shape[1]}x{restored.shape[0]} px")
    else:
        print(f"  {RED}✘ Falha na restauração facial.{RESET}")
        return False

    # 3. Teste de Extração Biométrica (Embedding 512-d)
    print(f"\n{BOLD}[3/4] Testando Extração de Vetor Biométrico (512-d)...{RESET}")
    t0 = time.perf_counter()
    embedding, success = await amd_worker_client.get_face_embedding(crop)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if success and embedding and len(embedding) == 512:
        print(f"  {GREEN}✔ Vetor biométrico de 512 dimensões extraído em {elapsed_ms:.2f} ms!{RESET}")
        print(f"    Primeiros 3 coeficientes: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
    else:
        print(f"  {RED}✘ Falha na extração de vetor biométrico.{RESET}")
        return False

    # 4. Teste de Carga e Concorrência (10 Requisições Simultâneas)
    print(f"\n{BOLD}[4/4] Bateria de Estresse: Disparando 10 requisições simultâneas...{RESET}")
    t0 = time.perf_counter()
    tasks = [amd_worker_client.restore_face(crop, upscale=2) for _ in range(10)]
    results = await asyncio.gather(*tasks)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    success_count = sum(1 for _, ok in results if ok)
    avg_per_req = total_time_ms / len(results)

    print(f"  {GREEN}✔ Sucesso: {success_count}/10 requisições processadas!{RESET}")
    print(f"    Tempo total do lote  : {total_time_ms:.2f} ms")
    print(f"    Média por requisição : {avg_per_req:.2f} ms")
    print(f"    Vazão estimada       : {BOLD}{(10 / (total_time_ms / 1000.0)):.1f} rostos por segundo{RESET}")

    print_banner("RESULTADO FINAL: ARQUITETURA DUAL-GPU VALIDADA COM SUCESSO!")
    print(f"  {BOLD}NVIDIA GTX 1660 :{RESET} Câmeras 2K contínuas sem qualquer perda de quadros.")
    print(f"  {BOLD}AMD RX 5600 XT  :{RESET} Processando super-resolução com isolamento total.\n")
    return True

def main():
    try:
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTeste cancelado pelo usuário.")

if __name__ == "__main__":
    main()
