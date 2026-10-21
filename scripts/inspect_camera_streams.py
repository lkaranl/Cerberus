#!/usr/bin/env python3
"""
Cerberus — Inspetor de Resolução e Streams RTSP (stream1 vs stream2)
Conecta a cada câmera em config/cameras.json e inspeciona:
- Resolução nativa de vídeo (Largura x Altura);
- Taxa de quadros declarada (FPS);
- Proporção de aspecto (Aspect Ratio);
- Tempo de abertura do fluxo (latência de handshake RTSP).
"""

import sys
import time
import json
import os
from pathlib import Path

# Suprime logs verbosos de socket/RTSP do FFmpeg
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "0"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def print_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def inspect_url(url: str, label: str):
    import cv2
    t0 = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        open_time = (time.time() - t0) * 1000
        return {
            "status": "FAIL",
            "open_ms": open_time,
            "width": 0,
            "height": 0,
            "fps": 0,
            "codec": "N/A"
        }

    # Tenta ler 1 frame para confirmar decodificação e dimensões reais
    ret, frame = cap.read()
    open_time = (time.time() - t0) * 1000

    if not ret or frame is None:
        cap.release()
        return {
            "status": "NO_FRAME",
            "open_ms": open_time,
            "width": 0,
            "height": 0,
            "fps": 0,
            "codec": "N/A"
        }

    real_h, real_w = frame.shape[:2]
    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    return {
        "status": "OK",
        "open_ms": open_time,
        "width": real_w,
        "height": real_h,
        "fps": reported_fps
    }

def main():
    print_header("CERBERUS — COMPARAÇÃO DE RESOLUÇÃO RTSP (STREAM1 vs STREAM2)")

    config_path = Path("config/cameras.json")
    if not config_path.exists():
        print("❌ Arquivo config/cameras.json não encontrado.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    active_cams = [c for c in cameras if c.get("active", True)]
    if not active_cams:
        print("❌ Nenhuma câmera ativa em config/cameras.json.")
        sys.exit(1)

    print(f"Detectadas {len(active_cams)} câmeras ativas. Analisando fluxos RTSP...\n")

    results = []

    for cam in active_cams:
        cid = cam["id"]
        cname = cam["name"]
        base_url = cam["url"]

        # Gera URLs para stream1 e stream2
        if "/stream1" in base_url:
            url_stream1 = base_url
            url_stream2 = base_url.replace("/stream1", "/stream2")
        elif "/stream2" in base_url:
            url_stream1 = base_url.replace("/stream2", "/stream1")
            url_stream2 = base_url
        else:
            url_stream1 = base_url
            url_stream2 = base_url + "/stream2"

        print(f"🔍 Testando [{cid}] {cname}...")
        print(f"   • stream1: {url_stream1.split('@')[-1] if '@' in url_stream1 else url_stream1}")
        res1 = inspect_url(url_stream1, f"{cid} stream1")
        if res1["status"] == "OK":
            print(f"     └─ Resolução: {res1['width']}x{res1['height']} ({res1['width']*res1['height']/1e6:.1f} MP) | FPS: {res1['fps']:.1f} | Handshake: {res1['open_ms']:.0f} ms")
        else:
            print(f"     └─ Status: {res1['status']} (Falha na conexão)")

        print(f"   • stream2: {url_stream2.split('@')[-1] if '@' in url_stream2 else url_stream2}")
        res2 = inspect_url(url_stream2, f"{cid} stream2")
        if res2["status"] == "OK":
            print(f"     └─ Resolução: {res2['width']}x{res2['height']} ({res2['width']*res2['height']/1e6:.1f} MP) | FPS: {res2['fps']:.1f} | Handshake: {res2['open_ms']:.0f} ms")
        else:
            print(f"     └─ Status: {res2['status']} (Falha na conexão ou desabilitado na câmera)")

        results.append({
            "camera": cname,
            "id": cid,
            "res1": res1,
            "res2": res2
        })
        print()

    # Tabela Resumo
    print("=" * 85)
    print(f"{'Câmera':<24} | {'stream1 (Main)':<20} | {'stream2 (Sub)':<20} | {'Diferença':<12}")
    print("-" * 85)

    for item in results:
        r1 = item["res1"]
        r2 = item["res2"]
        str1 = f"{r1['width']}x{r1['height']}" if r1["status"] == "OK" else "Indisponível"
        str2 = f"{r2['width']}x{r2['height']}" if r2["status"] == "OK" else "Indisponível"

        if r1["status"] == "OK" and r2["status"] == "OK":
            ratio = (r1["width"] * r1["height"]) / max(1, (r2["width"] * r2["height"]))
            diff = f"{ratio:.1f}x mais pixels"
        else:
            diff = "N/A"

        print(f"{item['camera'][:23]:<24} | {str1:<20} | {str2:<20} | {diff:<12}")

    print("=" * 85)
    print("\n💡 RECOMENDAÇÃO TÉCNICA CERBERUS:")
    print("• stream1 (Main Stream): É o fluxo de ALTA RESOLUÇÃO (2K / 1080p).")
    print("  👉 Ideal para Reconhecimento Facial SOTA e Leitura de Placas (LPR), pois preserva detalhes finos.")
    print("• stream2 (Sub Stream): É o fluxo de BAIXA RESOLUÇÃO (geralmente 640x360 ou 640x480).")
    print("  👉 Usado apenas se a rede for instável ou a GPU estiver sobrecarregada, mas prejudica rostos distantes.")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
