#!/usr/bin/env python3
"""
Cerberus — Benchmark Completo de FPS (Taxa de Quadros em Tempo Real)
Mede com precisão cirúrgica o desempenho em todas as camadas do sistema:
1. FPS do Sensor Físico das Câmeras (RTSP nativo);
2. FPS da Decodificação de Vídeo (NVDEC / FFmpeg);
3. FPS do Gateway WebRTC go2rtc (Taxa de envio WebRTC H.264);
4. FPS da Inferência de IA (YOLO11 + ByteTrack + ArcFace);
5. Telemetria Ativa do Servidor Cerberus (Porta 8095).
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

# Suprime logs verbosos de socket/RTSP do OpenCV
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Cores ANSI para terminal
BOLD = "\033[1m"
GREEN = "\033[38;2;118;230;0m"
CYAN = "\033[38;2;0;240;255m"
YELLOW = "\033[38;2;255;200;0m"
RED = "\033[38;2;255;70;70m"
RESET = "\033[0m"

def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {BOLD}{CYAN}⚡ {title}{RESET}")
    print("=" * 80)

def test_camera_native_fps(cam_name: str, rtsp_url: str, duration_sec: float = 4.0):
    """
    Conecta diretamente ao stream RTSP da câmera e conta o FPS real do sensor.
    Isso identifica se a câmera está configurada para 15 FPS ou 25 FPS no app Tapo.
    """
    import cv2
    print(f"  🔍 Analisando taxa de quadros física de: {BOLD}{cam_name}{RESET}...")
    
    t0 = time.time()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"  {RED}❌ Falha ao conectar ao RTSP ({cam_name}){RESET}")
        return 0.0, (0, 0)

    # Descarta os 3 primeiros frames para estabilização de buffer
    for _ in range(3):
        cap.grab()

    frame_count = 0
    w, h = 0, 0
    start_time = time.time()
    end_time = start_time + duration_sec

    while time.time() < end_time:
        if cap.grab():
            frame_count += 1
            if w == 0:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    h, w = frame.shape[:2]

    elapsed = time.time() - start_time
    cap.release()

    fps = round(frame_count / elapsed, 1) if elapsed > 0 else 0.0
    return fps, (w, h)

def test_go2rtc_api():
    """Consulta o gateway go2rtc local (porta 1984) e verifica os streams WebRTC ativos."""
    url = "http://127.0.0.1:1984/api/streams"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Cerberus-Benchmark"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, data
    except Exception as e:
        return False, str(e)

def test_cerberus_telemetry():
    """Consulta o status consolidado de câmeras e inferência do Cerberus (porta 8095)."""
    url = "http://127.0.0.1:8095/api/cameras"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Cerberus-Benchmark"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, data
    except Exception as e:
        return False, str(e)

def main():
    print_banner("CERBERUS AI — DIAGNÓSTICO E BENCHMARK DE TAXA DE QUADROS (FPS)")
    print(f" Servidor Local: {BOLD}Nobara Linux / Python 3.12{RESET}")
    print(f" Aceleração:     {BOLD}NVIDIA NVDEC / CUDA / TensorRT / WebRTC (go2rtc){RESET}")

    # 1. Carrega lista de câmeras
    config_file = REPO_ROOT / "config" / "cameras.json"
    cameras = []
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cameras = json.load(f)
    else:
        print(f"{RED}Arquivo config/cameras.json não encontrado.{RESET}")
        return

    # 2. Teste da Taxa de Quadros Físicos (Sensor RTSP)
    print_banner("FASE 1: TAXA DE QUADROS NATIVA DO SENSOR DAS CÂMERAS (RTSP)")
    print(" Medindo quantos quadros físicos cada câmera IP envia na rede local...")

    camera_results = {}
    for cam in cameras:
        if cam.get("active", True):
            fps, res = test_camera_native_fps(cam["name"], cam["url"], duration_sec=3.5)
            camera_results[cam["id"]] = {
                "name": cam["name"],
                "sensor_fps": fps,
                "res": f"{res[0]}x{res[1]}" if res[0] > 0 else "N/A"
            }
            color = GREEN if fps >= 20.0 else (YELLOW if fps >= 14.0 else RED)
            print(f"  👉 {BOLD}{cam['name']}{RESET}: {color}{fps} FPS{RESET} (Resolução: {res[0]}x{res[1]})")

    # 3. Teste do Gateway WebRTC go2rtc
    print_banner("FASE 2: STATUS DO GATEWAY WEBRTC (go2rtc - Porta 1984)")
    go2rtc_ok, go2rtc_data = test_go2rtc_api()
    if go2rtc_ok:
        print(f"  {GREEN}✅ Gateway go2rtc está ATIVO e respondendo na porta 1984!{RESET}")
        print("  Canais WebRTC registrados no go2rtc:")
        for stream_name, stream_info in go2rtc_data.items():
            producers = stream_info.get("producers", [])
            consumers = stream_info.get("consumers", [])
            print(f"    • Stream {BOLD}{stream_name}{RESET}: {len(producers)} fonte(s) ativa(s), {len(consumers)} cliente(s) WebRTC conectado(s)")
    else:
        print(f"  {YELLOW}⚠️ Gateway go2rtc offline ou não respondeu: {go2rtc_data}{RESET}")
        print(f"  Dica: Execute {BOLD}sudo systemctl restart go2rtc{RESET}")

    # 4. Teste da Telemetria do Cerberus
    print_banner("FASE 3: TELEMETRIA EM TEMPO REAL DO CERBERUS (Porta 8095)")
    cerberus_ok, cerberus_data = test_cerberus_telemetry()
    if cerberus_ok:
        print(f"  {GREEN}✅ Servidor Cerberus Vision está ATIVO e monitorando!{RESET}")
        print("\n  Tabela de Desempenho Consolidado:")
        print(f"  {'Câmera':<22} | {'Sensor RTSP':<13} | {'Stream Web':<12} | {'IA Inferência':<14} | {'Status':<10}")
        print("  " + "-" * 78)
        for cam_stat in cerberus_data:
            c_name = cam_stat.get("name", "Desconhecida")
            capt_fps = cam_stat.get("capture_fps", 0.0)
            stream_fps = cam_stat.get("stream_fps", 0.0)
            inf_fps = cam_stat.get("inference_fps", 0.0)
            conn = "ONLINE" if cam_stat.get("connected") else "OFFLINE"
            conn_color = GREEN if conn == "ONLINE" else RED

            print(f"  {c_name:<22} | {capt_fps:>6.1f} FPS    | {stream_fps:>6.1f} FPS   | {inf_fps:>6.1f} FPS     | {conn_color}{conn:<10}{RESET}")
    else:
        print(f"  {YELLOW}⚠️ Cerberus não respondeu em /api/cameras ({cerberus_data}){RESET}")
        print(f"  Dica: Execute {BOLD}sudo systemctl restart cerberus{RESET}")

    # 5. Conclusão e Recomendações
    print_banner("ANÁLISE E CONCLUSÃO DE DESEMPENHO")
    print(" 💡 Resumo do Diagnóstico:")
    for cid, res in camera_results.items():
        s_fps = res["sensor_fps"]
        if s_fps <= 16.0 and s_fps > 0:
            print(f"  ⚠️  {BOLD}{res['name']}{RESET}: O sensor físico da câmera está emitindo a {BOLD}{s_fps} FPS{RESET}.")
            print(f"     👉 Para alcançar 25 FPS, altere a taxa de quadros no app da Tapo no celular:")
            print(f"        (Configurações da Câmera ➡️ Vídeo e Exibição ➡️ Taxa de Quadros / Qualidade de Vídeo).")
        elif s_fps >= 20.0:
            print(f"  ✅ {BOLD}{res['name']}{RESET}: Taxa física nativa excelente ({BOLD}{s_fps} FPS{RESET})!")

    print(f"\n{BOLD}Para monitorar o stream WebRTC ao vivo no navegador:{RESET}")
    print(f"  Acesse: {CYAN}http://192.168.68.72:8095{RESET}\n")

if __name__ == "__main__":
    main()
