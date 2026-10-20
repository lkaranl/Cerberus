#!/usr/bin/env python3
"""
Cerberus - Teste e Diagnóstico de Decodificação por Hardware NVIDIA NVDEC
Verifica o suporte a hardware decoding (NVDEC / CUVID) no Nobara Linux (GTX 1660)
comparando com a decodificação tradicional por software na CPU.
"""

import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"   {title}")
    print("=" * 65)

def check_system_ffmpeg():
    print("🔍 [1/3] Verificando FFmpeg do sistema e aceleração NVIDIA...")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("❌ FFmpeg não encontrado no PATH do sistema.")
        return False, []

    print(f"✓ Binário FFmpeg: {ffmpeg_bin}")
    try:
        hw_res = subprocess.run([ffmpeg_bin, "-hwaccels"], capture_output=True, text=True, timeout=5)
        hwaccels = hw_res.stdout.splitlines()
        print(f"✓ Aceleradores de hardware detectados no FFmpeg: {[h.strip() for h in hwaccels if h.strip() and not h.startswith('Hardware')]}")

        dec_res = subprocess.run([ffmpeg_bin, "-decoders"], capture_output=True, text=True, timeout=5)
        nvdec_codecs = [line.strip() for line in dec_res.stdout.splitlines() if "cuvid" in line]
        if nvdec_codecs:
            print(f"✓ Decodificadores NVDEC/CUVID encontrados ({len(nvdec_codecs)}):")
            for c in nvdec_codecs[:5]:
                print(f"    • {c}")
            return True, nvdec_codecs
        else:
            print("⚠️ Decodificadores CUVID (h264_cuvid/hevc_cuvid) não listados no FFmpeg.")
            return False, []
    except Exception as e:
        print(f"❌ Erro ao inspecionar FFmpeg: {e}")
        return False, []

def check_opencv_build():
    print("\n🔍 [2/3] Verificando suporte a aceleração no OpenCV...")
    try:
        import cv2
        print(f"✓ OpenCV Version: {cv2.__version__}")
        has_cuda_accel = hasattr(cv2, "VIDEO_ACCELERATION_CUDA")
        has_any_accel = hasattr(cv2, "VIDEO_ACCELERATION_ANY")
        print(f"✓ Flag VIDEO_ACCELERATION_CUDA disponível: {has_cuda_accel}")
        print(f"✓ Flag VIDEO_ACCELERATION_ANY disponível: {has_any_accel}")

        build_info = cv2.getBuildInformation()
        has_gstreamer = "GStreamer:                   YES" in build_info
        print(f"✓ Suporte a GStreamer no OpenCV: {'SIM' if has_gstreamer else 'NÃO'}")
        return has_gstreamer
    except Exception as e:
        print(f"❌ Erro ao verificar OpenCV: {e}")
        return False

def test_camera_stream():
    print("\n🔍 [3/3] Testando decodificação de stream RTSP...")
    cam_config = Path("config/cameras.json")
    if not cam_config.exists():
        print("❌ config/cameras.json não encontrado.")
        return

    with open(cam_config, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    active_cams = [c for c in cameras if c.get("active", True)]
    if not active_cams:
        print("❌ Nenhuma câmera ativa em config/cameras.json.")
        return

    cam = active_cams[0]
    rtsp_url = cam["url"]
    cam_name = cam["name"]
    print(f"✓ Testando com câmera: '{cam_name}'")
    print(f"✓ URL: {rtsp_url.split('@')[-1] if '@' in rtsp_url else rtsp_url}")

    import cv2
    import os

    # Teste 1: OpenCV Padrão (Software na CPU)
    print("\n--- Teste A: OpenCV Padrão (Decodificação CPU) ---")
    t0 = time.time()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Não foi possível conectar à câmera no Teste A.")
        return

    frames_cpu = 0
    t_start = time.time()
    for _ in range(60):
        ret, frame = cap.read()
        if ret:
            frames_cpu += 1
        else:
            break
    elapsed_cpu = time.time() - t_start
    cap.release()
    fps_cpu = frames_cpu / elapsed_cpu if elapsed_cpu > 0 else 0
    print(f"✓ CPU Decode: {frames_cpu} frames lidos em {elapsed_cpu:.2f}s ({fps_cpu:.1f} FPS)")

    # Função auxiliar para ler decodificador na GPU via nvidia-smi
    def get_gpu_decoder_utilization():
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.decoder,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0:
                parts = res.stdout.strip().split(",")
                dec_util = parts[0].strip()
                gpu_util = parts[1].strip() if len(parts) > 1 else "?"
                return f"NVDEC: {dec_util}% | GPU Core: {gpu_util}%"
        except Exception:
            pass
        return "N/A"

    # Teste 2: OpenCV com Opções FFmpeg NVDEC
    print("\n--- Teste B: OpenCV com Flag FFmpeg NVDEC (hwaccel;cuda) ---")
    orig_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|hwaccel;cuda|hwaccel_device;0|vsync;0"

    cap_hw = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap_hw.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap_hw.isOpened():
        frames_hw = 0
        t_start_hw = time.time()
        for i in range(60):
            ret, frame = cap_hw.read()
            if ret:
                frames_hw += 1
                if i == 30:
                    gpu_stat_b = get_gpu_decoder_utilization()
            else:
                break
        elapsed_hw = time.time() - t_start_hw
        cap_hw.release()
        fps_hw = frames_hw / elapsed_hw if elapsed_hw > 0 else 0
        print(f"✓ NVDEC Decode (OpenCV env): {frames_hw} frames em {elapsed_hw:.2f}s ({fps_hw:.1f} FPS) [{gpu_stat_b if 'gpu_stat_b' in locals() else 'N/A'}]")
    else:
        print("⚠️ OpenCV não conseguiu inicializar com opções hwaccel;cuda.")

    # Restaura opções
    if orig_options:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = orig_options
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

    # Teste 2.1: OpenCV com parâmetros diretos CAP_PROP_HW_ACCELERATION (sem CAP_PROP_HW_DEVICE)
    print("\n--- Teste B.2: OpenCV com CAP_PROP_HW_ACCELERATION (VIDEO_ACCELERATION_ANY) ---")
    try:
        cap_hw_prop = cv2.VideoCapture(
            rtsp_url,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        )
        cap_hw_prop.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap_hw_prop.isOpened():
            frames_prop = 0
            t_start_prop = time.time()
            for i in range(60):
                ret, frame = cap_hw_prop.read()
                if ret:
                    frames_prop += 1
                    if i == 30:
                        gpu_stat_prop = get_gpu_decoder_utilization()
                else:
                    break
            elapsed_prop = time.time() - t_start_prop
            cap_hw_prop.release()
            fps_prop = frames_prop / elapsed_prop if elapsed_prop > 0 else 0
            print(f"✓ Direct Prop Decode: {frames_prop} frames em {elapsed_prop:.2f}s ({fps_prop:.1f} FPS) [{gpu_stat_prop if 'gpu_stat_prop' in locals() else 'N/A'}]")
        else:
            print("⚠️ OpenCV não abriu com CAP_PROP_HW_ACCELERATION.")
    except Exception as e_prop:
        print(f"⚠️ Erro ao testar CAP_PROP_HW_ACCELERATION: {e_prop}")

    # Teste 3: FFmpeg Nativo Pipe NVDEC
    print("\n--- Teste C: FFmpeg Nativo com NVDEC (Pipe Subprocess) ---")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        w, h = 1920, 1080
        if 'frame' in locals() and frame is not None:
            h, w = frame.shape[:2]
        frame_bytes_size = w * h * 3

        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-loglevel", "error",
            "-hwaccel", "cuda",
            "-hwaccel_device", "0",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        p = None
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes_size * 2)
            frames_pipe = 0
            t_pipe_start = time.time()
            for i in range(60):
                raw = p.stdout.read(frame_bytes_size)
                if len(raw) == frame_bytes_size:
                    frames_pipe += 1
                    if i == 30:
                        gpu_stat_pipe = get_gpu_decoder_utilization()
                else:
                    break
            elapsed_pipe = time.time() - t_pipe_start
            fps_pipe = frames_pipe / elapsed_pipe if elapsed_pipe > 0 else 0
            print(f"✓ FFmpeg NVDEC Pipe: {frames_pipe} frames ({w}x{h}) em {elapsed_pipe:.2f}s ({fps_pipe:.1f} FPS) [{gpu_stat_pipe if 'gpu_stat_pipe' in locals() else 'N/A'}]")
        except Exception as pe:
            print(f"⚠️ Teste FFmpeg Pipe falhou: {pe}")
        finally:
            if p is not None:
                try:
                    p.kill()
                    p.wait(timeout=1)
                except Exception:
                    pass

    # Teste 4: Benchmark Direto do FFmpeg na linha de comando
    print("\n--- Teste D: Benchmark Direto do FFmpeg (-hwaccel cuda -> null sink) ---")
    if ffmpeg_bin:
        try:
            bench_cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-rtsp_transport", "tcp",
                "-hwaccel", "cuda",
                "-hwaccel_device", "0",
                "-i", rtsp_url,
                "-frames:v", "60",
                "-f", "null",
                "-",
                "-benchmark"
            ]
            t_bench = time.time()
            res = subprocess.run(bench_cmd, capture_output=True, text=True, timeout=10)
            elapsed_bench = time.time() - t_bench
            gpu_bench = get_gpu_decoder_utilization()
            stderr_lines = [l for l in res.stderr.splitlines() if "bench:" in l or "frame=" in l or "speed=" in l]
            print(f"✓ FFmpeg Benchmark concluído em {elapsed_bench:.2f}s [{gpu_bench}]")
            for l in stderr_lines[-3:]:
                print(f"    • {l}")
        except Exception as be:
            print(f"⚠️ Benchmark FFmpeg falhou: {be}")

    print("\n" + "=" * 65)
    print("✓ Diagnóstico concluído com sucesso!")
    print("=" * 65)

if __name__ == "__main__":
    print_header("Cerberus - Diagnóstico e Teste de Decodificação NVDEC")
    has_nvdec, codecs = check_system_ffmpeg()
    has_gst = check_opencv_build()
    test_camera_stream()
