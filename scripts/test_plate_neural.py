#!/usr/bin/env python3
"""
Script de validação e benchmark do motor LPR / OCR Neural (PP-OCRv4).
Testa leitura da placa do carro do usuário e recortes gravados pelo sistema.
"""

import os
import sys
import glob
import time
import cv2

# Assegura que o diretório raiz esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.detector.plate_reader import plate_reader
from app.detector.visual_matcher import visual_matcher

def main():
    print("=" * 65)
    print("  CERBERUS — BENCHMARK DO MOTOR LPR NEURAL (FASE 1)")
    print("=" * 65)
    print(f"Motor OCR Ativo: {plate_reader.ocr_engine}")
    if plate_reader.ocr_engine == "rapidocr":
        print("  -> Status: PP-OCRv4 Neural SOTA ativo (DBNet + SVTR / ONNX Runtime)")
    elif plate_reader.ocr_engine == "pytesseract":
        print("  -> Status: PyTesseract ativo (Fallback clássico)")
    else:
        print(f"  -> Status: {plate_reader.ocr_engine}")

    visual_matcher.load_vehicles()
    print("\nVeículos Cadastrados na Base:")
    for name, data in visual_matcher.known_vehicles.items():
        print(f"  🚗 {name} — Placa Registrada: '{data['plate']}'")

    # 1. Teste na foto de referência cadastrada
    ref_image_path = "data/known_vehicles/Meu_Carro_[NEC4J83]/image.png"
    if os.path.exists(ref_image_path):
        print("\n" + "-" * 65)
        print(f"[TESTE 1] Foto de Referência: {ref_image_path}")
        img = cv2.imread(ref_image_path)
        if img is not None:
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            plate = plate_reader.read_plate(img)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            print(f"  Dimensões da Imagem: {w}x{h}")
            print(f"  Tempo de Inferência: {elapsed_ms:.1f} ms")
            print(f"  Placa Reconhecida:   '{plate}'")
            if plate == "NEC4J83":
                print("  Resultado:           ✅ SUCESSO ABSOLUTO (Placa Mercosul correta)")
            elif plate:
                print(f"  Resultado:           ⚠️ Leitura parcial ou divergente: '{plate}'")
            else:
                print("  Resultado:           ❌ Não identificada nesta foto")
    else:
        print(f"\nFoto de referência não encontrada em {ref_image_path}")

    # 2. Teste nos recortes reais de eventos gravados pelas câmeras
    event_crops = sorted(glob.glob("data/events/*vehicle*crop.jpg"))
    if event_crops:
        print("\n" + "-" * 65)
        print(f"[TESTE 2] Recortes Reais de Eventos das Câmeras ({len(event_crops)} encontrados)")
        for f in event_crops[-5:]:
            img = cv2.imread(f)
            if img is None:
                continue
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            plate = plate_reader.read_plate(img)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            v_name, conf, is_known, p_final = visual_matcher.identify_vehicle(img, plate)
            print(f"\n  Arquivo: {os.path.basename(f)} ({w}x{h})")
            print(f"    -> Tempo: {elapsed_ms:.1f} ms | Placa: '{plate}'")
            print(f"    -> Identificação: {v_name} (confiança: {conf:.2f}, conhecido: {is_known})")
    else:
        print("\nNenhum recorte em data/events/*vehicle*crop.jpg encontrado para teste secundário.")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
