import os
import glob
import cv2
import numpy as np

from app.detector.plate_reader import plate_reader
from app.detector.visual_matcher import visual_matcher

def main():
    visual_matcher.load_vehicles()
    print("Veículos cadastrados:", list(visual_matcher.known_vehicles.keys()))
    for k, v in visual_matcher.known_vehicles.items():
        print(f"  {k} -> Placa: {v['plate']}")

    crops = sorted(glob.glob("data/events/*vehicle*crop.jpg"))
    print(f"\nTotal de recortes de veículos encontrados: {len(crops)}")

    for f in crops[-5:]:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        plate = plate_reader.read_plate(img)
        v_name, conf, is_known, p_final = visual_matcher.identify_vehicle(img, plate)
        print(f"\nArquivo: {os.path.basename(f)} ({w}x{h})")
        print(f"  -> Placa lida: '{plate}'")
        print(f"  -> Identificação: '{v_name}' (confiança: {conf}, conhecido: {is_known})")

if __name__ == "__main__":
    main()
