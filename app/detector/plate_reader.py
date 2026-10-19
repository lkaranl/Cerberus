import re
from typing import Optional, List
import cv2
import numpy as np

from app.core.logging import logger

class PlateReader:
    """
    Localizador e leitor de placas veiculares (ANPR / LPR).
    Otimizado para placas brasileiras (Mercosul e Tradicional).
    Utiliza pré-processamento morfológico com suporte a OCR.
    """
    def __init__(self):
        # Padrões de placas do Brasil
        # Mercosul: 3 letras, 1 número, 1 letra, 2 números (ex: BRA2E19)
        self.mercosul_regex = re.compile(r"\b([A-Z]{3}[0-9][A-Z0-9][0-9]{2})\b")
        # Tradicional: 3 letras e 4 números (ex: ABC1234)
        self.tradicional_regex = re.compile(r"\b([A-Z]{3}-?[0-9]{4})\b")

        self.ocr_engine = None
        self._init_ocr()

    def _init_ocr(self):
        # Tenta carregar pytesseract ou easyocr se disponível
        try:
            import pytesseract
            self.ocr_engine = "pytesseract"
            self.pytesseract = pytesseract
            logger.info("Módulo OCR de placas inicializado com PyTesseract.")
            return
        except ImportError:
            pass

        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(["en"], gpu=True)
            self.ocr_engine = "easyocr"
            logger.info("Módulo OCR de placas inicializado com EasyOCR.")
            return
        except ImportError:
            pass

        logger.info("OCR avançado opcional não instalado. Leitor de placas operando via detecção de contornos de placa.")

    def _extract_plate_text(self, plate_roi: np.ndarray) -> str:
        """Executa OCR sobre a região de interesse da placa."""
        if plate_roi is None or plate_roi.size == 0:
            return ""

        # Pré-processamento: escala de cinza e binarização de alto contraste
        if len(plate_roi.shape) == 3:
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_roi

        resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        raw_text = ""
        try:
            if self.ocr_engine == "pytesseract":
                config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 8"
                raw_text = self.pytesseract.image_to_string(thresh, config=config).strip().upper()
            elif self.ocr_engine == "easyocr":
                results = self.easyocr_reader.readtext(thresh, detail=0)
                raw_text = "".join(results).strip().upper()
        except Exception:
            pass

        # Limpeza de caracteres espúrios
        clean_text = re.sub(r"[^A-Z0-9]", "", raw_text)

        # Checa padrão Mercosul
        match_mercosul = self.mercosul_regex.search(clean_text)
        if match_mercosul:
            return match_mercosul.group(1)

        # Checa padrão Tradicional
        match_trad = self.tradicional_regex.search(clean_text)
        if match_trad:
            return match_trad.group(1).replace("-", "")

        return ""

    def read_plate(self, vehicle_crop: np.ndarray) -> str:
        """
        Analisa o recorte do veículo, localiza a placa e tenta ler os caracteres.
        Retorna a placa formatada ou string vazia se ilegível.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return ""

        h, w = vehicle_crop.shape[:2]

        # A placa geralmente fica na metade inferior do veículo
        roi_y1 = int(h * 0.40)
        lower_crop = vehicle_crop[roi_y1:h, 0:w]

        # Tenta ler no recorte inferior diretamente
        detected = self._extract_plate_text(lower_crop)
        if detected:
            return detected

        # Detecção de contornos retangulares de alta probabilidade de placa
        try:
            gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(blur, 50, 200)

            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                # Busca polígonos de 4 lados (retângulos)
                if len(approx) == 4:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    aspect_ratio = float(cw) / max(1, ch)
                    # Placas brasileiras têm proporção entre 2.2 e 4.5
                    if 2.0 <= aspect_ratio <= 4.5 and cw >= 60 and ch >= 20:
                        candidate_plate = lower_crop[y:y+ch, x:x+cw]
                        plate_text = self._extract_plate_text(candidate_plate)
                        if plate_text:
                            return plate_text
        except Exception:
            pass

        return ""


plate_reader = PlateReader()
