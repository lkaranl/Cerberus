import re
from typing import Optional, List, Tuple
import cv2
import numpy as np

from app.core.logging import logger

class PlateReader:
    """
    Localizador e leitor neural de placas veiculares (ANPR / LPR).
    Otimizado para placas brasileiras (Mercosul e Tradicional).
    Utiliza PP-OCRv4 (DBNet + SVTR via ONNX Runtime) como motor SOTA primário,
    com reparo morfológico de caracteres e fallback resiliente.
    """
    def __init__(self):
        # Padrões de placas do Brasil
        # Mercosul: 3 letras, 1 número, 1 letra, 2 números (ex: BRA2E19, NEC4J83)
        self.mercosul_regex = re.compile(r"\b([A-Z]{3}[0-9][A-Z0-9][0-9]{2})\b")
        # Tradicional: 3 letras e 4 números (ex: ABC1234)
        self.tradicional_regex = re.compile(r"\b([A-Z]{3}-?[0-9]{4})\b")

        # Mapeamentos de confusão comum em OCR entre letras e dígitos
        self._char_to_digit = {"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0"}
        self._digit_to_char = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

        self.ocr_engine = None
        self.rapid_ocr = None
        self.pytesseract = None
        self.easyocr_reader = None

        self._init_ocr()

    def _init_ocr(self):
        """Inicializa o motor de OCR priorizando PP-OCRv4 (RapidOCR / ONNX Runtime)."""
        # 1. Tenta RapidOCR (PP-OCRv4 SOTA via ONNX Runtime)
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.rapid_ocr = RapidOCR()
            self.ocr_engine = "rapidocr"
            logger.info("⚡ Módulo LPR inicializado com PP-OCRv4 Neural SOTA (RapidOCR / ONNX Runtime).")
            return
        except Exception as e:
            logger.debug(f"RapidOCR não disponível ou não instalado: {e}")

        # 2. Fallback para PyTesseract
        try:
            import pytesseract
            self.ocr_engine = "pytesseract"
            self.pytesseract = pytesseract
            logger.info("Módulo OCR de placas inicializado com PyTesseract (Fallback).")
            return
        except ImportError:
            pass

        # 3. Fallback para EasyOCR
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(["en"], gpu=True)
            self.ocr_engine = "easyocr"
            logger.info("Módulo OCR de placas inicializado com EasyOCR (Fallback).")
            return
        except ImportError:
            pass

        logger.warning("Nenhum motor de OCR instalado. Leitor de placas operando exclusivamente por heurística.")

    def validate_and_repair_plate(self, raw_candidate: str) -> Optional[str]:
        """
        Valida e repara potenciais erros de confusão letra/número em placas brasileiras
        respeitando a sintaxe estrita do padrão Mercosul (LLL N L NN) e Tradicional (LLL NNNN).
        Suporta placas completas (7 caracteres) e prefixos truncados na borda (6 caracteres).
        """
        if not raw_candidate:
            return None

        # Limpeza preliminar: remove caracteres não alfanuméricos e converte para maiúsculo
        clean = re.sub(r"[^A-Z0-9]", "", raw_candidate.upper().strip())

        # Caso 1: Candidato tem exatamente 7 caracteres
        if len(clean) == 7:
            chars = list(clean)
            # Regra 1: Posições 0, 1, 2 DEVEM ser letras
            for i in range(3):
                if chars[i].isdigit() and chars[i] in self._digit_to_char:
                    chars[i] = self._digit_to_char[chars[i]]
            # Regra 2: Posição 3 DEVE ser dígito numérico (0-9)
            if chars[3].isalpha() and chars[3] in self._char_to_digit:
                chars[3] = self._char_to_digit[chars[3]]
            # Regra 3: Posições 5 e 6 DEVEM ser dígitos numéricos (0-9)
            for i in (5, 6):
                if chars[i].isalpha() and chars[i] in self._char_to_digit:
                    chars[i] = self._char_to_digit[chars[i]]

            repaired = "".join(chars)
            if self.mercosul_regex.fullmatch(repaired) or self.tradicional_regex.fullmatch(repaired):
                return repaired

        # Caso 2: Candidato é maior que 7 caracteres (ex: texto concatenado 'BRASILNEC4J83')
        elif len(clean) > 7:
            for i in range(len(clean) - 6):
                sub = clean[i:i+7]
                res = self.validate_and_repair_plate(sub)
                if res:
                    return res

        # Caso 3: Candidato tem 6 caracteres (ex: último caractere parcialmente cortado na borda como 'NEC4J8')
        elif len(clean) == 6:
            chars = list(clean)
            for i in range(3):
                if chars[i].isdigit() and chars[i] in self._digit_to_char:
                    chars[i] = self._digit_to_char[chars[i]]
            if chars[3].isalpha() and chars[3] in self._char_to_digit:
                chars[3] = self._char_to_digit[chars[3]]
            if chars[5].isalpha() and chars[5] in self._char_to_digit:
                chars[5] = self._char_to_digit[chars[5]]

            repaired = "".join(chars)
            if re.match(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]$", repaired):
                return repaired

        return None

    def _read_with_rapidocr(self, img_bgr: np.ndarray) -> Optional[str]:
        """Executa inferência neural com PP-OCRv4 via RapidOCR."""
        if self.rapid_ocr is None or img_bgr is None or img_bgr.size == 0:
            return None

        try:
            h, w = img_bgr.shape[:2]
            # Otimização de latência e resolução:
            # Imagens 4K ou excessivamente grandes causam lentidão desnecessária e prejudicam o receptive field do DBNet
            if max(h, w) > 1280:
                scale = 1280.0 / float(max(h, w))
                proc_img = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                proc_img = img_bgr

            results, elapse = self.rapid_ocr(proc_img)
            if not results:
                return None

            best_plate = None
            best_score = 0.0

            for item in results:
                if len(item) < 3:
                    continue
                _box, text, score = item[0], item[1], float(item[2])

                plate = self.validate_and_repair_plate(text)
                if plate:
                    # Prioridade para placas completas de 7 caracteres
                    weight = score if len(plate) == 7 else score * 0.92
                    if weight > best_score:
                        best_score = weight
                        best_plate = plate

            if best_plate:
                logger.info(f"⚡ [PP-OCRv4] Placa reconhecida: {best_plate} (conf: {best_score:.2f})")
                return best_plate

        except Exception as e:
            logger.debug(f"Erro na inferência PP-OCRv4: {e}")

        return None

    def _read_with_legacy_ocr(self, plate_roi: np.ndarray) -> str:
        """Executa OCR clássico (Tesseract / EasyOCR) como contingência."""
        if plate_roi is None or plate_roi.size == 0:
            return ""

        if len(plate_roi.shape) == 3:
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_roi

        resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        raw_text = ""
        try:
            if self.ocr_engine == "pytesseract" and self.pytesseract:
                config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 8"
                raw_text = self.pytesseract.image_to_string(thresh, config=config).strip().upper()
            elif self.ocr_engine == "easyocr" and self.easyocr_reader:
                results = self.easyocr_reader.readtext(thresh, detail=0)
                raw_text = "".join(results).strip().upper()
        except Exception:
            pass

        repaired = self.validate_and_repair_plate(raw_text)
        return repaired if repaired else ""

    def read_plate(self, vehicle_crop: np.ndarray) -> str:
        """
        Analisa o recorte do veículo, localiza e decodifica a placa veicular.
        Retorna a placa formatada (ex: 'NEC4J83') ou string vazia se não identificável.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return ""

        h, w = vehicle_crop.shape[:2]

        # -------------------------------------------------------------
        # 1. Rota SOTA: PP-OCRv4 Neural (RapidOCR)
        # -------------------------------------------------------------
        if self.ocr_engine == "rapidocr":
            # A placa em 99% dos casos está localizada nos 65% inferiores do veículo
            roi_y1 = int(h * 0.35)
            lower_crop = vehicle_crop[roi_y1:h, 0:w]

            plate = self._read_with_rapidocr(lower_crop)
            if plate:
                return plate

            # Fallback 1.1: Tenta no veículo completo caso o enquadramento seja atípico
            plate_full = self._read_with_rapidocr(vehicle_crop)
            if plate_full:
                return plate_full

        # -------------------------------------------------------------
        # 2. Rota Legada / Fallback (Tesseract / EasyOCR / Contornos)
        # -------------------------------------------------------------
        roi_y1 = int(h * 0.40)
        lower_crop = vehicle_crop[roi_y1:h, 0:w]

        # Tentativa no recorte inferior
        detected = self._read_with_legacy_ocr(lower_crop)
        if detected:
            return detected

        # Detecção geométrica de contornos retangulares para isolar a placa
        try:
            gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(blur, 50, 200)

            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 4:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    aspect_ratio = float(cw) / max(1, ch)
                    if 2.0 <= aspect_ratio <= 4.5 and cw >= 60 and ch >= 20:
                        candidate_plate = lower_crop[y:y+ch, x:x+cw]
                        plate_text = self._read_with_legacy_ocr(candidate_plate)
                        if plate_text:
                            return plate_text
        except Exception:
            pass

        return ""


plate_reader = PlateReader()
