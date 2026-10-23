import numpy as np
import pytest
from app.detector.motion import MotionDetector

def test_motion_detector_static_frames():
    detector = MotionDetector(min_area_ratio=0.01)
    static_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    
    # Executa warmup
    for _ in range(6):
        has_motion, ratio = detector.detect(static_frame)
    
    # Em fundo estático após warmup, não deve acusar movimento
    has_motion, ratio = detector.detect(static_frame)
    assert has_motion is False
    assert ratio < 0.01

def test_motion_detector_with_movement():
    detector = MotionDetector(min_area_ratio=0.01)
    base_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
    
    # Inicializa fundo
    for _ in range(10):
        detector.detect(base_frame)
        
    # Insere alteração significativa
    moved_frame = base_frame.copy()
    moved_frame[200:500, 400:800] = 255
    has_motion, ratio = detector.detect(moved_frame)
    assert has_motion is True
    assert ratio >= 0.01

def test_motion_detector_empty_frame():
    detector = MotionDetector()
    has_motion, ratio = detector.detect(None)
    assert has_motion is False
    assert ratio == 0.0
