import time
import pytest
from unittest.mock import MagicMock
from app.detector.motion import MotionDetector

def test_motion_gating_decision_logic():
    # Simula variáveis de decisão
    motion_gating_enabled = True
    grace_period = 3.0
    
    # Caso 1: Movimento detectado -> deve inferir
    has_motion = True
    time_since_motion = 10.0
    has_tracks = False
    in_grace_period = time_since_motion <= grace_period
    should_infer = (not motion_gating_enabled) or has_motion or in_grace_period or has_tracks
    assert should_infer is True

    # Caso 2: Sem movimento, mas dentro do período de graça -> deve inferir
    has_motion = False
    time_since_motion = 1.5
    has_tracks = False
    in_grace_period = time_since_motion <= grace_period
    should_infer = (not motion_gating_enabled) or has_motion or in_grace_period or has_tracks
    assert should_infer is True

    # Caso 3: Sem movimento, fora do período de graça, mas com rastros ativos -> deve inferir
    has_motion = False
    time_since_motion = 10.0
    has_tracks = True
    in_grace_period = time_since_motion <= grace_period
    should_infer = (not motion_gating_enabled) or has_motion or in_grace_period or has_tracks
    assert should_infer is True

    # Caso 4: Sem movimento, fora do período de graça e sem rastros -> PULA inferência (Modo Eco)
    has_motion = False
    time_since_motion = 10.0
    has_tracks = False
    in_grace_period = time_since_motion <= grace_period
    should_infer = (not motion_gating_enabled) or has_motion or in_grace_period or has_tracks
    assert should_infer is False

    # Caso 5: Motion gating desativado globalmente -> sempre infere
    motion_gating_enabled = False
    should_infer = (not motion_gating_enabled) or has_motion or in_grace_period or has_tracks
    assert should_infer is True
