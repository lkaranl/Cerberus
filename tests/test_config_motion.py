import os
import pytest
from app.core.config import Settings

def test_motion_gating_default_settings():
    s = Settings()
    assert s.motion_gating_enabled is True
    assert s.motion_min_area == 0.005
    assert s.motion_grace_period == 3.0

def test_motion_gating_env_override(monkeypatch):
    monkeypatch.setenv("MOTION_GATING_ENABLED", "false")
    monkeypatch.setenv("MOTION_MIN_AREA", "0.02")
    monkeypatch.setenv("MOTION_GRACE_PERIOD", "5.5")
    s = Settings()
    assert s.motion_gating_enabled is False
    assert s.motion_min_area == 0.02
    assert s.motion_grace_period == 5.5
