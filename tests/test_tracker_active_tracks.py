import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.detector.tracker import CameraTrackerManager, TrackedEntity

def test_has_active_tracks_empty():
    manager = CameraTrackerManager("cam-test", "Camera Teste")
    assert manager.has_active_tracks() is False

def test_has_active_tracks_with_entity():
    manager = CameraTrackerManager("cam-test", "Camera Teste")
    entity = TrackedEntity(tracking_id=1, entity_type="person", class_label="Pessoa")
    manager.active_tracks[1] = entity
    assert manager.has_active_tracks() is True
