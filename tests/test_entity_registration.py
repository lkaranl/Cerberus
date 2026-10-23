import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.api.routes import router, sanitize_entity_name, ClassifyEntityRequest

def test_sanitize_entity_name():
    assert sanitize_entity_name("Karan") == "Karan"
    assert sanitize_entity_name("Meu/Carro:Honda*") == "MeuCarroHonda"
    assert sanitize_entity_name("  Gato   Felix  ") == "Gato Felix"
    assert sanitize_entity_name("Morador... ") == "Morador"
    assert sanitize_entity_name("Thor (Cachorro)") == "Thor (Cachorro)"

def test_classify_entity_request_model():
    req1 = ClassifyEntityRequest(event_id="evt_123", person_name="Karan")
    assert req1.event_id == "evt_123"
    assert req1.entity_type == "person"
    assert req1.person_name == "Karan"

    req2 = ClassifyEntityRequest(
        event_id="evt_456",
        entity_name="Meu Civic",
        entity_type="vehicle",
        license_plate="BRA-2E19"
    )
    assert req2.entity_name == "Meu Civic"
    assert req2.entity_type == "vehicle"
    assert req2.license_plate == "BRA-2E19"

    req3 = ClassifyEntityRequest(
        event_id="evt_789",
        entity_name="Thor",
        entity_type="pet"
    )
    assert req3.entity_name == "Thor"
    assert req3.entity_type == "pet"

def test_routes_registered():
    route_paths = [r.path for r in router.routes]
    
    # Valida presença das novas rotas e das rotas retrocompatíveis
    assert "/api/candidates" in route_paths
    assert "/api/faces/candidates" in route_paths
    assert "/api/entities/known" in route_paths
    assert "/api/entities/classify-event" in route_paths
    assert "/api/faces/classify-event" in route_paths
    assert "/api/entities/upload" in route_paths
