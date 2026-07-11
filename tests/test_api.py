from fastapi.testclient import TestClient

from src.nlp_project.api import app


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["author"] == "Vina RAHARITSIFA - M1 I2AD, INSI"
