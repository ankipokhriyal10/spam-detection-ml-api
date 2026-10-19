
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_prediction_schema():
    response = client.post(
        "/predict",
        json={"text": "Congratulations! You won a free prize."}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in ["spam", "not_spam"]
    assert 0.0 <= data["probability"] <= 1.0
    assert "text" in data


def test_empty_text_rejected():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_missing_text_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_invalid_json_field_type():
    response = client.post("/predict", json={"text": 123})
    assert response.status_code == 422


def test_not_found():
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404
