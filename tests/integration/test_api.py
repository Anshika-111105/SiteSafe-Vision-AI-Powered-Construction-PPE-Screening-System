import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.fixtures.create_fixtures import generate_test_image, generate_corrupt_image, generate_empty_image

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "SiteSafe Vision API"


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "timestamp_utc" in data


def test_api_version():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"
    assert data["application_name"] == "SiteSafe Vision"


def test_api_metadata():
    response = client.get("/metadata")
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "class_mapping" in data


def test_api_predict_valid_image():
    img_bytes = generate_test_image(224, 224)
    response = client.post(
        "/predict",
        files={"file": ("sample_worker.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["FULL_PPE", "PARTIAL_PPE", "NO_PPE"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "recommendation" in data
    assert "request_id" in data
    assert "probabilities" in data


def test_api_predict_batch():
    img1 = generate_test_image(224, 224, color=(100, 120, 140))
    img2 = generate_test_image(224, 224, color=(150, 170, 190))

    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("worker_1.jpg", img1, "image/jpeg")),
            ("files", ("worker_2.jpg", img2, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 2
    assert len(data["results"]) == 2


def test_api_predict_corrupt_image():
    corrupt_bytes = generate_corrupt_image()
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Corrupted or unreadable" in response.json()["detail"]


def test_api_predict_empty_image():
    empty_bytes = generate_empty_image()
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", empty_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Empty file uploaded" in response.json()["detail"]
