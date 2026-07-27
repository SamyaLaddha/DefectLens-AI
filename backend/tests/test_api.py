"""
Unit tests for the FastAPI backend API endpoints (/api/health and /api/predict).
"""

import io
import pytest
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from PIL import Image
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app

@pytest.fixture
def client():
    """
    FastAPI TestClient fixture.
    """
    return TestClient(app)

def test_health_endpoint_degraded(client):
    """
    Verifies /api/health returns degraded state when predictor is not loaded.
    """
    # Temporarily force predictor in app.state to None
    app.state.predictor = None

    response = client.get("/api/health")
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False
    assert data["device"] == "none"
    assert data["category"] == "none"

def test_predict_endpoint_service_unavailable(client):
    """
    Verifies /api/predict returns HTTP 503 when model is not loaded.
    """
    app.state.predictor = None

    # Create dummy image upload bytes
    img = Image.new("RGB", (64, 64), (200, 200, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/api/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Model checkpoint is not loaded" in response.json()["message"]

def test_predict_endpoint_bad_request_non_image(client):
    """
    Verifies /api/predict returns HTTP 400 when uploaded file is not an image MIME type.
    """
    # Mock predictor in app state to bypass 503 check
    mock_predictor = MagicMock()
    app.state.predictor = mock_predictor

    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "not a valid image format" in response.json()["message"]

def test_predict_endpoint_bad_request_corrupt_image(client):
    """
    Verifies /api/predict returns HTTP 400 when image parsing fails.
    """
    mock_predictor = MagicMock()
    app.state.predictor = mock_predictor

    response = client.post(
        "/api/predict",
        files={"file": ("corrupt.png", b"corrupt image bytes", "image/png")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Failed to parse uploaded image" in response.json()["message"]

def test_health_and_predict_endpoint_success(client):
    """
    Verifies /api/health and /api/predict return healthy and correct schema outputs.
    """
    # 1. Setup mock predictor
    mock_predictor = MagicMock()
    mock_predictor.threshold = 0.5
    mock_predictor.threshold_method = "youden"
    mock_predictor.device = torch.device("cpu")
    mock_predictor.category = "bottle"
    
    # Mock prediction return values
    dummy_anomaly_map = np.random.rand(256, 256)
    mock_predictor.predict.return_value = {
        "is_anomaly": 1,
        "anomaly_score": 0.85,
        "anomaly_map": dummy_anomaly_map,
        "inference_time": 0.045
    }
    app.state.predictor = mock_predictor

    # 2. Test /api/health returns healthy
    health_response = client.get("/api/health")
    assert health_response.status_code == status.HTTP_200_OK
    health_data = health_response.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True
    assert health_data["device"] == "cpu"
    assert health_data["category"] == "bottle"
    assert health_data["threshold"] == 0.5

    # 3. Test /api/predict returns complete prediction response schema
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    predict_response = client.post(
        "/api/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert predict_response.status_code == status.HTTP_200_OK
    
    predict_data = predict_response.json()
    
    # Assert complete prediction response schema
    assert "is_anomaly" in predict_data
    assert "anomaly_score" in predict_data
    assert "anomaly_map" in predict_data
    assert "threshold" in predict_data
    assert "inference_time" in predict_data

    # Verify data types and values
    assert predict_data["is_anomaly"] == 1
    assert predict_data["anomaly_score"] == 0.85
    assert predict_data["threshold"] == 0.5
    assert predict_data["inference_time"] == 0.045
    
    # Verify continuous anomaly heatmap serialization (list of lists of shape 256x256)
    anomaly_map = predict_data["anomaly_map"]
    assert isinstance(anomaly_map, list)
    assert len(anomaly_map) == 256
    assert isinstance(anomaly_map[0], list)
    assert len(anomaly_map[0]) == 256
