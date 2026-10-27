"""
Unit tests for the FastAPI backend API endpoints (/api/health, /api/categories, and /api/predict).
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
from app.inference.model_registry import ModelRegistry

@pytest.fixture
def client():
    """
    FastAPI TestClient fixture.
    """
    return TestClient(app)

def test_health_endpoint_degraded(client):
    """
    Verifies /api/health returns degraded state when model registry is not loaded.
    """
    app.state.model_registry = None

    response = client.get("/api/health")
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False
    assert data["device"] == "none"
    assert data["category"] == "none"
    assert data["loaded_categories"] == []
    assert data["thresholds"] == {}

def test_predict_endpoint_service_unavailable(client):
    """
    Verifies /api/predict returns HTTP 503 when model registry is not loaded.
    """
    app.state.model_registry = None

    img = Image.new("RGB", (64, 64), (200, 200, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/api/predict?category=bottle",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Model registry is not initialized" in response.json()["message"]

def test_predict_endpoint_missing_category(client):
    """
    Verifies /api/predict returns HTTP 422 when category query parameter is omitted.
    """
    mock_registry = MagicMock()
    app.state.model_registry = mock_registry

    img = Image.new("RGB", (64, 64), (200, 200, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    # Call predict without category parameter
    response = client.post(
        "/api/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_predict_endpoint_invalid_category(client):
    """
    Verifies /api/predict returns HTTP 400 when an invalid/unloaded category is requested.
    """
    mock_registry = MagicMock()
    mock_registry.predictors = {"bottle": MagicMock(), "cable": MagicMock()}
    mock_registry.get_predictor.return_value = None
    app.state.model_registry = mock_registry

    img = Image.new("RGB", (64, 64), (200, 200, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/api/predict?category=unknown_item",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "is invalid or not loaded" in response.json()["message"]

def test_predict_endpoint_bad_request_non_image(client):
    """
    Verifies /api/predict returns HTTP 400 when uploaded file is not an image MIME type.
    """
    mock_registry = MagicMock()
    mock_registry.predictors = {"bottle": MagicMock()}
    mock_registry.get_predictor.return_value = MagicMock()
    app.state.model_registry = mock_registry

    response = client.post(
        "/api/predict?category=bottle",
        files={"file": ("test.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "not a valid image format" in response.json()["message"]

def test_predict_endpoint_bad_request_corrupt_image(client):
    """
    Verifies /api/predict returns HTTP 400 when image parsing fails.
    """
    mock_registry = MagicMock()
    mock_registry.predictors = {"bottle": MagicMock()}
    mock_registry.get_predictor.return_value = MagicMock()
    app.state.model_registry = mock_registry

    response = client.post(
        "/api/predict?category=bottle",
        files={"file": ("corrupt.png", b"corrupt image bytes", "image/png")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Failed to parse uploaded image" in response.json()["message"]

def test_health_categories_and_predict_success(client):
    """
    Verifies /api/health, /api/categories, and /api/predict with proper routing and schema.
    """
    # 1. Setup mock predictors for bottle, cable, screw
    mock_bottle = MagicMock()
    mock_bottle.threshold = 0.1182
    mock_bottle.threshold_method = "youden"
    mock_bottle.device = torch.device("cpu")
    mock_bottle.category = "bottle"
    mock_bottle.predict.return_value = {
        "is_anomaly": 0,
        "anomaly_score": 0.05,
        "anomaly_map": np.random.rand(256, 256),
        "inference_time": 0.035
    }

    mock_screw = MagicMock()
    mock_screw.threshold = 0.0625
    mock_screw.threshold_method = "youden"
    mock_screw.device = torch.device("cpu")
    mock_screw.category = "screw"
    mock_screw.predict.return_value = {
        "is_anomaly": 1,
        "anomaly_score": 0.88,
        "anomaly_map": np.random.rand(256, 256),
        "inference_time": 0.042
    }

    mock_registry = MagicMock()
    mock_registry.predictors = {"bottle": mock_bottle, "screw": mock_screw}
    mock_registry.get_predictor.side_effect = lambda cat: mock_registry.predictors.get(cat)
    app.state.model_registry = mock_registry

    # 2. Test /api/health
    health_response = client.get("/api/health")
    assert health_response.status_code == status.HTTP_200_OK
    health_data = health_response.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True
    assert "bottle" in health_data["loaded_categories"]
    assert "screw" in health_data["loaded_categories"]
    assert health_data["thresholds"]["screw"] == 0.0625

    # 3. Test /api/categories
    categories_response = client.get("/api/categories")
    assert categories_response.status_code == status.HTTP_200_OK
    assert categories_response.json() == ["bottle", "screw"]

    # 4. Test /api/predict routing to screw
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    predict_response = client.post(
        "/api/predict?category=screw",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert predict_response.status_code == status.HTTP_200_OK
    data = predict_response.json()

    # Assert screw predictor was called and returned screw-specific threshold
    mock_screw.predict.assert_called_once()
    assert data["category"] == "screw"
    assert data["threshold"] == 0.0625
    assert data["is_anomaly"] == 1
    assert data["anomaly_score"] == 0.88
    assert len(data["anomaly_map"]) == 256

def test_registry_startup_and_threshold_loading():
    """
    Verifies ModelRegistry loads the 3 verified checkpoints with category-specific thresholds.
    """
    registry = ModelRegistry()
    registry.load_models()

    loaded = list(registry.predictors.keys())
    assert "bottle" in loaded
    assert "cable" in loaded
    assert "screw" in loaded

    # Verify category-specific thresholds match calibrated values
    bottle_pred = registry.get_predictor("bottle")
    cable_pred = registry.get_predictor("cable")
    screw_pred = registry.get_predictor("screw")

    assert bottle_pred is not None
    assert cable_pred is not None
    assert screw_pred is not None

    # Check calibrated threshold values (~0.1182 for bottle, ~0.2265 for cable, ~0.0625 for screw)
    assert pytest.approx(bottle_pred.threshold, rel=1e-2) == 0.1182
    assert pytest.approx(cable_pred.threshold, rel=1e-2) == 0.2265
    assert pytest.approx(screw_pred.threshold, rel=1e-2) == 0.0625
