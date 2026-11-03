"""
Unit tests for the multi-image batch inspection endpoint (POST /api/predict/batch).
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app
from app.inference.model_registry import ModelRegistry

@pytest.fixture
def client():
    """
    FastAPI TestClient fixture with mocked model registry for fast, deterministic testing.
    """
    # Create mock predictors for bottle, cable, screw
    mock_bottle = MagicMock()
    mock_bottle.threshold = 0.118206
    mock_bottle.category = "bottle"
    mock_bottle.predict.return_value = {
        "is_anomaly": 0,
        "anomaly_score": 0.021,
        "anomaly_map": np.full((256, 256), 0.021, dtype=np.float32),
        "inference_time": 0.035
    }

    mock_screw = MagicMock()
    mock_screw.threshold = 0.062534
    mock_screw.category = "screw"
    mock_screw.predict.return_value = {
        "is_anomaly": 0,
        "anomaly_score": 0.015,
        "anomaly_map": np.full((256, 256), 0.015, dtype=np.float32),
        "inference_time": 0.038
    }

    mock_cable = MagicMock()
    mock_cable.threshold = 0.086241
    mock_cable.category = "cable"
    mock_cable.predict.return_value = {
        "is_anomaly": 0,
        "anomaly_score": 0.030,
        "anomaly_map": np.full((256, 256), 0.030, dtype=np.float32),
        "inference_time": 0.036
    }

    registry = MagicMock()
    registry.predictors = {
        "bottle": mock_bottle,
        "screw": mock_screw,
        "cable": mock_cable
    }
    registry.get_predictor.side_effect = lambda c: registry.predictors.get(c)
    app.state.model_registry = registry

    return TestClient(app)

def make_test_image(size=(64, 64), color=(180, 180, 180)):
    """
    Helper creating PNG bytes for a synthetic test image.
    """
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_batch_predict_service_unavailable(client):
    """
    Verifies /api/predict/batch returns 503 when model registry is uninitialized.
    """
    app.state.model_registry = None

    files = [("files", (f"img_{i}.png", make_test_image(), "image/png")) for i in range(4)]
    response = client.post("/api/predict/batch?category=bottle", files=files)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Model registry is not initialized" in response.json()["message"]

def test_batch_predict_missing_category(client):
    """
    Verifies /api/predict/batch returns 422 if category parameter is omitted.
    """
    files = [("files", (f"img_{i}.png", make_test_image(), "image/png")) for i in range(4)]
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_batch_predict_invalid_category(client):
    """
    Verifies /api/predict/batch returns 400 for unknown category.
    """
    files = [("files", (f"img_{i}.png", make_test_image(), "image/png")) for i in range(4)]
    response = client.post("/api/predict/batch?category=unknown_part", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "is invalid or not loaded" in response.json()["message"]

def test_batch_predict_min_boundary_violation(client):
    """
    Verifies /api/predict/batch rejects batches with fewer than 4 images.
    """
    # Send 3 images (below MIN_BATCH_SIZE of 4)
    files = [("files", (f"img_{i}.png", make_test_image(), "image/png")) for i in range(3)]
    response = client.post("/api/predict/batch?category=bottle", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "requires at least 4 images" in response.json()["message"]

def test_batch_predict_max_boundary_violation(client):
    """
    Verifies /api/predict/batch rejects batches exceeding 12 images.
    """
    # Send 13 images (above MAX_BATCH_SIZE of 12)
    files = [("files", (f"img_{i}.png", make_test_image(), "image/png")) for i in range(13)]
    response = client.post("/api/predict/batch?category=bottle", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "allows at most 12 images per batch" in response.json()["message"]

def test_batch_predict_image_id_count_mismatch(client):
    """
    Verifies /api/predict/batch returns 400 if image_ids count does not match files count.
    """
    files = [("files", (f"img_{i}.png", make_test_image(), "image/png")) for i in range(4)]
    # Provide 3 IDs for 4 files
    data = {"image_ids": [f"uuid_{i}" for i in range(3)]}

    response = client.post("/api/predict/batch?category=bottle", files=files, data=data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "does not match number of files" in response.json()["message"]

def test_batch_predict_success_all_clear(client):
    """
    Verifies successful batch inspection where all 5 views are nominal (ALL_CLEAR).
    Verifies that client UUIDs are preserved and anomaly maps are exactly 256x256.
    """
    files = [("files", (f"screw_{i}.png", make_test_image(), "image/png")) for i in range(5)]
    client_ids = [f"uuid_test_screw_{i}" for i in range(5)]
    data = {"image_ids": client_ids}

    response = client.post("/api/predict/batch?category=screw", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK

    res_data = response.json()
    assert res_data["category"] == "screw"
    assert res_data["total_submitted"] == 5
    assert res_data["total_processed"] == 5
    assert res_data["total_failed"] == 0
    assert res_data["batch_status"] == "ALL_CLEAR"
    assert res_data["threshold"] == pytest.approx(0.062534, rel=1e-4)
    assert len(res_data["results"]) == 5

    for idx, item in enumerate(res_data["results"]):
        assert item["status"] == "success"
        assert item["image_id"] == client_ids[idx]
        assert item["filename"] == f"screw_{idx}.png"
        assert item["is_anomaly"] == 0
        assert isinstance(item["anomaly_score"], float)
        assert item["threshold"] == pytest.approx(0.062534, rel=1e-4)
        assert isinstance(item["inference_time"], float)
        # Check exact 256x256 anomaly map dimensions
        assert len(item["anomaly_map"]) == 256
        assert len(item["anomaly_map"][0]) == 256

def test_batch_predict_success_defect_detected(client):
    """
    Verifies batch verdict DEFECT_DETECTED when one or more images exceed threshold.
    """
    mock_screw = app.state.model_registry.get_predictor("screw")
    # Make prediction 2 return an anomaly
    call_count = 0
    def mock_predict(img):
        nonlocal call_count
        call_count += 1
        is_anom = 1 if call_count == 2 else 0
        score = 0.095 if is_anom else 0.015
        return {
            "is_anomaly": is_anom,
            "anomaly_score": score,
            "anomaly_map": np.full((256, 256), score, dtype=np.float32),
            "inference_time": 0.038
        }
    mock_screw.predict.side_effect = mock_predict

    files = [("files", (f"screw_{i}.png", make_test_image(), "image/png")) for i in range(4)]
    response = client.post("/api/predict/batch?category=screw", files=files)
    assert response.status_code == status.HTTP_200_OK

    res_data = response.json()
    assert res_data["total_submitted"] == 4
    assert res_data["total_processed"] == 4
    assert res_data["total_anomalous"] == 1
    assert res_data["total_normal"] == 3
    assert res_data["total_failed"] == 0
    assert res_data["batch_status"] == "DEFECT_DETECTED"
    assert res_data["results"][1]["is_anomaly"] == 1

def test_batch_predict_partial_failure_incomplete(client):
    """
    Verifies partial failure handling: 3 valid images + 1 corrupted non-image file.
    Assures failed image is NEVER counted as normal and batch verdict is INCOMPLETE.
    """
    corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_STREAM"
    files = [
        ("files", ("valid_1.png", make_test_image(), "image/png")),
        ("files", ("corrupt_2.png", corrupt_bytes, "image/png")),
        ("files", ("valid_3.png", make_test_image(), "image/png")),
        ("files", ("valid_4.png", make_test_image(), "image/png"))
    ]
    client_ids = [f"id_partial_{i}" for i in range(4)]
    data = {"image_ids": client_ids}

    response = client.post("/api/predict/batch?category=cable", files=files, data=data)
    assert response.status_code == status.HTTP_200_OK

    res_data = response.json()
    assert res_data["total_submitted"] == 4
    assert res_data["total_processed"] == 3
    assert res_data["total_failed"] == 1

    # Corrupt item must be flagged with status "error"
    corrupt_item = res_data["results"][1]
    assert corrupt_item["status"] == "error"
    assert corrupt_item["image_id"] == "id_partial_1"
    assert "Failed to parse uploaded image" in corrupt_item["error_message"]
    assert corrupt_item["is_anomaly"] is None
    assert corrupt_item["anomaly_map"] is None

    # Valid items must have succeeded
    assert res_data["results"][0]["status"] == "success"
    assert res_data["results"][2]["status"] == "success"
    assert res_data["results"][3]["status"] == "success"

    # Overall batch status must be INCOMPLETE
    assert res_data["batch_status"] == "INCOMPLETE"

def test_batch_predict_partial_failure_incomplete_defect_detected(client):
    """
    Verifies batch verdict INCOMPLETE_DEFECT_DETECTED when there is a defect AND a failed image.
    """
    mock_bottle = app.state.model_registry.get_predictor("bottle")
    mock_bottle.predict.return_value = {
        "is_anomaly": 1,
        "anomaly_score": 0.85,
        "anomaly_map": np.full((256, 256), 0.85, dtype=np.float32),
        "inference_time": 0.04
    }

    corrupt_bytes = b"INVALID"
    files = [
        ("files", ("valid_anom.png", make_test_image(), "image/png")),
        ("files", ("corrupt.png", corrupt_bytes, "image/png")),
        ("files", ("valid_2.png", make_test_image(), "image/png")),
        ("files", ("valid_3.png", make_test_image(), "image/png"))
    ]

    response = client.post("/api/predict/batch?category=bottle", files=files)
    assert response.status_code == status.HTTP_200_OK

    res_data = response.json()
    assert res_data["total_failed"] == 1
    assert res_data["total_anomalous"] >= 1
    assert res_data["batch_status"] == "INCOMPLETE_DEFECT_DETECTED"

def test_batch_predict_dimension_validation(client):
    """
    Verifies that an image with sub-minimal dimensions (<16x16) is caught and handled as an error item.
    """
    tiny_image = make_test_image(size=(8, 8))
    files = [
        ("files", ("tiny.png", tiny_image, "image/png")),
        ("files", ("norm_1.png", make_test_image(), "image/png")),
        ("files", ("norm_2.png", make_test_image(), "image/png")),
        ("files", ("norm_3.png", make_test_image(), "image/png"))
    ]

    response = client.post("/api/predict/batch?category=bottle", files=files)
    assert response.status_code == status.HTTP_200_OK

    res_data = response.json()
    assert res_data["total_failed"] == 1
    assert "smaller than minimum allowed 16x16" in res_data["results"][0]["error_message"]
