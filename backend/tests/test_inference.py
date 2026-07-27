"""
Unit tests for the STFPM single-image inference pipeline.
"""

import os
import json
import tempfile
import pytest
import numpy as np
# pyrefly: ignore [missing-import]
import torch
import torchvision.models as models
from PIL import Image
from unittest.mock import patch

from app.config import settings
from app.models.stfpm import STFPM, save_checkpoint
from app.inference.predictor import STFPMPredictor
from app.inference.heatmap import generate_heatmap_overlay

@pytest.fixture
def mock_resnet_builder():
    """
    Patch resnet constructors so they always instantiate without pretrained weights
    to avoid internet downloads.
    """
    original_resnet18 = models.resnet18
    def mock_r18(*args, **kwargs):
        kwargs_copy = kwargs.copy()
        if "weights" in kwargs_copy:
            kwargs_copy["weights"] = None
        if "pretrained" in kwargs_copy:
            kwargs_copy["pretrained"] = False
        return original_resnet18(*args, **kwargs_copy)

    with patch("torchvision.models.resnet18", side_effect=mock_r18):
        yield

@pytest.fixture
def mock_inference_setup(tmpdir, mock_resnet_builder):
    """
    Sets up a temporary directory with a mock model best.pth and threshold_config.json.
    """
    checkpoint_dir = os.path.join(tmpdir, "checkpoints", "bottle", "run_1")
    best_path = os.path.join(checkpoint_dir, "best.pth")
    config_path = os.path.join(checkpoint_dir, "threshold_config.json")

    # Create dummy model and save weights
    model = STFPM(backbone_name="resnet18")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_checkpoint(
        model, 
        best_path, 
        epoch=1, 
        metadata={"category": "bottle", "run_name": "run_1"}
    )

    # Save threshold configuration
    threshold_config = {
        "threshold": 0.5,
        "threshold_method": "youden",
        "image_auroc": 0.99,
        "pixel_auroc": 0.98,
        "pro_score": 0.95,
        "precision": 0.9,
        "recall": 0.9,
        "f1_score": 0.9
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=4)

    # Patch settings configuration paths
    with patch.object(settings, "CHECKPOINT_DIR", os.path.join(tmpdir, "checkpoints")), \
         patch.object(settings, "DEVICE", "cpu"):
        yield checkpoint_dir

def test_predictor_initialization(mock_inference_setup):
    """
    Verifies that the predictor instantiates, loads configurations, and binds device parameters.
    """
    predictor = STFPMPredictor(category="bottle", run_name="run_1")
    assert predictor.threshold == 0.5
    assert predictor.threshold_method == "youden"
    assert predictor.device.type == "cpu"
    assert predictor.is_cold is True

def test_predictor_inference_timing(mock_inference_setup):
    """
    Verifies prediction dictionary structure and that cold-start transitions to warm timing.
    """
    predictor = STFPMPredictor(category="bottle", run_name="run_1")
    
    # Create test dummy image
    img = Image.new("RGB", (128, 128), (200, 200, 200))

    # Pass 1: Cold-start run
    assert predictor.is_cold is True
    res1 = predictor.predict(img)
    
    assert "is_anomaly" in res1
    assert "anomaly_score" in res1
    assert "anomaly_map" in res1
    assert "inference_time" in res1
    assert isinstance(res1["is_anomaly"], int)
    assert isinstance(res1["anomaly_score"], float)
    assert isinstance(res1["anomaly_map"], np.ndarray)
    assert isinstance(res1["inference_time"], float)
    assert res1["anomaly_map"].shape == (settings.IMAGE_SIZE, settings.IMAGE_SIZE)

    # Flag should switch to False
    assert predictor.is_cold is False

    # Pass 2: Warm inference run
    res2 = predictor.predict(img)
    assert res2["inference_time"] > 0.0
    assert predictor.is_cold is False

def test_predictor_image_inputs(mock_inference_setup, tmp_path):
    """
    Verifies that predictor accepts PIL Images, numpy arrays, and path string inputs.
    """
    predictor = STFPMPredictor(category="bottle", run_name="run_1")

    # 1. PIL Image
    img_pil = Image.new("RGB", (64, 64), (100, 150, 200))
    res_pil = predictor.predict(img_pil)
    assert res_pil["is_anomaly"] in [0, 1]

    # 2. Numpy Array
    img_np = np.array(img_pil)
    res_np = predictor.predict(img_np)
    assert res_np["is_anomaly"] in [0, 1]

    # 3. Path String
    img_file = tmp_path / "temp.png"
    img_pil.save(img_file)
    res_path = predictor.predict(str(img_file))
    assert res_path["is_anomaly"] in [0, 1]

    # 4. Invalid Type
    with pytest.raises(TypeError) as exc:
        predictor.predict([1, 2, 3])
    assert "Input image must be a file path string" in str(exc.value)

def test_heatmap_overlay_helper():
    """
    Verifies that generate_heatmap_overlay blends colors correctly and clips outputs between 0 and 1.
    """
    raw_img = np.random.rand(64, 64, 3)
    anomaly_map = np.random.rand(64, 64)

    overlay = generate_heatmap_overlay(raw_img, anomaly_map)
    assert overlay.shape == (64, 64, 3)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0
