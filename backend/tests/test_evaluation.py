"""
Unit tests for the STFPM evaluation pipeline.
"""

import os
import json
import tempfile
import pytest
import numpy as np
# pyrefly: ignore [missing-import]
import torch
import torchvision.models as models
from unittest.mock import patch, MagicMock

from app.config import settings
from app.utils.seed import set_seed
from app.training.evaluate import evaluate_model, calculate_pro_score, select_threshold
from app.models.stfpm import STFPM, save_checkpoint

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
def mock_mvtec_dir(tmp_path):
    """
    Creates a minimal mock MVTec AD directory structure for test evaluation splits.
    """
    from PIL import Image
    category = "bottle"
    category_dir = tmp_path / category

    # Create test directories
    test_good = category_dir / "test" / "good"
    test_good.mkdir(parents=True)
    test_anomaly = category_dir / "test" / "broken_large"
    test_anomaly.mkdir(parents=True)
    gt_anomaly = category_dir / "ground_truth" / "broken_large"
    gt_anomaly.mkdir(parents=True)

    # Save 2 normal test images
    for i in range(2):
        img = Image.new("RGB", (64, 64), (200, 200, 200))
        img.save(test_good / f"{i:03d}.png")

    # Save 2 anomalous test images and their masks
    for i in range(2):
        img = Image.new("RGB", (64, 64), (150, 100, 100))
        img.save(test_anomaly / f"{i:03d}.png")

        # Binary mask with anomaly region
        mask = Image.new("L", (64, 64), 0)
        # Add anomalous region
        for y in range(20, 45):
            for x in range(20, 45):
                mask.putpixel((x, y), 255)
        mask.save(gt_anomaly / f"{i:03d}_mask.png")

    return str(tmp_path)

def test_select_threshold():
    """
    Verifies that the threshold selectors correctly fit Youden's J and F1-max thresholds.
    """
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    # Youden's threshold should sit between 0.2 and 0.8
    youden_th = select_threshold(y_true, y_scores, method="youden")
    assert 0.2 < youden_th <= 0.8

    # F1-max threshold should sit between 0.2 and 0.8
    f1_max_th = select_threshold(y_true, y_scores, method="f1_max")
    assert 0.2 < f1_max_th <= 0.8

def test_calculate_pro_score():
    """
    Verifies that PRO-score computes per-region overlaps for continuous anomaly maps.
    """
    # Create simple dummy masks and anomaly maps
    y_true_masks = np.zeros((2, 8, 8), dtype=int)
    # Labeled regions in ground truth masks
    y_true_masks[0, 2:5, 2:5] = 1
    y_true_masks[1, 1:3, 1:3] = 1

    predicted_anomaly_maps = np.zeros((2, 8, 8), dtype=float)
    # Overlap region detections
    predicted_anomaly_maps[0, 2:4, 2:4] = 0.8  # partial overlap
    predicted_anomaly_maps[1, 1:3, 1:3] = 0.9  # complete overlap

    pro = calculate_pro_score(y_true_masks, predicted_anomaly_maps, fpr_limit=0.3)
    assert isinstance(pro, float)
    assert 0.0 <= pro <= 1.0

def test_evaluation_smoke_run(mock_mvtec_dir, mock_resnet_builder):
    """
    Executes a model evaluation smoke run on a mock dataset and verifies configuration file outputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "checkpoints", "bottle", "run_1")
        best_path = os.path.join(checkpoint_dir, "best.pth")

        # Create a mock checkpoint file to load
        model = STFPM(backbone_name="resnet18")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(
            model, 
            best_path, 
            epoch=5, 
            metadata={"category": "bottle", "run_name": "run_1", "best_val_loss": 0.3}
        )

        # Get original file modification time to verify immutability
        original_mtime = os.path.getmtime(best_path)

        # Clean up any existing screenshot overlays to ensure independent assertions
        screenshots_dir = "docs/screenshots"
        for name in ["false_positive.png", "false_negative.png"]:
            p = os.path.join(screenshots_dir, name)
            if os.path.exists(p):
                os.remove(p)

        # Patch configs to write to temp directories
        with patch.object(settings, "DATASET_PATH", mock_mvtec_dir), \
             patch.object(settings, "CHECKPOINT_DIR", os.path.join(tmpdir, "checkpoints")), \
             patch.object(settings, "NUM_WORKERS", 0), \
             patch.object(settings, "DEVICE", "cpu"), \
             patch("app.training.evaluate.select_threshold", return_value=999.0):
            
            # Execute model evaluation (forces all anomalous test samples to resolve as False Negatives)
            evaluate_model()

            # 1. Verify original best.pth remains unmodified (mtime remains exactly identical)
            assert os.path.getmtime(best_path) == original_mtime

            # 2. Verify threshold_config.json was created and is valid
            config_path = os.path.join(checkpoint_dir, "threshold_config.json")
            assert os.path.exists(config_path)

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                assert config["threshold"] == 999.0
                assert config["threshold_method"] == "youden"
                assert "image_auroc" in config
                assert "pixel_auroc" in config
                assert "pro_score" in config
                assert "precision" in config
                assert "recall" in config
                assert "f1_score" in config

            # 3. Verify false negative overlay image was created
            assert os.path.exists(os.path.join(screenshots_dir, "false_negative.png"))
