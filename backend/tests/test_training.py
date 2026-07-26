"""
Unit tests for the STFPM training pipeline, including early stopping and checkpointing.
"""

import os
import csv
import tempfile
import pytest
# pyrefly: ignore [missing-import]
import torch
import torchvision.models as models
from unittest.mock import patch, MagicMock

from app.config import settings
from app.utils.seed import set_seed
from app.training.train import train_model, run_sanity_check
from app.models.stfpm import STFPM, load_checkpoint

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
    Creates a minimal mock MVTec AD directory structure for training smoke tests.
    """
    from PIL import Image
    category = "bottle"
    category_dir = tmp_path / category

    # Create directories
    train_good = category_dir / "train" / "good"
    train_good.mkdir(parents=True)

    # Save 5 normal train images
    for i in range(5):
        img = Image.new("RGB", (64, 64), (200, 200, 200))
        img.save(train_good / f"{i:03d}.png")

    return str(tmp_path)

def test_auto_device_selection():
    """
    Asserts that device auto-selection successfully falls back to CPU when CUDA is unavailable.
    """
    with patch("torch.cuda.is_available", return_value=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device.type == "cpu"

    with patch("torch.cuda.is_available", return_value=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device.type == "cuda"

def test_run_sanity_check():
    """
    Verifies that the dataset pre-run sanity check raises errors on overlap or empty splits.
    """
    mock_train = MagicMock()
    mock_val = MagicMock()

    # Case 1: normal splits with zero overlap
    mock_train.image_paths = ["/data/1.png", "/data/2.png"]
    mock_train.__len__.return_value = 2
    mock_val.image_paths = ["/data/3.png"]
    mock_val.__len__.return_value = 1
    
    # Should pass without raising exception
    run_sanity_check(mock_train, mock_val)

    # Case 2: empty training split
    mock_train.__len__.return_value = 0
    with pytest.raises(ValueError) as exc:
        run_sanity_check(mock_train, mock_val)
    assert "Training dataset cannot have 0 samples" in str(exc.value)

    # Case 3: overlap detected
    mock_train.__len__.return_value = 2
    mock_train.image_paths = ["/data/1.png", "/data/2.png"]
    mock_val.image_paths = ["/data/2.png"]
    with pytest.raises(ValueError) as exc:
        run_sanity_check(mock_train, mock_val)
    assert "Overlap detected" in str(exc.value)

def test_training_smoke_run(mock_mvtec_dir, mock_resnet_builder):
    """
    Executes a 2-epoch training smoke run on a mock dataset and verifies log files and checkpoints.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch config directories to write to our temp folder
        with patch.object(settings, "DATASET_PATH", mock_mvtec_dir), \
             patch.object(settings, "CHECKPOINT_DIR", os.path.join(tmpdir, "checkpoints")), \
             patch.object(settings, "LOG_DIR", os.path.join(tmpdir, "logs")), \
             patch.object(settings, "BATCH_SIZE", 2), \
             patch.object(settings, "NUM_WORKERS", 0), \
             patch.object(settings, "DEVICE", "cpu"), \
             patch.object(settings, "VAL_SPLIT", 0.20):  # 1 val, 4 train
            
            # Execute training for 2 epochs
            train_model(epochs=2)

            checkpoint_dir = os.path.join(settings.CHECKPOINT_DIR, settings.CATEGORY, settings.RUN_NAME)
            log_dir = os.path.join(settings.LOG_DIR, settings.CATEGORY, settings.RUN_NAME)

            # Verify files exist in run-specific paths
            assert os.path.exists(os.path.join(checkpoint_dir, "best.pth"))
            assert os.path.exists(os.path.join(checkpoint_dir, "final.pth"))
            
            csv_path = os.path.join(log_dir, "history.csv")
            assert os.path.exists(csv_path)

            # Verify CSV formatting
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                assert reader[0] == ["epoch", "train_loss", "val_loss", "lr"]
                assert len(reader) == 3  # Header + 2 epochs

def test_early_stopping_and_best_checkpoint(mock_mvtec_dir, mock_resnet_builder):
    """
    Mocks loss calculations to verify early stopping and best checkpoint selection.
    """
    # Preset sequence of validation discrepancy loss values:
    # Epoch 1: val loss = 0.5 (best checkpoint saved)
    # Epoch 2: val loss = 0.4 (new best checkpoint saved)
    # Epoch 3: val loss = 0.6 (no checkpoint saved)
    # Epoch 4: val loss = 0.6 (no checkpoint saved, patience count = 2 -> triggers early stop)
    loss_sequence = [0.5, 0.4, 0.6, 0.6, 0.6]
    loss_iter = iter(loss_sequence)

    def mock_compute_loss(*args, **kwargs):
        # args[1] represents student_features
        student_feats = args[1]
        if isinstance(student_feats, list) and len(student_feats) > 0 and student_feats[0].requires_grad:
            # Training phase: return dummy loss linked to student gradients
            return student_feats[0].sum() * 0.0 + torch.tensor(1.0)
        # Validation phase: return preset validation loss
        return torch.tensor(next(loss_iter))

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(settings, "DATASET_PATH", mock_mvtec_dir), \
             patch.object(settings, "CHECKPOINT_DIR", os.path.join(tmpdir, "checkpoints")), \
             patch.object(settings, "LOG_DIR", os.path.join(tmpdir, "logs")), \
             patch.object(settings, "BATCH_SIZE", 2), \
             patch.object(settings, "NUM_WORKERS", 0), \
             patch.object(settings, "DEVICE", "cpu"), \
             patch.object(settings, "PATIENCE", 2), \
             patch.object(settings, "VAL_SPLIT", 0.20), \
             patch("app.training.train.compute_discrepancy_loss", side_effect=mock_compute_loss):

            # Run training. Patience is 2, training should terminate early at Epoch 4
            train_model(epochs=10)

            checkpoint_dir = os.path.join(settings.CHECKPOINT_DIR, settings.CATEGORY, settings.RUN_NAME)
            best_path = os.path.join(checkpoint_dir, "best.pth")
            final_path = os.path.join(checkpoint_dir, "final.pth")

            # Verify checkpoints
            model = STFPM(backbone_name="resnet18")
            
            # Best checkpoint should be from Epoch 2 (val loss = 0.4)
            best_info = load_checkpoint(model, best_path)
            assert best_info["epoch"] == 2
            assert best_info["metadata"]["best_val_loss"] == pytest.approx(0.4)

            # Final checkpoint should be from Epoch 4
            final_info = load_checkpoint(model, final_path)
            assert final_info["epoch"] == 4
