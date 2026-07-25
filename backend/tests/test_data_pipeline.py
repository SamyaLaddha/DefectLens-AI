"""
Unit tests for the data pipeline, configurations, and reproducibility seeding.
"""

import os
import pytest
from PIL import Image
import numpy as np
import torch

from app.config import settings
from app.utils.seed import set_seed
from app.data.dataset import MVTecDataset
from app.data.transforms import get_transforms

@pytest.fixture
def mock_mvtec_dir(tmp_path):
    """
    Creates a mock MVTec AD directory structure with dummy images.
    """
    category = "bottle"
    category_dir = tmp_path / category

    # Create directory tree
    train_good = category_dir / "train" / "good"
    test_good = category_dir / "test" / "good"
    test_broken = category_dir / "test" / "broken"
    gt_broken = category_dir / "ground_truth" / "broken"

    train_good.mkdir(parents=True)
    test_good.mkdir(parents=True)
    test_broken.mkdir(parents=True)
    gt_broken.mkdir(parents=True)

    # Helper to write dummy images
    def save_dummy_img(path, color, size=(128, 128)):
        img = Image.new("RGB", size, color)
        img.save(path)

    # Helper to write dummy masks
    def save_dummy_mask(path, size=(128, 128)):
        # Binary mask with specific anomalous patch in the center
        arr = np.zeros(size, dtype=np.uint8)
        arr[40:80, 40:80] = 255
        img = Image.fromarray(arr, mode="L")
        img.save(path)

    # Create 10 normal train images
    for i in range(10):
        save_dummy_img(train_good / f"{i:03d}.png", color=(200, 200, 200))

    # Create 1 normal test image
    save_dummy_img(test_good / "000.png", color=(200, 200, 200))

    # Create 1 anomalous test image and its ground truth mask
    save_dummy_img(test_broken / "000.png", color=(100, 50, 50))
    save_dummy_mask(gt_broken / "000_mask.png")

    return str(tmp_path)

def test_configuration_loading():
    """
    Asserts that the settings object loads expected fields correctly.
    """
    assert hasattr(settings, "DATASET_NAME")
    assert hasattr(settings, "CATEGORY")
    assert hasattr(settings, "VAL_SPLIT")
    assert hasattr(settings, "HORIZONTAL_FLIP")
    assert settings.CATEGORY == "bottle"
    assert settings.VAL_SPLIT == 0.10
    assert settings.HORIZONTAL_FLIP is True

def test_deterministic_seeding():
    """
    Asserts that seeding results in deterministic output across generators.
    """
    # Test NumPy reproducibility
    set_seed(42, deterministic=False)
    arr1 = np.random.rand(5)
    set_seed(42, deterministic=False)
    arr2 = np.random.rand(5)
    assert np.allclose(arr1, arr2)

    # Test PyTorch reproducibility
    set_seed(100, deterministic=False)
    t1 = torch.rand(5)
    set_seed(100, deterministic=False)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2)

def test_train_val_splits_integrity(mock_mvtec_dir):
    """
    Verifies train/validation splits contain only normal images, respect ratio, and have zero overlap.
    """
    train_dataset = MVTecDataset(
        dataset_path=mock_mvtec_dir,
        category="bottle",
        split="train",
        val_split=0.10,
        seed=42
    )

    val_dataset = MVTecDataset(
        dataset_path=mock_mvtec_dir,
        category="bottle",
        split="val",
        val_split=0.10,
        seed=42
    )

    # 1. train contains only normal images (label 0)
    assert all(label == 0 for label in train_dataset.labels)

    # 2. validation contains only normal images (label 0)
    assert all(label == 0 for label in val_dataset.labels)

    # 3. train and validation have no overlapping files
    train_paths = set(train_dataset.image_paths)
    val_paths = set(val_dataset.image_paths)
    overlap = train_paths.intersection(val_paths)
    assert len(overlap) == 0

    # 4. Respects the val_split ratio (1 out of 10 for 0.10)
    assert len(val_dataset) == 1
    assert len(train_dataset) == 9

def test_seed_consistency(mock_mvtec_dir):
    """
    Verifies that the same seed produces the same train/validation splits.
    """
    train_ds_1 = MVTecDataset(
        dataset_path=mock_mvtec_dir,
        category="bottle",
        split="train",
        val_split=0.30,
        seed=42
    )

    train_ds_2 = MVTecDataset(
        dataset_path=mock_mvtec_dir,
        category="bottle",
        split="train",
        val_split=0.30,
        seed=42
    )

    assert train_ds_1.image_paths == train_ds_2.image_paths

    # Different seed shuffles differently, causing different splits
    train_ds_diff = MVTecDataset(
        dataset_path=mock_mvtec_dir,
        category="bottle",
        split="train",
        val_split=0.30,
        seed=999
    )
    assert train_ds_1.image_paths != train_ds_diff.image_paths

def test_test_split_loading(mock_mvtec_dir):
    """
    Verifies that test split loads both normal and anomalous samples with correct masks.
    """
    test_dataset = MVTecDataset(
        dataset_path=mock_mvtec_dir,
        category="bottle",
        split="test",
        image_size=64
    )

    # Test dataset has 2 images: 1 normal, 1 anomalous
    assert len(test_dataset) == 2
    assert 0 in test_dataset.labels
    assert 1 in test_dataset.labels

    # Identify indices
    normal_idx = test_dataset.labels.index(0)
    anomalous_idx = test_dataset.labels.index(1)

    # Load normal sample
    _, normal_label, normal_mask = test_dataset[normal_idx]
    assert normal_label == 0
    # Normal test samples receive an appropriate empty/zero mask representation
    assert torch.all(normal_mask == 0.0)
    assert normal_mask.shape == (1, 64, 64)

    # Load anomalous sample
    _, anomalous_label, anomalous_mask = test_dataset[anomalous_idx]
    assert anomalous_label == 1
    # Anomalous test images resolve to the correct ground-truth masks (anomalous patch is not all zeros)
    assert torch.any(anomalous_mask > 0.0)
    assert anomalous_mask.shape == (1, 64, 64)

def test_masks_remain_binary(mock_mvtec_dir):
    """
    Verifies that resized ground truth masks remain strictly binary (0.0 or 1.0 values only).
    """
    test_dataset = MVTecDataset(
        dataset_path=mock_mvtec_dir,
        category="bottle",
        split="test",
        image_size=64
    )

    # Check mask values
    for i in range(len(test_dataset)):
        _, _, mask = test_dataset[i]
        unique_vals = torch.unique(mask)
        for val in unique_vals:
            assert val.item() in [0.0, 1.0]
