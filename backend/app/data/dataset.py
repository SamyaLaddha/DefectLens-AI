"""
MVTec AD Custom Dataset Loader.

This module loads images from the MVTec AD dataset directory structure,
partitioning normal images into zero-overlap train and validation splits
and loading both normal and anomalous images with masks for testing.
"""

import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MVTecDataset(Dataset):
    """
    Dataset class for loading MVTec AD images and ground-truth masks.
    """
    def __init__(
        self,
        dataset_path: str,
        category: str,
        split: str = "train",
        transform = None,
        val_split: float = 0.10,
        seed: int = 42,
        image_size: int = 256
    ):
        """
        Initialize directory paths, list image paths, and prepare splits.
        """
        self.dataset_path = dataset_path
        self.category = category
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.image_paths = []
        self.labels = []
        self.mask_paths = []

        # Validate split name
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")

        # Resolve paths
        category_dir = os.path.join(dataset_path, category)
        train_good_dir = os.path.join(category_dir, "train", "good")
        test_dir = os.path.join(category_dir, "test")

        if split in ["train", "val"]:
            if not os.path.exists(train_good_dir):
                raise FileNotFoundError(f"Training directory not found: {train_good_dir}")

            # Gather all normal training images
            extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
            all_files = []
            for ext in extensions:
                all_files.extend(glob.glob(os.path.join(train_good_dir, ext)))

            # Deduplicate because Windows case-insensitive matching can cause duplicate entries
            all_files = list(set(all_files))
            # Sort files first to ensure starting point is identical across platforms
            all_files = sorted(all_files)

            # Shuffle deterministically using a local numpy random generator
            rng = np.random.default_rng(seed)
            rng.shuffle(all_files)

            # Compute split index
            num_val = int(len(all_files) * val_split)

            # Zero-overlap split
            if split == "val":
                self.image_paths = all_files[:num_val]
            else:
                self.image_paths = all_files[num_val:]

            # Train/validation splits contain only normal images (label 0, no masks needed)
            self.labels = [0] * len(self.image_paths)
            self.mask_paths = [None] * len(self.image_paths)

        elif split == "test":
            if not os.path.exists(test_dir):
                raise FileNotFoundError(f"Testing directory not found: {test_dir}")

            # Gather test subdirectories (e.g., 'good', 'broken_large', etc.)
            subdirs = sorted(os.listdir(test_dir))
            for subdir in subdirs:
                subdir_path = os.path.join(test_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
                files = []
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(subdir_path, ext)))
                # Deduplicate because Windows case-insensitive matching can cause duplicate entries
                files = list(set(files))
                files = sorted(files)

                is_anomaly = (subdir != "good")
                label = 1 if is_anomaly else 0

                for f in files:
                    self.image_paths.append(f)
                    self.labels.append(label)

                    if is_anomaly:
                        # Construct ground-truth mask path
                        # Expected path structure: ground_truth/<subdir>/<filename>_mask.png
                        base_name = os.path.basename(f)
                        name_no_ext, _ = os.path.splitext(base_name)
                        mask_name = f"{name_no_ext}_mask.png"
                        mask_path = os.path.join(category_dir, "ground_truth", subdir, mask_name)
                        self.mask_paths.append(mask_path)
                    else:
                        self.mask_paths.append(None)

    def __len__(self) -> int:
        """
        Return the total number of images in this split.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Retrieve an image item and its ground-truth annotations.

        Note: During training, test-set labels are NEVER touched. Using test labels
        or test performance for model selection or early stopping would cause data
        leakage and invalidate the generalization metrics.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        original_size = img.size

        # Apply image transform
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.functional.to_tensor(img)

        # Load and resize mask
        # If mask is None (normal sample), we return a zero-filled binary tensor
        if mask_path is not None:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing ground-truth mask for anomalous sample: {mask_path}")
            mask = Image.open(mask_path).convert("L")
        else:
            # For normal images, return an empty/zero mask of the original size
            mask = Image.new("L", original_size, 0)

        # Resize mask to the configured image_size using NEAREST mode to avoid interpolation artifacts
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_tensor = T.functional.to_tensor(mask)

        # Ensure mask remains strictly binary (0.0 or 1.0)
        mask_tensor = (mask_tensor >= 0.5).float()

        return img_tensor, label, mask_tensor
