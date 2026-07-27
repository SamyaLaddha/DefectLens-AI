"""
Single-image inference predictor for DefectLens AI.

Loads weights and thresholds, processes input images, and runs predictions.
"""

import os
import json
import time
import torch
import numpy as np
from PIL import Image

from app.config import settings
from app.data.transforms import get_transforms
from app.models.stfpm import STFPM, load_checkpoint
from app.inference.heatmap import compute_anomaly_map
from app.utils.logging_utils import get_logger

logger = get_logger("inference")

class STFPMPredictor:
    """
    STFPM predictor class executing single-image anomaly classification.
    """
    def __init__(self, category: str = None, run_name: str = None):
        # Fall back to configurations if parameters are omitted
        self.category = category or settings.CATEGORY
        self.run_name = run_name or settings.RUN_NAME

        # Resolve checkpoint paths
        checkpoint_dir = os.path.join(settings.CHECKPOINT_DIR, self.category, self.run_name)
        best_path = os.path.join(checkpoint_dir, "best.pth")
        config_path = os.path.join(checkpoint_dir, "threshold_config.json")

        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Trained checkpoint not found at: {best_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Calibrated threshold config not found at: {config_path}")

        # Load calibrated threshold value
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            self.threshold = float(config["threshold"])
            self.threshold_method = config.get("threshold_method", "youden")
            logger.info(f"Loaded threshold: {self.threshold:.6f} fitted via {self.threshold_method}")

        # Determine execution device
        if settings.DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(settings.DEVICE)

        # Load model architecture and checkpoints
        self.model = STFPM(backbone_name=settings.BACKBONE)
        load_checkpoint(self.model, best_path)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform pipelines
        _, self.transform = get_transforms(
            image_size=settings.IMAGE_SIZE,
            horizontal_flip=False  # Inference uses fixed transforms without random augmentations
        )

        # Flag tracking cold-start timing
        self.is_cold = True
        logger.info(f"STFPMPredictor initialized on device: {self.device}")

    def predict(self, image) -> dict:
        """
        Runs anomaly detection and classification on a single input image.

        - image: PIL Image, numpy array, or file path string.
        Returns a dictionary containing predicted labels, anomaly scores,
        continuous anomaly maps, and execution timing metrics.
        """
        t_start = time.perf_counter()

        # Load PIL image if path string is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Inference input image not found: {image}")
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype("uint8")).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError("Input image must be a file path string, numpy array, or PIL Image object.")

        # Apply preprocessing transforms
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Run model inference pass without gradient tracking
        with torch.no_grad():
            t_feats, s_feats = self.model(img_tensor)
            
            # Reusable anomaly map generation from heatmap module
            anomaly_map = compute_anomaly_map(t_feats, s_feats, settings.IMAGE_SIZE)

        # Image-level anomaly score is the maximum pixel value of the anomaly map
        anomaly_score = float(anomaly_map.max())
        
        # Apply threshold criterion for classification
        is_anomaly = 1 if anomaly_score >= self.threshold else 0

        # Compute execution timing
        elapsed = time.perf_counter() - t_start
        run_type = "cold-start" if self.is_cold else "warm"
        
        # Log device type and speed
        logger.info(
            f"Prediction completed | Time: {elapsed:.6f}s ({run_type}) | "
            f"Device: {self.device.type.upper()} | Score: {anomaly_score:.6f} | Anomaly: {is_anomaly}"
        )
        
        # Reset cold-start flag
        self.is_cold = False

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
            "inference_time": elapsed
        }
