"""
Configuration settings loader.

Loads configurations from app/training/config.yaml using PyYAML.
"""

import os
import yaml
from dotenv import load_dotenv

# Load any custom env vars from a local .env file
load_dotenv()

class Settings:
    """
    Settings loader that reads from training/config.yaml and environment variables.
    """
    def __init__(self):
        # Resolve config.yaml relative to this file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "training", "config.yaml")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # Load configurations with fallback values
        dataset_cfg = config.get("dataset", {})
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})

        # Server configs
        self.PORT: int = int(os.getenv("PORT", "8000"))
        self.HOST: str = os.getenv("HOST", "127.0.0.1")

        # Dataset configs
        self.DATASET_NAME: str = dataset_cfg.get("name", "mvtec")
        self.CATEGORIES: list = dataset_cfg.get("categories", ["bottle", "cable", "screw"])
        self.CATEGORY_RUNS: dict = dataset_cfg.get("category_runs", {
            "bottle": "run_2",
            "cable": "run_3",
            "screw": "run_2"
        })
        self.CATEGORY: str = dataset_cfg.get("category", "bottle")
        self.DATASET_PATH: str = os.getenv("MVTEC_DATASET_DIR", dataset_cfg.get("dataset_path", "./data/mvtec"))
        self.IMAGE_SIZE: int = int(dataset_cfg.get("image_size", 256))
        self.BATCH_SIZE: int = int(dataset_cfg.get("batch_size", 32))
        self.VAL_SPLIT: float = float(dataset_cfg.get("val_split", 0.10))
        self.HORIZONTAL_FLIP: bool = bool(dataset_cfg.get("horizontal_flip", True))
        self.NUM_WORKERS: int = int(dataset_cfg.get("num_workers", 0))

        # Model configs
        self.BACKBONE: str = model_cfg.get("backbone", "resnet18")

        # Training/Reproducibility configs
        self.EPOCHS: int = int(train_cfg.get("epochs", 100))
        self.LEARNING_RATE: float = float(train_cfg.get("learning_rate", 0.0004))
        self.SEED: int = int(train_cfg.get("seed", 42))
        self.DETERMINISTIC: bool = bool(train_cfg.get("deterministic", True))
        self.PATIENCE: int = int(train_cfg.get("patience", 10))
        self.DEVICE: str = train_cfg.get("device", "auto")
        self.RUN_NAME: str = train_cfg.get("run_name", "run_1")
        self.REQUIRE_ALL_CATEGORIES: bool = bool(train_cfg.get("require_all_categories", False))

        # Paths
        self.CHECKPOINT_DIR: str = train_cfg.get("checkpoint_dir", "./checkpoints")
        self.LOG_DIR: str = train_cfg.get("log_dir", "./logs")

        # Evaluation configs
        self.THRESHOLD_METHOD: str = eval_cfg.get("threshold_method", "youden")
        self.PRO_FPR_LIMIT: float = float(eval_cfg.get("pro_fpr_limit", 0.3))

settings = Settings()
