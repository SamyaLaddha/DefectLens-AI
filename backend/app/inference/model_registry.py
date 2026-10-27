"""
Model Registry for preloading and serving category-specific predictors in DefectLens AI.
"""

from typing import Optional, Dict
from app.config import settings
from app.inference.predictor import STFPMPredictor
from app.utils.logging_utils import get_logger

logger = get_logger("model_registry")

class ModelRegistry:
    """
    Manages preloaded model instances for active dataset categories.
    """
    def __init__(self):
        self.predictors: Dict[str, STFPMPredictor] = {}

    def load_models(self):
        """
        Iterates and preloads all category models listed in config.yaml.
        Missing model checkpoints are logged and skipped in development
        but cause startup crashes when REQUIRE_ALL_CATEGORIES is enabled.
        """
        logger.info(f"Initiating preloading for categories: {settings.CATEGORIES}")
        
        for category in settings.CATEGORIES:
            run_name = settings.CATEGORY_RUNS.get(category, settings.RUN_NAME)
            logger.info(f"Attempting to load predictor for: '{category}' (run: '{run_name}')")
            try:
                predictor = STFPMPredictor(category=category, run_name=run_name)
                self.predictors[category] = predictor
                logger.info(
                    f"Successfully loaded category model: '{category}' "
                    f"(run: '{run_name}', threshold: {predictor.threshold:.6f})"
                )
            except Exception as e:
                error_msg = f"Failed to preload model for category '{category}' (run '{run_name}'): {str(e)}"
                if settings.REQUIRE_ALL_CATEGORIES:
                    logger.critical(f"CRITICAL STARTUP FAILURE: {error_msg}")
                    raise RuntimeError(f"Startup check failed: {error_msg}")
                else:
                    logger.error(f"DEVELOPMENT MODEL SKIP: {error_msg}")

        logger.info(f"Model preloading complete. Operational categories: {list(self.predictors.keys())}")

    def get_predictor(self, category: str) -> Optional[STFPMPredictor]:
        """
        Retrieves the preloaded predictor instance. Returns None if not loaded.
        """
        return self.predictors.get(category)
