"""
Logging utilities for DefectLens AI.

Provides console-based logger initialization for training and evaluation.
"""

import logging

def get_logger(name: str) -> logging.Logger:
    """
    Initialize and return a configured logger with console handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        # Formatted using standard Normal hyphens only
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
