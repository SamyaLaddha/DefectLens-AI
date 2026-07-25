"""
Reproducibility seeding utilities.

Configures seeds for Python, NumPy, and PyTorch, applying deterministic settings.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seeds for Python, NumPy, and PyTorch generators.

    Ensures that random generators yield consistent results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Configuring workspace config for determinism to avoid CUDA errors in some PyTorch versions
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Applying deterministic algorithms has performance implications (speed tradeoff).
        # Certain optimized algorithms on CPU and GPU (like fast convolution paths) are
        # skipped in favor of slower but reproducible deterministic execution paths.
        torch.use_deterministic_algorithms(True)
        # Ensure CUDA convolution operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Revert to standard non-deterministic behaviors
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
