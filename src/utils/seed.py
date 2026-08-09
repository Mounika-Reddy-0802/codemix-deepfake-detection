"""Global seed control for reproducibility.

Seeds Python's ``random`` and numpy always; seeds torch (and enables deterministic
cudnn) only if torch is importable, so this module stays usable without the ML
stack. Call :func:`set_seed` once at the start of training/eval.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 1234, deterministic: bool = True) -> int:
    """Seed all RNGs for reproducibility. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
