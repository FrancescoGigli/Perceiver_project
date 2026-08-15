# src/utils/seed.py
# Fissa ogni sorgente di casualita' del training.

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Rende il training riproducibile: pesi, shuffling, augmentation, kernel cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
