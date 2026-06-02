"""
Reproducibility helpers.

The headline result is a small test-accuracy gap between methods, which is only
meaningful if run-to-run variance is controlled. Two practices support that:

  1. Seed Python, numpy, and torch, and record the seed in every result file.
  2. Run every config across multiple seeds (3-5) and report mean +/- std.

set_seed covers (1); the experiment scripts cover (2) by looping over seeds.

cudnn.deterministic=True removes a source of GPU nondeterminism at a small
throughput cost, which is the right trade-off for a study that depends on
trustworthy, repeatable numbers.
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
