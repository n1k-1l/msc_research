"""
Data loading for MNIST and CIFAR-10.

A fixed validation slice is carved out of the official training set so that
hyperparameters can be tuned on validation rather than test. The test set is
evaluated once per final config, after all tuning decisions are made; tuning on
test would bias the reported test-accuracy gap.

Normalisation uses the standard per-dataset channel statistics. No data
augmentation is applied: augmentation is itself a regulariser and would
confound a comparison between dropout strategies, so dropout is kept as the
only varying regulariser.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# standard per-channel mean/std
_STATS = {
    "mnist":   ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
_INPUT_DIM = {"mnist": 784, "cifar10": 3072}
_NUM_CLASSES = 10


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    input_dim: int
    num_classes: int


def get_data(
    dataset: str,
    data_root: str = "./data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    seed: int = 0,
) -> DataBundle:
    """
    Return train/validation/test DataLoaders.

    The train/validation split is seeded, so a given seed always yields the same
    validation set, which keeps comparisons across runs fair.
    """
    dataset = dataset.lower()
    if dataset not in _STATS:
        raise ValueError(f"unknown dataset {dataset!r}")

    mean, std = _STATS[dataset]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[dataset]
    full_train = ds_cls(data_root, train=True,  download=True, transform=tf)
    test_set   = ds_cls(data_root, train=False, download=True, transform=tf)

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=gen)

    # pin_memory only accelerates host-to-device copies for CUDA; on CPU/MPS it
    # page-locks memory for no benefit (and warns on MPS), so it is gated on
    # CUDA. persistent_workers keeps the worker pool alive across epochs instead
    # of re-spawning it each epoch, which matters where the start method is
    # 'spawn' (macOS).
    pin_memory = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory,
                  persistent_workers=(num_workers > 0))
    return DataBundle(
        train=DataLoader(train_set, shuffle=True,  drop_last=True,  **common),
        val=DataLoader(val_set,     shuffle=False, drop_last=False, **common),
        test=DataLoader(test_set,   shuffle=False, drop_last=False, **common),
        input_dim=_INPUT_DIM[dataset],
        num_classes=_NUM_CLASSES,
    )
