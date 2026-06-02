"""
Experiment configuration.

Every experiment is a named ExpConfig in REGISTRY. The code path is identical
across runs; only the named config changes. Centralising experiment definitions
here keeps runs reproducible and avoids drift between near-duplicate
per-experiment scripts.

The model architectures are defined here. Additional dropout strategies are
added by registering new configs with the relevant dropout_kind and
hyperparameters; no new configuration machinery is required.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Union


@dataclass
class ExpConfig:
    name: str
    dataset: str                       # "mnist" | "cifar10"
    widths: List[int]                  # full layer sizes incl. input/output
    dropout_kind: str = "uniform"      # "uniform" | "per_neuron" | "none"
    p: float = 0.5                     # base drop probability (p_base)
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    activation: str = "relu"
    val_fraction: float = 0.1
    # Hook-driven per-neuron dropout (Eq. 4). Active only when prob_source is set
    # (and dropout_kind == "per_neuron"); otherwise per-neuron dropout stays at
    # the static constant p. These fields are otherwise ignored.
    warmup_epochs: int = 0             # uniform-dropout warm-up before activation (N_warm)
    recompute_every: int = 0           # epochs between recomputations (Delta)
    alpha: Union[float, List[float]] = 2.0   # sigmoid sensitivity (scalar or per hidden layer)
    beta: Union[float, List[float]] = 0.0    # sigmoid offset (scalar or per hidden layer)
    standardize: bool = True                 # z-score each layer's signal before the sigmoid
    prob_source: Optional[str] = None        # None (static) | "forman" | "magnitude" | "random"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----- model architectures -----
MNIST_SMALL = [784, 512, 256, 10]
MNIST_DEEP  = [784, 1024, 512, 256, 10]
CIFAR_MAIN  = [3072, 1024, 512, 256, 10]


def _baselines() -> Dict[str, ExpConfig]:
    """Baseline grid: each (dataset, architecture) crossed with the dropout settings."""
    cfgs: Dict[str, ExpConfig] = {}

    specs = [
        ("mnist_small", "mnist",   MNIST_SMALL, 40),
        ("mnist_deep",  "mnist",   MNIST_DEEP,  40),
        ("cifar_main",  "cifar10", CIFAR_MAIN,  80),
    ]
    for tag, dataset, widths, epochs in specs:
        # No dropout (ablation).
        cfgs[f"{tag}_nodrop"] = ExpConfig(
            name=f"{tag}_nodrop", dataset=dataset, widths=widths,
            dropout_kind="none", p=0.0, epochs=epochs)
        # Uniform dropout (primary baseline).
        cfgs[f"{tag}_uniform"] = ExpConfig(
            name=f"{tag}_uniform", dataset=dataset, widths=widths,
            dropout_kind="uniform", p=0.5, epochs=epochs)
        # Per-neuron dropout at constant p; equivalent to uniform (sanity check).
        cfgs[f"{tag}_perneuron"] = ExpConfig(
            name=f"{tag}_perneuron", dataset=dataset, widths=widths,
            dropout_kind="per_neuron", p=0.5, epochs=epochs)
        # Curvature-aware dropout. warmup_epochs / recompute_every / alpha / beta
        # are tuned on validation; these are starting points.
        cfgs[f"{tag}_curvature"] = ExpConfig(
            name=f"{tag}_curvature", dataset=dataset, widths=widths,
            dropout_kind="per_neuron", p=0.5, epochs=epochs,
            warmup_epochs=5, recompute_every=5, alpha=2.0, beta=0.0,
            prob_source="forman")
        # Controls sharing the per-neuron hook mechanism but a different signal,
        # to show the geometric signal (not mere non-uniformity or raw magnitude)
        # drives any effect.
        cfgs[f"{tag}_magnitude"] = ExpConfig(
            name=f"{tag}_magnitude", dataset=dataset, widths=widths,
            dropout_kind="per_neuron", p=0.5, epochs=epochs,
            warmup_epochs=5, recompute_every=5, alpha=2.0, beta=0.0,
            prob_source="magnitude")
        cfgs[f"{tag}_random"] = ExpConfig(
            name=f"{tag}_random", dataset=dataset, widths=widths,
            dropout_kind="per_neuron", p=0.5, epochs=epochs,
            warmup_epochs=5, recompute_every=5, alpha=2.0, beta=0.0,
            prob_source="random")
        # Targeted-Dropout direction: negative alpha retains high-magnitude
        # neurons and drops low-magnitude ones (the opposite of _magnitude). A
        # soft per-neuron stand-in for Gomez et al. (2018), as a magnitude
        # comparator with the sensible importance direction.
        cfgs[f"{tag}_targeted"] = ExpConfig(
            name=f"{tag}_targeted", dataset=dataset, widths=widths,
            dropout_kind="per_neuron", p=0.5, epochs=epochs,
            warmup_epochs=5, recompute_every=5, alpha=-2.0, beta=0.0,
            prob_source="magnitude")
    return cfgs


REGISTRY: Dict[str, ExpConfig] = _baselines()


def get_config(name: str) -> ExpConfig:
    if name not in REGISTRY:
        raise KeyError(f"unknown config {name!r}. "
                       f"available: {sorted(REGISTRY)}")
    return REGISTRY[name]
