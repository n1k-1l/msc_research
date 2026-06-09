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
    # Targeted Dropout (Gomez et al. 2019): drop the target_gamma fraction of units
    # at the prunable end of the score with prob target_drop each step, so the net
    # is robust to post-hoc pruning. prob_mapping="targeted" uses this hard mask
    # instead of the Eq. 4 sigmoid; magnitude-TD vs curvature-TD then differ only in
    # prob_source and target_direction. Evaluated on accuracy-vs-sparsity.
    prob_mapping: str = "sigmoid"            # "sigmoid" (curvature-dropout) | "targeted" (Targeted Dropout)
    target_gamma: float = 0.5                # fraction of units targeted (gamma)
    target_drop: float = 0.5                 # drop probability for targeted units (alpha)
    target_direction: str = "low"            # "low" (small=prunable: magnitude) | "high" (large=prunable: curvature)
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
        # to show the geometric signal (not mere non-uniformity or weight
        # magnitude) drives any effect.
        cfgs[f"{tag}_random"] = ExpConfig(
            name=f"{tag}_random", dataset=dataset, widths=widths,
            dropout_kind="per_neuron", p=0.5, epochs=epochs,
            warmup_epochs=5, recompute_every=5, alpha=2.0, beta=0.0,
            prob_source="random")
        # Weight-magnitude comparator in the Targeted-Dropout direction (Gomez et
        # al. 2018): negative alpha retains high-magnitude neurons and drops
        # low-magnitude ones. A soft per-neuron stand-in for targeted dropout.
        cfgs[f"{tag}_targeted"] = ExpConfig(
            name=f"{tag}_targeted", dataset=dataset, widths=widths,
            dropout_kind="per_neuron", p=0.5, epochs=epochs,
            warmup_epochs=5, recompute_every=5, alpha=-2.0, beta=0.0,
            prob_source="magnitude")
    return cfgs


def _targeted_dropout_configs() -> Dict[str, ExpConfig]:
    """Curvature as the Targeted-Dropout criterion vs the magnitude baseline.

    Faithful Targeted Dropout (hard bottom/top-gamma mask + stochastic drop) on the
    small MNIST MLP, evaluated on pruning robustness (accuracy-vs-sparsity; see
    run_baseline.py -> prune_curve), which is the right metric for TD. Three arms
    differing only in the score source and which end of it is "prunable":
      td_magnitude - rank units by |w| (L2 norm), drop the lowest (vanilla TD).
      td_forman    - rank by Forman curvature, drop the highest (positive=redundant).
      td_random    - random scores (control: is any structure better than none?).
    gamma / drop are untuned starting points (tune on validation).
    """
    # Schedule follows standard Targeted Dropout (Gomez et al. 2019): no warm-up
    # (fixed gamma from the start) and the targeted set re-ranked every epoch
    # (the paper re-ranks every minibatch; once-per-epoch is the closest the epoch
    # hook allows -- noted as a faithfulness approximation). gamma/drop are in the
    # paper's tested range; gamma=0.5 trains robustness to dropping the bottom 50%.
    common = dict(dataset="mnist", widths=MNIST_SMALL, dropout_kind="per_neuron",
                  p=0.0, epochs=40, warmup_epochs=0, recompute_every=1,
                  prob_mapping="targeted", target_gamma=0.5, target_drop=0.5)
    return {
        "mnist_small_td_magnitude": ExpConfig(
            name="mnist_small_td_magnitude", prob_source="magnitude",
            target_direction="low", **common),
        "mnist_small_td_forman": ExpConfig(
            name="mnist_small_td_forman", prob_source="forman",
            target_direction="high", **common),
        "mnist_small_td_random": ExpConfig(
            name="mnist_small_td_random", prob_source="random",
            target_direction="low", **common),
    }


REGISTRY: Dict[str, ExpConfig] = {**_baselines(), **_targeted_dropout_configs()}


def get_config(name: str) -> ExpConfig:
    if name not in REGISTRY:
        raise KeyError(f"unknown config {name!r}. "
                       f"available: {sorted(REGISTRY)}")
    return REGISTRY[name]
