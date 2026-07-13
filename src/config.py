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
    widths: List[int]                  # full layer sizes incl. input/output (MLP)
    arch: str = "mlp"                  # "mlp" | "lenet" (CNN channel-graph pruning)
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
    # Iterative hard pruning (train dense -> prune a fraction -> fine-tune -> repeat),
    # to test whether curvature diverges from magnitude under NON-uniform connectivity.
    # When iterative_prune is set, the trained dense net is pruned by each of
    # magnitude / curvature / random (see src/iterative_prune.py) and the curves are
    # stored as `iterative_prune_curves`. Ignored otherwise.
    iterative_prune: bool = False
    prune_granularity: str = "edge"          # "edge" (unstructured; the divergence test) | "unit" (structured; control)
    prune_schedule: List[float] = field(     # cumulative sparsity reached each round
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
    finetune_epochs: int = 5                 # fine-tune epochs between prune rounds
    # Sparse-by-construction net (fixed random topology): gives genuinely non-uniform
    # degree independently of magnitude, so graph curvature (Forman/Ollivier) has real
    # geometric content and Ollivier-Ricci is tractable. When sparse_init, a fixed
    # random mask at sparse_density is applied per layer and held through training;
    # pruning then starts from that topology. prune_criteria selects which scores to
    # compare (defaults to the CRITERIA tuple when None).
    sparse_init: bool = False
    sparse_density: float = 0.15             # keep-fraction of the fixed random topology
    prune_criteria: Optional[List[str]] = None
    calib_batches: int = 2                   # train batches averaged for activation-dependent criteria
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


def _iterative_prune_configs() -> Dict[str, ExpConfig]:
    """Iterative hard pruning on the small MNIST MLP: train one dense net, then prune
    it by magnitude / curvature / random with fine-tuning between rounds.

    edge (unstructured) is the divergence test -- removing individual weights makes
    degree non-uniform, so Forman curvature finally carries geometric content beyond
    |w|; unit (structured) is the control that should stay ~ magnitude because the
    survivors remain fully interconnected. Trained with no dropout (a clean dense
    base), evaluated on accuracy-vs-sparsity; each seed writes all three curves.
    """
    common = dict(dataset="mnist", widths=MNIST_SMALL, dropout_kind="none",
                  p=0.0, epochs=40, iterative_prune=True, finetune_epochs=5)
    return {
        "mnist_small_ip": ExpConfig(
            name="mnist_small_ip", prune_granularity="edge", **common),
        "mnist_small_ip_unit": ExpConfig(
            name="mnist_small_ip_unit", prune_granularity="unit", **common),
    }


def _sparse_prune_configs() -> Dict[str, ExpConfig]:
    """Curvature vs magnitude on a sparse-by-construction MLP (fixed random topology).

    The dense/E9 setting forces graph curvature to be a magnitude proxy (uniform
    degree) or, when we impose sparsity ourselves, a destabilising degree feedback.
    Here the topology is non-uniform by construction and chosen independently of
    magnitude, and it is sparse enough that Ollivier-Ricci is tractable. Train one
    fixed-sparse net per seed, then prune further by each criterion (accuracy vs
    additional sparsity). ollivier_topo (unweighted metric) is the one signal that
    decouples from magnitude; ollivier (weighted 1/|w|) and forman track it.
    """
    return {
        "mnist_sparse_ip": ExpConfig(
            name="mnist_sparse_ip", dataset="mnist", widths=MNIST_SMALL,
            dropout_kind="none", p=0.0, epochs=40,
            iterative_prune=True, sparse_init=True, sparse_density=0.15,
            finetune_epochs=3,
            prune_schedule=[0.85, 0.88, 0.91, 0.94, 0.96],
            prune_criteria=["magnitude", "random", "forman", "ollivier",
                            "ollivier_topo", "ollivier_neural"]),
    }


def _cnn_prune_configs() -> Dict[str, ExpConfig]:
    """Channel-graph pruning on a LeNet CNN (Stage 1 of the CNN extension).

    Extends the curvature-vs-magnitude pruning comparison to convolutions by viewing
    each conv as a channel graph (see src/cnn_prune.py). MNIST/LeNet is the fast
    validation regime -- and the one where arXiv:2601.16366 reports curvature ties
    magnitude, so it doubles as a check that our pipeline reproduces that tie before
    the heavier CIFAR/VGG test. Prunes conv channel-edges + FC weights by each
    criterion; accuracy vs sparsity.
    """
    return {
        "mnist_lenet_ip": ExpConfig(
            name="mnist_lenet_ip", dataset="mnist", widths=[784, 10], arch="lenet",
            dropout_kind="none", p=0.0, epochs=15,
            iterative_prune=True, finetune_epochs=2,
            prune_schedule=[0.3, 0.5, 0.7, 0.85, 0.92],
            prune_criteria=["magnitude", "random", "forman", "ollivier",
                            "ollivier_topo", "ollivier_neural"]),
    }


REGISTRY: Dict[str, ExpConfig] = {**_baselines(), **_targeted_dropout_configs(),
                                  **_iterative_prune_configs(),
                                  **_sparse_prune_configs(),
                                  **_cnn_prune_configs()}


def get_config(name: str) -> ExpConfig:
    if name not in REGISTRY:
        raise KeyError(f"unknown config {name!r}. "
                       f"available: {sorted(REGISTRY)}")
    return REGISTRY[name]
