"""
Curvature-aware dropout.

Computes discrete Forman-Ricci curvature on the weight graph of an MLP, maps it
to per-neuron dropout probabilities, and exposes the whole thing as an
epoch_hook so it plugs into the existing training loop without changing it.

The hook mechanism is signal-agnostic: it drives per-neuron dropout from any
per-neuron "score" function (see SCORE_SOURCES), of which Forman curvature is the
method and weight-magnitude / random are controls. The Eq. 4 mapping below
applies to whatever score is supplied; only per_neuron_curvature / forman_edges
are about curvature specifically.

Weight graph (proposal section 3). For each adjacent layer pair (L_k, L_{k+1})
a weighted bipartite graph is formed whose edge weights are the absolute values
of the connecting parameters. The graph is fully connected: a neuron in layer k
has degree equal to the width of layer k+1. Each edge therefore belongs to
exactly one layer pair, and a hidden neuron belongs to the two pairs on either
side of it.

Forman-Ricci curvature (Eq. 2). For an edge e between vertices u, v with edge
weight w_e, vertex weights w_u, w_v, and adjacent edges e' incident to either
endpoint (excluding e):

    F(e) = w_e * ( w_u/w_e + w_v/w_e
                   - sum_{e' ~ u} w_u / sqrt(w_e w_e')
                   - sum_{e' ~ v} w_v / sqrt(w_e w_e') )

Defining a per-vertex strength S(v) = sum_{e incident to v} 1/sqrt(w_e) (the sum
runs over the edges of the one pair the vertex is considered in, and includes e
itself), this collapses to the closed form used below:

    F(e=(u,v)) = 2 (w_u + w_v) - sqrt(w_e) * ( w_u S(u) + w_v S(v) )

which is a single vectorised expression over the pair's weight matrix. The
implementation fixes vertex weights to 1 (the network supplies only edge
weights; the vertex weight is a free choice), so it evaluates
F = 4 - sqrt(w_e) * (S(u) + S(v)); in the fully unweighted case this is the
special case F = 4 - deg(u) - deg(v), used as a correctness check.

Per-neuron curvature (Eq. 3) averages the curvature of a neuron's incident edges,
spanning both pairs a hidden neuron belongs to.

Score-to-dropout (Eq. 4) maps the per-neuron score to a drop probability via a
shifted sigmoid, then renormalises so the expected number of dropped neurons per
layer matches uniform dropout at rate p_base:

    p~_i = p_base * sigmoid(alpha * (score_i - beta))
    p_i  = (n * p_base / sum_j p~_j) * p~_i

For curvature, higher curvature (redundant neighbourhoods) gives higher dropout;
negative curvature (bottlenecks) gives lower dropout, so such neurons are
retained more often.
"""
from __future__ import annotations
from typing import Callable, List, Dict, Any, Sequence, Union
import torch
import torch.nn as nn

from .dropout import PerNeuronDropout

# Largest drop probability handed to PerNeuronDropout; its forward divides by
# (1 - p), so p must stay strictly below 1.
_P_MAX = 1.0 - 1e-4

# A per-neuron score function: ordered Linear weight tensors -> one score vector
# per layer (input .. output).
ScoreFn = Callable[[List[torch.Tensor]], List[torch.Tensor]]


@torch.no_grad()
def forman_edges(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Forman-Ricci curvature of every edge in one bipartite layer pair.

    Vertex weights are fixed to 1: the network supplies only edge weights (|W|),
    and the vertex weight is a free choice we set to unity. The closed form is
    then F = 4 - sqrt(w_e) * (S(u) + S(v)), which reduces to F = 4 - deg(u) -
    deg(v) in the fully unweighted case (the correctness check).

    Args:
        A: (n_tgt, n_src) matrix of absolute edge weights for a pair
           (L_k, L_{k+1}); A[a, b] is the edge between source neuron b in L_k and
           target neuron a in L_{k+1} (the shape of a Linear's weight matrix).

    Returns:
        F with the same shape as A; F[a, b] is that edge's curvature.
    """
    inv_sqrt = A.clamp_min(eps).rsqrt()
    # vertex strength S(v) = sum over the pair's incident edges of 1/sqrt(w_e)
    s_src = inv_sqrt.sum(dim=0).unsqueeze(0)   # (1, n_src)
    s_tgt = inv_sqrt.sum(dim=1).unsqueeze(1)   # (n_tgt, 1)
    return 4.0 - A.sqrt() * (s_src + s_tgt)


@torch.no_grad()
def forman_edges_masked(A: torch.Tensor, mask: torch.Tensor,
                        eps: float = 1e-12) -> torch.Tensor:
    """Forman-Ricci curvature per edge with a boolean edge-presence mask.

    Unstructured pruning removes individual edges, so a vertex's *effective degree*
    is its number of surviving incident edges. Masked-out edges must therefore
    contribute nothing to the vertex strengths S(u), S(v) (as opposed to
    forman_edges, which clamps a zero weight to eps and so would treat a pruned
    edge as a huge 1/sqrt(w) term). The constant 4 = 2(w_u + w_v) is independent of
    degree and stays; degree enters only through the sums, so this reduces to
    F = 4 - deg_present(u) - deg_present(v) in the unweighted masked case -- i.e.
    the topological term now varies per vertex once the graph is non-uniform.

    With an all-True mask this is identical to forman_edges. Pruned edges are
    returned as NaN (their curvature is undefined); survivors carry finite values.

    Args:
        A: (n_tgt, n_src) absolute edge weights for one layer pair.
        mask: bool tensor, same shape as A; True = edge present.
    """
    inv_sqrt = torch.where(mask, A.clamp_min(eps).rsqrt(), torch.zeros_like(A))
    s_src = inv_sqrt.sum(dim=0, keepdim=True)   # S over surviving edges, (1, n_src)
    s_tgt = inv_sqrt.sum(dim=1, keepdim=True)   # (n_tgt, 1)
    F = 4.0 - A.sqrt() * (s_src + s_tgt)
    return torch.where(mask, F, torch.full_like(F, float("nan")))


@torch.no_grad()
def per_neuron_curvature_masked(weights: List[torch.Tensor],
                                masks: List[torch.Tensor]) -> List[torch.Tensor]:
    """Per-neuron Forman curvature on a masked (unstructured-pruned) weight graph.

    Like per_neuron_curvature, but each neuron averages over its *surviving*
    incident edges only, using forman_edges_masked so effective degree tracks the
    mask. Used by the unit-granularity iterative-pruning control (where whole
    neurons, i.e. full rows/cols, are removed).

    Args:
        weights: ordered Linear weight tensors [W_1, ..., W_L].
        masks: one bool mask per weight tensor (True = edge present).
    """
    A = [W.abs() for W in weights]
    sizes = [A[0].shape[1]] + [a.shape[0] for a in A]
    dev, dtype = A[0].device, A[0].dtype

    kappa_sum = [torch.zeros(n, device=dev, dtype=dtype) for n in sizes]
    degree = [torch.zeros(n, device=dev, dtype=dtype) for n in sizes]
    for k, (a, m) in enumerate(zip(A, masks)):
        F = forman_edges_masked(a, m)
        F0 = torch.nan_to_num(F, nan=0.0)      # masked edges add nothing
        kappa_sum[k] += F0.sum(dim=0)          # source side (layer k)
        kappa_sum[k + 1] += F0.sum(dim=1)      # target side (layer k+1)
        degree[k] += m.sum(dim=0)              # surviving incident edges
        degree[k + 1] += m.sum(dim=1)
    return [ks / d.clamp_min(1) for ks, d in zip(kappa_sum, degree)]


@torch.no_grad()
def per_neuron_curvature(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """Per-neuron Forman-Ricci curvature for every layer (Eq. 3).

    Each edge's curvature is computed within its bipartite layer pair; a neuron's
    curvature is then the mean over its incident edges, which for a hidden neuron
    spans the pairs on both sides of it.

    Args:
        weights: ordered Linear weight tensors [W_1, ..., W_L]; W_k has shape
            (size(L_k), size(L_{k-1})).

    Returns:
        A list of 1-D tensors, one per layer (input .. output). For driving
        dropout, the entry for hidden layer k is at index k.
    """
    A = [W.abs() for W in weights]
    sizes = [A[0].shape[1]] + [a.shape[0] for a in A]
    dev, dtype = A[0].device, A[0].dtype

    kappa_sum = [torch.zeros(n, device=dev, dtype=dtype) for n in sizes]
    degree = [torch.zeros(n, device=dev, dtype=dtype) for n in sizes]
    for k, a in enumerate(A):
        F = forman_edges(a)
        kappa_sum[k] += F.sum(dim=0)          # source side (layer k)
        kappa_sum[k + 1] += F.sum(dim=1)      # target side (layer k+1)
        degree[k] += F.shape[0]
        degree[k + 1] += F.shape[1]
    return [ks / d for ks, d in zip(kappa_sum, degree)]


@torch.no_grad()
def weight_magnitude(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """Per-neuron L2 norm of incident weights (weight-magnitude control).

    This is the unit-ranking statistic of Targeted Dropout (Gomez et al. 2019):
    units are ranked by the L2 norm of their weights and the smallest are the
    prunable ones. As in per_neuron_curvature, a hidden neuron's incident weights
    span the two layer pairs it belongs to, so the norm is taken over both sides --
    the *same* edge set curvature averages over -- which keeps the magnitude-vs-
    curvature comparison on identical neighbourhoods. Targeted Dropout ranks within
    each layer, where the degree is constant, so the L2 norm is left unnormalised
    (dividing by degree would not change any within-layer ranking).
    """
    sizes = [weights[0].shape[1]] + [W.shape[0] for W in weights]
    dev, dtype = weights[0].device, weights[0].dtype
    sq = [torch.zeros(n, device=dev, dtype=dtype) for n in sizes]
    for k, W in enumerate(weights):
        w2 = W * W
        sq[k] += w2.sum(dim=0)          # source side (layer k)
        sq[k + 1] += w2.sum(dim=1)      # target side (layer k+1)
    return [s.sqrt() for s in sq]


@torch.no_grad()
def random_curvature(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """Random per-neuron scores (random-curvature control).

    Drives the same renormalised mapping with random values, isolating whether
    the curvature signal (rather than mere non-uniformity) drives any effect.
    Redrawn at each recomputation; reproducible through the global seed.
    """
    w0 = weights[0]
    sizes = [w0.shape[1]] + [W.shape[0] for W in weights]
    return [torch.randn(n, device=w0.device, dtype=w0.dtype) for n in sizes]


# Selectable per-neuron score function: the method and its controls.
SCORE_SOURCES: Dict[str, ScoreFn] = {
    "forman": per_neuron_curvature,
    "magnitude": weight_magnitude,
    "random": random_curvature,
}


def get_source(name: str) -> ScoreFn:
    """Look up a per-neuron score function by name."""
    try:
        return SCORE_SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown score source {name!r}; "
                         f"choices: {sorted(SCORE_SOURCES)}")


@torch.no_grad()
def scores_to_probs(
    scores: torch.Tensor,
    p_base: float,
    alpha: float,
    beta: float,
    standardize: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Map a per-neuron score to renormalised drop probabilities (Eq. 4).

    The score is whatever the source produced (Forman curvature, weight
    magnitude, random); this mapping is signal-agnostic.

    Raw Forman curvature scales with layer width, so the location of the score
    distribution differs sharply between layers. With standardize=True the layer's
    score is z-scored (mean 0, unit std) before the sigmoid, which makes alpha and
    beta dimensionless and behave consistently across layers and architectures;
    this is equivalent to deriving a per-layer alpha = alpha/std and beta from the
    layer's own distribution.

    The renormalisation keeps the expected per-layer drop count equal to uniform
    dropout at rate p_base. The final clamp to [0, 1) is required because the
    renormalisation can push individual probabilities to or above 1; when it
    fires it perturbs the budget slightly (see clamped_fraction).
    """
    if standardize:
        scores = (scores - scores.mean()) / (scores.std(unbiased=False) + eps)
    p_tilde = p_base * torch.sigmoid(alpha * (scores - beta))
    n = scores.numel()
    p = (n * p_base / (p_tilde.sum() + eps)) * p_tilde
    return p.clamp(0.0, _P_MAX)


def clamped_fraction(probs: torch.Tensor) -> float:
    """Fraction of probabilities the clamp pinned at the maximum (budget leak)."""
    return float((probs >= _P_MAX).float().mean().item())


@torch.no_grad()
def scores_to_targeted_probs(
    scores: torch.Tensor,
    gamma: float,
    drop: float,
    direction: str = "low",
) -> torch.Tensor:
    """Targeted-Dropout mask as a per-neuron drop-probability vector.

    Targeted Dropout (Gomez et al. 2019) selects the gamma fraction of units at the
    *prunable* end of a ranking and drops each of them with probability `drop`,
    leaving the rest untouched, so the network learns to be robust to pruning that
    set. That is exactly a per-neuron probability vector ({drop on the targeted
    units, 0 elsewhere}), so it feeds PerNeuronDropout.set_probs unchanged and
    reuses the whole hook -- magnitude-TD and curvature-TD then differ only in the
    score source and which end is prunable.

    direction="low" targets the lowest-scoring units (weight magnitude: small |w| =
    prunable, the vanilla TD criterion); "high" targets the highest (Forman
    curvature: positive curvature = redundant = prunable). The criterion is a pure
    ranking -- no differentiability, no standardisation needed.
    """
    n = scores.numel()
    n_target = int(round(gamma * n))
    probs = torch.zeros_like(scores)
    if n_target <= 0:
        return probs
    if direction not in ("low", "high"):
        raise ValueError(f"direction must be 'low' or 'high', got {direction!r}")
    idx = torch.topk(scores, n_target, largest=(direction == "high")).indices
    probs[idx] = float(drop)
    return probs.clamp(0.0, _P_MAX)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson (linear) correlation rho of two vectors, pure torch.

    Logged each recompute as `pearson_curv_mag` to track how tightly curvature and
    magnitude are coupled as training proceeds: |rho| ~ 1 means curvature has
    collapsed to ~a function of weight magnitude here (explains the null -- the two
    Targeted-Dropout criteria then rank units the same way); smaller |rho| means
    curvature carries extra (neighbourhood) information.
    """
    if a.numel() < 2:
        return float("nan")
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)


def _layer_stats(scores: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
    return {
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "score_std": float(scores.std(unbiased=False)),
        "p_min": float(probs.min()),
        "p_mean": float(probs.mean()),
        "p_max": float(probs.max()),
        "clamped_fraction": clamped_fraction(probs),
    }


class DropoutProbHook:
    """epoch_hook that activates and refreshes hook-driven per-neuron dropout.

    Training protocol (proposal section 3): uniform dropout for the first
    warmup_epochs, then the per-neuron score is computed and per-neuron dropout
    activated, recomputed every recompute_every epochs as the weights evolve.
    During warm-up the PerNeuronDropout modules keep their initial constant p,
    which is equivalent to uniform dropout, so nothing special is needed for that
    phase.

    The per-recompute score / probability distribution is recorded in `log` so the
    "curvature distribution evolution" metric can be reported. The runner reads
    `hook.log` after training and appends it to the result JSON.

    Args:
        p_base: Eq. 4 dropout budget (shared across layers).
        alpha, beta: Eq. 4 sensitivity and offset. Each may be a scalar (used
            for every layer) or a per-hidden-layer sequence. Raw Forman curvature
            scales with layer width, so the score distributions of different
            layers sit at very different locations; per-layer alpha/beta let each
            sigmoid be placed on its own layer's distribution.
        warmup_epochs: epochs of uniform dropout before activation (N_warm).
        recompute_every: epochs between recomputations (Delta); <= 1 recomputes
            every epoch after warm-up.
        standardize: z-score each layer's score before the sigmoid, so a single
            (alpha, beta) transfers across layers of different widths.
        source: per-neuron score function (see SCORE_SOURCES); defaults to
            per_neuron_curvature (the method). Swap it for the weight-magnitude or
            random controls.
        record: whether to populate `log`.
    """

    def __init__(
        self,
        p_base: float,
        alpha: Union[float, Sequence[float]],
        beta: Union[float, Sequence[float]],
        warmup_epochs: int,
        recompute_every: int,
        standardize: bool = False,
        source: ScoreFn = per_neuron_curvature,
        record: bool = True,
        mapping: str = "sigmoid",
        gamma: float = 0.5,
        drop: float = 0.5,
        direction: str = "low",
    ):
        self.p_base = p_base
        self.alpha = alpha
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.recompute_every = max(1, recompute_every)
        self.standardize = standardize
        self.source = source
        self.record = record
        # mapping=="targeted" -> Targeted-Dropout mask (scores_to_targeted_probs),
        # using gamma/drop/direction; otherwise the Eq. 4 sigmoid (scores_to_probs),
        # using alpha/beta/standardize.
        self.mapping = mapping
        self.gamma = gamma
        self.drop = drop
        self.direction = direction
        self.log: List[Dict[str, Any]] = []

    def _should_recompute(self, epoch: int) -> bool:
        if epoch < self.warmup_epochs:
            return False
        # Below is true every self.recompute_every epochs
        return (epoch - self.warmup_epochs) % self.recompute_every == 0

    @staticmethod
    def _resolve(value: Union[float, Sequence[float]], k: int, n: int,
                 name: str) -> float:
        """Per-layer value: scalars broadcast, sequences are indexed by layer."""
        if isinstance(value, (int, float)):
            return float(value)
        seq = list(value)
        assert len(seq) == n, (
            f"{name} has {len(seq)} entries but there are {n} hidden layers")
        return float(seq[k])

    @torch.no_grad()
    def __call__(self, model: nn.Module, epoch: int) -> None:
        if not self._should_recompute(epoch):
            return

        linears = model.linear_layers()
        drops = [m for m in model.dropout_modules()
                 if isinstance(m, PerNeuronDropout)]
        assert len(drops) == len(linears) - 1, (
            "the per-neuron dropout hook expects a PerNeuronDropout after every "
            f"hidden layer; got {len(drops)} dropouts for {len(linears)} linear "
            "layers")

        weights = [lin.weight for lin in linears]
        scores = self.source(weights)
        n_hidden = len(drops)
        # Diagnostic: how different are the curvature and magnitude orderings as
        # training proceeds -- the key read on whether curvature is more than a
        # magnitude statistic here. Cheap (about two forward-pass-cost passes).
        diag = (per_neuron_curvature(weights), weight_magnitude(weights)) \
            if self.record else None
        layer_log: List[Dict[str, float]] = []
        for k, dmod in enumerate(drops):
            # dmod k follows linears[k], whose output is hidden layer (k + 1).
            score = scores[k + 1]
            if self.mapping == "targeted":
                probs = scores_to_targeted_probs(score, self.gamma, self.drop,
                                                 direction=self.direction)
            else:
                alpha_k = self._resolve(self.alpha, k, n_hidden, "alpha")
                beta_k = self._resolve(self.beta, k, n_hidden, "beta")
                probs = scores_to_probs(score, self.p_base, alpha_k, beta_k,
                                        standardize=self.standardize)
            dmod.set_probs(probs)
            if self.record:
                curv_k, mag_k = diag[0][k + 1], diag[1][k + 1]
                stats = _layer_stats(score, probs)
                stats["pearson_curv_mag"] = _pearson(curv_k, mag_k)
                # Raw per-neuron arrays so curvature-vs-magnitude scatter plots can
                # be drawn at each checkpoint (curv = Forman kappa, mag = L2 norm).
                stats["curv"] = curv_k.cpu().tolist()
                stats["mag"] = mag_k.cpu().tolist()
                layer_log.append(stats)

        if self.record:
            self.log.append({"epoch": epoch, "layers": layer_log})


def make_dropout_prob_hook(
    p_base: float,
    alpha: Union[float, Sequence[float]],
    beta: Union[float, Sequence[float]],
    warmup_epochs: int,
    recompute_every: int,
    standardize: bool = False,
    source: ScoreFn = per_neuron_curvature,
    record: bool = True,
    mapping: str = "sigmoid",
    gamma: float = 0.5,
    drop: float = 0.5,
    direction: str = "low",
) -> DropoutProbHook:
    """Build a DropoutProbHook. See DropoutProbHook for the arguments."""
    return DropoutProbHook(p_base, alpha, beta, warmup_epochs, recompute_every,
                           standardize=standardize, source=source, record=record,
                           mapping=mapping, gamma=gamma, drop=drop, direction=direction)
