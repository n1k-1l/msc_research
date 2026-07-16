"""
Iterative hard pruning: does curvature diverge from magnitude under NON-uniform
connectivity?

The dense-TD result is a mechanistic null -- on a fully connected layer every
neuron has the same degree, so Forman's topological term is constant and curvature
collapses to ~a function of weight magnitude. The prediction that falls out of that
mechanism: the two scores decouple only once connectivity stops being uniform. This
module tests it directly with iterative magnitude-pruning-style rounds (train ->
prune a fraction -> fine-tune -> repeat), comparing three edge/unit scores.

Granularity is the whole point:
  - "edge" (unstructured): remove individual weights. Different neurons keep
    different numbers of edges => degree becomes non-uniform => Forman gains genuine
    geometric content. THIS is the divergence test.
  - "unit" (structured): remove whole neurons (row + column). The survivors stay
    fully interconnected (complete bipartite) => degree stays uniform => curvature
    should still track magnitude. The confirming control.

Three arms differ only in the score used to rank what to prune:
  magnitude - |w| (edge) or L2 norm (unit): the standard unstructured-pruning floor.
  curvature - Forman F(e) (edge) or kappa(v) (unit): prune the most positive
              (redundant) first, keep negative (bottleneck).
  random    - control.

All three prune the SAME dense net (deep-copied per criterion), so the comparison is
unconfounded: identical starting weights, only the criterion differs. Each round logs
test/val accuracy AND the correlation between the curvature and magnitude orderings
over the survivors -- the headline diagnostic for whether the coupling actually
breaks as the graph sparsifies.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import itertools

import numpy as np
import torch
import torch.nn as nn

from .curvature import (forman_dc_edges_masked, forman_edges_masked,
                        per_neuron_curvature_masked, weight_magnitude)
from .train import evaluate

CRITERIA = ("magnitude", "curvature", "random")

# Criteria that need per-neuron activation vectors (data-dependent scoring).
_ACT_CRITERIA = ("forman_act",)
# Criteria that call the (heavier) Ollivier-Ricci solver (library implementation).
_OLLIVIER_CRITERIA = ("ollivier", "ollivier_topo")


@torch.no_grad()
def make_hetero_masks(linears, density: float, sigma: float,
                      generator: torch.Generator):
    """Fixed sparse masks with HEAVY-TAILED degree heterogeneity at a given density.

    Disentangles sparsity from degree heterogeneity: ER masks (make_sparse_masks)
    concentrate -- Binomial degrees fluctuate ~ sqrt((1-d)/(n d)) -- so curvature
    stays a magnitude proxy at ANY uniform-random sparsity. Here each vertex gets a
    lognormal propensity x = exp(sigma*Z) and edge (u,v) is kept with probability
    proportional to x_u * x_v (clipped at 1, rescaled to hit the target density), so
    sigma directly controls the degree coefficient of variation at FIXED density.
    sigma = 0 reduces to ER.
    """
    masks = []
    for lin in linears:
        n_tgt, n_src = lin.weight.shape
        x = torch.exp(sigma * torch.randn(n_src, generator=generator))
        y = torch.exp(sigma * torch.randn(n_tgt, generator=generator))
        P = y[:, None] * x[None, :]
        c = density / P.mean()
        for _ in range(4):                       # fix clipping-induced density loss
            realized = (c * P).clamp(max=1.0).mean()
            c = c * density / realized.clamp_min(1e-9)
        P = (c * P).clamp(max=1.0)
        masks.append(torch.rand(lin.weight.shape, generator=generator) < P)
    return masks


@torch.no_grad()
def make_sparse_masks(linears, density: float, generator: torch.Generator):
    """Fixed random (Erdos-Renyi) edge mask per Linear at the given keep-density.

    Gives a sparse-by-construction, non-uniform-degree topology chosen independently
    of weight magnitude -- the honest setting for comparing curvature vs magnitude
    (and where Ollivier-Ricci becomes tractable). Held fixed through training.
    """
    masks = []
    for lin in linears:
        r = torch.rand(lin.weight.shape, generator=generator)
        masks.append(r < density)
    return masks


# --------------------------------------------------------------------------- #
# correlation helpers (over survivors) -- the decoupling diagnostic
# --------------------------------------------------------------------------- #
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    return _pearson(rx, ry)


# --------------------------------------------------------------------------- #
# fine-tuning with pruned edges held at zero
# --------------------------------------------------------------------------- #
def _register_grad_masks(linears: List[nn.Linear], masks: List[torch.Tensor]) -> None:
    """Zero the gradient of pruned weights so the optimiser never revives them.

    masks are mutated in place across rounds, and the hook closes over the tensor
    object, so a single registration tracks every subsequent prune.
    """
    for lin, m in zip(linears, masks):
        lin.weight.register_hook(lambda g, m=m: g * m)


def _finetune(model: nn.Module, train_loader, device, epochs: int, lr: float,
              linears: List[nn.Linear], masks: List[torch.Tensor]) -> None:
    """Short fine-tune; pruned edges are re-zeroed after every step (belt-and-braces
    on top of the gradient mask, since Adam momentum could otherwise nudge them)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for lin, m in zip(linears, masks):
                    lin.weight.mul_(m)


# --------------------------------------------------------------------------- #
# edge (unstructured) pruning
# --------------------------------------------------------------------------- #
def _act_gate(A: torch.Tensor, act: Tuple[torch.Tensor, torch.Tensor],
              eps: float = 1e-6) -> torch.Tensor:
    """Gate edge strengths by endpoint activations (data-dependence): an edge touching
    a rarely-firing neuron looks weaker. act = (a_src[n_src], a_tgt[n_tgt])."""
    a_src, a_tgt = act
    g = torch.sqrt(a_src[None, :].clamp_min(eps) * a_tgt[:, None].clamp_min(eps))
    return A * g


def _edge_scores(W: torch.Tensor, mask: torch.Tensor, criterion: str,
                 act: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                 ) -> Tuple[torch.Tensor, bool]:
    """Per-edge prunability score and whether the most prunable is the LARGEST.

    magnitude:     smallest |w| prunable (largest=False).
    curvature/forman: most positive Forman F(e) prunable -- redundant (largest=True).
    forman_dc:     DEGREE-CORRECTED Forman (neighbour mean, not sum) -- the causal
                   ablation for the collapse mechanism; most positive prunable.
    forman_act:    Forman on activation-gated weights (data-dependent).
    ollivier:      Ollivier-Ricci, weighted 1/|w| metric (library implementation);
                   most positive prunable (largest=True).
    ollivier_topo: Ollivier-Ricci on the unweighted graph -- pure topology.
    random:        control.
    """
    A = W.abs()
    if criterion == "magnitude":
        return A, False
    if criterion == "random":
        return torch.rand_like(A), False
    if criterion in ("curvature", "forman"):
        return forman_edges_masked(A, mask), True
    if criterion == "forman_dc":
        return forman_dc_edges_masked(A, mask), True
    if criterion == "forman_act":
        return forman_edges_masked(_act_gate(A, act), mask), True
    if criterion in _OLLIVIER_CRITERIA:
        from .ollivier import ollivier_edges_masked
        if criterion == "ollivier_topo":
            # Unweighted metric: isolates graph geometry from weight magnitude (a
            # weighted 1/|w| metric makes ORC a near-perfect magnitude proxy).
            return ollivier_edges_masked(torch.ones_like(W), mask), True
        return ollivier_edges_masked(W, mask), True
    raise ValueError(f"unknown criterion {criterion!r}")


@torch.no_grad()
def _prune_edges_to(linears, masks, criterion: str, target: float,
                    acts: Optional[List[torch.Tensor]] = None) -> List[Dict]:
    """Prune each layer's surviving edges down to cumulative sparsity `target`
    (per-layer, so no layer can be wiped out). `acts` (per-layer activation vectors,
    length n_layers+1) is required for the data-dependent criteria.

    Returns per-layer FORENSICS on what this step killed -- the mechanism autopsy:
    where the pruned edges sat in the |w| ranking, the degree of their endpoints
    relative to the surviving population, and how many vertices the step fully
    disconnected. The collapse mechanism predicts curvature preferentially prunes
    low-degree-endpoint edges regardless of |w| (few neighbours => few negative
    terms => F more positive), orphaning vertices; magnitude should show neither.
    """
    forensics: List[Dict] = []
    for k, (lin, m) in enumerate(zip(linears, masks)):
        total = m.numel()
        n_new = int(round(target * total)) - int((~m).sum())
        if n_new <= 0:
            forensics.append({"layer": k, "n_pruned": 0})
            continue                            # skip scoring (esp. costly Ollivier) on no-op
        act = (acts[k], acts[k + 1]) if acts is not None else None
        scores, largest = _edge_scores(lin.weight, m, criterion, act=act)
        surv = m.view(-1).nonzero(as_tuple=True)[0]
        n_take = min(n_new, surv.numel())
        sel = torch.topk(scores.view(-1)[surv], n_take, largest=largest).indices
        prune_idx = surv[sel]

        # --- forensics, measured on the pre-prune state -----------------------
        n_src = m.shape[1]
        A = lin.weight.abs().view(-1)
        deg_src, deg_tgt = m.sum(dim=0), m.sum(dim=1)
        rows = torch.div(prune_idx, n_src, rounding_mode="floor")
        cols = prune_idx % n_src
        pruned_deg = (deg_tgt[rows] + deg_src[cols]).float()
        surv_rows = torch.div(surv, n_src, rounding_mode="floor")
        surv_deg = (deg_tgt[surv_rows] + deg_src[surv % n_src]).float()
        # percentile of each pruned edge's |w| among pre-prune survivors
        mag_sorted = A[surv].sort().values
        pct = torch.searchsorted(mag_sorted, A[prune_idx].contiguous()).float() \
            / max(surv.numel(), 1)
        alive_src, alive_tgt = deg_src > 0, deg_tgt > 0
        # ----------------------------------------------------------------------

        m.view(-1)[prune_idx] = False
        lin.weight.view(-1)[prune_idx] = 0.0
        forensics.append({
            "layer": k,
            "n_pruned": int(n_take),
            "pruned_deg_mean": float(pruned_deg.mean()),
            "pruned_deg_median": float(pruned_deg.median()),
            "surv_deg_mean": float(surv_deg.mean()),
            "pruned_mag_pctile_mean": float(pct.mean()),
            "pruned_mag_pctile_median": float(pct.median()),
            "new_disconnected_src": int((alive_src & (m.sum(dim=0) == 0)).sum()),
            "new_disconnected_tgt": int((alive_tgt & (m.sum(dim=1) == 0)).sum()),
        })
    return forensics


# --------------------------------------------------------------------------- #
# unit (structured) pruning -- confirming control
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _unit_scores(linears, masks, criterion: str) -> List[Tuple[torch.Tensor, bool]]:
    """Per-hidden-neuron prunability score for each hidden layer.

    Hidden layer h sits between linears[h-1] (incoming, rows) and linears[h]
    (outgoing, cols); a neuron's incident edges span both.
    """
    n_hidden = len(linears) - 1
    weights = [lin.weight for lin in linears]
    if criterion == "magnitude":
        mags = weight_magnitude(weights)                 # per-layer L2 norm
        return [(mags[h], False) for h in range(1, n_hidden + 1)]
    if criterion == "curvature":
        kappa = per_neuron_curvature_masked(weights, masks)
        return [(kappa[h], True) for h in range(1, n_hidden + 1)]
    if criterion == "random":
        return [(torch.rand(weights[h - 1].shape[0], device=weights[0].device),
                 False) for h in range(1, n_hidden + 1)]
    raise ValueError(f"unknown criterion {criterion!r}")


@torch.no_grad()
def _unit_alive(masks, h: int) -> torch.Tensor:
    """Boolean over hidden-layer-h neurons: a unit is alive while any incident edge
    survives (its incoming row in masks[h-1] or outgoing col in masks[h])."""
    return masks[h - 1].any(dim=1) | masks[h].any(dim=0)


@torch.no_grad()
def _prune_units_to(linears, masks, criterion: str, target: float) -> None:
    scored = _unit_scores(linears, masks, criterion)
    n_hidden = len(linears) - 1
    for h in range(1, n_hidden + 1):
        scores, largest = scored[h - 1]
        alive = _unit_alive(masks, h)
        total = scores.numel()
        n_new = int(round(target * total)) - int((~alive).sum())
        if n_new <= 0:
            continue
        surv = alive.nonzero(as_tuple=True)[0]
        k = min(n_new, surv.numel())
        sel = torch.topk(scores[surv], k, largest=largest).indices
        idx = surv[sel]
        masks[h - 1][idx, :] = False          # kill incoming row(s)
        linears[h - 1].weight[idx, :] = 0.0
        masks[h][:, idx] = False              # kill outgoing col(s)
        linears[h].weight[:, idx] = 0.0


# --------------------------------------------------------------------------- #
# per-round measurement
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _measure(linears, masks, model, val_loader, test_loader, device,
             r: int, target: float, granularity: str) -> Dict:
    """Accuracy + the curvature-vs-magnitude decoupling diagnostics for one round.

    Correlations are per hidden layer over the survivors: for "edge", between edge
    |w| and edge F(e) over surviving edges; for "unit", between per-neuron L2 norm
    and kappa over surviving units. degree_cv is the coefficient of variation of
    vertex degree -- the check that connectivity actually became non-uniform.
    """
    total_edges = sum(m.numel() for m in masks)
    pruned_edges = sum(int((~m).sum()) for m in masks)
    pearson, spearman, degree_cv = [], [], []
    pearson_dc, spearman_dc = [], []
    weights = [lin.weight for lin in linears]

    for k, (lin, m) in enumerate(zip(linears, masks)):
        A = lin.weight.abs()
        F = forman_edges_masked(A, m)
        Fdc = forman_dc_edges_masked(A, m)
        surv = m
        if surv.sum() >= 2:
            mag = A[surv].cpu().numpy()
            cur = F[surv].cpu().numpy()
            pearson.append(_pearson(mag, cur))
            spearman.append(_spearman(mag, cur))
            # degree-corrected coupling over edges where F_dc is defined
            # (bridges are -inf by construction and drop out of the correlation)
            fin = surv & torch.isfinite(Fdc)
            if fin.sum() >= 2:
                pearson_dc.append(_pearson(A[fin].cpu().numpy(),
                                           Fdc[fin].cpu().numpy()))
                spearman_dc.append(_spearman(A[fin].cpu().numpy(),
                                             Fdc[fin].cpu().numpy()))
            else:
                pearson_dc.append(float("nan"))
                spearman_dc.append(float("nan"))
        else:
            pearson.append(float("nan"))
            spearman.append(float("nan"))
            pearson_dc.append(float("nan"))
            spearman_dc.append(float("nan"))
        deg = torch.cat([m.sum(dim=0).float(), m.sum(dim=1).float()])
        deg = deg[deg > 0]
        degree_cv.append(float(deg.std(unbiased=False) / deg.mean())
                         if deg.numel() and deg.mean() > 0 else float("nan"))

    # cumulative dead hidden units (no surviving incident edge on either side)
    n_hidden = len(linears) - 1
    dead_units = [int((~_unit_alive(masks, h)).sum()) for h in range(1, n_hidden + 1)]

    return {
        "round": r,
        "target_sparsity": float(target),
        "sparsity": pruned_edges / total_edges,   # achieved edge sparsity
        "val_acc": float(evaluate(model, val_loader, device)),
        "test_acc": float(evaluate(model, test_loader, device)),
        "pearson_curv_mag": pearson,               # per hidden-adjacent layer
        "spearman_curv_mag": spearman,
        "pearson_dc_mag": pearson_dc,              # degree-corrected Forman vs |w|
        "spearman_dc_mag": spearman_dc,
        "degree_cv": degree_cv,
        "dead_units": dead_units,                  # cumulative, per hidden layer
    }


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def iterative_prune_one(model: nn.Module, criterion: str, loaders, device,
                        schedule: Sequence[float], finetune_epochs: int, lr: float,
                        granularity: str = "edge",
                        init_masks: Optional[Sequence[torch.Tensor]] = None,
                        calib_batches: int = 2) -> List[Dict]:
    """Prune one model by `criterion` over `schedule`, fine-tuning between rounds.

    init_masks: starting edge masks. None => dense (all-ones). For the sparse-by-
    construction study, pass the fixed sparse topology so pruning starts from it and
    Forman/Ollivier see genuine non-uniform degree (not self-imposed).
    calib_batches: how many train batches to average activations over (data-dependent
    criteria only).
    """
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    linears = model.linear_layers()
    if init_masks is not None:
        masks = [m.clone().to(device) for m in init_masks]
    else:
        masks = [torch.ones_like(lin.weight, dtype=torch.bool) for lin in linears]
    _register_grad_masks(linears, masks)
    needs_act = criterion in _ACT_CRITERIA

    records = []
    for r, target in enumerate(schedule):
        acts = None
        if needs_act:
            from .activations import collect_activations
            batches = list(itertools.islice(train_loader, calib_batches))
            acts = collect_activations(model, batches, device)
        forensics = None
        if granularity == "edge":
            forensics = _prune_edges_to(linears, masks, criterion, target, acts=acts)
        elif granularity == "unit":
            _prune_units_to(linears, masks, criterion, target)
        else:
            raise ValueError(f"unknown granularity {granularity!r}")
        _finetune(model, train_loader, device, finetune_epochs, lr, linears, masks)
        rec = _measure(linears, masks, model, val_loader, test_loader,
                       device, r, target, granularity)
        if forensics is not None:
            rec["forensics"] = forensics           # what THIS step killed (per layer)
        records.append(rec)
    return records


def iterative_prune_by_criteria(dense_model: nn.Module, loaders, device,
                                schedule: Sequence[float], finetune_epochs: int,
                                lr: float, granularity: str = "edge",
                                criteria: Sequence[str] = CRITERIA,
                                init_masks: Optional[Sequence[torch.Tensor]] = None,
                                calib_batches: int = 2) -> Dict[str, List[Dict]]:
    """Prune the SAME trained net by each criterion (identical start weights + topology),
    return {criterion: [per-round record]}. Mirrors prune_by_criteria. For the sparse
    study pass init_masks (the fixed topology) and the full criteria set."""
    return {c: iterative_prune_one(copy.deepcopy(dense_model), c, loaders, device,
                                   schedule, finetune_epochs, lr, granularity,
                                   init_masks=init_masks, calib_batches=calib_batches)
            for c in criteria}
