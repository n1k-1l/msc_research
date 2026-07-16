"""
Channel-graph pruning for CNNs (extends the curvature-vs-magnitude study to convs).

A conv layer's weight tensor is (out_ch, in_ch, kH, kW) with weight sharing, so it is
not a bipartite weight graph like a Linear. The standard, tractable abstraction is the
**channel graph**: nodes = channels, and the edge between in-channel j and out-channel
i carries the L2 norm of the kernel slice W[i, j, :, :]. That collapses each conv to an
(out_ch x in_ch) matrix -- exactly the object the existing curvature scorers eat
(forman/ollivier via iterative_prune._edge_scores). A Linear is its own
weight matrix. Pruning a channel-graph edge zeros the whole kernel slice (or the single
weight, for a Linear).

This driver mirrors iterative_prune but over a mixed list of Conv/Linear "graph views":
score channel-edges by each criterion, prune the most-prunable, fine-tune with the
pruned edges frozen at zero, and measure accuracy vs sparsity. Per-channel activations
(mean |.| over batch + spatial) feed the data-dependent criteria.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import itertools

import numpy as np
import torch
import torch.nn as nn

from .curvature import per_neuron_curvature_masked, weight_magnitude
from .iterative_prune import _edge_scores, _pearson, _spearman, _ACT_CRITERIA
from .train import evaluate


# --------------------------------------------------------------------------- #
# graph views: turn a Conv2d / Linear into a 2D (out, in) weight graph
# --------------------------------------------------------------------------- #
class _View:
    module: nn.Module

    @property
    def matrix(self) -> torch.Tensor:      # (out, in) edge-weight matrix for scoring
        raise NotImplementedError

    def apply_mask(self, mask: torch.Tensor) -> None:   # zero real weights of pruned edges
        raise NotImplementedError

    def register_grad_mask(self, mask: torch.Tensor) -> None:
        raise NotImplementedError


class ConvView(_View):
    """Conv2d as a channel graph; edge (out i, in j) weight = ||W[i,j,:,:]||_2."""
    def __init__(self, conv: nn.Conv2d):
        self.module = conv

    @property
    def matrix(self) -> torch.Tensor:
        return self.module.weight.flatten(2).norm(dim=2)     # (out, in)

    def apply_mask(self, mask: torch.Tensor) -> None:
        with torch.no_grad():
            self.module.weight.mul_(mask[:, :, None, None])

    def register_grad_mask(self, mask: torch.Tensor) -> None:
        self.module.weight.register_hook(lambda g, m=mask: g * m[:, :, None, None])


class LinearView(_View):
    """Linear as its own weight graph (edge (out i, in j) weight = W[i,j])."""
    def __init__(self, lin: nn.Linear):
        self.module = lin

    @property
    def matrix(self) -> torch.Tensor:
        return self.module.weight

    def apply_mask(self, mask: torch.Tensor) -> None:
        with torch.no_grad():
            self.module.weight.mul_(mask)

    def register_grad_mask(self, mask: torch.Tensor) -> None:
        self.module.weight.register_hook(lambda g, m=mask: g * m)


def graph_views(model: nn.Module, include_linear: bool = True) -> List[_View]:
    """Conv (channel-graph) and optionally Linear views, in module order."""
    views: List[_View] = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            views.append(ConvView(m))
        elif include_linear and isinstance(m, nn.Linear):
            views.append(LinearView(m))
    return views


# --------------------------------------------------------------------------- #
# per-channel activations (data-dependent criteria)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_channel_activations(model, views, batches, device
                                ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Mean |activation| per channel at each view's input and output, over `batches`.

    Returns [(a_in, a_out)] aligned with `views`: a_in has length = view's in-channels
    (source nodes), a_out = out-channels (target nodes). Conv maps are reduced over
    batch + spatial; Linear over batch. Uses module output (pre-activation) magnitude.
    """
    model.eval()
    sums = [[None, None] for _ in views]
    count = 0

    def mk(k):
        def hook(mod, inp, out, k=k):
            x, y = inp[0].detach(), out.detach()
            ain = x.abs().mean(dim=(0, 2, 3)) if x.dim() == 4 else x.abs().mean(dim=0)
            aout = y.abs().mean(dim=(0, 2, 3)) if y.dim() == 4 else y.abs().mean(dim=0)
            s = sums[k]
            s[0] = ain if s[0] is None else s[0] + ain
            s[1] = aout if s[1] is None else s[1] + aout
        return hook

    handles = [v.module.register_forward_hook(mk(k)) for k, v in enumerate(views)]
    try:
        for x, _ in batches:
            model(x.to(device, non_blocking=True))
            count += 1
    finally:
        for h in handles:
            h.remove()
    return [(s[0] / count, s[1] / count) for s in sums]


# --------------------------------------------------------------------------- #
# pruning
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _prune_views_to(views, masks, criterion, target, acts=None) -> None:
    for k, (v, m) in enumerate(zip(views, masks)):
        total = m.numel()
        n_new = int(round(target * total)) - int((~m).sum())
        if n_new <= 0:
            continue
        act = acts[k] if acts is not None else None
        scores, largest = _edge_scores(v.matrix, m, criterion, act=act)
        surv = m.view(-1).nonzero(as_tuple=True)[0]
        n_take = min(n_new, surv.numel())
        sel = torch.topk(scores.view(-1)[surv], n_take, largest=largest).indices
        prune_idx = surv[sel]
        m.view(-1)[prune_idx] = False
        v.apply_mask(m)


def _finetune_views(model, views, masks, train_loader, device, epochs, lr) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
            with torch.no_grad():
                for v, m in zip(views, masks):
                    v.apply_mask(m)


@torch.no_grad()
def _measure(model, views, masks, val_loader, test_loader, device, r, target) -> Dict:
    total = sum(m.numel() for m in masks)
    pruned = sum(int((~m).sum()) for m in masks)
    return {
        "round": r,
        "target_sparsity": float(target),
        "sparsity": pruned / total,
        "val_acc": float(evaluate(model, val_loader, device)),
        "test_acc": float(evaluate(model, test_loader, device)),
    }


def cnn_prune_one(model, criterion, loaders, device, schedule, finetune_epochs, lr,
                  include_linear=True, calib_batches=2) -> List[Dict]:
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    views = graph_views(model, include_linear=include_linear)
    masks = [torch.ones_like(v.matrix, dtype=torch.bool) for v in views]
    for v, m in zip(views, masks):
        v.register_grad_mask(m)
    needs_act = criterion in _ACT_CRITERIA

    records = []
    for r, target in enumerate(schedule):
        acts = None
        if needs_act:
            batches = list(itertools.islice(train_loader, calib_batches))
            acts = collect_channel_activations(model, views, batches, device)
        _prune_views_to(views, masks, criterion, target, acts=acts)
        _finetune_views(model, views, masks, train_loader, device, finetune_epochs, lr)
        records.append(_measure(model, views, masks, val_loader, test_loader,
                                device, r, target))
    return records


def cnn_prune_by_criteria(model, loaders, device, schedule, finetune_epochs, lr,
                          criteria, include_linear=True, calib_batches=2
                          ) -> Dict[str, List[Dict]]:
    """Prune the SAME trained CNN by each criterion (identical start weights)."""
    return {c: cnn_prune_one(copy.deepcopy(model), c, loaders, device, schedule,
                             finetune_epochs, lr, include_linear=include_linear,
                             calib_batches=calib_batches)
            for c in criteria}


# --------------------------------------------------------------------------- #
# whole-FILTER (structured) pruning -- the CNN analogue of unit pruning (E13)
# --------------------------------------------------------------------------- #
# Removing a whole filter (output channel) keeps the channel graph complete
# bipartite among the survivors, so degree stays uniform: the mechanism predicts
# Forman curvature remains a magnitude proxy and no collapse occurs -- the
# structured counterpart to iterative_prune's unit control. The one non-trivial
# bit is the flatten boundary: killing conv-filter i must also kill its spatial
# block of columns in the first Linear (feature order after Flatten is
# (channel, h, w), so filter i owns columns [i*B, (i+1)*B), B = fc_in / c_out).


@torch.no_grad()
def _reduced_chain(views, masks) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Node-graph matrices with CHAINED dims: mats[k] is (out_k, nodes_in_k) where
    nodes_in_k equals out_{k-1}. Flatten boundaries (in-dim = prev_out * block) are
    L2-reduced over the spatial block; masks reduce by `any`. On this chain the
    per-neuron curvature / magnitude functions apply unchanged."""
    mats, nmasks, prev_out = [], [], None
    for v, m in zip(views, masks):
        M = v.matrix * m                      # zeroed entries stay zero
        mm = m
        if prev_out is not None and M.shape[1] != prev_out:
            block = M.shape[1] // prev_out
            assert block * prev_out == M.shape[1], "flatten boundary not divisible"
            M = M.view(M.shape[0], prev_out, block).norm(dim=2)
            mm = m.view(m.shape[0], prev_out, block).any(dim=2)
        mats.append(M)
        nmasks.append(mm)
        prev_out = v.matrix.shape[0]
    return mats, nmasks


@torch.no_grad()
def _filter_scores(mats, nmasks, criterion: str) -> Tuple[List[torch.Tensor], bool]:
    """Per-node scores over the reduced chain (lists indexed input..output).
    magnitude: two-sided L2 of incident node-edges, prune lowest.
    curvature: per-node Forman kappa on the (masked) node graph, prune highest.
    random: control."""
    if criterion == "magnitude":
        return weight_magnitude(mats), False
    if criterion in ("curvature", "forman"):
        return per_neuron_curvature_masked(mats, nmasks), True
    if criterion == "random":
        sizes = [mats[0].shape[1]] + [M.shape[0] for M in mats]
        return [torch.rand(n) for n in sizes], False
    raise ValueError(f"unknown criterion {criterion!r}")


@torch.no_grad()
def _kill_node(views, masks, h: int, j: int) -> None:
    """Remove hidden node j of node-layer h: its incoming row in view h-1 and its
    outgoing column(s) in view h (spatial block at a flatten boundary)."""
    masks[h - 1][j, :] = False
    v_out, m_out = views[h], masks[h]
    n_in, prev_out = m_out.shape[1], views[h - 1].matrix.shape[0]
    if n_in != prev_out:                       # flatten boundary
        block = n_in // prev_out
        m_out[:, j * block:(j + 1) * block] = False
    else:
        m_out[:, j] = False


@torch.no_grad()
def _prune_filters_to(views, masks, criterion: str, target: float) -> None:
    """Prune each hidden node layer down to cumulative fraction `target` removed."""
    mats, nmasks = _reduced_chain(views, masks)
    scores, largest = _filter_scores(mats, nmasks, criterion)
    for h in range(1, len(views)):             # hidden node layers only
        alive = nmasks[h - 1].any(dim=1) | nmasks[h].any(dim=0) \
            if h < len(views) else nmasks[h - 1].any(dim=1)
        total = alive.numel()
        n_new = int(round(target * total)) - int((~alive).sum())
        if n_new <= 0:
            continue
        surv = alive.nonzero(as_tuple=True)[0]
        n_take = min(n_new, surv.numel())
        sel = torch.topk(scores[h][surv], n_take, largest=largest).indices
        for j in surv[sel].tolist():
            _kill_node(views, masks, h, j)
    for v, m in zip(views, masks):
        v.apply_mask(m)


@torch.no_grad()
def _measure_filters(model, views, masks, val_loader, test_loader, device,
                     r: int, target: float) -> Dict:
    """Accuracy + node-level coupling diagnostics (schema-compatible with
    iterative_prune records so plot_iterative_prune works unchanged)."""
    mats, nmasks = _reduced_chain(views, masks)
    kappa = per_neuron_curvature_masked(mats, nmasks)
    mag = weight_magnitude(mats)
    pear, spear, deg_cv = [], [], []
    for h in range(1, len(views)):             # hidden node layers
        alive = nmasks[h - 1].any(dim=1)
        if int(alive.sum()) >= 2:
            k = kappa[h][alive].cpu().numpy()
            m_ = mag[h][alive].cpu().numpy()
            pear.append(_pearson(m_, k))
            spear.append(_spearman(m_, k))
        else:
            pear.append(float("nan")); spear.append(float("nan"))
        deg = torch.cat([nmasks[h - 1].sum(dim=1).float()[alive],
                         nmasks[h].sum(dim=0).float()[alive]]) \
            if h < len(nmasks) else nmasks[h - 1].sum(dim=1).float()[alive]
        deg = deg[deg > 0]
        deg_cv.append(float(deg.std(unbiased=False) / deg.mean())
                      if deg.numel() and deg.mean() > 0 else float("nan"))
    total = sum(m.numel() for m in masks)
    pruned = sum(int((~m).sum()) for m in masks)
    return {
        "round": r, "target_sparsity": float(target),
        "sparsity": pruned / total,            # weight-level sparsity (plot axis)
        "val_acc": float(evaluate(model, val_loader, device)),
        "test_acc": float(evaluate(model, test_loader, device)),
        "pearson_curv_mag": pear, "spearman_curv_mag": spear,
        "degree_cv": deg_cv,
    }


def filter_prune_one(model, criterion, loaders, device, schedule,
                     finetune_epochs, lr) -> List[Dict]:
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    views = graph_views(model, include_linear=True)
    masks = [torch.ones_like(v.matrix, dtype=torch.bool) for v in views]
    for v, m in zip(views, masks):
        v.register_grad_mask(m)
    records = []
    for r, target in enumerate(schedule):
        _prune_filters_to(views, masks, criterion, target)
        _finetune_views(model, views, masks, train_loader, device,
                        finetune_epochs, lr)
        records.append(_measure_filters(model, views, masks, val_loader,
                                        test_loader, device, r, target))
    return records


def filter_prune_by_criteria(model, loaders, device, schedule, finetune_epochs,
                             lr, criteria) -> Dict[str, List[Dict]]:
    """Whole-filter pruning of the SAME trained CNN by each criterion."""
    return {c: filter_prune_one(copy.deepcopy(model), c, loaders, device,
                                schedule, finetune_epochs, lr)
            for c in criteria}
