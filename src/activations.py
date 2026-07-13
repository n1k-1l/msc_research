"""
Per-neuron activation statistics over a calibration batch (for data-dependent pruning).

The 2601.16366 result shows the signal that lets graph curvature beat magnitude is
*activation/data-dependence*, not geometry per se: a neuron's edges matter in
proportion to how strongly that neuron actually fires on real inputs. This module
collects, for a small seeded calibration set, the mean |activation| of every neuron,
aligned with the curvature graph's per-layer `sizes` list so it can gate edge weights
in the Forman/Ollivier scorers.

Returned list is indexed by layer: entry 0 = input features (mean |x|), entries
1..L-1 = hidden neurons (mean post-activation magnitude), entry L = outputs
(mean |logit|). L = number of Linear layers.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple
import torch
import torch.nn as nn

ACT_TYPES = ("ReLU", "GELU")


@torch.no_grad()
def collect_activations(model: nn.Module, batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                        device: torch.device) -> List[torch.Tensor]:
    """Mean |activation| per neuron for every layer, averaged over `batches`.

    `batches` is any iterable of (x, y); pass a handful of calibration batches. Hooks
    the model's activation modules (one per hidden layer) plus its flattened input and
    final output, so the result aligns one-to-one with the weight graph's layer sizes.
    """
    model.eval()
    linears = model.linear_layers()
    L = len(linears)
    sizes = [linears[0].weight.shape[1]] + [lin.weight.shape[0] for lin in linears]
    sums = [torch.zeros(n, device=device) for n in sizes]
    count = 0

    act_mods = [m for m in model.net if m.__class__.__name__ in ACT_TYPES]
    assert len(act_mods) == L - 1, (
        f"expected one activation per hidden layer ({L - 1}); got {len(act_mods)}")

    store: dict = {}
    handles = [m.register_forward_hook(
        lambda mod, inp, out, k=k: store.__setitem__(k, out.detach()))
        for k, m in enumerate(act_mods, start=1)]
    try:
        for x, _ in batches:
            x = x.to(device, non_blocking=True)
            xf = x.flatten(start_dim=1)
            out = model(x)                       # fires the hidden hooks
            sums[0] += xf.abs().sum(dim=0)
            for k in range(1, L):
                sums[k] += store[k].abs().sum(dim=0)
            sums[L] += out.abs().sum(dim=0)
            count += xf.shape[0]
    finally:
        for h in handles:
            h.remove()
    return [s / max(count, 1) for s in sums]
