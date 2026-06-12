"""
Pruning-robustness evaluation (accuracy-vs-sparsity).

This is the headline metric for the curvature-vs-magnitude Targeted-Dropout
comparison -- and the *right* metric for Targeted Dropout (the dropout study used
generalisation, which is the wrong one). After training, units are ranked by the
same per-neuron score and direction used during training, the most-prunable
fraction is removed at a range of sparsity levels, and test accuracy is measured.
The resulting curve says how gracefully the network degrades as it is pruned by
its own training criterion.

A unit is pruned by zeroing its outgoing weights (its column in the next Linear),
removing its contribution to the rest of the network. Each sparsity level prunes a
fresh deep copy so the original model is untouched.
"""
from __future__ import annotations
from typing import List, Dict, Sequence
import copy
import torch
import torch.nn as nn

from .curvature import ScoreFn, get_source
from .train import evaluate

# Criterion set for unconfounded cross-pruning: prune the SAME trained model by
# each and compare. (label, score source, prunable direction). Curvature prunes
# the highest (positive = redundant), magnitude the lowest (small |w|), random is
# the control. Used to compare criteria on a network that was NOT trained to
# favour any of them (e.g. a uniform-dropout / no-dropout net).
CROSS_PRUNE_CRITERIA = [
    ("forman", "forman", "high"),
    ("magnitude", "magnitude", "low"),
    ("random", "random", "low"),
]


@torch.no_grad()
def prune_and_evaluate(
    model: nn.Module,
    source: ScoreFn,
    direction: str,
    sparsities: Sequence[float],
    loader,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Test accuracy at each sparsity when pruning hidden units by `source`.

    direction: "low" prunes the lowest-scoring units (weight magnitude: small =
    prunable); "high" prunes the highest (Forman curvature: positive = redundant).
    The ranking is computed once on the trained weights and reused at every level,
    so the levels are nested (a unit pruned at sparsity s is also pruned at s' > s).
    """
    assert direction in ("low", "high"), f"direction must be low/high, got {direction!r}"
    linears = model.linear_layers()
    n_hidden = len(linears) - 1
    base_scores = source([lin.weight for lin in linears])

    curve: List[Dict[str, float]] = []
    for s in sparsities:
        pruned = copy.deepcopy(model).to(device)
        plinears = pruned.linear_layers()
        for k in range(n_hidden):
            score = base_scores[k + 1]          # hidden layer k+1 = output of linears[k]
            n_prune = int(round(s * score.numel()))
            if n_prune <= 0:
                continue
            idx = torch.topk(score, n_prune, largest=(direction == "high")).indices
            plinears[k + 1].weight[:, idx] = 0.0   # zero the unit's outgoing column
        curve.append({"sparsity": float(s),
                      "test_acc": float(evaluate(pruned, loader, device))})
    return curve


@torch.no_grad()
def prune_by_criteria(
    model: nn.Module,
    sparsities: Sequence[float],
    loader,
    device: torch.device,
    criteria=CROSS_PRUNE_CRITERIA,
) -> Dict[str, List[Dict[str, float]]]:
    """Prune the SAME trained model by each criterion; return {label: prune_curve}.

    The point is an unconfounded criterion comparison: applied to one fixed network
    (ideally one not trained to favour any criterion), does Forman curvature prune
    more gracefully than weight magnitude? Each criterion ranks the trained weights
    independently, so the curves are directly comparable on identical weights.
    """
    return {label: prune_and_evaluate(model, get_source(src), direction,
                                      sparsities, loader, device)
            for label, src, direction in criteria}
