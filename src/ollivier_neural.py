"""
Faithful activation-weighted Ollivier-Ricci curvature (the 2601.16366 mechanism).

The naive `ollivier_edges_masked` folds |w| into the edge *metric*, which just
re-encodes magnitude (spearman ~ -0.94 with |w|). The paper's signal instead lives
in the transport *mass distributions*: the mass on a neuron's neighbour is weighted
by how strongly that neighbour actually fires on real data (a "neural neighbour
distribution"). The Wasserstein transport then compares the *actually-used*
neighbourhoods of two neurons -- information orthogonal to weight magnitude.

This module puts the activations where they belong -- in the measures m_x, m_y --
and uses a purely topological ground metric (adjacent = 1 hop, else `far`), so the
only data-dependent ingredient is activation-weighted mass. That is the faithful test
of "does activation-weighted structure carry pruning signal magnitude lacks?", and it
is what GraphRicciCurvature cannot express (its measure is derived from edge weights,
symmetric, so it can't be keyed on external per-node activations).

Only tractable on a sparse graph and intended for one-shot scoring (compute once on
the trained net). Same interface/convention as forman_edges_masked: per-edge tensor
shaped like W, NaN on pruned edges; higher = more redundant (prunable).

    kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y),   d(x,y) = 1 (adjacent)
    m_x(u) proportional to activation(u) over x's neighbours  (idem m_y)
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import ot


@torch.no_grad()
def neural_ollivier_edges_masked(
    W: torch.Tensor,
    mask: torch.Tensor,
    act_src: torch.Tensor,
    act_tgt: torch.Tensor,
    far: float = 3.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Activation-weighted Ollivier-Ricci per surviving edge of one bipartite pair.

    Args:
        W: (n_tgt, n_src) weight matrix; W[i,j] connects source j -> target i.
        mask: bool, same shape; True = edge present.
        act_src: per-source-neuron activation (length n_src).
        act_tgt: per-target-neuron activation (length n_tgt).
        far: ground distance for a non-adjacent target/source pair (adjacent = 1;
             bipartite, so non-adjacent opposite-side nodes are >= 3 hops apart).

    Returns:
        Tensor shaped like W: curvature on surviving edges, NaN on pruned edges.
    """
    F = torch.full_like(W.abs(), float("nan"))
    m = mask.detach().cpu().numpy()
    n_tgt, n_src = m.shape
    a_src = act_src.detach().cpu().numpy().astype(np.float64) + eps
    a_tgt = act_tgt.detach().cpu().numpy().astype(np.float64) + eps

    # neighbour index lists (topology only)
    col_nbrs = [np.nonzero(m[:, j])[0] for j in range(n_src)]   # targets adjacent to source j
    row_nbrs = [np.nonzero(m[i, :])[0] for i in range(n_tgt)]   # sources adjacent to target i

    idx = np.argwhere(m)                     # (E, 2) row-major: [target i, source j]
    out = np.empty(idx.shape[0], dtype=np.float64)
    for e in range(idx.shape[0]):
        i, j = idx[e]
        Tj = col_nbrs[j]                     # support of m_x (source j's neighbours)
        Si = row_nbrs[i]                     # support of m_y (target i's neighbours)
        if Tj.size == 0 or Si.size == 0:
            out[e] = 0.0
            continue
        mx = a_tgt[Tj]; mx = mx / mx.sum()   # activation-weighted neighbour mass
        my = a_src[Si]; my = my / my.sum()
        sub = m[np.ix_(Tj, Si)]              # (|Tj|,|Si|) bool: are these pairs adjacent?
        D = np.where(sub, 1.0, far)
        w1 = ot.emd2(mx, my, D)              # exact Wasserstein-1
        out[e] = 1.0 - w1                    # d(x,y) = 1 hop

    F[mask] = torch.tensor(out, dtype=F.dtype, device=F.device)
    return F
