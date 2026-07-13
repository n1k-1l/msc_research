"""
Ollivier-Ricci curvature per edge on a masked bipartite layer.

Ollivier-Ricci (ORC) is the optimal-transport discretisation of Ricci curvature:
kappa(x,y) = 1 - W1(m_x, m_y) / d(x,y), where m_x is a mass distribution on x's
neighbours and W1 is the earth-mover distance under the graph metric. Unlike Forman
-- which on a weighted graph reduces to F = 2 - sum sqrt(w_e/w_e') and is therefore a
near-function of magnitude + degree -- ORC has no such trivial reduction, so it is the
discretisation the 2601.16366 result uses to beat magnitude.

This is only tractable on a *sparse* graph (dense: ~500k edges x an OT solve each). On
a sparse-by-construction net (~50-80k edges) a full-layer computation is ~2-3s.

`ollivier_edges_masked` matches the `forman_edges_masked` interface exactly (per-edge
tensor shaped like W, NaN on pruned edges), so it drops straight into the pruning
scorer. Edge distance = 1 / (|w| * gate), gate optionally folding endpoint activations
(data-dependence): an edge touching a rarely-firing neuron becomes "longer". Higher
positive curvature = redundant (prunable), as with Forman.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci


@torch.no_grad()
def ollivier_edges_masked(
    W: torch.Tensor,
    mask: torch.Tensor,
    act_src: Optional[torch.Tensor] = None,
    act_tgt: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    method: str = "OTDSinkhornMix",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Ollivier-Ricci curvature of every surviving edge in one bipartite layer pair.

    Args:
        W: (n_tgt, n_src) weight matrix (Linear.weight); W[i,j] connects source j to
           target i. |W| gives edge strength.
        mask: bool, same shape; True = edge present.
        act_src, act_tgt: optional per-neuron activation vectors (source side length
           n_src, target side n_tgt) to gate edge distance (data-dependence). None =
           static ORC.
        alpha: Ollivier idleness (mass kept at the node); 0.5 is the common default.

    Returns:
        Tensor shaped like W: ORC on surviving edges, NaN on pruned edges.
    """
    A = W.abs()
    F = torch.full_like(A, float("nan"))
    idx = mask.nonzero(as_tuple=False)          # (E, 2): [:,0]=tgt row, [:,1]=src col
    if idx.shape[0] == 0:
        return F
    tgt = idx[:, 0]
    src = idx[:, 1]
    w = A[tgt, src].clamp_min(eps)

    gate = torch.ones_like(w)
    if act_src is not None:
        gate = gate * act_src.to(w.device)[src].clamp_min(eps).sqrt()
    if act_tgt is not None:
        gate = gate * act_tgt.to(w.device)[tgt].clamp_min(eps).sqrt()
    dist = (1.0 / (w * gate)).cpu().numpy()

    s_np = src.cpu().numpy()
    t_np = tgt.cpu().numpy()
    G = nx.Graph()
    G.add_weighted_edges_from(
        ((("s", int(a)), ("t", int(b)), float(d)) for a, b, d in zip(s_np, t_np, dist)))

    # proc=1: GraphRicciCurvature's multiprocessing forks workers, which segfaults
    # when torch is already loaded (macOS fork). Single-process is fast enough on a
    # sparse layer (~30k edges/s) and safe.
    orc = OllivierRicci(G, alpha=alpha, weight="weight", method=method,
                        proc=1, verbose="ERROR")
    orc.compute_ricci_curvature()

    out = np.fromiter(
        (orc.G[("s", int(a))][("t", int(b))]["ricciCurvature"]
         for a, b in zip(s_np, t_np)), dtype=np.float64, count=len(s_np))
    F[tgt, src] = torch.tensor(out, dtype=F.dtype, device=F.device)
    return F
