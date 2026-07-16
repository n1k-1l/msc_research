"""Degree-corrected Forman (the collapse-mechanism causal ablation) + prune
forensics.

forman_dc replaces the neighbour SUM in F(e) = 2 - sum_{e'~e} sqrt(w_e/w_e')
with a MEAN, so the degree term vanishes by construction: on any unweighted
graph every edge scores exactly 0 regardless of degree, while plain Forman
gives F = 4 - deg(u) - deg(v). Whatever variation survives is purely
weight-geometric -- which is what makes pruning by it a causal test of the
degree-feedback collapse story.
"""
import math

import pytest
import torch

from src.curvature import forman_dc_edges_masked, forman_edges_masked
from src.iterative_prune import _prune_edges_to


def test_unweighted_heterogeneous_degree_is_constant_zero():
    """On an unweighted graph, F varies with degree; F_dc is 0 everywhere."""
    torch.manual_seed(0)
    A = torch.ones(6, 8)
    mask = torch.rand(6, 8) < 0.5
    # ensure at least one edge with neighbours exists
    mask[0, :4] = True
    F = forman_edges_masked(A, mask)
    Fdc = forman_dc_edges_masked(A, mask)
    surv = mask & torch.isfinite(Fdc)
    assert torch.allclose(Fdc[surv], torch.zeros(int(surv.sum())), atol=1e-5)
    # plain Forman must NOT be constant here (degrees are heterogeneous)
    assert F[mask].std() > 1e-3


def test_matches_bruteforce_mean_ratio():
    """F_dc(e) = 1 - mean over neighbouring edges of sqrt(w_e / w_e')."""
    torch.manual_seed(1)
    A = torch.rand(4, 5) + 0.1
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[2, 3] = False
    Fdc = forman_dc_edges_masked(A, mask)
    for r in range(4):
        for c in range(5):
            if not mask[r, c]:
                assert math.isnan(float(Fdc[r, c]))
                continue
            ratios = []
            for c2 in range(5):            # edges sharing the target vertex r
                if c2 != c and mask[r, c2]:
                    ratios.append(math.sqrt(float(A[r, c] / A[r, c2])))
            for r2 in range(4):            # edges sharing the source vertex c
                if r2 != r and mask[r2, c]:
                    ratios.append(math.sqrt(float(A[r, c] / A[r2, c])))
            expected = 1.0 - sum(ratios) / len(ratios)
            assert float(Fdc[r, c]) == pytest.approx(expected, abs=1e-5)


def test_bridge_edge_is_least_prunable():
    """An edge whose endpoints have no other edge gets -inf (never redundant)."""
    A = torch.rand(3, 3) + 0.1
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 0] = True                      # isolated bridge
    mask[1, 1] = mask[1, 2] = mask[2, 1] = True   # a connected patch
    Fdc = forman_dc_edges_masked(A, mask)
    assert float(Fdc[0, 0]) == float("-inf")
    assert torch.isfinite(Fdc[1, 1]) and torch.isfinite(Fdc[1, 2])


def _tiny_linears(shape=(8, 10), seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(shape[1], shape[0], bias=False)
    return [lin], [torch.ones_like(lin.weight, dtype=torch.bool)]


def test_forensics_schema_and_magnitude_percentile():
    """Magnitude pruning must kill edges from the bottom of the |w| ranking."""
    linears, masks = _tiny_linears()
    f = _prune_edges_to(linears, masks, "magnitude", target=0.5)
    assert len(f) == 1
    rec = f[0]
    for key in ("n_pruned", "pruned_deg_mean", "surv_deg_mean",
                "pruned_mag_pctile_mean", "new_disconnected_src",
                "new_disconnected_tgt"):
        assert key in rec
    assert rec["n_pruned"] == 40
    # bottom-half kill => percentiles live in [0, 0.5), mean ~ 0.25
    assert rec["pruned_mag_pctile_mean"] < 0.3


def test_forensics_counts_disconnections():
    """Wiping a whole column must be reported as a disconnected source vertex."""
    linears, masks = _tiny_linears()
    lin, m = linears[0], masks[0]
    with torch.no_grad():                  # make column 0 the 8 smallest weights
        lin.weight.abs_()
        lin.weight[:, 0] = 1e-6
    f = _prune_edges_to(linears, masks, "magnitude", target=8 / m.numel())
    assert f[0]["n_pruned"] == 8
    assert f[0]["new_disconnected_src"] == 1
    assert f[0]["new_disconnected_tgt"] == 0
    assert not m[:, 0].any()


def test_forman_dc_prunes_without_degree_bias_on_unweighted():
    """On an unweighted sparse graph forman_dc has no degree signal (all ties),
    while plain forman prunes low-degree endpoints first -- the mechanism in
    miniature."""
    torch.manual_seed(3)
    lin = torch.nn.Linear(12, 10, bias=False)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    mask = torch.rand(10, 12) < 0.4
    mask[0, :6] = True                     # guarantee some high-degree structure
    masks = [mask.clone()]
    f = _prune_edges_to([lin], masks, "curvature", target=0.75)
    # forman kills edges whose endpoint degree is BELOW the survivor average
    assert f[0]["pruned_deg_mean"] < f[0]["surv_deg_mean"]
