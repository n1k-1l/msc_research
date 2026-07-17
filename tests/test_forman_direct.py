"""The literal Eq.(2) evaluation must agree with the closed form (Eq. 4)."""
import torch

from src.curvature import (forman_edges, forman_edges_direct,
                           per_neuron_curvature, per_neuron_curvature_direct)


def test_direct_matches_closed_form_weighted():
    torch.manual_seed(0)
    A = torch.rand(7, 5) + 0.05
    assert torch.allclose(forman_edges_direct(A), forman_edges(A), atol=1e-5)


def test_direct_matches_unweighted_special_case():
    A = torch.ones(6, 4)
    F = forman_edges_direct(A)
    # complete bipartite, unweighted: F = 4 - deg(u) - deg(v) = 4 - 6 - 4
    assert torch.allclose(F, torch.full_like(F, 4.0 - 6.0 - 4.0), atol=1e-5)


def test_per_neuron_direct_matches_closed():
    torch.manual_seed(1)
    weights = [torch.randn(6, 8), torch.randn(4, 6)]
    closed = per_neuron_curvature(weights)
    direct = per_neuron_curvature_direct(weights)
    for c, d in zip(closed, direct):
        assert torch.allclose(c, d, atol=1e-5)
