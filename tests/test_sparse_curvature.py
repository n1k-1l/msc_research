"""
Sparse-net building blocks: fixed sparse masks, activation collection, and the two
Ollivier-Ricci curvatures (weighted via GraphRicciCurvature, and the faithful
activation-weighted one). Small graphs keep the OT solves fast.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MLP
from src.iterative_prune import make_sparse_masks
from src.activations import collect_activations
from src.ollivier import ollivier_edges_masked
from src.ollivier_neural import neural_ollivier_edges_masked


def test_make_sparse_masks_density():
    lin = MLP([20, 16, 10], dropout_kind="none").linear_layers()
    masks = make_sparse_masks(lin, 0.3, torch.Generator().manual_seed(0))
    assert len(masks) == len(lin)
    for m, l in zip(masks, lin):
        assert m.shape == l.weight.shape and m.dtype == torch.bool
        assert abs(float(m.float().mean()) - 0.3) < 0.06


def test_activations_shapes_and_sign():
    model = MLP([20, 16, 10], dropout_kind="none")
    x, y = torch.randn(8, 20), torch.randint(0, 10, (8,))
    acts = collect_activations(model, [(x, y)], torch.device("cpu"))
    assert [a.numel() for a in acts] == [20, 16, 10]     # input, hidden, output
    assert (acts[1] >= 0).all()                          # post-ReLU hidden is non-negative


def test_ollivier_masked_finite_and_nan():
    torch.manual_seed(0)
    W = torch.randn(6, 5)
    mask = torch.rand(6, 5) < 0.6
    F = ollivier_edges_masked(W, mask)
    assert F.shape == W.shape
    assert torch.isnan(F[~mask]).all()                   # pruned edges undefined
    assert torch.isfinite(F[mask]).all()


def test_neural_ollivier_finite_and_bounded():
    torch.manual_seed(1)
    W = torch.randn(6, 5)
    mask = torch.rand(6, 5) < 0.7
    a_src, a_tgt = torch.rand(5) + 0.1, torch.rand(6) + 0.1
    F = neural_ollivier_edges_masked(W, mask, a_src, a_tgt)
    assert torch.isnan(F[~mask]).all()
    fv = F[mask]
    assert torch.isfinite(fv).all()
    assert (fv <= 1.0 + 1e-6).all()                      # kappa = 1 - W1 <= 1


if __name__ == "__main__":
    test_make_sparse_masks_density()
    test_activations_shapes_and_sign()
    test_ollivier_masked_finite_and_nan()
    test_neural_ollivier_finite_and_bounded()
    print("ok")
