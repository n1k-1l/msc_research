"""
Sparse-net building blocks: fixed sparse masks, activation collection, and the
library Ollivier-Ricci curvature (GraphRicciCurvature). Small graphs keep the OT
solves fast.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MLP
from src.iterative_prune import make_sparse_masks
from src.activations import collect_activations
from src.ollivier import ollivier_edges_masked


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


@pytest.mark.skip(reason="GraphRicciCurvature/networkit segfaults next to torch's "
                         "OpenMP on macOS (crash in _get_all_pairs_shortest_path). "
                         "Running with OMP_NUM_THREADS=1 set before imports fixes "
                         "it if this ever needs re-enabling. ORC is appendix-only; "
                         "not worth stabilising now (2026-07-14).")
def test_ollivier_masked_finite_and_nan():
    torch.manual_seed(0)
    W = torch.randn(6, 5)
    mask = torch.rand(6, 5) < 0.6
    F = ollivier_edges_masked(W, mask)
    assert F.shape == W.shape
    assert torch.isnan(F[~mask]).all()                   # pruned edges undefined
    assert torch.isfinite(F[mask]).all()


if __name__ == "__main__":
    test_make_sparse_masks_density()
    test_activations_shapes_and_sign()
    test_ollivier_masked_finite_and_nan()
    print("ok")
