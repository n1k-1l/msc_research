"""Sparse-net building blocks: the fixed sparse and heavy-tailed mask makers."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MLP
from src.iterative_prune import make_hetero_masks, make_sparse_masks


def test_make_sparse_masks_density():
    lin = MLP([20, 16, 10], dropout_kind="none").linear_layers()
    masks = make_sparse_masks(lin, 0.3, torch.Generator().manual_seed(0))
    assert len(masks) == len(lin)
    for m, l in zip(masks, lin):
        assert m.shape == l.weight.shape and m.dtype == torch.bool
        assert abs(float(m.float().mean()) - 0.3) < 0.06


def test_make_hetero_masks_density_and_spread():
    """Heavy-tailed masks hit the target density but with wider degree spread
    than the ER masks at the same density."""
    lin = MLP([64, 48, 10], dropout_kind="none").linear_layers()
    gen = torch.Generator().manual_seed(0)
    het = make_hetero_masks(lin, 0.5, 1.5, gen)
    er = make_sparse_masks(lin, 0.5, torch.Generator().manual_seed(0))
    for h, e, l in zip(het, er, lin):
        assert h.shape == l.weight.shape
        assert abs(float(h.float().mean()) - 0.5) < 0.08
        deg_h = torch.cat([h.sum(0).float(), h.sum(1).float()])
        deg_e = torch.cat([e.sum(0).float(), e.sum(1).float()])
        cv = lambda d: float(d.std(unbiased=False) / d.mean())
        assert cv(deg_h) > cv(deg_e)
