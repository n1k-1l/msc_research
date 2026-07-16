"""
Whole-filter (structured) CNN pruning: the reduced node chain has chained dims
across the flatten boundary, killing a filter removes its row AND its spatial
block of FC columns, and filter pruning preserves degree uniformity among
survivors (the structural fact behind the forman ~ magnitude prediction).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import LeNet
from src.cnn_prune import (graph_views, _reduced_chain, _kill_node,
                           _prune_filters_to)


def _fresh(conv_channels=(8, 12)):
    model = LeNet(conv_channels=conv_channels)
    views = graph_views(model, include_linear=True)
    masks = [torch.ones_like(v.matrix, dtype=torch.bool) for v in views]
    return model, views, masks


def test_reduced_chain_dims_chain():
    model, views, masks = _fresh(conv_channels=(8, 12))
    mats, nmasks = _reduced_chain(views, masks)
    # raw view shapes: (8,1),(12,8),(120, 12*16),(84,120),(10,84)
    dims = [tuple(M.shape) for M in mats]
    assert dims == [(8, 1), (12, 8), (120, 12), (84, 120), (10, 84)]
    for k in range(1, len(dims)):
        assert dims[k][1] == dims[k - 1][0], "node dims must chain"


def test_kill_node_at_flatten_boundary():
    model, views, masks = _fresh(conv_channels=(8, 12))
    # kill conv2 filter j=3: node layer h=2 (views: conv1, conv2, fc1, ...)
    _kill_node(views, masks, h=2, j=3)
    assert not masks[1][3, :].any(), "incoming channel-row must be dead"
    block = masks[2].shape[1] // 12          # fc_in / c2 = 16
    assert not masks[2][:, 3 * block:(3 + 1) * block].any(), \
        "the filter's spatial block of FC columns must be dead"
    assert masks[2][:, :3 * block].all(), "other FC columns untouched"


def test_filter_pruning_preserves_degree_uniformity():
    torch.manual_seed(0)
    model, views, masks = _fresh(conv_channels=(8, 12))
    _prune_filters_to(views, masks, "magnitude", target=0.4)
    mats, nmasks = _reduced_chain(views, masks)
    for h in range(1, len(nmasks)):
        alive_rows = nmasks[h - 1].any(dim=1)
        deg = nmasks[h - 1].sum(dim=1).float()[alive_rows]
        # every surviving node keeps ALL surviving in-neighbours: degree uniform
        assert deg.numel() == 0 or float(deg.std()) == 0.0, \
            f"degree non-uniform among survivors at node layer {h}"
    # weights actually zeroed
    for v, m in zip(views, masks):
        M = v.matrix
        assert torch.count_nonzero(M[~m]) == 0


if __name__ == "__main__":
    test_reduced_chain_dims_chain()
    test_kill_node_at_flatten_boundary()
    test_filter_pruning_preserves_degree_uniformity()
    print("ok")
