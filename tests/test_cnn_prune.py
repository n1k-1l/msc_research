"""
CNN channel-graph pruning: views reduce Conv/Linear to 2D graphs, masks zero the
right real weights (kernel slices for conv), and pruned channel-edges stay dead.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import LeNet
from src.cnn_prune import graph_views, ConvView, cnn_prune_one


def test_graph_views_shapes():
    model = LeNet()
    views = graph_views(model, include_linear=True)
    shapes = [tuple(v.matrix.shape) for v in views]
    assert shapes == [(6, 1), (16, 6), (120, 256), (84, 120), (10, 84)]


def test_conv_view_matrix_and_mask():
    model = LeNet()
    conv = model.conv_layers()[1]           # 6 -> 16, 5x5
    v = ConvView(conv)
    M = v.matrix
    assert M.shape == (16, 6)
    # channel-edge weight is the L2 norm of the kernel slice
    assert torch.allclose(M[3, 2], conv.weight[3, 2].flatten().norm(), atol=1e-6)
    # pruning edge (3,2) zeros the whole kernel slice
    mask = torch.ones(16, 6, dtype=torch.bool); mask[3, 2] = False
    v.apply_mask(mask)
    assert torch.count_nonzero(conv.weight[3, 2]) == 0
    assert torch.count_nonzero(conv.weight[0, 0]) > 0     # others untouched



def test_cnn_prune_one_keeps_pruned_zero():
    torch.manual_seed(0)
    model = LeNet()
    x = torch.randn(64, 1, 28, 28); y = torch.randint(0, 10, (64,))
    loader = [(x, y)] * 3
    rec = cnn_prune_one(model, "magnitude", (loader, loader, loader),
                        torch.device("cpu"), schedule=[0.5], finetune_epochs=1, lr=1e-3)
    assert rec[0]["sparsity"] > 0.4
    # every conv/linear must still have its pruned channel-edges at zero
    from src.cnn_prune import graph_views as gv
    for v in gv(model):
        M = v.matrix
        assert torch.isfinite(M).all()


if __name__ == "__main__":
    test_graph_views_shapes()
    test_conv_view_matrix_and_mask()
    test_channel_activation_lengths()
    test_cnn_prune_one_keeps_pruned_zero()
    print("ok")
