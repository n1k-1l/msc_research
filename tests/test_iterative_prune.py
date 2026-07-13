"""
Iterative-pruning sanity checks: masked curvature is degree-aware, edge pruning
actually breaks degree uniformity, and pruned edges stay dead through fine-tuning.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curvature import forman_edges, forman_edges_masked
from src.iterative_prune import _prune_edges_to, _finetune, _register_grad_masks
from src.models import MLP


def test_masked_equals_dense_when_full():
    A = torch.rand(6, 5) + 0.1
    full = torch.ones_like(A, dtype=torch.bool)
    assert torch.allclose(forman_edges(A), forman_edges_masked(A, full), atol=1e-6)


def test_masked_unweighted_reduces_to_degree_formula():
    # Unweighted (all weights 1) with some edges masked: F = 4 - deg_present(u) - deg_present(v).
    A = torch.ones(4, 3)
    mask = torch.ones(4, 3, dtype=torch.bool)
    mask[0, 0] = False          # drop one edge
    F = forman_edges_masked(A, mask)
    deg_src = mask.sum(dim=0)   # present degree per source
    deg_tgt = mask.sum(dim=1)   # present degree per target
    expected = 4.0 - (deg_src.unsqueeze(0) + deg_tgt.unsqueeze(1)).float()
    assert torch.allclose(F[mask], expected[mask], atol=1e-6)
    assert torch.isnan(F[~mask]).all()          # pruned edge is undefined


def test_edge_pruning_breaks_degree_uniformity():
    torch.manual_seed(0)
    model = MLP(widths=[20, 16, 10], dropout_kind="none", p=0.0)
    linears = model.linear_layers()
    masks = [torch.ones_like(l.weight, dtype=torch.bool) for l in linears]
    _prune_edges_to(linears, masks, "magnitude", target=0.5)
    # after unstructured pruning, per-neuron degree varies (was constant when dense)
    deg = torch.cat([masks[0].sum(0).float(), masks[0].sum(1).float()])
    assert deg.std() > 0, "edge pruning left degree uniform"
    # sparsity actually reached ~50%
    frac = 1 - masks[0].float().mean()
    assert abs(frac - 0.5) < 0.05


def test_pruned_edges_stay_zero_after_finetune():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = MLP(widths=[20, 16, 10], dropout_kind="none", p=0.0).to(device)
    linears = model.linear_layers()
    masks = [torch.ones_like(l.weight, dtype=torch.bool) for l in linears]
    _register_grad_masks(linears, masks)
    _prune_edges_to(linears, masks, "magnitude", target=0.5)
    x = torch.randn(64, 20)
    y = torch.randint(0, 10, (64,))
    loader = [(x, y)] * 5
    _finetune(model, loader, device, epochs=3, lr=1e-3, linears=linears, masks=masks)
    for lin, m in zip(linears, masks):
        assert torch.equal(lin.weight[~m], torch.zeros_like(lin.weight[~m])), \
            "a pruned edge was revived during fine-tuning"


if __name__ == "__main__":
    test_masked_equals_dense_when_full()
    test_masked_unweighted_reduces_to_degree_formula()
    test_edge_pruning_breaks_degree_uniformity()
    test_pruned_edges_stay_zero_after_finetune()
    print("ok")
