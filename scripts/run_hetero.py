#!/usr/bin/env python3
"""
Sparsity vs degree heterogeneity: the disentangling study (E12).

E9/E10 decoupled curvature from magnitude only at extreme (>0.85) sparsity, inviting
the objection that the regime is not meaningful. Synthetic probes show sparsity was
never the axis: ER masks stay coupled to 98% sparsity (degree concentration), while
degree heterogeneity at FIXED 0.5 sparsity moves the coupling. This study makes that
a trained-network result: fixed density, lognormal-propensity masks with tail
parameter sigma (see make_hetero_masks), train sparse, then

  (a) measure the trained Forman<->magnitude coupling per layer  -> rho vs degree-CV
  (b) prune further by magnitude / forman / random over a MODERATE sparsity schedule
      -> does decoupled Forman help/tie/lose at meaningful sparsity?

    python scripts/run_hetero.py --sigmas 0.0 0.5 1.0 1.5 --seeds 0 1 2 \
        --density 0.5 --results-dir results_hetero
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import get_data
from src.models import MLP
from src.train import train
from src.curvature import forman_edges_masked
from src.iterative_prune import (make_hetero_masks, iterative_prune_by_criteria,
                                 _pearson, _spearman)
from src.utils import set_seed


@torch.no_grad()
def mask_and_coupling_stats(linears, masks):
    """Per layer-pair: degree CV of the mask and trained rho(F, |w|) over survivors."""
    out = []
    for lin, m in zip(linears, masks):
        A = lin.weight.abs()
        F = forman_edges_masked(A, m)
        mag = A[m].cpu().numpy()
        cur = F[m].cpu().numpy()
        deg = torch.cat([m.sum(0).float(), m.sum(1).float()])
        deg = deg[deg > 0]
        out.append({
            "density": float(m.float().mean()),
            "degree_cv": float(deg.std(unbiased=False) / deg.mean()),
            "pearson": _pearson(mag, cur),
            "spearman": _spearman(mag, cur),
        })
    return out


def run_one(sigma: float, seed: int, args) -> dict:
    print(f"\n=== sigma={sigma} | seed {seed} ===")
    set_seed(seed)
    device = torch.device("cpu")
    data = get_data("mnist", data_root=args.data_root, batch_size=128,
                    val_fraction=0.1, num_workers=0, seed=seed)
    model = MLP(widths=[784, 512, 256, 10], dropout_kind="none", p=0.0)
    gen = torch.Generator().manual_seed(seed)
    masks = make_hetero_masks(model.linear_layers(), args.density, sigma, gen)
    print("  densities:", [round(float(m.float().mean()), 3) for m in masks])

    res = train(model, data.train, data.val, data.test, device,
                epochs=args.epochs, lr=1e-3, config={}, weight_masks=masks,
                verbose=False)
    stats = mask_and_coupling_stats(model.linear_layers(), masks)
    print(f"  trained: test={res.test_acc:.4f}  "
          f"deg_cv={[round(s['degree_cv'],2) for s in stats]}  "
          f"spearman={[round(s['spearman'],2) for s in stats]}")

    curves = iterative_prune_by_criteria(
        model, loaders=(data.train, data.val, data.test), device=device,
        schedule=args.schedule, finetune_epochs=args.finetune_epochs, lr=1e-3,
        criteria=args.criteria, init_masks=masks)

    out = {
        "sigma": sigma, "seed": seed, "density": args.density,
        "epochs": args.epochs, "schedule": args.schedule,
        "test_acc": res.test_acc, "best_val_acc": res.best_val_acc,
        "mask_stats": stats,                    # per layer: deg_cv + trained coupling
        "iterative_prune_curves": curves,       # same schema as E9/E10
    }
    path = Path(args.results_dir) / f"hetero_s{sigma:g}_seed{seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  saved -> {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=3)
    ap.add_argument("--schedule", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    ap.add_argument("--criteria", nargs="+",
                    default=["magnitude", "curvature", "random"],
                    help="edge criteria to compare (add forman_dc for the "
                         "degree-corrected causal ablation)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--results-dir", default="results_hetero")
    args = ap.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    for sigma in args.sigmas:
        for seed in args.seeds:
            run_one(sigma, seed, args)


if __name__ == "__main__":
    main()
