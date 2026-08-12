#!/usr/bin/env python3
"""
Rank-concordance of the two pruning criteria over training (poster panel).

How often do Forman curvature and weight magnitude pick the SAME units to prune?
At sparsity s, forman prunes the top-s units by kappa and magnitude the bottom-s
by |w|; because the two scores are anti-correlated those sets nearly coincide.
This plots the top-k overlap |F ∩ M| / k vs epoch, per layer, for a few
sparsities, mean ± std over seeds -- a rank-concordance time series showing the
criteria converging on the same prunable set as training proceeds.

    python scripts/plot_concordance.py --results-dir results_td_cpu \
        --config mnist_small_td_forman
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 19, "axes.titlesize": 19, "axes.labelsize": 19,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16, "legend.title_fontsize": 16,
    "lines.linewidth": 2.5, "lines.markersize": 8})


def overlap(curv, mag, s: float) -> float:
    """Top-k agreement between forman's prune set (top-s kappa) and magnitude's
    (bottom-s |w|), as a fraction of k = round(s*n)."""
    curv = np.asarray(curv)
    mag = np.asarray(mag)
    n = len(curv)
    k = round(s * n)
    if k == 0:
        return 1.0
    F = set(np.argsort(curv)[-k:])      # highest curvature = forman-prunable
    M = set(np.argsort(mag)[:k])        # lowest magnitude = magnitude-prunable
    return len(F & M) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_td_cpu")
    ap.add_argument("--config", default="mnist_small_td_forman")
    ap.add_argument("--sparsities", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    logs = [json.loads(f.read_text()).get("curvature_log")
            for f in sorted(rdir.glob(f"{args.config}_seed*.json"))]
    logs = [l for l in logs if l]
    if not logs:
        raise SystemExit(f"no curvature_log data for {args.config} in {rdir}")
    epochs = [e["epoch"] for e in logs[0]]
    n_layers = len(logs[0][0]["layers"])

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4.5), squeeze=False)
    for L in range(n_layers):
        ax = axes[0][L]
        for s in args.sparsities:
            vals = np.array([[overlap(e["layers"][L]["curv"], e["layers"][L]["mag"], s)
                              for e in log] for log in logs])
            m, sd = vals.mean(0), vals.std(0)
            ax.plot(epochs, m, marker="o", label=f"s = {s:g}")
            ax.fill_between(epochs, m - sd, m + sd, alpha=0.15)
        ax.axhline(1.0, color="k", lw=1.2, ls=":")
        ax.set_ylim(0.5, 1.02)
        ax.set_xlabel("epoch")
        ax.set_ylabel("prunable-set overlap")
        ax.set_title(f"hidden layer {L}")
        ax.grid(alpha=0.3)
        ax.legend(title="sparsity")
    fig.tight_layout()
    out = rdir / f"{args.config}_concordance.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy (thesis)
    print(f"saved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()
