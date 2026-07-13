#!/usr/bin/env python3
"""
Heterogeneity study (E12) figure: sparsity vs degree heterogeneity, disentangled.

Panel A: trained Forman<->magnitude coupling (Spearman over surviving edges) against
the mask's degree CV, per hidden layer pair, coloured by the tail parameter sigma --
the coupling is a smooth function of degree heterogeneity at FIXED density.
Panel B: forman - magnitude test-accuracy difference over the moderate-sparsity
prune-further schedule, one line per sigma (mean +/- std over seeds), with
random - magnitude as the grey reference band. If decoupled curvature carried
importance signal, the sigma-ordered lines would rise above zero.

    python scripts/plot_hetero.py --results-dir results_hetero
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_hetero")
    args = ap.parse_args()
    R = [json.loads(Path(f).read_text())
         for f in sorted(glob.glob(f"{args.results_dir}/hetero_s*_seed*.json"))]
    if not R:
        raise SystemExit("no data")
    sigmas = sorted({r["sigma"] for r in R})
    cmap = plt.get_cmap("viridis")
    col = {s: cmap(i / max(len(sigmas) - 1, 1)) for i, s in enumerate(sigmas)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- A: coupling vs degree CV (hidden pairs only: layers 0, 1) ---
    for r in R:
        for L, mk in [(0, "o"), (1, "s")]:
            st = r["mask_stats"][L]
            axA.scatter(st["degree_cv"], st["spearman"], marker=mk, s=42,
                        color=col[r["sigma"]], edgecolor="k", linewidth=0.4,
                        zorder=3)
    for s in sigmas:
        axA.scatter([], [], color=col[s], label=f"$\\sigma={s:g}$")
    axA.scatter([], [], marker="o", color="0.5", label="pair 0 (784–512)")
    axA.scatter([], [], marker="s", color="0.5", label="pair 1 (512–256)")
    axA.set_xlabel("degree coefficient of variation (fixed density 0.5)")
    axA.set_ylabel(r"Spearman $\rho(F, |w|)$ over surviving edges")
    axA.set_title("Coupling is a function of degree heterogeneity,\nnot sparsity")
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8)

    # --- B: forman - magnitude gap vs sparsity, per sigma ---
    for s in sigmas:
        runs = [r for r in R if r["sigma"] == s]
        f = np.array([[x["test_acc"] for x in r["iterative_prune_curves"]["curvature"]]
                      for r in runs])
        m = np.array([[x["test_acc"] for x in r["iterative_prune_curves"]["magnitude"]]
                      for r in runs])
        rd = np.array([[x["test_acc"] for x in r["iterative_prune_curves"]["random"]]
                       for r in runs])
        spars = [x["sparsity"] for x in runs[0]["iterative_prune_curves"]["magnitude"]]
        d = (f - m)
        axB.plot(spars, d.mean(0), marker="o", color=col[s], label=f"$\\sigma={s:g}$")
        axB.fill_between(spars, d.mean(0) - d.std(0), d.mean(0) + d.std(0),
                         alpha=0.12, color=col[s])
        if s == sigmas[-1]:
            dr = (rd - m)
            axB.plot(spars, dr.mean(0), ls=":", color="0.4",
                     label="random $-$ magnitude")
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xlabel("total edge sparsity (from fixed density 0.5)")
    axB.set_ylabel("forman $-$ magnitude test accuracy")
    axB.set_title("Decoupled curvature at moderate sparsity:\ndoes it earn its keep?")
    axB.grid(alpha=0.3)
    axB.legend(fontsize=8)

    fig.suptitle("Fixed density 0.5, heavy-tailed degree heterogeneity "
                 "($\\sigma$ = lognormal tail parameter); mean $\\pm$ std over seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.results_dir) / "hetero.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy (thesis)
    print(f"saved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()
