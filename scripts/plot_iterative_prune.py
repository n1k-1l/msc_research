#!/usr/bin/env python3
"""
Iterative hard pruning: curvature vs magnitude under non-uniform connectivity.

Reads the `iterative_prune_curves` field written by run_baseline for an iterative-
pruning config (magnitude / curvature / random, each a list of per-round records)
and draws two panels:

  A. Accuracy vs edge sparsity per criterion (mean +/- std over seeds). A star marks
     sparsities where the paired curvature-vs-magnitude difference is significant
     (|t| > 2.26, p < 0.05, df = n-1, matched seeds), coloured by the winner.
  B. |Pearson(curvature, magnitude)| over surviving edges vs sparsity (mean over the
     layers, +/- std over seeds) -- the decoupling diagnostic. If the mechanism is
     right this decays from ~1 as the graph becomes non-uniform. degree_cv (dashed,
     right axis) confirms connectivity actually stopped being uniform.

    python scripts/plot_iterative_prune.py --results-dir results_ip --config mnist_small_ip
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

# Shared poster palette: curvature (method) / magnitude (baseline) / random (control).
COL = {"curvature": "#1f77b4", "magnitude": "#ff7f0e", "random": "#7f7f7f"}


def load(rdir: str, cfg: str):
    R = [json.loads(Path(f).read_text())
         for f in sorted(glob.glob(f"{rdir}/{cfg}_seed*.json"))]
    if not R:
        raise SystemExit(f"no data for {cfg} in {rdir}")
    crit = {}
    for c in COL:
        rounds = [r["iterative_prune_curves"][c] for r in R]
        acc = np.array([[x["test_acc"] for x in rr] for rr in rounds])       # (seeds, rounds)
        spars = np.array([[x["sparsity"] for x in rr] for rr in rounds])
        # mean over layers of |Pearson| and of degree_cv, per round
        absr = np.array([[np.nanmean(np.abs(x["pearson_curv_mag"])) for x in rr]
                         for rr in rounds])
        dcv = np.array([[np.nanmean(x["degree_cv"]) for x in rr] for rr in rounds])
        crit[c] = dict(acc=acc, spars=spars, absr=absr, dcv=dcv)
    return crit, len(R)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_ip")
    ap.add_argument("--config", default="mnist_small_ip")
    args = ap.parse_args()

    crit, n = load(args.results_dir, args.config)
    x = crit["magnitude"]["spars"].mean(0)          # shared sparsity axis
    tcrit = 2.262 if n == 10 else 2.26              # df = n-1 ~ 9

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Panel A: accuracy vs sparsity ---
    for c in ("curvature", "magnitude", "random"):
        m, sd = crit[c]["acc"].mean(0), crit[c]["acc"].std(0)
        axA.plot(x, m, marker="o", ms=4, color=COL[c], label=c)
        axA.fill_between(x, m - sd, m + sd, alpha=0.15, color=COL[c])
    d = crit["curvature"]["acc"] - crit["magnitude"]["acc"]
    t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(n) + 1e-12)
    for i in range(len(x)):
        if abs(t[i]) > tcrit:
            y = max(crit["curvature"]["acc"][:, i].mean(),
                    crit["magnitude"]["acc"][:, i].mean())
            win = COL["curvature"] if t[i] > 0 else COL["magnitude"]
            axA.annotate("*", (x[i], y + 0.005), ha="center", color=win, fontsize=16)
    axA.set_xlabel("edge sparsity (fraction of weights pruned)")
    axA.set_ylabel("test accuracy")
    axA.set_title(f"Accuracy vs sparsity  (n = {n})")
    axA.grid(alpha=0.3)
    axA.legend()

    # --- Panel B: decoupling diagnostic ---
    for c in ("curvature", "magnitude", "random"):
        m, sd = crit[c]["absr"].mean(0), crit[c]["absr"].std(0)
        axB.plot(x, m, marker="o", ms=4, color=COL[c], label=f"{c}: |$\\rho$|")
        axB.fill_between(x, m - sd, m + sd, alpha=0.12, color=COL[c])
    axB.set_xlabel("edge sparsity (fraction of weights pruned)")
    axB.set_ylabel(r"|Pearson($\kappa$, $|w|$)| over survivors")
    axB.set_ylim(0, 1.05)
    axB.set_title("Do curvature and magnitude decouple?")
    axB.grid(alpha=0.3)
    axr = axB.twinx()
    axr.plot(x, crit["curvature"]["dcv"].mean(0), ls="--", color=COL["curvature"],
             lw=1.2, label="curvature degree CV")
    axr.set_ylabel("degree CV, curvature arm (dashed)")
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")

    fig.suptitle(f"Iterative hard pruning -- {args.config}  "
                 "(* = significant curvature$-$magnitude difference, p<0.05)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.results_dir) / f"{args.config}_iterative_prune.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy (thesis)
    print(f"saved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()
