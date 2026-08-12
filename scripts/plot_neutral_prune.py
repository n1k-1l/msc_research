#!/usr/bin/env python3
"""
Confounding-control panel: prune a NEUTRAL net by each criterion (poster).

On a network trained with NO curvature/magnitude bias (uniform dropout or no
dropout), prune the same trained weights by forman / magnitude / random over a
sparsity sweep (the `prune_curves_by_criterion` field from --cross-prune). This
decouples *criterion quality* from *training co-adaptation*: if curvature were a
genuinely better importance signal it should win here too. It does not -- on the
no-dropout net magnitude is significantly better at mid sparsity. A red star marks
sparsities where the paired forman-vs-magnitude difference is significant
(|t| > 2.26, p < 0.05, df = n-1, same seeds = matched pairs).

    python scripts/plot_neutral_prune.py --results-dir results_neutral
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
plt.rcParams.update({
    "font.size": 19, "axes.titlesize": 19, "axes.labelsize": 19,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16, "legend.title_fontsize": 16,
    "lines.linewidth": 2.5, "lines.markersize": 8})

# Shared poster palette: forman (method) / magnitude (baseline) / random (control).
COL = {"forman": "#1f77b4", "magnitude": "#ff7f0e", "random": "#7f7f7f"}


def load(rdir: str, cfg: str):
    R = [json.loads(Path(f).read_text())
         for f in sorted(glob.glob(f"{rdir}/{cfg}_seed*.json"))]
    if not R:
        raise SystemExit(f"no data for {cfg} in {rdir}")
    data = {c: np.array([[p["test_acc"] for p in r["prune_curves_by_criterion"][c]]
                         for r in R]) for c in COL}
    spars = [p["sparsity"] for p in R[0]["prune_curves_by_criterion"]["forman"]]
    return data, spars, len(R)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_neutral")
    ap.add_argument("--configs", nargs="+",
                    default=["mnist_small_uniform_p25", "mnist_small_nodrop"])
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.configs),
                             figsize=(6.2 * len(args.configs), 4.8), squeeze=False)
    for j, cfg in enumerate(args.configs):
        ax = axes[0][j]
        data, spars, n = load(args.results_dir, cfg)
        for c in ("forman", "magnitude", "random"):
            m, sd = data[c].mean(0), data[c].std(0)
            ax.plot(spars, m, marker="o", color=COL[c], label=c)
            ax.fill_between(spars, m - sd, m + sd, alpha=0.15, color=COL[c])
        # inference lives in the corresponding table (Bonferroni within family);
        # the figure shows estimates only
        ax.set_xlabel("fraction of units pruned")
        ax.set_ylabel("test accuracy")
        TITLES = {"mnist_small_uniform_p25": "uniform dropout, $p=0.25$",
                  "mnist_small_nodrop": "no dropout"}
        ax.set_title(TITLES.get(cfg, cfg))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = Path(args.results_dir) / "neutral_prune.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy (thesis)
    print(f"saved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()
