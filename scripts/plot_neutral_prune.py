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
            ax.plot(spars, m, marker="o", ms=4, color=COL[c], label=c)
            ax.fill_between(spars, m - sd, m + sd, alpha=0.15, color=COL[c])
        # paired forman - magnitude significance (matched seeds)
        d = data["forman"] - data["magnitude"]
        t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(n) + 1e-12)
        for i, s in enumerate(spars):
            if abs(t[i]) > 2.26:
                y = max(data["forman"][:, i].mean(), data["magnitude"][:, i].mean())
                # colour the star by the winner: blue = forman better, orange = magnitude
                win = COL["forman"] if t[i] > 0 else COL["magnitude"]
                ax.annotate("*", (s, y + 0.015), ha="center", color=win, fontsize=16)
        ax.set_xlabel("sparsity (fraction of hidden units pruned)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{cfg.replace('mnist_small_', '')}  (n = {n})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Neutral net pruned by each criterion\n"
                 "(* = significant forman$-$magnitude difference, p<0.05)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.results_dir) / "neutral_prune.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy (thesis)
    print(f"saved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()
