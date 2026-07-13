#!/usr/bin/env python3
"""
Sparse-by-construction pruning: curvature flavours vs magnitude on a fixed
non-uniform topology (accuracy vs additional sparsity, N criteria).

Reads `iterative_prune_curves` written by run_baseline for mnist_sparse_ip and plots
each criterion's mean +/- std test accuracy over the sparsity sweep. Stars mark
sparsities where a curvature criterion differs significantly from magnitude (paired
t, |t| > 2.26, p < 0.05, matched seeds). The scientific question: does any
graph-geometric criterion -- especially the magnitude-decoupled `ollivier_topo` --
beat magnitude, or is decoupled geometry simply uninformative (~ random)?

    python scripts/plot_sparse_prune.py --results-dir results_sparse --config mnist_sparse_ip
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

# Fixed palette / order; magnitude = baseline, ollivier_topo = the decoupled signal.
STYLE = {
    "magnitude":     ("#d9761a", "-",  "o"),
    "random":        ("#7d838f", ":",  "x"),
    "forman":        ("#2ca02c", "--", "s"),
    "ollivier":      ("#9467bd", "--", "^"),
    "ollivier_topo": ("#1f77b4", "-",  "D"),
    "ollivier_neural": ("#e377c2", "-", "P"),
}


def load(rdir: str, cfg: str):
    R = [json.loads(Path(f).read_text())
         for f in sorted(glob.glob(f"{rdir}/{cfg}_seed*.json"))]
    if not R:
        raise SystemExit(f"no data for {cfg} in {rdir}")
    crits = list(R[0]["iterative_prune_curves"].keys())
    data = {}
    for c in crits:
        acc = np.array([[x["test_acc"] for x in r["iterative_prune_curves"][c]] for r in R])
        data[c] = acc
    spars = np.array([[x["sparsity"] for x in r["iterative_prune_curves"][crits[0]]]
                      for r in R]).mean(0)
    return data, spars, len(R)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_sparse")
    ap.add_argument("--config", default="mnist_sparse_ip")
    args = ap.parse_args()

    data, x, n = load(args.results_dir, args.config)
    tcrit = 2.262 if n == 10 else 2.26

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    base = data.get("magnitude")
    for c, acc in data.items():
        color, ls, mk = STYLE.get(c, ("#333333", "-", "."))
        m, sd = acc.mean(0), acc.std(0)
        ax.plot(x, m, ls=ls, marker=mk, ms=5, color=color, label=c)
        ax.fill_between(x, m - sd, m + sd, alpha=0.10, color=color)
        # significance vs magnitude (matched seeds)
        if base is not None and c not in ("magnitude",):
            d = acc - base
            t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(n) + 1e-12)
            for i in range(len(x)):
                if abs(t[i]) > tcrit:
                    ax.annotate("*", (x[i], m[i] - 0.006), ha="center",
                                color=color, fontsize=13)

    ax.set_xlabel("total edge sparsity (fixed sparse net, pruned further)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Sparse-by-construction pruning: geometry vs magnitude "
                 f"(n = {n})\n* = significant vs magnitude (p<0.05, paired)")
    ax.grid(alpha=0.3)
    ax.legend(title="criterion", fontsize=9)
    fig.tight_layout()
    out = Path(args.results_dir) / f"{args.config}_sparse_prune.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy (thesis)
    print(f"saved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()
