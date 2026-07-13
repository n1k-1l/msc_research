#!/usr/bin/env python3
"""
Plot pruning-robustness (accuracy-vs-sparsity) for Targeted-Dropout configs.

Reads a results dir, groups the per-seed `prune_curve` fields by config, and
overlays the mean +/- std curve for each criterion (forman / magnitude / random).
Also reports the mean `pearson_curv_mag` (curvature-vs-magnitude correlation rho)
per config -- the diagnostic for how tightly the two are coupled, i.e. whether
curvature is more than a magnitude statistic here.

    python scripts/plot_prune.py --results-dir results_td
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_td")
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    curves: dict[str, list] = defaultdict(list)   # config -> [prune_curve per seed]
    spear: dict[str, list] = defaultdict(list)    # config -> [mean rho per seed]
    for f in sorted(rdir.glob("*_seed*.json")):
        d = json.loads(f.read_text())
        name = d["config"]["name"]
        if "prune_curve" in d:
            curves[name].append(d["prune_curve"])
        log = d.get("curvature_log") or []
        if log:
            vals = [l["pearson_curv_mag"] for l in log[-1]["layers"]
                    if "pearson_curv_mag" in l]
            if vals:
                spear[name].append(float(np.mean(vals)))

    if not curves:
        print(f"no prune_curve data found in {rdir}")
        return

    # Shared poster palette: same criterion = same colour across every figure.
    col = {"forman": "#1f77b4", "magnitude": "#ff7f0e", "random": "#7f7f7f"}
    plt.figure(figsize=(7, 5))
    print(f"\nPruning robustness ({rdir}):")
    for name in sorted(curves):
        runs = curves[name]
        spars = [p["sparsity"] for p in runs[0]]
        accs = np.array([[p["test_acc"] for p in run] for run in runs])  # (seeds, sparsity)
        mean, std = accs.mean(0), accs.std(0)
        label = name.replace("mnist_small_td_", "")
        sp = np.mean(spear[name]) if spear.get(name) else float("nan")
        c = col.get(label)
        plt.plot(spars, mean, marker="o", color=c, label=f"{label} (rho={sp:.2f})")
        plt.fill_between(spars, mean - std, mean + std, alpha=0.2, color=c)
        print(f"\n  {name}  (pearson curv~mag rho = {sp:.3f}, n={accs.shape[0]} seeds)")
        for i, s in enumerate(spars):
            print(f"    sparsity {s:.1f}: {mean[i]:.4f} +/- {std[i]:.4f}")

    plt.xlabel("sparsity (fraction of hidden units pruned)")
    plt.ylabel("test accuracy")
    plt.title("Targeted Dropout: pruning robustness by criterion\n(MNIST MLP, mean +/- std)")
    plt.legend()
    plt.grid(alpha=0.3)
    out = rdir / "prune_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector copy (thesis)
    print(f"\nsaved -> {out} (+.pdf)")


if __name__ == "__main__":
    main()
