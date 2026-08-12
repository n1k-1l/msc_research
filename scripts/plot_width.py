#!/usr/bin/env python3
"""
Width dose-response study (E16) figure.

(a) Trained curvature<->magnitude coupling vs hidden width: the concentration
    prediction of Proposition 1 (coupling weakens as layers narrow), measured.
(b) Paired forman-TD minus magnitude-TD test accuracy across the prune sweep,
    one line per width: a systematic trade-off -- worse within the targeted
    range (<= gamma), better past it -- whose size scales inversely with width.
(c) The mechanism: magnitude-TD's targeted set freezes (~gamma of units ever
    targeted), curvature-TD's churns through most of the network, spreading
    robustness training past the nominal gamma set.

    python scripts/plot_width.py
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

WIDTHS = [(32, "results_width", "mnist_w32"), (64, "results_width", "mnist_w64"),
          (128, "results_width", "mnist_w128"), (256, "results_width", "mnist_w256"),
          (512, "results_td_cpu", "mnist_small")]
GAMMA = 0.5
MAG, FOR = "#d9761a", "#2ca02c"


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def load(d, pre, src):
    return [json.loads(p.read_text())
            for p in sorted(Path(d).glob(f"{pre}_td_{src}_seed*.json"))]


def targeted_sets(run, layer, key, high):
    sets = []
    for e in run["curvature_log"]:
        s = np.array(e["layers"][layer][key])
        k = int(round(GAMMA * s.size))
        idx = np.argsort(s)
        sets.append(set((idx[-k:] if high else idx[:k]).tolist()))
    return sets, s.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_width/width")
    args = ap.parse_args()

    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, :])
    ws = [w for w, _, _ in WIDTHS]

    # (a) coupling vs width
    for L, mk, lab in [(0, "o", "layer pair 0"), (1, "s", "layer pair 1")]:
        mean, std = [], []
        for w, d, pre in WIDTHS:
            rhos = []
            for r in load(d, pre, "forman"):
                last = r["curvature_log"][-1]["layers"][L]
                rhos.append(spearman(np.array(last["curv"]), np.array(last["mag"])))
            mean.append(np.mean(rhos)); std.append(np.std(rhos, ddof=1))
        axA.errorbar(ws, mean, yerr=std, marker=mk, color="#1f77b4",
                     ls="-" if L == 0 else "--", label=lab, capsize=3)
    axA.set_xscale("log", base=2)
    axA.set_xticks(ws); axA.set_xticklabels(ws)
    axA.set_xlabel("hidden width $h$ (layers $h$, $h/2$)")
    axA.set_ylabel(r"trained Spearman $\rho(\kappa, \|w\|)$")
    axA.set_title("(a) trained coupling")
    axA.grid(alpha=0.3); axA.legend()

    # (b) paired forman - magnitude across the sweep, per width
    cmap = plt.get_cmap("viridis")
    col = {w: cmap(i / (len(ws) - 1)) for i, w in enumerate(ws)}
    for w, d, pre in WIDTHS:
        f = np.array([[x["test_acc"] for x in r["prune_curve"]]
                      for r in load(d, pre, "forman")])
        m = np.array([[x["test_acc"] for x in r["prune_curve"]]
                      for r in load(d, pre, "magnitude")])
        sp = [x["sparsity"] for x in load(d, pre, "forman")[0]["prune_curve"]]
        diff = f - m
        axB.plot(sp, diff.mean(0), marker="o", color=col[w], label=f"$h={w}$")
        axB.fill_between(sp, diff.mean(0) - diff.std(0, ddof=1) / np.sqrt(len(diff)),
                         diff.mean(0) + diff.std(0, ddof=1) / np.sqrt(len(diff)),
                         alpha=0.15, color=col[w])
    axB.axhline(0, color="k", lw=0.8)
    axB.axvline(GAMMA, color="0.5", lw=0.8, ls=":")
    axB.annotate(r"$\gamma$", (GAMMA, axB.get_ylim()[0]), fontsize=18,
                 xytext=(GAMMA + 0.01, 0.9), textcoords=("data", "axes fraction"))
    axB.set_xlabel("pruned fraction (units)")
    axB.set_ylabel("accuracy difference")
    axB.set_title("(c) accuracy difference, forman-TD $-$ magnitude-TD")
    axB.grid(alpha=0.3)
    axB.legend(ncol=3, fontsize=14, loc="upper left", framealpha=0.9)

    # (c) fraction of units ever targeted, per arm
    for src, key, high, colr, lab in [("magnitude", "mag", False, MAG, "magnitude-TD"),
                                      ("forman", "curv", True, FOR, "forman-TD")]:
        mean, std = [], []
        for w, d, pre in WIDTHS:
            fr = []
            for r in load(d, pre, src):
                per_layer = []
                for L in range(2):
                    sets, n = targeted_sets(r, L, key, high)
                    per_layer.append(len(set().union(*sets)) / n)
                fr.append(np.mean(per_layer))
            mean.append(np.mean(fr)); std.append(np.std(fr, ddof=1))
        axC.errorbar(ws, mean, yerr=std, marker="o", color=colr, label=lab, capsize=3)
    axC.axhline(GAMMA, color="k", lw=0.8, ls=":",
                label=r"$\gamma$ (frozen set)")
    axC.set_xscale("log", base=2)
    axC.set_xticks(ws); axC.set_xticklabels(ws)
    axC.set_ylim(0.4, 1.02)
    axC.set_xlabel("hidden width $h$")
    axC.set_ylabel("")
    axC.set_title("(b) units ever targeted")
    axC.grid(alpha=0.3); axC.legend(fontsize=14)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved -> {out}.png (+.pdf)")


if __name__ == "__main__":
    main()
