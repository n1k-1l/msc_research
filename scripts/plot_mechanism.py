#!/usr/bin/env python3
"""
Mechanism study (E14) figure: the collapse, explained and removed.

Top row — the causal ablation. (a) accuracy vs sparsity on heavy-tailed masks
(density 0.5, sigma=1.0): forman collapses from ~0.65 while forman_dc (degree term
removed by construction) stays on magnitude. (b) the same on the ER-0.15 protocol
(E10). (c) the coupling accounting along the forman arm: F decouples from |w| as
degree heterogeneity grows; F_dc never does — Forman = magnitude (+) degree, and
degree is the only part that decouples.

Bottom row — the autopsy (what each criterion kills, per prune step, sigma=1.0).
(d) endpoint degree of pruned edges relative to the surviving average: forman
targets low-degree vertices from the very first round, before any accuracy damage.
(e) where the pruned edges sat in the survivor |w| ranking: forman drifts from
deadwood (bottom 2%) to the middle of the distribution (~0.42) — it ends up
executing functional weights on thinly-connected paths.

    python scripts/plot_mechanism.py --mech-dir results_mech --hetero-dir results_mech_hetero
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

# Shared palette: same criterion = same colour across every figure.
STYLE = {
    "magnitude": ("#d9761a", "-",  "o", "magnitude"),
    "forman":    ("#2ca02c", "--", "s", "forman"),
    "forman_dc": ("#1f77b4", "-",  "D", "forman (degree-corrected)"),
    "random":    ("#7d838f", ":",  "x", "random"),
}
SIGMA = 1.0     # hetero regime shown in the figure (sigma=1.5 is near-identical)


def load(pattern: str, alias: dict[str, str] | None = None):
    """{criterion: acc[seed, round]}, plus the per-round records of every seed."""
    R = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(pattern))]
    if not R:
        raise SystemExit(f"no data for {pattern}")
    alias = alias or {}
    curves, recs = {}, {}
    for c in R[0]["iterative_prune_curves"]:
        cc = alias.get(c, c)
        curves[cc] = np.array([[x["test_acc"] for x in r["iterative_prune_curves"][c]]
                               for r in R])
        recs[cc] = [r["iterative_prune_curves"][c] for r in R]
    spars = np.array([x["sparsity"] for x in R[0]["iterative_prune_curves"]
                      [list(R[0]["iterative_prune_curves"])[0]]])
    return spars, curves, recs


def acc_panel(ax, spars, curves, title, xlabel):
    for c, (col, ls, mk, lab) in STYLE.items():
        if c not in curves:
            continue
        a = curves[c]
        ax.plot(spars, a.mean(0), ls, color=col, marker=mk, label=lab)
        ax.fill_between(spars, a.mean(0) - a.std(0), a.mean(0) + a.std(0),
                        alpha=0.15, color=col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def forensic_series(recs, key_fn):
    """Per-seed, per-round n_pruned-weighted mean of a forensic quantity."""
    out = []
    for seed_recs in recs:
        row = []
        for rec in seed_recs:
            fs = [f for f in rec.get("forensics", []) if f.get("n_pruned", 0) > 0]
            n = sum(f["n_pruned"] for f in fs)
            row.append(sum(key_fn(f) * f["n_pruned"] for f in fs) / n if n
                       else np.nan)
        out.append(row)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech-dir", default="results_mech")
    ap.add_argument("--hetero-dir", default="results_mech_hetero")
    ap.add_argument("--out", default=None, help="output stem (default: <mech-dir>/mechanism)")
    args = ap.parse_args()

    sp_h, cv_h, rec_h = load(f"{args.hetero_dir}/hetero_s{SIGMA:g}_seed*.json",
                             alias={"curvature": "forman"})
    sp_e, cv_e, rec_e = load(f"{args.mech_dir}/mnist_sparse_mech_seed*.json")

    fig = plt.figure(figsize=(13, 12.5))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axL = fig.add_subplot(gs[2, 1])
    axL.axis("off")

    # (a) / (b) accuracy: the ablation removes the collapse in both regimes
    acc_panel(axA, sp_h, cv_h,
              f"(a) heavy-tailed ($\\sigma$={SIGMA:g})",
              "total edge sparsity")
    acc_panel(axB, sp_e, cv_e, "(b) ER topology",
              "total edge sparsity")
    axB.set_ylabel("")

    # (c) coupling along the forman arm: F decouples, F_dc never does
    for spars, recs, regime, alpha in [(sp_h, rec_h, "heavy-tailed", 1.0),
                                       (sp_e, rec_e, "ER", 0.45)]:
        f = np.array([[np.nanmean(x["spearman_curv_mag"]) for x in sr]
                      for sr in recs["forman"]])
        fdc = np.array([[np.nanmean(x["spearman_dc_mag"]) for x in sr]
                        for sr in recs["forman"]])
        axC.plot(spars, f.mean(0), "--", marker="s", alpha=alpha,
                 color=STYLE["forman"][0],
                 label=f"$F$, {regime}")
        axC.plot(spars, fdc.mean(0), "-", marker="D", alpha=alpha,
                 color=STYLE["forman_dc"][0],
                 label=f"$F_{{dc}}$, {regime}")
    axC.set_xlabel("total edge sparsity")
    axC.set_ylabel(r"$\rho_S$ with $|w|$")
    axC.set_title(r"(c) coupling with $|w|$")
    axC.grid(alpha=0.3)
    axC.legend(loc="upper left", fontsize=13)

    # (d) / (e) autopsy, heavy-tailed regime
    for c, (col, ls, mk, lab) in STYLE.items():
        dg = forensic_series(rec_h[c],
                             lambda f: f["pruned_deg_mean"] / f["surv_deg_mean"])
        pc = forensic_series(rec_h[c], lambda f: f["pruned_mag_pctile_mean"])
        axD.plot(sp_h, np.nanmean(dg, 0), ls, color=col, marker=mk, label=lab)
        axE.plot(sp_h, np.nanmean(pc, 0), ls, color=col, marker=mk, label=lab)
    axD.axhline(1.0, color="k", lw=0.8)
    axD.set_xlabel("total edge sparsity")
    axD.set_ylabel("kill degree / survivor mean")
    axD.set_title("(d) endpoint degree of kills")
    axD.grid(alpha=0.3)
    axE.axhline(0.5, color="k", lw=0.8)
    axE.set_xlabel("total edge sparsity")
    axE.set_ylabel("kill $|w|$ percentile")
    axE.set_title(r"(e) kill $|w|$ percentile")
    axE.grid(alpha=0.3)
    handles, labels = axD.get_legend_handles_labels()
    axL.legend(handles, labels, loc="center", fontsize=19, frameon=False,
               title="criterion", title_fontsize=19)

    out = Path(args.out) if args.out else Path(args.mech_dir) / "mechanism"
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")   # vector copy (thesis)
    print(f"saved -> {out}.png (+.pdf)")


if __name__ == "__main__":
    main()
