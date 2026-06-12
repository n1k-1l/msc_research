#!/usr/bin/env python3
"""
Curvature-vs-magnitude diagnostics (the "why do they behave the same?" analysis).

On a fully connected net the hypothesis is that Forman curvature collapses to a
near-function of weight magnitude, which would explain why curvature-TD and
magnitude-TD give near-identical pruning curves. This script quantifies that from
the per-neuron `curv` (Forman kappa) and `mag` (L2 norm) arrays logged each
recompute in `curvature_log` (see DropoutProbHook), producing:

  1. Correlation vs training: Pearson (linear) and Spearman (rank) correlation
     between curvature and magnitude per hidden layer, mean +/- std over seeds,
     across every recompute checkpoint -- shows whether the relationship persists/
     strengthens through optimisation. Both are computed here from the raw curv/mag
     arrays, so this works regardless of which (if any) scalar the hook logged.
  2. Scatter plots: curvature vs magnitude per neuron at six evenly-spaced
     checkpoints, pooled over seeds, each annotated with Pearson and Spearman.

    python scripts/plot_curv_mag.py --results-dir results_td_cpu \
        --config mnist_small_td_forman

A strong, persistent correlation here is itself the result: it explains the null.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_logs(rdir: Path, config: str) -> list[list[dict]]:
    """Per-seed curvature_log lists for one config (skipping any without one)."""
    logs = []
    for f in sorted(rdir.glob(f"{config}_seed*.json")):
        d = json.loads(f.read_text())
        log = d.get("curvature_log")
        if log:
            logs.append(log)
    if not logs:
        raise SystemExit(f"no curvature_log with data found for {config} in {rdir}")
    return logs


def _pearson(x, y) -> float:
    """Pearson (linear) correlation, computed from raw arrays."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float:
    """Spearman (rank) correlation = Pearson on ranks, computed from raw arrays."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    rx = x.argsort().argsort()
    ry = y.argsort().argsort()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def correlation_figure(logs: list[list[dict]], out: Path, title: str) -> None:
    """Pearson and Spearman rho(curv, mag) vs epoch, per layer, mean +/- std over seeds.

    Both coefficients are recomputed from the logged raw per-neuron curv/mag arrays
    and overlaid on the same axes: Pearson is linear strength, Spearman is the
    rank/monotone strength (the one that decides which units TD prunes).
    """
    epochs = [e["epoch"] for e in logs[0]]
    n_layers = len(logs[0][0]["layers"])

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4.5), squeeze=False)
    for L in range(n_layers):
        ax = axes[0][L]
        for stat, name, color in ((_pearson, r"Pearson $\rho$", "C1"),
                                  (_spearman, r"Spearman $\rho$", "C0")):
            # (seeds, epochs); a checkpoint may legitimately be NaN (constant input)
            vals = np.array([[stat(e["layers"][L]["curv"], e["layers"][L]["mag"])
                              for e in log] for log in logs], dtype=float)
            mean = np.nanmean(vals, axis=0)
            std = np.nanstd(vals, axis=0)
            ax.plot(epochs, mean, marker="o", ms=3, color=color, label=name)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("epoch")
        ax.set_ylabel("correlation(curvature, magnitude)")
        ax.set_title(f"hidden layer {L}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")


def scatter_figure(logs: list[list[dict]], out: Path, title: str) -> None:
    """Curvature vs magnitude per neuron at six evenly-spaced checkpoints, pooled.

    Six rows (checkpoints spanning training) x n_layers columns; points pooled over
    seeds, each panel annotated with Pearson (P) and Spearman (S) on the pooled points.
    """
    n_ckpt = len(logs[0])
    n_panels = min(6, n_ckpt)
    ckpts = sorted(set(np.linspace(0, n_ckpt - 1, n_panels).round().astype(int).tolist()))
    n_layers = len(logs[0][0]["layers"])

    fig, axes = plt.subplots(len(ckpts), n_layers,
                             figsize=(5 * n_layers, 3.2 * len(ckpts)), squeeze=False)
    for r, ci in enumerate(ckpts):
        epoch = logs[0][ci]["epoch"]
        for L in range(n_layers):
            ax = axes[r][L]
            mag = np.concatenate([np.asarray(log[ci]["layers"][L]["mag"]) for log in logs])
            curv = np.concatenate([np.asarray(log[ci]["layers"][L]["curv"]) for log in logs])
            ax.scatter(mag, curv, s=6, alpha=0.4)
            ax.set_title(f"layer {L}, epoch {epoch}  "
                         f"(P={_pearson(mag, curv):.2f}, S={_spearman(mag, curv):.2f})")
            ax.set_xlabel("magnitude (L2 norm of incident weights)")
            ax.set_ylabel("Forman curvature kappa")
            ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_td_cpu")
    ap.add_argument("--config", default="mnist_small_td_forman",
                    help="config whose curvature_log to analyse")
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    logs = load_logs(rdir, args.config)
    print(f"{args.config}: {len(logs)} seeds, "
          f"{len(logs[0])} checkpoints, {len(logs[0][0]['layers'])} hidden layers")
    correlation_figure(logs, rdir / f"{args.config}_corr_vs_epoch.png",
                        f"curvature-vs-magnitude correlation over training\n{args.config}")
    scatter_figure(logs, rdir / f"{args.config}_scatter.png",
                   f"curvature vs magnitude per neuron\n{args.config}")


if __name__ == "__main__":
    main()
