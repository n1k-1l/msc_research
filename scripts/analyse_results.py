#!/usr/bin/env python3
"""
Analyse results: turn results/*_seed*.json into a report.

Reads every per-seed result file, groups by config, and reports the three
metric families (performance, generalization gap, convergence) using the shared
definitions in src/metrics.py, so nothing is recomputed with a private formula
here. Output:

  * results/analysis.md          -- markdown tables, one section per family
  * results/plots/<config>.png   -- per-config learning curve (val + train)
  * results/plots/<group>_val.png -- val-curve overlay per (dataset, arch),
                                      the convergence comparison figure

    python scripts/analyse_results.py
    python scripts/analyse_results.py --results-dir ./results --no-plots

Plots need matplotlib; if it is missing the tables still render and plotting is
skipped with a notice.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import summarize_config, VAL_THRESHOLD


def load_runs(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Group per-seed result files by config name (ignores *_summary.json)."""
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(results_dir.glob("*_seed*.json")):
        run = json.loads(path.read_text())
        by_config[run["config"]["name"]].append(run)
    for runs in by_config.values():
        runs.sort(key=lambda r: r["config"]["seed"])
    return by_config


def _fmt(stat: Dict[str, Any], places: int = 4, signed: bool = False) -> str:
    if stat["mean"] is None:
        return "n/a"
    m, s = stat["mean"], stat["std"]
    sign = "+" if signed else ""
    return f"{m:{sign}.{places}f} ± {s:.{places}f}"


def build_markdown(summaries: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Baseline results", ""]
    lines.append(f"Configs: {len(summaries)}. "
                 "Every cell is mean ± population std across seeds.")
    lines.append("")

    # ---- Family 1: performance ----
    lines += ["## 1. Performance", "",
              "| Config | Dataset | Seeds | Test accuracy |",
              "|---|---|---|---|"]
    for s in summaries:
        perf = s["performance"]["test_acc"]
        lines.append(f"| {s['config_name']} | {s['dataset']} | {perf['n']} "
                     f"| {_fmt(perf)} |")
    lines.append("")

    # ---- Family 2: generalization gap ----
    lines += ["## 2. Generalization gap (train−test, eval mode)", "",
              "Positive gap = overfitting. Train accuracy here is eval-mode "
              "(dropout off) so it is comparable to test.", "",
              "| Config | Train (eval) | Test | Gap |",
              "|---|---|---|---|"]
    for s in summaries:
        g = s["generalization"]
        note = "" if g["gap_available"] else " ⚠️ legacy run"
        lines.append(f"| {s['config_name']} | {_fmt(g['train_acc_eval'])} "
                     f"| {_fmt(g['test_acc'])} | {_fmt(g['gap'], signed=True)}{note} |")
    lines.append("")

    # ---- Family 3: convergence ----
    lines += ["## 3. Convergence speed", "",
              "Epochs to reach the val-accuracy threshold (95% MNIST / 50% "
              "CIFAR), normalized val-AUC (= mean val accuracy over training, "
              "higher is faster), and training wall-clock (excludes final "
              "eval).", "",
              "| Config | Epochs to threshold | Val AUC (mean) | Train seconds |",
              "|---|---|---|---|"]
    for s in summaries:
        c = s["convergence"]
        thr = c["val_threshold"]
        lines.append(
            f"| {s['config_name']} (≥{thr:.2f}) | {_fmt(c['epochs_to_threshold'], places=1)} "
            f"| {_fmt(c['val_auc'])} | {_fmt(c['train_seconds'], places=1)} |")
    lines.append("")
    return "\n".join(lines)


def _arch_key(run: Dict[str, Any]) -> str:
    cfg = run["config"]
    return f"{cfg['dataset']}_{'x'.join(str(w) for w in cfg['widths'])}"


def make_plots(by_config: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots "
              "(tables are unaffected). Install matplotlib to enable.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0

    # per-config learning curve: val (eval) vs train (dropout-on), seeds averaged
    for name, runs in by_config.items():
        epochs = [r["epoch"] for r in runs[0]["history"]]
        val = _mean_curve(runs, "val_acc")
        train = _mean_curve(runs, "train_acc")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, val, label="val acc (eval)")
        ax.plot(epochs, train, label="train acc (dropout on)", linestyle="--")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
        ax.set_title(name); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig); n += 1

    # convergence comparison: overlay val curves of configs sharing an arch
    groups: Dict[str, List[str]] = defaultdict(list)
    for name, runs in by_config.items():
        groups[_arch_key(runs[0])].append(name)
    for gkey, names in groups.items():
        if len(names) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in sorted(names):
            runs = by_config[name]
            epochs = [r["epoch"] for r in runs[0]["history"]]
            ax.plot(epochs, _mean_curve(runs, "val_acc"), label=name)
        dataset = by_config[names[0]][0]["config"]["dataset"].lower()
        ax.axhline(VAL_THRESHOLD[dataset], color="grey", linestyle=":",
                   label=f"threshold {VAL_THRESHOLD[dataset]:.2f}")
        ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy")
        ax.set_title(f"convergence: {gkey}"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{gkey}_val.png", dpi=120)
        plt.close(fig); n += 1

    return n


def _mean_curve(runs: List[Dict[str, Any]], key: str) -> List[float]:
    """Mean of `key` across seeds at each epoch (assumes aligned epoch counts)."""
    per_seed = [[rec[key] for rec in r["history"]] for r in runs]
    length = min(len(s) for s in per_seed)
    return [sum(s[i] for s in per_seed) / len(per_seed) for i in range(length)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    by_config = load_runs(results_dir)
    if not by_config:
        print(f"no *_seed*.json files found in {results_dir}")
        return

    summaries = [summarize_config(runs)
                 for _, runs in sorted(by_config.items())]
    md = build_markdown(summaries)
    (results_dir / "analysis.md").write_text(md)
    print(md)
    print(f"\nwrote {results_dir / 'analysis.md'}")

    if not args.no_plots:
        n = make_plots(by_config, results_dir / "plots")
        if n:
            print(f"wrote {n} plot(s) to {results_dir / 'plots'}")


if __name__ == "__main__":
    main()
