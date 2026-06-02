#!/usr/bin/env python3
"""
Run a baseline experiment across multiple seeds.

    python scripts/run_baseline.py --config mnist_small_uniform --seeds 0 1 2
    python scripts/run_baseline.py --config cifar_main_uniform  --seeds 0 1 2 3 4

Each (config, seed) pair writes its own JSON to results/, and a summary with
mean +/- std across seeds is printed and saved. The std indicates whether a
difference between configs is meaningful or within noise.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# make `src` importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.data import get_data
from src.metrics import summarize_config
from src.models import MLP
from src.train import train
from src.utils import set_seed, get_device


def run_one(cfg, seed: int, data_root: str, results_dir: Path,
            device: "torch.device | None" = None) -> dict:
    print(f"\n=== {cfg.name} | seed {seed} ===")
    set_seed(seed)
    device = device if device is not None else get_device()

    data = get_data(cfg.dataset, data_root=data_root,
                    batch_size=cfg.batch_size, val_fraction=cfg.val_fraction,
                    seed=seed)
    model = MLP(widths=cfg.widths, dropout_kind=cfg.dropout_kind,
                p=cfg.p, activation=cfg.activation)
    print(f"  params: {model.num_parameters():,} | device: {device}")

    record = {**cfg.to_dict(), "seed": seed}
    result = train(model, data.train, data.val, data.test, device,
                   epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
                   config=record, epoch_hook=None)

    out = {
        "config": record,
        "test_acc": result.test_acc,
        "train_acc_eval": result.train_acc_eval,
        "best_val_acc": result.best_val_acc,
        "best_epoch": result.best_epoch,
        "train_seconds": result.train_seconds,
        "total_seconds": result.total_seconds,
        "history": result.history,
    }
    path = results_dir / f"{cfg.name}_seed{seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  saved -> {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="name from src/config.py REGISTRY")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--device", default=None,
                    help="force a device (cpu|cuda|mps); default = auto-detect")
    args = ap.parse_args()

    cfg = get_config(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device(args.device) if args.device else None
    runs = [run_one(cfg, s, args.data_root, results_dir, device) for s in args.seeds]

    # Metric families computed by the shared metrics module so the summary
    # always agrees with the analysis report.
    summary = summarize_config(runs)
    (results_dir / f"{cfg.name}_summary.json").write_text(
        json.dumps(summary, indent=2))

    perf = summary["performance"]["test_acc"]
    gen = summary["generalization"]["gap"]
    conv = summary["convergence"]
    e2t = conv["epochs_to_threshold"]
    print(f"\n{'='*60}")
    print(f"  {cfg.name}  (n={perf['n']} seeds)")
    print(f"  performance | test acc      : {perf['mean']:.4f} +/- {perf['std']:.4f}")
    if summary["generalization"]["gap_available"]:
        print(f"  gap         | train-test    : {gen['mean']:+.4f} +/- {gen['std']:.4f}")
    print(f"  convergence | ep to val>={conv['val_threshold']:.2f}: "
          f"{e2t['mean']:.1f} +/- {e2t['std']:.1f}")
    print(f"              | val AUC (mean) : {conv['val_auc']['mean']:.4f}")
    print(f"              | train seconds  : {conv['train_seconds']['mean']:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
