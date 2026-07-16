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
import dataclasses
import json
import sys
from pathlib import Path

# make `src` importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.curvature import make_dropout_prob_hook, get_source
from src.data import get_data
from src.prune_eval import prune_and_evaluate, prune_by_criteria
from src.iterative_prune import iterative_prune_by_criteria, CRITERIA
from src.metrics import summarize_config
from src.models import MLP
from src.train import train
from src.utils import set_seed, get_device


def run_one(cfg, seed: int, data_root: str, results_dir: Path,
            device: "torch.device | None" = None, num_workers: int = 2,
            cross_prune: bool = False) -> dict:
    print(f"\n=== {cfg.name} | seed {seed} ===")
    set_seed(seed)
    device = device if device is not None else get_device()

    data = get_data(cfg.dataset, data_root=data_root,
                    batch_size=cfg.batch_size, val_fraction=cfg.val_fraction,
                    num_workers=num_workers, seed=seed)
    if getattr(cfg, "arch", "mlp") == "lenet":
        from src.models import LeNet
        model = LeNet(num_classes=cfg.widths[-1], activation=cfg.activation,
                      conv_channels=tuple(cfg.extra.get("conv_channels", (6, 16))))
    else:
        model = MLP(widths=cfg.widths, dropout_kind=cfg.dropout_kind,
                    p=cfg.p, activation=cfg.activation)
    print(f"  params: {model.num_parameters():,} | device: {device}")

    # Hook-driven per-neuron dropout (curvature method and its controls) is added
    # through the epoch hook when prob_source is set; baselines pass None.
    hook = None
    if cfg.prob_source is not None:
        hook = make_dropout_prob_hook(
            p_base=cfg.p, alpha=cfg.alpha, beta=cfg.beta,
            warmup_epochs=cfg.warmup_epochs, recompute_every=cfg.recompute_every,
            standardize=cfg.standardize,
            source=get_source(cfg.prob_source),
            mapping=cfg.prob_mapping, gamma=cfg.target_gamma,
            drop=cfg.target_drop, direction=cfg.target_direction)

    # Sparse-by-construction topology: a fixed random mask per layer, seeded per run,
    # held through training (weight_masks) and used as the starting point for pruning.
    sparse_masks = None
    if getattr(cfg, "sparse_init", False):
        import torch as _torch
        from src.iterative_prune import make_sparse_masks
        gen = _torch.Generator().manual_seed(seed)
        sparse_masks = make_sparse_masks(model.linear_layers(), cfg.sparse_density, gen)
        dens = [round(float(m.float().mean()), 3) for m in sparse_masks]
        print(f"  sparse topology densities: {dens}")

    record = {**cfg.to_dict(), "seed": seed}
    result = train(model, data.train, data.val, data.test, device,
                   epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
                   config=record, epoch_hook=hook, weight_masks=sparse_masks)

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
    # Curvature distribution evolution (proposal secondary metric); append-only.
    if hook is not None:
        out["curvature_log"] = hook.log
    # Pruning robustness (accuracy-vs-sparsity): for Targeted-Dropout configs, prune
    # the most-prunable units by the same score/direction used in training and
    # re-evaluate. train() leaves `model` holding the best-val checkpoint.
    sparsities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if cfg.prob_mapping == "targeted" and cfg.prob_source is not None:
        out["prune_curve"] = prune_and_evaluate(
            model, get_source(cfg.prob_source), cfg.target_direction,
            sparsities=sparsities, loader=data.test, device=device)
    # Unconfounded criterion comparison (--cross-prune): prune this one trained
    # model by forman / magnitude / random and store each curve. On a neutral net
    # (uniform / no dropout) this isolates the pruning criterion from any training
    # bias toward it. Works for any config, append-only field.
    if cross_prune:
        out["prune_curves_by_criterion"] = prune_by_criteria(
            model, sparsities=sparsities, loader=data.test, device=device)
    # Iterative hard pruning: prune the trained dense net by each criterion with
    # fine-tuning between rounds (the non-uniform-connectivity divergence test).
    if getattr(cfg, "iterative_prune", False):
        criteria = getattr(cfg, "prune_criteria", None) or CRITERIA
        if getattr(cfg, "arch", "mlp") == "lenet":
            if cfg.prune_granularity == "unit":
                # whole-filter (structured): the CNN analogue of unit pruning
                from src.cnn_prune import filter_prune_by_criteria
                out["iterative_prune_curves"] = filter_prune_by_criteria(
                    model, loaders=(data.train, data.val, data.test),
                    device=device, schedule=cfg.prune_schedule,
                    finetune_epochs=cfg.finetune_epochs, lr=cfg.lr,
                    criteria=criteria)
            else:
                from src.cnn_prune import cnn_prune_by_criteria
                out["iterative_prune_curves"] = cnn_prune_by_criteria(
                    model, loaders=(data.train, data.val, data.test),
                    device=device, schedule=cfg.prune_schedule,
                    finetune_epochs=cfg.finetune_epochs,
                    lr=cfg.lr, criteria=criteria,
                    calib_batches=getattr(cfg, "calib_batches", 2))
        else:
            out["iterative_prune_curves"] = iterative_prune_by_criteria(
                model, loaders=(data.train, data.val, data.test), device=device,
                schedule=cfg.prune_schedule, finetune_epochs=cfg.finetune_epochs,
                lr=cfg.lr, granularity=cfg.prune_granularity,
                criteria=criteria, init_masks=sparse_masks,
                calib_batches=getattr(cfg, "calib_batches", 2))
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
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers; use 0 on macOS+MPS (worker/MPS deadlock)")
    ap.add_argument("--p", type=float, default=None, help="override p_base")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override sigmoid sensitivity alpha")
    ap.add_argument("--recompute-every", type=int, default=None,
                    help="override recompute interval Delta")
    ap.add_argument("--target-gamma", type=float, default=None,
                    help="override Targeted-Dropout targeting fraction gamma")
    ap.add_argument("--cross-prune", action="store_true",
                    help="also prune the trained model by forman/magnitude/random "
                         "and store each curve (unconfounded criterion comparison)")
    args = ap.parse_args()

    cfg = get_config(args.config)
    # Hyperparameter-sweep overrides: each provided value overrides the config and
    # adds a suffix to the saved name (e.g. _p30_a4_d10) so every grid point is a
    # distinct config in the results dir / analysis.
    overrides, suffix = {}, ""
    if args.p is not None:
        overrides["p"] = args.p
        suffix += f"_p{round(args.p * 100):02d}"
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
        suffix += f"_a{args.alpha:g}"
    if args.recompute_every is not None:
        overrides["recompute_every"] = args.recompute_every
        suffix += f"_d{args.recompute_every}"
    if args.target_gamma is not None:
        overrides["target_gamma"] = args.target_gamma
        suffix += f"_g{round(args.target_gamma * 100):02d}"
    if overrides:
        cfg = dataclasses.replace(cfg, name=cfg.name + suffix, **overrides)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device(args.device) if args.device else None
    runs = [run_one(cfg, s, args.data_root, results_dir, device, args.num_workers,
                    cross_prune=args.cross_prune)
            for s in args.seeds]

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
    # e2t mean is None when no seed reached the threshold (e.g. a run that
    # diverged) -- a real, reportable outcome, so don't crash formatting it.
    e2t_str = (f"{e2t['mean']:.1f} +/- {e2t['std']:.1f}"
               if e2t["mean"] is not None else "not reached")
    print(f"  convergence | ep to val>={conv['val_threshold']:.2f}: {e2t_str}")
    print(f"              | val AUC (mean) : {conv['val_auc']['mean']:.4f}")
    print(f"              | train seconds  : {conv['train_seconds']['mean']:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
