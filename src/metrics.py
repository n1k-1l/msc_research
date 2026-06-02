"""
Metrics: the single definition of how results are scored.

Each metric is defined once here as a pure function over a parsed per-seed
result dict (the JSON that run_baseline.py writes). Both the run script (for its
printed summary) and the analysis script import these, so a given metric always
means the same thing.

Three families:

  1. Performance        -- final test accuracy on the best-validation checkpoint.
  2. Generalization gap -- train_acc_eval - test_acc, both measured in eval mode
                           (dropout off) on the same checkpoint so they are
                           comparable. The per-epoch train_acc logged during
                           training is dropout-suppressed and is not used for the
                           gap; train_acc_eval exists for that purpose.
  3. Convergence        -- epochs to reach a dataset-specific val-accuracy
                           threshold, the normalised AUC of the val curve (mean
                           val accuracy over training, comparable across runs
                           with different epoch counts), and training wall-clock
                           (kept separate from final-eval time).

A "run" is the dict parsed from one <config>_seed<k>.json file.

Aggregation across seeds uses population std (ddof=0), matching the original
summaries. The choice is made once here so it is applied consistently.
"""
from __future__ import annotations
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

# Val-accuracy target for the "epochs-to-threshold" convergence metric, by
# dataset. Picked well below the final accuracy so every run reaches it.
VAL_THRESHOLD: Dict[str, float] = {"mnist": 0.95, "cifar10": 0.50}


# ----------------------------------------------------------------------------
# Per-run metrics (one parsed JSON dict in, one number out)
# ----------------------------------------------------------------------------
def final_test_acc(run: Dict[str, Any]) -> float:
    """Final test accuracy on the best-validation checkpoint."""
    return float(run["test_acc"])


def generalization_gap(run: Dict[str, Any]) -> Optional[float]:
    """train_acc_eval - test_acc, both eval-mode on the best checkpoint.

    Returns None for legacy runs that predate train_acc_eval, so the caller can
    flag them instead of silently computing a dropout-biased gap.
    """
    train_eval = run.get("train_acc_eval")
    if train_eval is None:
        return None
    return float(train_eval) - float(run["test_acc"])


def epochs_to_val_threshold(run: Dict[str, Any],
                            threshold: Optional[float] = None) -> Optional[int]:
    """First epoch whose eval-mode val accuracy >= threshold.

    Returns None if the threshold was never reached (a valid, reportable
    outcome).
    """
    if threshold is None:
        threshold = VAL_THRESHOLD[run["config"]["dataset"].lower()]
    for rec in run["history"]:
        if rec["val_acc"] >= threshold:
            return int(rec["epoch"])
    return None


def val_auc(run: Dict[str, Any]) -> float:
    """Trapezoidal AUC of the val curve, normalised by (#epochs - 1).

    Normalising makes this the mean val accuracy over training, so runs with
    different epoch counts are on the same scale. Higher means good accuracy was
    reached sooner and held; complementary to the threshold-crossing epoch.
    """
    accs = [r["val_acc"] for r in run["history"]]
    n = len(accs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(accs[0])
    trapezoid = 0.5 * (accs[0] + accs[-1]) + sum(accs[1:-1])
    return float(trapezoid / (n - 1))


def train_wall_seconds(run: Dict[str, Any]) -> float:
    """Training wall-clock, excluding the final eval passes.

    Falls back to summing per-epoch seconds for legacy runs without
    train_seconds.
    """
    ts = run.get("train_seconds")
    if ts is not None:
        return float(ts)
    return float(sum(r["seconds"] for r in run["history"]))


# ----------------------------------------------------------------------------
# Aggregation across seeds
# ----------------------------------------------------------------------------
@dataclass
class Stat:
    mean: Optional[float]
    std: Optional[float]
    n: int                       # number of seeds that produced a value

    def as_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "n": self.n}


def aggregate(values: List[Optional[float]]) -> Stat:
    """Mean +/- population std over the non-None values."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return Stat(mean=None, std=None, n=0)
    return Stat(mean=mean(vals),
                std=(pstdev(vals) if len(vals) > 1 else 0.0),
                n=len(vals))


def summarize_config(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed runs for one config into the three metric families.

    Called by both run_baseline (after a sweep) and analyse_results, so the
    summary JSON and the analysis report stay consistent.
    """
    if not runs:
        raise ValueError("summarize_config got no runs")

    cfg = runs[0]["config"]
    dataset = cfg["dataset"].lower()
    threshold = VAL_THRESHOLD[dataset]

    gaps = [generalization_gap(r) for r in runs]
    return {
        "config_name": cfg["name"],
        "dataset": dataset,
        "seeds": [r["config"]["seed"] for r in runs],
        "performance": {
            "test_acc": aggregate([final_test_acc(r) for r in runs]).as_dict(),
        },
        "generalization": {
            "train_acc_eval": aggregate([r.get("train_acc_eval") for r in runs]).as_dict(),
            "test_acc": aggregate([final_test_acc(r) for r in runs]).as_dict(),
            "gap": aggregate(gaps).as_dict(),
            "gap_available": all(g is not None for g in gaps),
        },
        "convergence": {
            "val_threshold": threshold,
            "epochs_to_threshold": aggregate(
                [epochs_to_val_threshold(r, threshold) for r in runs]).as_dict(),
            "val_auc": aggregate([val_auc(r) for r in runs]).as_dict(),
            "train_seconds": aggregate([train_wall_seconds(r) for r in runs]).as_dict(),
        },
        "per_seed_test_acc": {r["config"]["seed"]: final_test_acc(r) for r in runs},
    }
