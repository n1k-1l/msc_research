"""
Training loop.

A single loop is shared by every experiment, so the dropout strategy is the
only thing that varies between runs. Two design points:

1. Epoch hook. The loop accepts an optional epoch_hook(model, epoch) callback,
   fired after each training epoch. It is None for the baselines. Curvature-
   aware dropout uses it to recompute curvature and update each
   PerNeuronDropout's probabilities, so the loop itself does not change.

2. Per-epoch logging. Train and validation accuracy are recorded every epoch so
   the train-vs-test gap and convergence curves are available without
   re-running.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, List, Dict, Any
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_acc: float
    seconds: float


@dataclass
class RunResult:
    config: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)
    test_acc: float = 0.0
    train_acc_eval: float = 0.0   # train acc in eval mode on the best ckpt; for the gap
    best_val_acc: float = 0.0
    best_epoch: int = 0
    train_seconds: float = 0.0    # training wall-clock, excludes final eval passes
    total_seconds: float = 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy. Sets eval mode (dropout becomes identity)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
    epoch_hook: Optional[Callable[[nn.Module, int], None]] = None,
    verbose: bool = True,
) -> RunResult:
    """
    Train and return a RunResult.

    epoch_hook(model, epoch) is called after each epoch; pass None for the
    baselines (see the module docstring).

    The test set is evaluated once, at the end, on the best-validation
    checkpoint, and is never used for tuning. The train set is also evaluated
    once at the end in eval mode (dropout off); that train_acc_eval is the train
    number comparable to test and is what the generalization gap is built from.
    The per-epoch train_acc is measured with dropout on and reflects the
    optimisation view only.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    result = RunResult(config=config or {})
    best_state = None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = correct = total = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * y.numel()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        rec = EpochRecord(epoch, train_loss, train_acc, val_acc,
                          time.time() - t0)
        result.history.append(asdict(rec))

        if val_acc > result.best_val_acc:
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            # CPU copy so the checkpoint survives even when GPU memory is tight.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if verbose:
            print(f"  epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"train {train_acc:.4f} | val {val_acc:.4f} | "
                  f"{rec.seconds:.1f}s")

        # Epoch hook: no-op for the baselines.
        if epoch_hook is not None:
            epoch_hook(model, epoch)

    # Training wall-clock stops here, before the final eval passes, so the
    # convergence metric is not inflated by evaluation time.
    result.train_seconds = time.time() - t_start

    # Final evaluation on the best-validation checkpoint. Test and train are
    # both measured in eval mode (dropout off) so the gap is comparable.
    # train_loader has drop_last=True, so up to (batch_size - 1) samples are
    # skipped; negligible for an accuracy estimate over the train set.
    if best_state is not None:
        model.load_state_dict(best_state)
    result.test_acc = evaluate(model, test_loader, device)
    result.train_acc_eval = evaluate(model, train_loader, device)
    result.total_seconds = time.time() - t_start

    if verbose:
        gap = result.train_acc_eval - result.test_acc
        print(f"  >> best val {result.best_val_acc:.4f} @ epoch "
              f"{result.best_epoch} | test {result.test_acc:.4f} | "
              f"train(eval) {result.train_acc_eval:.4f} | gap {gap:+.4f}")
    return result
