# Curvature-Aware Dropout — Experiment Harness
Contains basline MLP experiments 

It is built so that adding curvature-aware dropout is a small,
contained change — see "Extending this" below.

## Layout

```
src/
  dropout.py   pluggable dropout modules (the key abstraction)
  models.py    one configurable MLP class
  data.py      MNIST / CIFAR-10 loaders with train/val/test split
  train.py     one training loop, with an end-of-epoch hook
  config.py    named experiment configs (the only thing that varies)
  utils.py     seeding / device
scripts/
  run_baseline.py   run a config across seeds, aggregate mean ± std
configs/     (reserved for YAML configs)
results/     per-run JSON + per-config summary JSON
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch matching GPU's driver — see
https://pytorch.org/get-started/locally/ for the exact command.

## Run the baselines

From the repo root:

```bash
# MNIST sanity checks
python scripts/run_baseline.py --config mnist_small_uniform --seeds 0 1 2
python scripts/run_baseline.py --config mnist_small_nodrop  --seeds 0 1 2

# the equivalence sanity check (should match mnist_small_uniform)
python scripts/run_baseline.py --config mnist_small_perneuron --seeds 0 1 2

# CIFAR-10 primary baseline
python scripts/run_baseline.py --config cifar_main_uniform --seeds 0 1 2 3 4
python scripts/run_baseline.py --config cifar_main_nodrop  --seeds 0 1 2 3 4
```

Available config names: see `REGISTRY` in `src/config.py`. Tags are
`mnist_small`, `mnist_deep`, `cifar_main`, each with `_uniform`, `_nodrop`,
`_perneuron`.

## Expected numbers (sanity check)

These are ballpark targets. 

| Config                    | Test accuracy (rough) |
|---------------------------|-----------------------|
| MNIST, MLP, uniform drop  | ~98%                  |
| MNIST, MLP, no dropout    | ~98% (train acc ~100%)|
| CIFAR-10, MLP, uniform    | ~55–60%               |
| CIFAR-10, MLP, no dropout | ~52–58%, larger train/test gap |

A plain MLP on CIFAR-10 capping in the high-50s is **expected** — MLPs throw
away spatial structure. That low ceiling is also why CIFAR-10 is a useful
stress test for the curvature method.

Two things to look for in the baselines, both feeding directly into the
proposal's metrics:
- `no dropout` should show a **larger train-minus-test gap** than uniform
  dropout. If it does not, dropout is not regularising and something is off.
- Across seeds, note the **std** of test accuracy. That std sets the bar a
  curvature-vs-uniform improvement must clear to be meaningful.

## Extending this (curvature-aware dropout)

The repo is deliberately shaped so this is contained:

1. `PerNeuronDropout` already holds a per-neuron probability buffer and a
   `set_probs()` method. Nothing in it needs to change.
2. Write a new `src/curvature.py`: build the weighted weight-graph per layer
   pair, compute Forman–Ricci curvature per edge, average to per-neuron
   curvature (Eq. 3), map to probabilities via the shifted sigmoid (Eq. 4),
   and renormalise to the `pbase` budget.
3. Wrap that as an `epoch_hook(model, epoch)`: after `warmup_epochs`, and
   every `recompute_every` epochs, recompute and call `set_probs` on each
   module from `model.dropout_modules()`.
4. Add `dropout_kind="curvature"` configs in `src/config.py`.

The training loop, model, data, and runner do **not** change. That is the
point of the baseline being built this way.
```
