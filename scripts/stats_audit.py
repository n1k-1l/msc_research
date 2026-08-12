#!/usr/bin/env python3
"""
Statistical audit of every paired comparison reported in the thesis.

For each comparison: mean paired difference (percentage points), 95% CI on the
difference (t-interval, df = n-1), Cohen's d_z (mean/sd of the paired
differences), raw two-sided p, and Bonferroni-adjusted p within the comparison's
FAMILY. A family is one experiment x one criterion pair (i.e. one table or
figure panel): the set of tests a reader would scan together.

    python scripts/stats_audit.py
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load(pattern):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(pattern))]


def prune_curve(runs, key="prune_curve"):
    sp = [x["sparsity"] for x in runs[0][key]]
    acc = np.array([[x["test_acc"] for x in r[key]] for r in runs])
    return sp, acc


def ip_curve(runs, criterion):
    recs = [r["iterative_prune_curves"][criterion] for r in runs]
    sp = [round(x["target_sparsity"], 2) for x in recs[0]]
    acc = np.array([[x["test_acc"] for x in rec] for rec in recs])
    return sp, acc


def paired(a, b):
    """a, b: (n,) matched by seed. Returns dict of stats for diff = a - b."""
    d = a - b
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t = m / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if se > 0 else np.nan
    return {"n": n, "diff": m, "lo": m - tcrit * se, "hi": m + tcrit * se,
            "dz": m / sd if sd > 0 else np.nan, "t": t, "p": p}


def bonferroni(pvals):
    """Bonferroni adjusted p-values: raw p times family size, capped at 1."""
    p = np.asarray(pvals, dtype=float)
    return np.minimum(1.0, p * len(p))


FAMILIES = []


def family(name, rows):
    """rows: list of (label, a, b) with a, b seed-matched accuracy vectors."""
    tests = [(label, paired(a, b)) for label, a, b in rows]
    adj = bonferroni([t[1]["p"] for t in tests])
    for (label, st), pa in zip(tests, adj):
        st["p_bonf"] = pa
        st["label"] = label
    FAMILIES.append((name, [t[1] for t in tests]))


# ---- E5: TD curvature vs magnitude (main network) -------------------------
mag = load("results_td_cpu/mnist_small_td_magnitude_seed*.json")
cur = load("results_td_cpu/mnist_small_td_forman_seed*.json")
sp, am = prune_curve(mag)
_, ac = prune_curve(cur)
family("E5 TD: curvature - magnitude (h=512)",
       [(f"s={s}", ac[:, i], am[:, i]) for i, s in enumerate(sp) if s >= 0.5])

# ---- E7: neutral cross-prune, no-dropout net ------------------------------
neu = load("results_neutral/*nodrop*seed*.json")
if neu:
    key = "prune_curves_by_criterion"
    sp = [x["sparsity"] for x in neu[0][key]["forman"]]
    af = np.array([[x["test_acc"] for x in r[key]["forman"]] for r in neu])
    am = np.array([[x["test_acc"] for x in r[key]["magnitude"]] for r in neu])
    family("E7 neutral (no-dropout): curvature - magnitude",
           [(f"s={s}", af[:, i], am[:, i]) for i, s in enumerate(sp)
            if 0.3 <= s <= 0.7])

# ---- E16: width sweep, curvature vs magnitude -----------------------------
for h, d, pre in [(32, "results_width", "mnist_w32"), (64, "results_width", "mnist_w64"),
                  (128, "results_width", "mnist_w128"), (256, "results_width", "mnist_w256"),
                  (512, "results_td_cpu", "mnist_small")]:
    m = load(f"{d}/{pre}_td_magnitude_seed*.json")
    c = load(f"{d}/{pre}_td_forman_seed*.json")
    sp, am = prune_curve(m)
    _, ac = prune_curve(c)
    family(f"E16 width h={h}: curvature - magnitude",
           [(f"s={s}", ac[:, i], am[:, i]) for i, s in enumerate(sp)
            if s in (0.4, 0.5, 0.6, 0.7, 0.8)])

# ---- E17: noise sigma=0.5 vs magnitude, and vs curvature ------------------
for wtag, d, pre in [("w64", "results_width", "mnist_w64"),
                     ("w512", "results_td_cpu", "mnist_small")]:
    m = load(f"{d}/{pre}_td_magnitude_seed*.json")
    c = load(f"{d}/{pre}_td_forman_seed*.json")
    nz = load(f"results_magnoise/mnist_{wtag}_td_magnoise050_seed*.json")
    sp, am = prune_curve(m)
    _, ac = prune_curve(c)
    _, an = prune_curve(nz)
    idx = [(i, s) for i, s in enumerate(sp) if s in (0.5, 0.6, 0.7, 0.8)]
    family(f"E17 {wtag}: noise(0.5) - magnitude",
           [(f"s={s}", an[:, i], am[:, i]) for i, s in idx])
    family(f"E17 {wtag}: curvature - noise(0.5)",
           [(f"s={s}", ac[:, i], an[:, i]) for i, s in idx])

# ---- E9: iterative edge pruning -------------------------------------------
ip = load("results_ip/mnist_small_ip_seed*.json")
if ip:
    sp, am = ip_curve(ip, "magnitude")
    _, ac = ip_curve(ip, "curvature")
    family("E9 edge: curvature - magnitude",
           [(f"s={s}", ac[:, i], am[:, i]) for i, s in enumerate(sp)])

# ---- E14a: ER protocol (forman & F_dc vs magnitude) -----------------------
mech = load("results_mech/mnist_sparse_mech_seed*.json")
if mech:
    sp, am = ip_curve(mech, "magnitude")
    _, af = ip_curve(mech, "forman")
    _, ad = ip_curve(mech, "forman_dc")
    family("E14a ER: forman - magnitude",
           [(f"s={s}", af[:, i], am[:, i]) for i, s in enumerate(sp)])
    family("E14a ER: F_dc - magnitude",
           [(f"s={s}", ad[:, i], am[:, i]) for i, s in enumerate(sp)])

# ---- E14b: heavy-tailed sigma=1.0 -----------------------------------------
het = load("results_mech_hetero/hetero_s1_seed*.json")
if het:
    sp, am = ip_curve(het, "magnitude")
    _, af = ip_curve(het, "curvature")
    _, ad = ip_curve(het, "forman_dc")
    family("E14b hetero(1.0): forman - magnitude",
           [(f"s={s}", af[:, i], am[:, i]) for i, s in enumerate(sp)])
    family("E14b hetero(1.0): F_dc - magnitude",
           [(f"s={s}", ad[:, i], am[:, i]) for i, s in enumerate(sp)])

# ---- report ---------------------------------------------------------------
for name, tests in FAMILIES:
    print(f"\n### {name}  (m = {len(tests)} tests in family)")
    print(f"{'':8s} {'diff(pp)':>9s} {'95% CI':>18s} {'d_z':>6s} "
          f"{'t':>6s} {'p':>8s} {'p_bonf':>8s}  sig")
    for s in tests:
        sig = "**" if s["p_bonf"] < 0.05 else ("(*)" if s["p"] < 0.05 else "")
        print(f"{s['label']:8s} {100*s['diff']:>+9.2f} "
              f"[{100*s['lo']:>+7.2f},{100*s['hi']:>+7.2f}] {s['dz']:>6.2f} "
              f"{s['t']:>6.2f} {s['p']:>8.4f} {s['p_bonf']:>8.4f}  {sig}")
print("\n**: survives Bonferroni within family at 0.05 | (*): raw p<0.05 only")
