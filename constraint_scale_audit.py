#!/usr/bin/env python3
"""constraint_scale_audit.py -- what does each constraint actually contribute
to pymoo's violation sum, under the scaling in use and under alternatives,
measured on a real archive rather than argued.

WHY
---
When every individual in the surrogate population is infeasible, pymoo
stops Pareto sorting and minimises the sum of positive constraint values.
Whatever scale each constraint carries then decides which one the search
chases. "Divide by own limit" makes the entries dimensionless, but it does
not make a unit of violation MEAN the same thing across constraints.

THREE SCHEMES COMPARED
    limit   g / own limit                      (CONSTRAINT-NORM, in use)
    noise   g / (3 x measurement sigma)        one unit = one resolvable
                                               step above the limit
    spread  g / IQR of g over the archive      one unit = one typical
                                               archive spread of that g

For each scheme the audit prints, per constraint, the share of the total
violation sum it carries, and the Spearman correlation between the
rankings of the infeasible designs under the scheme in use and each
alternative. If the rankings agree, the choice is cosmetic; if they
diverge, it decides which designs the optimiser would pursue.

Measurement sigmas default to the campaign's own seed studies: k 0.0005,
F_dH 0.012, and are settable.

USAGE
    python constraint_scale_audit.py out_c6/optimization_checkpoint.json
    python constraint_scale_audit.py out_c7/optimization_checkpoint.json \\
        --sigma-k 0.0005 --sigma-f 0.012
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--sigma-k", type=float, default=0.0005,
                help="one-sigma Monte Carlo noise of a core eigenvalue")
ap.add_argument("--sigma-f", type=float, default=0.012,
                help="one-sigma seed noise of F_dH")
ap.add_argument("--sigma-enr", type=float, default=0.05)
ap.add_argument("--sigma-geom", type=float, default=0.05)
a = ap.parse_args()

ck = json.loads(Path(a.checkpoint).read_text())
cn = ck["constraint_names"]
raw = ck["all_raw"]
lim = ck["meta"].get("limits", {})

# scheme "limit": reproduce the scales run_optimization syncs from limits
k_max = lim.get("k_max") or 1.35
limit_scale = {"g_kmin": lim.get("k_min", 1.02), "g_kmax": k_max,
               "g_enr": lim.get("enr_max", 19.75), "g_peak": lim.get("f_max", 2.0),
               "g_geom": 90.0 - 2.0, "g_ctrl": 1.0}
noise_scale = {"g_kmin": 3 * a.sigma_k, "g_kmax": 3 * a.sigma_k,
               "g_ctrl": 3 * a.sigma_k, "g_peak": 3 * a.sigma_f,
               "g_enr": 3 * a.sigma_enr, "g_geom": 3 * a.sigma_geom}


def iqr(v):
    s = sorted(v)
    n = len(s)
    if n < 4:
        return max(s) - min(s) or 1.0
    q1, q3 = s[n // 4], s[(3 * n) // 4]
    return (q3 - q1) or 1.0


spread_scale = {c: iqr([float(r[c]) for r in raw]) for c in cn}
schemes = {"limit": limit_scale, "noise": noise_scale, "spread": spread_scale}

infeas = [r for r in raw if any(float(r[c]) > 0 for c in cn)]
print(f"{len(raw)} designs, {len(infeas)} infeasible, constraints {cn}")
print(f"\nper-constraint share of the violation SUM over infeasible designs")
print(f"{'constraint':>10} " + " ".join(f"{s:>8}" for s in schemes)
      + f"   {'violated':>8}")
share = {}
for s, sc in schemes.items():
    tot = {c: sum(max(0.0, float(r[c])) / sc.get(c, 1.0) for r in infeas)
           for c in cn}
    T = sum(tot.values()) or 1.0
    share[s] = {c: tot[c] / T for c in cn}
for c in cn:
    nv = sum(1 for r in raw if float(r[c]) > 0)
    print(f"{c:>10} " + " ".join(f"{100*share[s][c]:7.1f}%" for s in schemes)
          + f"   {nv:>8}")


def cv(r, sc):
    return sum(max(0.0, float(r[c])) / sc.get(c, 1.0) for c in cn)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    for pos, i in enumerate(o):
        rk[i] = pos
    return rk


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    return num / (dx * dy) if dx * dy else float("nan")


if len(infeas) >= 3:
    base = [cv(r, limit_scale) for r in infeas]
    print("\nranking of the infeasible designs by violation sum, "
          "Spearman rho against the scheme in use ('limit'):")
    for s, sc in schemes.items():
        print(f"  {s:>7}: rho = {spearman(base, [cv(r, sc) for r in infeas]):+.3f}")
    print("  rho near 1: the scaling only rescales, the search chases the "
          "same designs.\n  rho well below 1: the scaling decides which "
          "infeasible designs the optimiser pursues.")
else:
    print("\nfewer than 3 infeasible designs: the violation-sum regime does "
          "not arise on this archive, so the scaling is moot here.")
