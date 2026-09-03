#!/usr/bin/env python
"""
cost_vs_design.py -- does one Monte Carlo transport solve cost the same
everywhere in the design space?

campaign_timing.py measures how long each evaluation took. The optimisation
checkpoint stores what each evaluation WAS. Joining them by case index tests
whether the per-solve transport cost depends on the design variables. If it
does, the cost of a campaign is not simply the number of evaluations times a
constant, which matters for any budget-based stopping rule and for the
computational-cost section of the thesis.

The response variable is the mean seconds per transport solve of one design,

    t_solve = solve_s / n_solves

which removes the effect of the depletion chain length (a design that survives
to a higher burnup runs more solves, and that is a separate effect, reported
here as n_dep_solves).

usage
    python cost_vs_design.py --cases timing_c4_cases.csv \\
        --checkpoint out_c4/optimization_checkpoint.json --campaign C4 \\
        --out cost_c4

what it reports
    1. Spearman rank correlation of t_solve against every design variable and
       against n_dep_solves, with a permutation p-value (no SciPy needed)
    2. an ordinary least squares fit of t_solve on the standardised design
       variables, so the coefficients are directly comparable
    3. the same for total evaluation wall time, which mixes solve cost and
       chain length
    4. <out>_joined.csv with one row per design, ready for plotting

Spearman is used rather than Pearson because it needs no assumption of
linearity and is insensitive to the outliers a censored design can produce.
The p-value comes from a permutation test, which makes no distributional
assumption at all.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Equivalent to scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average the ranks of tied values
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for k in np.flatnonzero(counts > 1):
        ranks[inv == k] = ranks[inv == k].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def perm_p(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
           seed: int = 0) -> float:
    """Two-sided permutation p-value for a Spearman correlation."""
    rho0 = abs(spearman(a, b))
    rng = np.random.default_rng(seed)
    bb = b.copy()
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(bb)
        if abs(spearman(a, bb)) >= rho0:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def ols(X: np.ndarray, y: np.ndarray):
    """Least squares with an intercept. Returns (coefficients, R^2)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True,
                    help="<out>_cases.csv written by campaign_timing.py")
    ap.add_argument("--checkpoint", required=True,
                    help="optimization_checkpoint.json of the same campaign")
    ap.add_argument("--campaign", default="campaign", help="label for the report")
    ap.add_argument("--n-perm", type=int, default=20000,
                    help="permutations for the p-value, default 20000")
    ap.add_argument("--out", default="cost_vs_design", help="output prefix")
    args = ap.parse_args()

    ck = json.loads(Path(args.checkpoint).read_text())
    names = list(ck["design_variables"])
    raw = ck["all_raw"]

    rows = list(csv.DictReader(open(args.cases)))
    if not rows:
        sys.exit(f"{args.cases} is empty")

    # ---- join on the case index -------------------------------------------
    joined = []
    for r in rows:
        i = int(r["idx"])
        if i >= len(raw):
            continue
        n_solves = int(r["n_solves"])
        if n_solves < 1:
            continue
        rec = raw[i]
        if any(nm not in rec for nm in names):
            continue
        joined.append(dict(
            idx=i,
            t_solve=float(r["solve_s"]) / n_solves,
            wall_min=float(r["wall_s"]) / 60.0,
            n_solves=n_solves,
            n_dep_solves=int(r["n_dep_solves"]),
            censored=int(bool(rec.get("censored", False))),
            **{nm: float(rec[nm]) for nm in names},
        ))
    if len(joined) < 8:
        sys.exit(f"only {len(joined)} designs joined. Check that the cases CSV "
                 f"and the checkpoint belong to the same campaign.")

    n_dropped = len(rows) - len(joined)
    y_solve = np.array([j["t_solve"] for j in joined])
    y_wall = np.array([j["wall_min"] for j in joined])
    ndep = np.array([j["n_dep_solves"] for j in joined], dtype=float)
    X = np.column_stack([[j[nm] for j in joined] for nm in names])

    print(f"=== {args.campaign}: cost versus design ===")
    print(f"designs joined: {len(joined)}"
          + (f" ({n_dropped} row(s) in the CSV had no match)" if n_dropped else ""))
    print(f"seconds per transport solve: mean {y_solve.mean():.0f}, "
          f"sd {y_solve.std(ddof=1):.0f}, min {y_solve.min():.0f}, "
          f"max {y_solve.max():.0f}, max/min {y_solve.max()/y_solve.min():.2f}")
    print(f"evaluation wall time [min] : mean {y_wall.mean():.1f}, "
          f"sd {y_wall.std(ddof=1):.1f}")
    print(f"depletion solves per design: mean {ndep.mean():.1f}, "
          f"sd {ndep.std(ddof=1):.1f}")

    print("\n-- Spearman rank correlation with SECONDS PER SOLVE --")
    print(f"   (permutation p-value, {args.n_perm} permutations)")
    corr_rows = []
    for k, nm in enumerate(names):
        rho = spearman(X[:, k], y_solve)
        p = perm_p(X[:, k], y_solve, args.n_perm)
        corr_rows.append((nm, rho, p))
        print(f"   {nm:<14s} rho = {rho:+.3f}   p = {p:.4f}"
              + ("   <-- significant at 0.05" if p < 0.05 else ""))
    rho_nd = spearman(ndep, y_solve)
    p_nd = perm_p(ndep, y_solve, args.n_perm)
    print(f"   {'n_dep_solves':<14s} rho = {rho_nd:+.3f}   p = {p_nd:.4f}"
          + ("   <-- significant at 0.05" if p_nd < 0.05 else ""))

    print("\n-- Spearman rank correlation with EVALUATION WALL TIME --")
    for k, nm in enumerate(names):
        rho = spearman(X[:, k], y_wall)
        print(f"   {nm:<14s} rho = {rho:+.3f}")
    print(f"   {'n_dep_solves':<14s} rho = {spearman(ndep, y_wall):+.3f}")

    # ---- linear model on standardised variables ---------------------------
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    beta, r2 = ols(Z, y_solve)
    print("\n-- least squares, seconds per solve on standardised variables --")
    print(f"   intercept {beta[0]:.1f} s")
    for k, nm in enumerate(names):
        print(f"   {nm:<14s} {beta[k+1]:+7.2f} s per standard deviation")
    print(f"   R^2 = {r2:.3f}")

    out = Path(args.out)
    with open(f"{out}_joined.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(joined[0].keys()))
        w.writeheader()
        w.writerows(joined)

    summary = dict(
        campaign=args.campaign, n_designs=len(joined),
        mean_s_per_solve=float(y_solve.mean()),
        sd_s_per_solve=float(y_solve.std(ddof=1)),
        min_s_per_solve=float(y_solve.min()),
        max_s_per_solve=float(y_solve.max()),
        ratio_max_min=float(y_solve.max() / y_solve.min()),
        spearman_vs_s_per_solve={nm: dict(rho=rho, p=p)
                                 for nm, rho, p in corr_rows},
        spearman_ndep_vs_s_per_solve=dict(rho=rho_nd, p=p_nd),
        ols_standardised={nm: float(beta[k + 1]) for k, nm in enumerate(names)},
        ols_intercept_s=float(beta[0]), ols_r2=float(r2),
    )
    Path(f"{out}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}_joined.csv and {out}_summary.json")


if __name__ == "__main__":
    main()
