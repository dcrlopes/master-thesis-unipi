#!/usr/bin/env python3
"""test_constraint_norm.py -- validate the normalization on REAL archive data.

Standard-library only (csv, json, math): runs anywhere, needs no OpenMC,
no pymoo, no numpy. Reads a campaign summary CSV that carries the five g
columns (default campaign5/c5_full.csv) and reports:

  1. SIGN CHECK      the feasible set is identical before and after
                     normalization (it must be, scales are positive).
  2. SHARE           which constraint dominates the summed violation,
                     raw versus normalized.
  3. RANKING         Spearman rank correlation between the raw-CV order
                     and the normalized-CV order of the infeasible designs,
                     and the overlap of the two top-6 lists (6 = n_infill,
                     the designs the least-infeasible fallback would hand
                     to the acquisition).

Usage:
    python test_constraint_norm.py [path/to/campaign_full.csv]
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

G_NAMES = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom"]

# Must mirror the scales installed by apply_constraint_norm.py.
SCALES = {
    "g_kmin": 1.02,
    "g_kmax": 1.35,
    "g_enr":  19.75,
    "g_peak": 2.0,
    "g_geom": 90.0,       # R_VESSEL_INNER - VESSEL_CLEARANCE_CM = 90.0 - 0.0
}


def cv(row: dict, scales: dict | None) -> float:
    """pymoo-style total violation: sum of positive parts."""
    total = 0.0
    for name in G_NAMES:
        g = float(row[name])
        if scales is not None:
            g /= scales[name]
        if g > 0.0:
            total += g
    return total


def ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    rk = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            rk[order[t]] = avg
        i = j + 1
    return rk


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "campaign5/c5_full.csv")
    if not path.exists():
        sys.exit(f"ABORT: {path} not found")
    rows = list(csv.DictReader(path.open()))
    missing = [c for c in G_NAMES if c not in rows[0]]
    if missing:
        sys.exit(f"ABORT: columns {missing} not in {path}")

    cv_raw = [cv(r, None) for r in rows]
    cv_nrm = [cv(r, SCALES) for r in rows]

    # 1 -- sign check ------------------------------------------------------
    feas_raw = [i for i, v in enumerate(cv_raw) if v <= 0.0]
    feas_nrm = [i for i, v in enumerate(cv_nrm) if v <= 0.0]
    assert feas_raw == feas_nrm, "feasible set changed: scales must be > 0"
    print(f"[1] sign check: feasible set identical "
          f"({len(feas_raw)} feasible of {len(rows)})")

    infeas = [i for i in range(len(rows)) if cv_raw[i] > 0.0]

    # 2 -- violation share -------------------------------------------------
    print("[2] share of the summed violation over the infeasible archive:")
    print(f"    {'constraint':10s} {'raw':>8s} {'normalized':>12s}")
    for name in G_NAMES:
        s_raw = sum(max(float(rows[i][name]), 0.0) for i in infeas)
        s_nrm = sum(max(float(rows[i][name]) / SCALES[name], 0.0)
                    for i in infeas)
        t_raw = sum(cv_raw[i] for i in infeas)
        t_nrm = sum(cv_nrm[i] for i in infeas)
        print(f"    {name:10s} {100 * s_raw / t_raw:7.2f}% "
              f"{100 * s_nrm / t_nrm:11.2f}%")

    # 3 -- ranking change --------------------------------------------------
    a = [cv_raw[i] for i in infeas]
    b = [cv_nrm[i] for i in infeas]
    rho = spearman(a, b)
    top_raw = [infeas[j] for j in sorted(range(len(a)), key=lambda j: a[j])[:6]]
    top_nrm = [infeas[j] for j in sorted(range(len(b)), key=lambda j: b[j])[:6]]
    overlap = len(set(top_raw) & set(top_nrm))
    print(f"[3] Spearman rho over {len(infeas)} infeasible designs: {rho:.4f}")
    print(f"    top-6 by raw CV        : {[rows[i]['idx'] for i in top_raw]}")
    print(f"    top-6 by normalized CV : {[rows[i]['idx'] for i in top_nrm]}")
    print(f"    overlap: {overlap} of 6")

    # 4 -- interpretation, computed from the numbers above ------------------
    k_raw = sum(max(float(rows[i][n]), 0.0)
                for i in infeas for n in ("g_kmin", "g_kmax"))
    k_nrm = sum(max(float(rows[i][n]) / SCALES[n], 0.0)
                for i in infeas for n in ("g_kmin", "g_kmax"))
    t_raw = sum(cv_raw[i] for i in infeas)
    t_nrm = sum(cv_nrm[i] for i in infeas)
    sh_raw = 100.0 * k_raw / t_raw
    sh_nrm = 100.0 * k_nrm / t_nrm
    print()
    print("[4] interpretation")
    if overlap == 6 and rho > 0.99:
        print("    The ARCHIVE ranking is essentially unchanged. That is the")
        print("    expected result when the DOE was pre-screened by the exact")
        print("    geometry and LEU constraints, so only the peaking limit and")
        print("    the reactivity window ever fire in the archive, and peaking")
        print("    dominates under either weighting.")
        print("    This is NOT evidence that the change is inert. The bias it")
        print("    removes acts on the SURROGATE population, where NSGA-II")
        print("    roams the full box and g_geom spans tens of centimetres")
        print("    against a reactivity window spanning tenths of delta-k.")
    else:
        print("    The archive ranking itself changes: the raw sum was")
        print("    steering the infeasible search differently from the")
        print("    fraction-of-limit ranking.")
    print(f"    Measurable effect on the archive: the reactivity window")
    print(f"    (g_kmin + g_kmax) carries {sh_raw:.2f}% of the raw violation")
    print(f"    and {sh_nrm:.2f}% of the normalized violation, a factor of")
    print(f"    {sh_nrm / sh_raw:.2f} more relative weight against peaking.")


if __name__ == "__main__":
    main()
