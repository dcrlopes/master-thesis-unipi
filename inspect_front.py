#!/usr/bin/env python3
"""
inspect_front.py -- read a LIVE optimisation checkpoint written by
run_optimization.py and report, without running any OpenMC (Open source Monte
Carlo particle transport code) evaluation:

  1. what the checkpoint actually contains (schema, limits, block structure),
  2. every evaluation with its design variables, objectives and constraint
     margins,
  3. the feasible non-dominated (Pareto) front in the objective plane
     cycle length in EFPD (Effective Full Power Days, maximise) against
     F_dH (radial enthalpy-rise hot channel factor, minimise),
  4. optional licensing-style screens applied a posteriori: a discharge
     burnup limit in MWd/kgHM and a peaking limit on F_dH,
  5. CSV dumps of the full archive and of the front.

Only the Python standard library is used. There is NO OpenMC import, NO numpy
and NO matplotlib, so this runs with the plain system python3 on the AWS host,
on wks720 in any environment, and on WSL2. It must NOT be run through the
Docker `lab` alias: the container is for transport, not for reading JSON.

USAGE
-----
  python3 inspect_front.py --checkpoint out_c7/optimization_checkpoint.json

  python3 inspect_front.py --checkpoint out_c7/optimization_checkpoint.json \
      --burnup-screen 75 --f-screen 1.65 --csv-prefix c7_partial

EXIT CODES
----------
  0 normal, 2 environment or schema check failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import math
import statistics
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 0. Environment and schema guard. Nothing is printed about the physics until  #
#    the file has been proven to be the object this script expects.            #
# --------------------------------------------------------------------------- #

REQUIRED_KEYS = ["design_variables", "objectives", "constraint_names", "all_raw"]

# Keys every evaluation record must carry for the analysis to be meaningful.
REQUIRED_RECORD_KEYS = ["cycle_length", "peaking"]


def guard(path: Path) -> dict:
    """Refuse to proceed unless the environment and the file are the expected
    ones. Returns the parsed checkpoint."""
    print("=" * 78)
    print("ENVIRONMENT AND SCHEMA CHECK")
    print("=" * 78)
    print(f"  python           : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  working directory: {os.getcwd()}")

    if sys.version_info < (3, 7):
        print("  FAIL: Python 3.7 or newer is required.")
        sys.exit(2)

    # This script must not be run inside the transport container. Detect the
    # usual Docker markers and warn, because inside the container the relative
    # path out_c7/... resolves under /work and may not be what you expect.
    in_docker = Path("/.dockerenv").exists()
    print(f"  inside a container: {'yes' if in_docker else 'no'}")
    if in_docker:
        print("  NOTE: this script needs no OpenMC. Prefer the host python3 so "
              "that relative paths are the ones you see in your shell.")

    # OpenMC must NOT be needed. Say so explicitly so the check is meaningful.
    print("  OpenMC required  : no (pure JSON post-processing)")

    if not path.exists():
        print(f"  FAIL: checkpoint not found: {path}")
        print("        The checkpoint is written at the END of the design of "
              "experiments stage and after EVERY infill iteration, never in "
              "the middle of a block. If the file is missing, the campaign is "
              "still inside its first block.")
        sys.exit(2)

    size_mb = path.stat().st_size / 1.0e6
    print(f"  checkpoint       : {path}  ({size_mb:.2f} MB)")

    try:
        ckpt = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"  FAIL: the file is not valid JSON ({exc}).")
        print("        A checkpoint is written atomically at the end of an "
              "iteration. If it is truncated, the write was interrupted. Use "
              "the previous copy or wait for the next iteration.")
        sys.exit(2)

    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        print(f"  FAIL: the checkpoint lacks the keys {missing}.")
        sys.exit(2)

    raw = ckpt["all_raw"]
    if not raw:
        print("  FAIL: the archive is empty.")
        sys.exit(2)

    bad = [k for k in REQUIRED_RECORD_KEYS if k not in raw[0]]
    if bad:
        print(f"  FAIL: evaluation records lack the keys {bad}.")
        sys.exit(2)

    print("  schema           : OK")
    print()
    return ckpt


# --------------------------------------------------------------------------- #
# 1. Reading                                                                   #
# --------------------------------------------------------------------------- #

def describe(ckpt: dict) -> None:
    """Print what the campaign is, from the checkpoint alone."""
    meta = ckpt.get("meta") or {}
    dv = ckpt["design_variables"]
    cn = ckpt["constraint_names"]
    n = len(ckpt["all_raw"])

    print("=" * 78)
    print("CAMPAIGN AS RECORDED IN THE CHECKPOINT")
    print("=" * 78)
    print(f"  evaluations in the archive : {n}")
    print(f"  n_real_evaluations field   : {ckpt.get('n_real_evaluations')}")
    print(f"  design variables           : {', '.join(dv)}")
    print(f"  objectives                 : "
          f"{', '.join(f'{a} ({b})' for a, b in ckpt['objectives'])}")
    print(f"  constraints                : {', '.join(cn)}")

    lim = meta.get("limits") or {}
    if lim:
        print("  limits                     : "
              + ", ".join(f"{k}={v}" for k, v in lim.items()))
    sched = meta.get("schedule") or {}
    if sched:
        print(f"  burnup ceiling             : "
              f"{sched.get('max_burnup')} MWd/kgHM")
    if meta.get("ctrl_margin_dk") is not None:
        print(f"  control-rod screen margin  : {meta.get('ctrl_margin_dk')} dk")
    print(f"  host / threads             : "
          f"{meta.get('host')} / {meta.get('omp_threads')}")
    print(f"  OpenMC version             : {meta.get('openmc_version')}")

    blocks = meta.get("block_started_utc") or []
    if blocks:
        print(f"  blocks started (UTC)       : {len(blocks)}")
        for i, b in enumerate(blocks, 1):
            print(f"      block {i}: {b}")

    # Hypervolume history: one entry per block boundary, so its length tells
    # you how many times the archive has been closed and checkpointed.
    hv = ckpt.get("hv_history") or []
    if hv:
        print(f"  hypervolume history        : {len(hv)} entries, "
              f"last {hv[-1]:.6g}")
        if len(hv) >= 2 and hv[-2] != 0:
            gain = 100.0 * (hv[-1] - hv[-2]) / abs(hv[-2])
            print(f"  last hypervolume gain      : {gain:+.2f} %")
    ref = ckpt.get("hv_ref")
    if ref:
        print(f"  frozen reference point     : "
              f"EFPD {-float(ref[0]):.1f}, F_dH {float(ref[1]):.4f}")
    print()


def rows_from(ckpt: dict, tol: float) -> list:
    """Flatten the archive into a list of plain dictionaries, one per
    evaluation, with a feasibility verdict and the list of violated
    constraints."""
    cn = ckpt["constraint_names"]
    out = []
    for i, r in enumerate(ckpt["all_raw"]):
        viol = []
        for c in cn:
            if c not in r:
                viol.append(f"{c}(MISSING)")
                continue
            if float(r[c]) > tol:
                viol.append(c)
        row = dict(r)
        row["pos"] = i
        row["feasible"] = (len(viol) == 0)
        row["violated"] = ",".join(viol)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# 2. Pareto front                                                              #
# --------------------------------------------------------------------------- #

def nondominated(rows: list, efpd_key: str = "cycle_length",
                 fdh_key: str = "peaking") -> list:
    """Return the non-dominated subset for (EFPD maximise, F_dH minimise).

    Design j dominates design i when j is at least as good on both objectives
    and strictly better on at least one."""
    front = []
    for i, a in enumerate(rows):
        ea, fa = float(a[efpd_key]), float(a[fdh_key])
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            eb, fb = float(b[efpd_key]), float(b[fdh_key])
            if eb >= ea and fb <= fa and (eb > ea or fb < fa):
                dominated = True
                break
        if not dominated:
            front.append(a)
    front.sort(key=lambda r: float(r[efpd_key]))
    return front


def specific_power(rows: list) -> float:
    """Recover the global specific power in W/gHM from the archive itself.

    The evaluator relates burnup and cycle length by
        EFPD = bu_eoc * 1000 / spec_power
    with spec_power computed once per evaluator, so the map is linear and
    design independent. The median over the archive is used, and the spread is
    reported so an inconsistent archive is visible."""
    vals = []
    for r in rows:
        try:
            bu = float(r["bu_eoc_mwd_kg"])
            ef = float(r["cycle_length"])
        except (KeyError, TypeError, ValueError):
            continue
        if bu > 0 and ef > 0:
            vals.append(bu * 1000.0 / ef)
    if not vals:
        return float("nan")
    med = statistics.median(vals)
    spread = (max(vals) - min(vals)) / med if med else float("inf")
    if spread > 1.0e-3:
        print(f"  WARNING: the burnup to EFPD map is not constant across the "
              f"archive (relative spread {spread:.2e}). The burnup screen "
              f"below is approximate.")
    return med


# --------------------------------------------------------------------------- #
# 3. Printing                                                                  #
# --------------------------------------------------------------------------- #

def fmt(r, key, spec="{:.4f}", default="-"):
    v = r.get(key)
    if v is None:
        return default
    try:
        return spec.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def print_archive(rows: list, dv: list) -> None:
    print("=" * 78)
    print("EVERY EVALUATION IN THE ARCHIVE")
    print("=" * 78)
    head = (f"{'pos':>4} {'e_in':>6} {'e_out':>6} {'Gd_wt':>5} {'pitch':>6} "
            f"{'refl':>6} {'Gd_pin':>6} {'EFPD':>8} {'F_dH':>7} "
            f"{'k_core':>7} {'B_EOC':>7} {'feas':>5}  violated")
    print(head)
    print("-" * len(head))
    for r in rows:
        cen = "*" if r.get("censored") else " "
        print(f"{r['pos']:>4} "
              f"{fmt(r, 'enrich_inner', '{:.2f}'):>6} "
              f"{fmt(r, 'enrich_outer', '{:.2f}'):>6} "
              f"{fmt(r, 'gd_wt', '{:.2f}'):>5} "
              f"{fmt(r, 'pitch', '{:.3f}'):>6} "
              f"{fmt(r, 'refl_thick', '{:.2f}'):>6} "
              f"{fmt(r, 'gd_pins_used', '{:.0f}'):>6} "
              f"{fmt(r, 'cycle_length', '{:.0f}'):>7}{cen} "
              f"{fmt(r, 'peaking', '{:.4f}'):>7} "
              f"{fmt(r, 'keff_core_bol', '{:.4f}'):>7} "
              f"{fmt(r, 'bu_eoc_mwd_kg', '{:.2f}'):>7} "
              f"{'yes' if r['feasible'] else 'NO':>5}  {r['violated']}")
    print("  * = censored, the cycle length is a lower bound at the burnup "
          "ceiling.")
    print()


def print_front(front: list, title: str, extra_keys: list) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    if not front:
        print("  the feasible set is empty, there is no front yet.")
        print()
        return
    head = (f"{'pos':>4} {'EFPD':>8} {'F_dH':>7} {'e_in':>6} {'e_out':>6} "
            f"{'Gd_wt':>5} {'pitch':>6} {'refl':>6} {'Gd_pin':>6} "
            f"{'k_core':>7} {'B_EOC':>7}")
    for k in extra_keys:
        head += f" {k:>8}"
    print(head)
    print("-" * len(head))
    for r in front:
        line = (f"{r['pos']:>4} "
                f"{fmt(r, 'cycle_length', '{:.0f}'):>8} "
                f"{fmt(r, 'peaking', '{:.4f}'):>7} "
                f"{fmt(r, 'enrich_inner', '{:.2f}'):>6} "
                f"{fmt(r, 'enrich_outer', '{:.2f}'):>6} "
                f"{fmt(r, 'gd_wt', '{:.2f}'):>5} "
                f"{fmt(r, 'pitch', '{:.3f}'):>6} "
                f"{fmt(r, 'refl_thick', '{:.2f}'):>6} "
                f"{fmt(r, 'gd_pins_used', '{:.0f}'):>6} "
                f"{fmt(r, 'keff_core_bol', '{:.4f}'):>7} "
                f"{fmt(r, 'bu_eoc_mwd_kg', '{:.2f}'):>7}")
        for k in extra_keys:
            line += f" {fmt(r, k, '{:.4f}'):>8}"
        print(line)

    efpd = [float(r["cycle_length"]) for r in front]
    fdh = [float(r["peaking"]) for r in front]
    print(f"\n  front size {len(front)}, "
          f"EFPD from {min(efpd):.0f} to {max(efpd):.0f}, "
          f"F_dH from {min(fdh):.4f} to {max(fdh):.4f}")
    print()


# --------------------------------------------------------------------------- #
# 4. Main                                                                      #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect a live optimisation checkpoint and print the "
                    "feasible Pareto front. No OpenMC needed.")
    ap.add_argument("--checkpoint", required=True,
                    help="path to optimization_checkpoint.json")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="feasibility tolerance on the raw constraint values, "
                         "a design is feasible when every g is at or below "
                         "this value (default 0.0)")
    ap.add_argument("--burnup-screen", type=float, default=None,
                    metavar="MWD_KG",
                    help="apply a discharge burnup limit a posteriori, for "
                         "example 75, and recompute the front on the "
                         "truncated cycle lengths")
    ap.add_argument("--f-screen", type=float, default=None, metavar="F_DH",
                    help="apply a licensing-style peaking limit, for example "
                         "1.65, and recompute the front on the survivors")
    ap.add_argument("--csv-prefix", default=None,
                    help="write <prefix>_all.csv and <prefix>_front.csv")
    ap.add_argument("--no-archive-table", action="store_true",
                    help="skip the full per-evaluation table")
    args = ap.parse_args()

    ckpt = guard(Path(args.checkpoint))
    describe(ckpt)

    dv = ckpt["design_variables"]
    cn = ckpt["constraint_names"]
    rows = rows_from(ckpt, args.tol)

    feas = [r for r in rows if r["feasible"]]
    print("=" * 78)
    print("FEASIBILITY")
    print("=" * 78)
    print(f"  feasible {len(feas)} of {len(rows)} "
          f"({100.0 * len(feas) / len(rows):.1f} %)")
    for c in cn:
        n_bad = sum(1 for r in rows
                    if c in r and float(r[c]) > args.tol)
        n_miss = sum(1 for r in rows if c not in r)
        note = f", {n_miss} records missing this key" if n_miss else ""
        print(f"    {c:<10} violated by {n_bad:>4} of {len(rows)}{note}")
    n_cen = sum(1 for r in rows if r.get("censored"))
    print(f"  censored at the burnup ceiling: {n_cen} of {len(rows)}")
    print()

    if not args.no_archive_table:
        print_archive(rows, dv)

    # extra columns worth showing when the control-rod screen is active
    extra = [k for k in ("k_allre", "F_allre", "g_ctrl") if k in rows[0]]

    front = nondominated(feas)
    print_front(front, "FEASIBLE PARETO FRONT AS EVALUATED", extra)

    front_screened = None
    if args.burnup_screen is not None or args.f_screen is not None:
        sp = specific_power(rows)
        work = []
        for r in feas:
            q = dict(r)
            if args.burnup_screen is not None and not math.isnan(sp):
                cap_efpd = args.burnup_screen * 1000.0 / sp
                q["cycle_length"] = min(float(q["cycle_length"]), cap_efpd)
                q["screened"] = True
            if args.f_screen is not None and float(q["peaking"]) > args.f_screen:
                continue
            work.append(q)
        title = "FRONT UNDER THE SELECTION SCREENS"
        bits = []
        if args.burnup_screen is not None:
            bits.append(f"discharge burnup at most {args.burnup_screen} "
                        f"MWd/kgHM (specific power {sp:.3f} W/gHM)")
        if args.f_screen is not None:
            bits.append(f"F_dH at most {args.f_screen}")
        print("  screens applied: " + ", ".join(bits))
        front_screened = nondominated(work)
        print_front(front_screened, title, extra)

    if args.csv_prefix:
        keys = ["pos", "feasible", "violated"] + dv + \
               ["cycle_length", "peaking", "censored", "bu_eoc_mwd_kg",
                "k_bol", "keff_core_bol", "k_target", "gd_pins_used",
                "peaking_asm", "core_entropy_conv", "t_eval_s"] + cn + extra
        def dump(path, data):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                for r in data:
                    w.writerow(r)
            print(f"  wrote {path}  ({len(data)} rows)")
        print("=" * 78)
        print("CSV OUTPUT")
        print("=" * 78)
        dump(f"{args.csv_prefix}_all.csv", rows)
        dump(f"{args.csv_prefix}_front.csv", front)
        if front_screened is not None:
            dump(f"{args.csv_prefix}_front_screened.csv", front_screened)
        print()


if __name__ == "__main__":
    main()
