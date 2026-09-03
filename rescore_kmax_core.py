#!/usr/bin/env python3
"""
rescore_kmax_core.py -- re-score a finished campaign on the core reactivity
basis, using only what the archive already contains.

Every evaluation from Campaign 4 onward records both readings:

    k_bol          assembly k_inf from the reflective-boundary depletion model
    keff_core_bol  k_eff of the 2-D core at beginning of life

The campaigns constrained g_kmax on the first. Excess reactivity is a
core-level budget, so the second is the physically meaningful quantity. This
script answers what the feasibility picture looks like under the core basis,
across a sweep of candidate limits, without rerunning any transport.

It changes nothing on disk except its own outputs.

Usage
-----
    python rescore_kmax_core.py --checkpoint out_c5/optimization_checkpoint.json \\
        --label C5 --out rescore_c5

    python rescore_kmax_core.py --checkpoint out_c4/optimization_checkpoint.json \\
        --label C4 --out rescore_c4 --limits 1.20,1.25,1.30,1.35,1.40

Flags
    --checkpoint  campaign checkpoint written by save_checkpoint()
    --label       campaign name used in the printed tables and the LaTeX file
    --limits      comma-separated candidate k_max values to sweep
                  (default 1.20,1.25,1.30,1.35,1.40,1.45)
    --k-min       lower reactivity bound, default 1.02
    --f-max       peaking limit, default 2.0
    --enr-max     enrichment cap in wt%, default 19.75
    --out         output directory, default rescore
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

REQUIRED = ("k_bol", "keff_core_bol")


def load(path):
    ck = json.loads(Path(path).read_text())
    raw = ck.get("all_raw") or []
    if not raw:
        sys.exit(f"{path} holds no all_raw record")
    missing = [k for k in REQUIRED if k not in raw[0]]
    if missing:
        sys.exit(f"{path} lacks {missing}. Campaigns before C4 did not record "
                 f"the core solve, so they cannot be re-scored this way.")
    return ck, raw


def feasible(r, k_ref, k_max, k_min, f_max, enr_max):
    """True when every constraint is satisfied on the given reactivity basis."""
    g = [k_min - k_ref,
         k_ref - k_max,
         max(r["enrich_inner"], r["enrich_outer"]) - enr_max,
         r["peaking"] - f_max,
         r.get("g_geom", 0.0)]
    return all(v <= 0.0 for v in g), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", default="campaign")
    ap.add_argument("--limits", default="1.20,1.25,1.30,1.35,1.40,1.45")
    ap.add_argument("--k-min", type=float, default=1.02)
    ap.add_argument("--f-max", type=float, default=2.0)
    ap.add_argument("--enr-max", type=float, default=19.75)
    ap.add_argument("--out", default="rescore")
    args = ap.parse_args()

    ck, raw = load(args.checkpoint)
    limits = [float(x) for x in args.limits.split(",")]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    gaps = [1.0e5 * (r["k_bol"] - r["keff_core_bol"]) for r in raw]
    print(f"\n{args.label}: {len(raw)} evaluations")
    print("-" * 66)
    print("assembly-to-core reactivity gap, k_inf minus k_eff,core [pcm]")
    print(f"  min {min(gaps):7.0f}   mean {st.mean(gaps):7.0f}   "
          f"max {max(gaps):7.0f}   sd {st.pstdev(gaps):6.0f}")

    # ---- feasibility sweep -------------------------------------------------
    print("\nfeasible designs, all five constraints, by reactivity basis")
    print(f"{'k_max':>7}  {'assembly':>9}  {'core':>9}")
    print("-" * 30)
    sweep = []
    for lim in limits:
        na = sum(feasible(r, r["k_bol"], lim, args.k_min,
                          args.f_max, args.enr_max)[0] for r in raw)
        nc = sum(feasible(r, r["keff_core_bol"], lim, args.k_min,
                          args.f_max, args.enr_max)[0] for r in raw)
        sweep.append(dict(k_max=lim, n_feasible_assembly=na,
                          n_feasible_core=nc, n_total=len(raw)))
        print(f"{lim:7.2f}  {na:9d}  {nc:9d}")

    # ---- which constraint binds, at the campaign's own limit ---------------
    print("\nbinding constraint at k_max = 1.35, core basis")
    names = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom"]
    counts = {n: 0 for n in names}
    for r in raw:
        ok, g = feasible(r, r["keff_core_bol"], 1.35, args.k_min,
                         args.f_max, args.enr_max)
        if not ok:
            for n, v in zip(names, g):
                if v > 0.0:
                    counts[n] += 1
    for n in names:
        print(f"  {n:8s} violated by {counts[n]:3d} of {len(raw)}")

    # ---- per-design table --------------------------------------------------
    csv_path = outdir / f"{args.label}_kbasis.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "k_bol_assembly", "keff_core_bol", "gap_pcm",
                    "peaking_core", "cycle_length_as_recorded",
                    "feasible_assembly_135", "feasible_core_135"])
        for i, r in enumerate(raw):
            w.writerow([
                i, f"{r['k_bol']:.5f}", f"{r['keff_core_bol']:.5f}",
                f"{1.0e5 * (r['k_bol'] - r['keff_core_bol']):.0f}",
                f"{r['peaking']:.4f}", f"{r.get('cycle_length', float('nan')):.1f}",
                int(feasible(r, r["k_bol"], 1.35, args.k_min,
                             args.f_max, args.enr_max)[0]),
                int(feasible(r, r["keff_core_bol"], 1.35, args.k_min,
                             args.f_max, args.enr_max)[0]),
            ])

    summary = dict(
        label=args.label, checkpoint=str(args.checkpoint),
        n_evaluations=len(raw),
        gap_pcm=dict(min=min(gaps), mean=st.mean(gaps), max=max(gaps),
                     sd=st.pstdev(gaps)),
        sweep=sweep,
        binding_at_135_core=counts,
        limits_used=dict(k_min=args.k_min, f_max=args.f_max,
                         enr_max=args.enr_max),
        note=("cycle_length is copied as recorded and is NOT corrected for "
              "the OpenMC 0.15.3 write_rates depletion restart defect. Use "
              "the salvage output for any statement about cycle length. The "
              "reactivity and peaking quantities used here are beginning-of-"
              "life and are unaffected by that defect."))
    (outdir / f"{args.label}_kbasis_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"\nwrote {csv_path}")
    print(f"wrote {outdir / (args.label + '_kbasis_summary.json')}")


if __name__ == "__main__":
    main()
