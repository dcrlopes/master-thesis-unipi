#!/usr/bin/env python3
"""
peaking_noise_test.py -- is the F_dH spread on the Pareto front physics, or
Monte Carlo (MC) noise?

WHY THIS IS NEEDED
------------------
The uncapped front spans only ~5% in F_dH (the radial peaking factor) while
spanning ~18% in EFPD (Effective Full Power Days). F_dH is computed by
`ReactorEvaluator._bol_peaking()` as the MAXIMUM over a 17x17 pin-fission mesh
tally -- and a maximum over many noisy cells is both noisy AND biased upward.
If the seed-to-seed scatter is comparable to the front's spread, the ORDERING
of the front is not meaningful.

Two independent estimates are produced:

  (1) SEED REPLICATION -- re-runs the same design with different random seeds
      and reports the spread of F_dH. This is the honest, total estimate: it
      captures everything the tally uncertainty misses (which pin happens to
      win the maximum can change between seeds).

  (2) TALLY UNCERTAINTY -- OpenMC (Open source Monte Carlo particle transport
      code) already computes a standard deviation for every mesh cell, but
      `_bol_peaking()` throws it away. This script keeps it, giving the
      relative error of the winning pin for free.

WHAT IT DOES NOT DO
-------------------
Only the BOL (Beginning of Life) fixed-composition k-eigenvalue transport is
repeated -- NO depletion. That is the whole point: F_dH is evaluated on a
FRESH assembly, so one replicate costs minutes, not hours. Nothing in the
campaign is modified; a separate workdir is used.

USAGE
  conda activate openmc-env
  python peaking_noise_test.py \
      --checkpoint results_campaign2/block2/out/optimization_checkpoint.json \
      --n-designs 3 --seeds 5 --threads 64 --out noise_test
"""
import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
import openmc

import reactor_model as rm

# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--n-designs", type=int, default=3,
                help="how many Pareto designs to replicate (lowest-F_dH first)")
ap.add_argument("--seeds", type=int, default=5,
                help="replicates per design (>=3 for a usable spread)")
ap.add_argument("--particles", type=int, default=None,
                help="override the campaign particle count, to test whether "
                     "more particles shrink the scatter")
ap.add_argument("--batches", type=int, default=None)
ap.add_argument("--inactive", type=int, default=None)
ap.add_argument("--threads", type=int, default=None,
                help="OpenMP (Open Multi-Processing) threads; 64 on wks720")
ap.add_argument("--out", default="noise_test")
args = ap.parse_args()

if args.threads:
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

ck = json.loads(Path(args.checkpoint).read_text())
raw = ck["all_raw"]
con = ck.get("constraint_names", [])
dv = ck["design_variables"]
meta = ck.get("meta") or {}
tr_ck = dict(meta.get("transport") or {"particles": 4000, "batches": 60,
                                       "inactive": 20})
transport = dict(
    particles=args.particles or tr_ck["particles"],
    batches=args.batches or tr_ck["batches"],
    inactive=args.inactive or tr_ck["inactive"],
)


def feasible(r, tol=1e-9):
    return all(float(r.get(c, 0.0)) <= tol for c in con)


def pareto(recs):
    out = []
    for a in recs:
        if not any((b["cycle_length"] >= a["cycle_length"]
                    and b["peaking"] <= a["peaking"]
                    and (b["cycle_length"] > a["cycle_length"]
                         or b["peaking"] < a["peaking"]))
                   for b in recs):
            out.append(a)
    return out


feas = [r for r in raw if feasible(r)]
front = sorted(pareto(feas), key=lambda r: r["peaking"])
picks = front[:args.n_designs]
front_spread = (max(r["peaking"] for r in front)
                - min(r["peaking"] for r in front))

outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# One replicate: rebuild the BOL model, set the seed, tally pin fission.       #
# Mirrors ReactorEvaluator._bol_peaking() exactly, but keeps std_dev.          #
# --------------------------------------------------------------------------- #
def peaking_once(design, seed, case_dir, geo, op):
    model, _cells, _lat = rm.make_assembly_model(
        design, op, geo, bc="reflective", **transport)
    model.settings.seed = seed          # _settings() hard-codes seed=1; the
                                        # campaign therefore used ONE stream

    N = geo.lattice
    pitch = design.get("pitch", 1.26)
    half = N * pitch / 2.0

    mesh = openmc.RegularMesh()
    mesh.dimension = (N, N)
    mesh.lower_left = (-half, -half)
    mesh.upper_right = (half, half)

    t = openmc.Tally(name="pin_fission")
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ["fission"]
    model.tallies = openmc.Tallies([t])

    sp_path = model.run(cwd=str(case_dir), output=False)
    with openmc.StatePoint(sp_path) as sp:
        tal = sp.get_tally(name="pin_fission")
        val = tal.get_values(scores=["fission"]).reshape((N, N))
        sd = tal.get_values(scores=["fission"], value="std_dev").reshape((N, N))
        keff = float(sp.keff.nominal_value)

    fm = np.ma.masked_equal(val, 0.0)
    norm = fm / fm.mean()
    fdh = float(norm.max())
    idx = np.unravel_index(np.ma.argmax(norm), norm.shape)
    rel_err = float(sd[idx] / val[idx])     # relative error of the hot pin
    return fdh, rel_err, keff, idx


geo = rm.Geometry17x17()
op = rm.Operating()
rows = []

print("=" * 78)
print(f"transport: {transport['particles']} particles x {transport['batches']} "
      f"batches ({transport['inactive']} inactive)   seeds: {args.seeds}")
print(f"front size {len(front)}, F_dH spread across the front = {front_spread:.4f}")
print("=" * 78)

for di, rec in enumerate(picks):
    design = {k: float(rec[k]) for k in dv}
    tag = f"design{di}"
    print(f"\n[{tag}] stored F_dH={rec['peaking']:.4f}  "
          f"EFPD={rec['cycle_length']:.0f}  "
          + "  ".join(f"{k}={design[k]:.3f}" for k in dv))

    vals, errs, keffs = [], [], []
    for s in range(1, args.seeds + 1):
        case = outdir / tag / f"seed{s}"
        case.mkdir(parents=True, exist_ok=True)
        fdh, rel, keff, idx = peaking_once(design, s, case, geo, op)
        vals.append(fdh); errs.append(rel); keffs.append(keff)
        print(f"   seed {s:2d}: F_dH={fdh:.4f}  hot pin {idx}  "
              f"rel.err={rel * 100:.2f}%  k={keff:.5f}")

    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    rng = max(vals) - min(vals)
    rows.append(dict(design=tag, stored=rec["peaking"], mean=mean, sd=sd,
                     rng=rng, rel_err=statistics.mean(errs),
                     **{k: design[k] for k in dv}))
    print(f"   -> mean {mean:.4f}   sd {sd:.4f}   range {rng:.4f}   "
          f"mean hot-pin rel.err {statistics.mean(errs) * 100:.2f}%")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
worst = max(r["rng"] for r in rows)
worst_sd = max(r["sd"] for r in rows)
print(f"largest seed-to-seed RANGE in F_dH      : {worst:.4f}")
print(f"largest seed-to-seed STANDARD DEVIATION : {worst_sd:.4f}")
print(f"F_dH spread across the Pareto front     : {front_spread:.4f}")
ratio = worst / front_spread if front_spread else float("inf")
print(f"noise / signal (range basis)            : {ratio:.2f}")
print("-" * 78)
if ratio > 0.5:
    print("NOISE DOMINATES. The seed scatter is comparable to the entire front\n"
          "spread, so the ordering of the front is not statistically\n"
          "meaningful. Re-score the front designs at higher particle count\n"
          "before drawing physics conclusions from the F_dH ranking.")
elif ratio > 0.2:
    print("MARGINAL. Neighbouring front points are not separable, though the\n"
          "extremes are. Quote F_dH with an uncertainty band and avoid\n"
          "claiming a fine ranking.")
else:
    print("SIGNAL DOMINATES. The F_dH differences along the front exceed the\n"
          "MC scatter; the ranking can be trusted at this particle count.")
print("-" * 78)

csv = outdir / "noise_summary.csv"
cols = ["design", "stored", "mean", "sd", "rng", "rel_err"] + list(dv)
csv.write_text(",".join(cols) + "\n"
               + "\n".join(",".join(str(r[c]) for c in cols) for r in rows) + "\n")
print(f"CSV written: {csv}")
