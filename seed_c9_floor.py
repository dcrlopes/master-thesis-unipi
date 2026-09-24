#!/usr/bin/env python3
"""
seed_c9_floor.py -- the Campaign 9 archive as a seed for the same campaign run
under a raised mission floor. Arithmetic only, no model, no transport.

WHY A CONSTANT FLOOR
    Seven Campaign 9 designs were depleted with eight axial layers. Their
    resolved cycle length is 0.745 to 0.847 of the assembly-level value,
    mean 0.805. Any floor between 1947 and 2283 d applied to the assembly-level
    value reproduces all seven measured verdicts against 1826 d. The mean loss
    gives 1826 / 0.805 = 2267 d; the campaign uses 2270 d.

WHAT CHANGES IN THE ARCHIVE
    Only g_efpd, which the evaluator defines as efpd_req - cycle_length. The
    loop rebuilds its constraint matrix from the stored records
    (reactor_optimization._seed_from_raw), so the seed must carry the value
    the new floor gives, or the surrogate would learn the old feasibility.
    Objectives and every other constraint are untouched. The Campaign 9 value
    is kept as g_efpd_c9.

USAGE
    python seed_c9_floor.py --floor 2270 --n-seed 60 --out out_c9f      # all evaluations
    python seed_c9_floor.py --floor 2270 --n-seed 24 --out out_c9f_doe  # design of experiments only
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default="out_c9/optimization_checkpoint.json")
ap.add_argument("--floor", type=float, default=2270.0)
ap.add_argument("--n-seed", type=int, default=60, help="60 for the whole archive, 24 for the design of experiments")
ap.add_argument("--out", default="out_c9f")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

ck = json.loads(Path(a.src).read_text())
con = ck["constraint_names"]
raw = ck["all_raw"][:a.n_seed]
others = [c for c in con if c != "g_efpd"]

feas_old = feas_new = 0
for r in raw:
    t = float(r["cycle_length"])
    r["g_efpd_c9"] = float(r["g_efpd"])
    r["g_efpd"] = a.floor - t
    rest = all(float(r.get(c, 0.0)) <= 1e-9 for c in others)
    feas_old += rest and r["g_efpd_c9"] <= 0
    feas_new += rest and r["g_efpd"] <= 0

ck["all_raw"] = raw
for k in ("all_X", "all_F", "all_G"):
    if k in ck and isinstance(ck[k], list):
        ck[k] = ck[k][:a.n_seed]
ck["hv_history"] = []                      # the hypervolume history restarts under the new floor
ck["n_real_evaluations"] = len(raw)
ck["meta"]["c9_floor"] = {"seeded_from": a.src, "n_seed": a.n_seed, "floor_efpd": a.floor,
                          "rule": "g_efpd = floor - cycle_length; measured axial loss, mean ratio 0.805"}
kept = [i for i, r in enumerate(raw) if r["g_efpd"] <= 0 and all(float(r.get(c, 0)) <= 1e-9 for c in others)]
print(f"seed: {len(raw)} evaluations, floor {a.floor:.0f} d")
print(f"feasible: {feas_old} at 1826 d -> {feas_new} at {a.floor:.0f} d: {kept}")
if a.dry_run:
    print("dry run, nothing written"); raise SystemExit(0)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "optimization_checkpoint.json").write_text(json.dumps(ck, indent=1))
print(f"wrote {out}/optimization_checkpoint.json")
