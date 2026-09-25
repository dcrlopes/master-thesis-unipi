#!/usr/bin/env python3
"""
seed_c9_axial.py -- rescore the cycle-length constraint of a checkpoint under
the axial-ratio floor of axial_ratio_model.py. Arithmetic only, no transport.

RULE, per record
    measured in 3D   g_efpd = E_mission - E_3D            (the measurement itself)
    otherwise        g_efpd = E_req(record) - E_asm       (the model, with its margin)

    The mission value E_mission - E_asm is kept as g_efpd_mission, and the
    values of the earlier readings, g_efpd_c9 (1826 d) and g_efpd_floor
    (2270 d), are kept where present. The loop rebuilds its constraint matrix
    from the records on resume (reactor_optimization._seed_from_raw), so the
    records must carry the value the rule gives.

TWO MODES
    --src ... --out DIR     a new seed from another campaign's checkpoint: the
                            hypervolume history restarts, hv_ref is kept so the
                            three Campaign 9 histories share one reference
    --inplace CKPT          between blocks of the running continuation, after
                            the model was refitted with the newly confirmed
                            designs: only g_efpd of the records changes

USAGE
    python seed_c9_axial.py --src out_c9f/optimization_checkpoint.json --model axial_ratio_model.json --out out_c9a
    python seed_c9_axial.py --inplace out_c9a/optimization_checkpoint.json --model axial_ratio_model.json
    add --dry-run to print without writing
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import axial_ratio_model as arm

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default=None)
ap.add_argument("--out", default="out_c9a")
ap.add_argument("--inplace", default=None, metavar="CHECKPOINT")
ap.add_argument("--model", default="axial_ratio_model.json")
ap.add_argument("--mission", type=float, default=arm.MISSION)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if bool(a.src) == bool(a.inplace):
    raise SystemExit("give exactly one of --src (new seed) or --inplace (rescore a running checkpoint)")

path = Path(a.inplace or a.src)
ck = json.loads(path.read_text())
model = arm.load(a.model)
con = ck["constraint_names"]
others = [c for c in con if c != "g_efpd"]
raw = ck["all_raw"]


def feasible(r, g):
    return g <= 0 and all(float(r.get(c, 0.0)) <= 1e-9 for c in others)


n_meas = 0
before = sum(feasible(r, float(r["g_efpd"])) for r in raw)
mission_feas = 0
# the floor the source archive was scored under: 2270 d for the first
# continuation, 1826 d for Campaign 9 itself. Its reading is kept per record.
prev_req = float(((ck.get("meta") or {}).get("campaign9") or {}).get("efpd_req", a.mission))
for r in raw:
    E = float(r["cycle_length"])
    r.setdefault("g_efpd_mission", a.mission - E)
    if abs(prev_req - a.mission) > 1e-9 and "g_efpd_floor" not in r and "axial_measured" not in r:
        r["g_efpd_floor"] = float(r["g_efpd"])
        r["efpd_req_floor"] = prev_req
    m = arm.measured_lookup(model, r)
    if m is not None:
        n_meas += 1
        r.update(axial_measured=True, efpd_3d=m["efpd_3d"], sigma_3d=m["sigma_3d"],
                 axial_ratio_pred=m["ratio"], axial_req_efpd=a.mission / m["ratio"],
                 cycle_length_axial=m["efpd_3d"], axial_clamped=[], axial_source=m["source"])
        r["g_efpd"] = a.mission - m["efpd_3d"]
    else:
        ax = arm.requirement(model, r, a.mission)
        r.update(ax)
        r["axial_measured"] = False
        r["g_efpd"] = ax["axial_req_efpd"] - E
    mission_feas += feasible(r, float(r["g_efpd_mission"]))
after = [i for i, r in enumerate(raw) if feasible(r, float(r["g_efpd"]))]
meas_feas = [i for i in after if raw[i]["axial_measured"]]

print(f"{path}: {len(raw)} records, model {arm.describe(model)}")
print(f"  measured in 3D: {n_meas} records take their measured cycle")
print(f"  feasible: {mission_feas} under the mission alone, {before} before this rescoring, {len(after)} now: {after}")
print(f"  of which measured: {meas_feas}")
if a.dry_run:
    print("dry run, nothing written")
    raise SystemExit(0)

stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
entry = dict(utc=stamp, model=arm.summary(model), mission_efpd=a.mission, n_measured=n_meas,
             feasible=after)
if a.src:
    ck["hv_history"] = []                      # a new reading of the constraint restarts the history
    ck["n_real_evaluations"] = len(raw)
    ck.setdefault("meta", {})["c9_axial"] = dict(seeded_from=a.src, rescored=[entry],
                                                 rule="g_efpd = E_req(record) - E_asm from axial_ratio_model.py; "
                                                      "measured designs take E_mission - E_3D")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "optimization_checkpoint.json"
else:
    ck.setdefault("meta", {}).setdefault("c9_axial", {}).setdefault("rescored", []).append(entry)
    dest = path
dest.write_text(json.dumps(ck, indent=1))
print(f"wrote {dest}")
