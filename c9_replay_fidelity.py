#!/usr/bin/env python3
"""c9_replay_fidelity.py -- how closely a replay of the Campaign 9 acquisition
reproduces the designs the campaign evaluated.

For every infill block, the distance from each design the campaign evaluated
to the nearest pick of the replay, in the unit design box (each variable
divided by its search range, the norm divided by sqrt(n_var), the measure the
loop uses for its minimum separation of 0.14). Also the count of picks below
1 wt% gadolinia and the mean pairwise separation, the two statistics compared
between rules.

USAGE (repository root)
    python c9_replay_fidelity.py --replay figs_c9_acq/c9_acquisition_replay_wks720_asrun.json \
        --out figs_c9_acq/c9_replay_fidelity_wks720
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reactor_optimization import campaign9_problem

ap = argparse.ArgumentParser()
ap.add_argument("--replay", required=True)
ap.add_argument("--rule", default="asrun")
ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
ap.add_argument("--out", required=True, help="output stem, .json and .txt are written")
a = ap.parse_args()

rep = json.loads(Path(a.replay).read_text())
ck = json.loads(Path(a.checkpoint).read_text())
names, raw = ck["design_variables"], ck["all_raw"]
ds = campaign9_problem(1826.0).design_space
span = np.asarray(ds.xu, float) - np.asarray(ds.xl, float)
R = rep["rules"][a.rule]
rows, dists = [], []
for b in R["blocks"]:
    s, e = rep["blocks"][b["block"] - 1]
    A = np.array([[float(r[n]) for n in names] for r in raw[s:e]])
    P = np.array([[p[n] for n in names] for p in b["picks"]])
    d = [float(np.min(np.linalg.norm((P - x) / span, axis=1)) / np.sqrt(len(names))) for x in A]
    dists += d
    rows.append(dict(block=b["block"], evaluated=list(range(s, e)), nearest_distance=d,
                     picks_below_1wt=int(b["n_low_gd"]), mean_sep=float(b["mean_sep"])))
dists = np.array(dists)
summary = dict(replay=a.replay, rule=a.rule, n=len(dists), n_identical=int(np.sum(dists < 1e-6)),
               median=float(np.median(dists)), max=float(np.max(dists)),
               below_0_075=int(np.sum(dists < 0.075)), loop_min_sep=0.14,
               picks_below_1wt=int(sum(r["picks_below_1wt"] for r in rows)),
               mean_sep=float(np.mean([r["mean_sep"] for r in rows])))
Path(a.out + ".json").write_text(json.dumps(dict(summary=summary, blocks=rows), indent=1))
L = [f"replay {a.replay}, rule {a.rule}",
     "distance from each evaluated design to the nearest replayed pick, unit design box:"]
for r in rows:
    L.append(f"  block {r['block']}: " + " ".join(f"{x:.3f}" for x in r["nearest_distance"])
             + f"   | below 1 wt% {r['picks_below_1wt']}, mean separation {r['mean_sep']:.3f}")
L.append(f"identical {summary['n_identical']} of {summary['n']}; median {summary['median']:.3f}, "
         f"max {summary['max']:.3f}, {summary['below_0_075']} of {summary['n']} below 0.075 "
         f"(the loop's own separation is 0.14)")
L.append(f"picks below 1 wt% gadolinia {summary['picks_below_1wt']} of {summary['n']}, "
         f"mean separation {summary['mean_sep']:.3f}")
Path(a.out + ".txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
