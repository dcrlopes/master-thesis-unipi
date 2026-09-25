#!/usr/bin/env python3
r"""
c9_eval_predicted_front.py -- evaluate, with the Campaign 9 transport
evaluator, the designs of the Pareto front that the surrogate predicts after
the full 60-evaluation archive, so that the predicted front of
c9_gp_predicted_fronts.py can be compared with measured objectives instead
of being read as a prediction only.

Two modes.

  --make     select designs along the final predicted front (from
             figs_c9_gp/gp_predicted_fronts.json, cut 60), round the rod
             count to an integer, write the enumeration list for
             run_optimization.py --eval-list, and print the run command.
             No OpenMC.

  --compare  after the run, read the enumeration archive and, for each
             listed design, put the measured peaking, c_max, cycle length
             and feasibility next to the surrogate's prediction. Writes
             c9_post/predicted_front_eval.json and .txt. No OpenMC.

USAGE (repository root)
  python c9_eval_predicted_front.py --make            # writes c9_post/predicted_front_evallist.json
  python c9_eval_predicted_front.py --compare         # after the run on wks720

The run itself (wks720, conda env openmc-env, the Campaign 9 flags of
run_c9.sh, enumeration mode):
  setsid nohup python -u run_optimization.py --out out_c9_pred --workdir openmc_runs_c9_pred \
      --ktarget-table ktarget_table_c8.json --k-basis core --k-max 1.166 --k-min 1.02 \
      --f-max 1.65 --enr-max 16 --ctrl-margin 1000 --objective-set c9 --efpd-req 1826 \
      --boron-objective floor --boron-step 2000 --boron-top 3000 --boron-ceiling 2763 \
      --hump-noise 400 --threads 64 \
      --eval-list c9_post/predicted_front_evallist.json > out_c9_pred.log 2>&1 < /dev/null &
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--make", action="store_true")
ap.add_argument("--compare", action="store_true")
ap.add_argument("--fronts", default="figs_c9_gp/gp_predicted_fronts.json")
ap.add_argument("--cut", default="60", help="archive size of the refit to evaluate")
ap.add_argument("--n", type=int, default=6, help="designs selected along the predicted front")
ap.add_argument("--list", default="c9_post/predicted_front_evallist.json")
ap.add_argument("--ckpt", default="out_c9_pred/optimization_checkpoint.json")
ap.add_argument("--out", default="c9_post/predicted_front_eval")
a = ap.parse_args()

NAMES = ["enrich", "gd_wt", "refl_thick", "gd_pins"]
fr = json.loads(Path(a.fronts).read_text())
P = fr["predicted"][a.cut]
F, X = np.array(P["F"]), np.array(P["X"])
o = np.argsort(F[:, 0])
F, X = F[o], X[o]


def pick(F, n):
    """n designs spread along the front: the two ends plus the designs
    nearest to equally spaced quantiles of the peaking axis."""
    q = np.linspace(F[:, 0].min(), F[:, 0].max(), n)
    idx = sorted({int(np.abs(F[:, 0] - v).argmin()) for v in q})
    return idx


if a.make:
    idx = pick(F, a.n)
    designs = []
    for i in idx:
        d = {k: float(v) for k, v in zip(NAMES, X[i])}
        d["gd_pins"] = float(round(d["gd_pins"]))     # the evaluator uses an integer rod count
        d["predicted_peaking"] = float(F[i, 0])       # extra keys are ignored by load_eval_list
        d["predicted_c_max"] = float(F[i, 1])
        designs.append(d)
    Path(a.list).parent.mkdir(parents=True, exist_ok=True)
    Path(a.list).write_text(json.dumps({"source": a.fronts, "cut": a.cut, "designs": designs}, indent=1))
    print(f"{len(designs)} designs of the predicted front after {a.cut} evaluations -> {a.list}")
    for d in designs:
        print("  " + "  ".join(f"{k}={d[k]:.4g}" for k in NAMES)
              + f"  predicted F={d['predicted_peaking']:.3f} c_max={d['predicted_c_max']:.0f} ppm")
    print("\nrun on wks720:\n" + __doc__.split("The run itself")[1].split(":\n", 1)[1])

if a.compare:
    lst = json.loads(Path(a.list).read_text())["designs"]
    ck = json.loads(Path(a.ckpt).read_text())
    raw, cons = ck["all_raw"], ck["constraint_names"]
    rows = []
    for d in lst:
        x = np.array([d[k] for k in NAMES])
        hit = [r for r in raw if np.allclose([float(r[k]) for k in NAMES], x, atol=1e-6)]
        if not hit:
            rows.append({**{k: d[k] for k in NAMES}, "status": "not evaluated"})
            continue
        r = hit[0]
        viol = [g for g in cons if float(r.get(g, 0.0)) > 0]
        rows.append({**{k: d[k] for k in NAMES},
                     "predicted_peaking": d["predicted_peaking"], "measured_peaking": float(r["peaking"]),
                     "predicted_c_max": d["predicted_c_max"], "measured_c_max": float(r["c_max"]),
                     "cycle_length": float(r["cycle_length"]), "violated": viol, "status": "evaluated"})
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    lines = [f"{'enrich':>7} {'gd_wt':>6} {'refl':>5} {'pins':>4} | {'F pred':>6} {'F meas':>6} {'dF':>6} | "
             f"{'c pred':>6} {'c meas':>6} {'dc':>6} | {'EFPD':>6} violated"]
    for r in rows:
        if r["status"] != "evaluated":
            lines.append(f"{r['enrich']:7.3f} {r['gd_wt']:6.3f} {r['refl_thick']:5.2f} {r['gd_pins']:4.0f} | not evaluated")
            continue
        lines.append(f"{r['enrich']:7.3f} {r['gd_wt']:6.3f} {r['refl_thick']:5.2f} {r['gd_pins']:4.0f} | "
                     f"{r['predicted_peaking']:6.3f} {r['measured_peaking']:6.3f} {r['measured_peaking']-r['predicted_peaking']:+6.3f} | "
                     f"{r['predicted_c_max']:6.0f} {r['measured_c_max']:6.0f} {r['measured_c_max']-r['predicted_c_max']:+6.0f} | "
                     f"{r['cycle_length']:6.0f} {','.join(r['violated']) or 'none'}")
    Path(a.out + ".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {a.out}.json and .txt")
