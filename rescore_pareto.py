"""
rescore_pareto.py  --  two-fidelity stage 2
===========================================
Re-evaluate ONLY the feasible non-dominated (Pareto) designs of a finished
checkpoint at HIGH transport fidelity. The optimization explores cheaply
(default 4000 particles x 60 batches, where Monte Carlo noise on k is
~220 pcm and the peaking spread across the front is smaller than the tally
noise); this script then buys statistical certainty ONLY for the handful of
designs you will actually put in the thesis.

    python rescore_pareto.py --checkpoint out/optimization_checkpoint.json \
        --ktarget-table ktarget_table.json --particles 20000 --batches 100

Writes rescore.json (per design: exploration vs high-fidelity objectives) and
prints a side-by-side table. Pareto membership can CHANGE after re-scoring --
that is the point: noise-artifact points fall off the front.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def nondominated(cyc, pk, feas):
    """Physical-space Pareto mask: MAXIMISE cycle, MINIMISE peaking, among
    feasible points only."""
    F = np.column_stack([-np.asarray(cyc), np.asarray(pk)])
    nd = np.array(feas, dtype=bool).copy()
    idx = np.where(nd)[0]
    for i in idx:
        others = F[idx]
        dom = np.all(others <= F[i], axis=1) & np.any(others < F[i], axis=1)
        if dom.any():
            nd[i] = False
    return nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="optimization_checkpoint.json from the finished run")
    ap.add_argument("--ktarget-table", default=None,
                    help="the SAME k_target table the run used (Route B)")
    ap.add_argument("--ktarget", type=float, default=None,
                    help="Route A frozen k_target (if no table)")
    ap.add_argument("--particles", type=int, default=20000,
                    help="high-fidelity particles per batch (default 20000; "
                         "use 40000 for ~2x tighter k statistics)")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--inactive", type=int, default=25)
    ap.add_argument("--max-burnup", type=float, default=100.0,
                    help="MUST match the optimization run's --max-burnup")
    ap.add_argument("--dep-step", type=float, default=4.0,
                    help="MUST match the optimization run's --dep-step")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="rescore.json")
    args = ap.parse_args()

    # OpenMP (Open Multi-Processing) thread count must be set BEFORE openmc
    # is imported anywhere in the process.
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.threads)))
    from reactor_optimization import example_reactor_problem
    from openmc_evaluator import OpenMCEvaluator

    ck = json.loads(Path(args.checkpoint).read_text())
    raws = ck["all_raw"]
    cons = ck.get("constraint_names",
                  ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom"])
    cyc = [r["cycle_length"] for r in raws]
    pk = [r["peaking"] for r in raws]
    feas = [all(r[c] <= 1e-9 for c in cons) for r in raws]
    nd = nondominated(cyc, pk, feas)
    picks = [i for i in range(len(raws)) if nd[i]]
    print(f"{len(picks)} feasible Pareto designs to re-score "
          f"(of {len(raws)} archived evaluations)")

    kt = args.ktarget_table or args.ktarget
    if kt is None:
        raise SystemExit("give --ktarget-table (Route B) or --ktarget (Route A)")
    spec = example_reactor_problem()
    ev = OpenMCEvaluator(
        spec, k_target=kt,
        transport=dict(particles=args.particles, batches=args.batches,
                       inactive=args.inactive),
        max_burnup=args.max_burnup, dep_step=args.dep_step,
        workdir="rescore_runs")

    names = spec.design_space.names
    out = []
    print(f"{'case':>4} {'EFPD(4k)':>9} {'EFPD(hi)':>9} {'F_dh(4k)':>9} "
          f"{'F_dh(hi)':>9} {'cens':>5}")
    for i in picks:
        d = {n: float(raws[i][n]) for n in names}
        hi = ev.evaluate_one(d)
        ev.n_calls += 1
        out.append({"design": d, "exploration": raws[i], "high_fidelity": hi})
        print(f"{i:>4} {raws[i]['cycle_length']:>9.0f} "
              f"{hi['cycle_length']:>9.0f} {raws[i]['peaking']:>9.3f} "
              f"{hi['peaking']:>9.3f} {str(hi['censored']):>5}")

    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out} -- report HIGH-FIDELITY numbers in the thesis; "
          f"the exploration values were only for steering the search.")


if __name__ == "__main__":
    main()
