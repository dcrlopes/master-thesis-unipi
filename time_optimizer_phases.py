#!/usr/bin/env python
"""
time_optimizer_phases.py -- measure the cost of every optimiser phase of a
finished campaign by REPLAYING it offline from its checkpoint.

The active-learning loop in reactor_optimization.py is deterministic given the
archive and the seed: the Gaussian-process (GP) fit uses random_state=0 and the
NSGA-II (Non-Dominated Sorting Genetic Algorithm II) search uses
seed = cfg.seed + iteration. The checkpoint stores every evaluated design
(all_raw) in evaluation order, so the state of the archive at the start of
iteration k is simply its first n_init + k * n_infill rows. Re-running the
cheap phases on that prefix reproduces, on the same machine, what the campaign
paid for them. No OpenMC call is made.

Phases timed per iteration
    fit_obj     GP ensemble on the objectives           (one GP per objective)
    fit_con     GP ensemble on the constraints          (one GP per constraint)
    nsga        NSGA-II on the surrogate (pop 60 x gen 80 in production)
    acq         candidate extraction, predictive std, de-duplication
    hv          hypervolume of the archive after the infill was added
    ckpt        writing the checkpoint JSON

Usage (from the simulation-code repository root, inside openmc-env):

    python time_optimizer_phases.py \
        --checkpoint out_c3/optimization_checkpoint.json \
        --campaign C3 --n-init 36 --n-infill 6 --out phases_c3

Flags
    --checkpoint   optimisation checkpoint of the finished campaign
    --campaign     label for the tables
    --n-init       DOE size of that campaign (24 for C2, 36 for C3 and C4)
    --n-infill     real evaluations per iteration (6 in every campaign)
    --nsga-pop     NSGA-II population, default 60 (production value)
    --nsga-gen     NSGA-II generations, default 80 (production value)
    --seed         cfg.seed used in production, default 1
    --repeats      timing repetitions per phase, default 1 (use 3 for the thesis)
    --out          output prefix, writes <out>_phases.csv, <out>_summary.json,
                   <out>_table.tex

Run it on the machine that ran the campaign (wks720), because the numbers are
machine-dependent. The script prints the hostname it ran on for the record.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.getcwd())

from sklearn.exceptions import ConvergenceWarning           # noqa: E402
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from pymoo.algorithms.moo.nsga2 import NSGA2                # noqa: E402
from pymoo.operators.sampling.lhs import LHS                # noqa: E402
from pymoo.optimize import minimize                         # noqa: E402

from reactor_optimization import (ActiveLearningMOO, AnalyticEvaluator,  # noqa: E402
                                  OptimizerConfig, _SurrogateProblem,
                                  example_reactor_problem)

PHASES = ("fit_obj", "fit_con", "nsga", "acq", "hv", "ckpt")


def timed(fn, repeats: int):
    """Mean wall time of fn() over `repeats` calls, plus the last return value."""
    ts, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), out


def replay(args) -> tuple[list[dict], dict]:
    spec = example_reactor_problem()
    cfg = OptimizerConfig(n_init=args.n_init, n_iter=0, n_infill=args.n_infill,
                          nsga_pop=args.nsga_pop, nsga_gen=args.nsga_gen,
                          surrogate="gp", seed=args.seed)
    opt = ActiveLearningMOO(spec, AnalyticEvaluator(spec), cfg)
    n_total = opt.load_checkpoint(args.checkpoint)
    X_all, F_all, G_all = opt.X.copy(), opt.F.copy(), opt.G.copy()
    raw_all = list(opt.raw)
    hv_ref = opt._hv_ref_frozen

    n_iter = (n_total - args.n_init) // args.n_infill
    if n_iter <= 0:
        sys.exit(f"checkpoint holds {n_total} evaluations, not more than the "
                 f"DOE of {args.n_init}: nothing to replay")
    if args.n_init + n_iter * args.n_infill != n_total:
        print(f"WARNING: {n_total} evaluations is not n_init + k * n_infill "
              f"({args.n_init} + k * {args.n_infill}). Check --n-init/--n-infill. "
              f"Replaying {n_iter} complete iterations.")

    rows = []
    for it in range(n_iter):
        n = args.n_init + it * args.n_infill
        Xn, Fn, Gn = X_all[:n], F_all[:n], G_all[:n]
        row = dict(iteration=it + 1, n_archive=n)

        row["fit_obj"], obj_sur = timed(
            lambda: opt._new_surrogate().fit(Xn, Fn), args.repeats)
        row["fit_con"], con_sur = timed(
            lambda: opt._new_surrogate().fit(Xn, Gn), args.repeats)

        def _nsga():
            prob = _SurrogateProblem(spec, obj_sur, con_sur)
            algo = NSGA2(pop_size=cfg.nsga_pop, sampling=LHS())
            return minimize(prob, algo, ("n_gen", cfg.nsga_gen),
                            seed=cfg.seed + it, verbose=False)
        row["nsga"], res = timed(_nsga, args.repeats)

        def _acq():
            cand = opt._least_infeasible_candidates(res)
            if cand is None or cand.shape[0] == 0:
                cand = np.atleast_2d(spec.design_space.lhs(
                    max(cfg.n_infill * 4, 32), seed=cfg.seed + 777 + it))
            _, std = obj_sur.predict(cand)
            score = (std / (std.max(axis=0) + 1e-12)).sum(axis=1)
            order = np.argsort(-score)
            chosen = []
            for idx in order:
                x = cand[idx]
                if Xn.size and np.min(np.linalg.norm(Xn - x, axis=1)) < 1e-6:
                    continue
                chosen.append(x)
                if len(chosen) >= cfg.n_infill:
                    break
            return np.array(chosen), cand.shape[0]
        row["acq"], (chosen, n_cand) = timed(_acq, args.repeats)
        row["n_candidates"] = int(n_cand)
        row["nsga_feasible"] = bool(getattr(res, "X", None) is not None)

        # archive AFTER the infill of this iteration, as the loop computes HV
        m = n + args.n_infill
        opt.X, opt.F, opt.G = X_all[:m], F_all[:m], G_all[:m]
        opt.raw = raw_all[:m]
        opt._hv_ref_frozen = hv_ref
        row["hv"], hv_val = timed(opt._hv, args.repeats)
        row["hv_value"] = float(hv_val)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            row["ckpt"], _ = timed(lambda: opt.save_checkpoint(str(p)), args.repeats)

        row["optimiser_total"] = sum(row[ph] for ph in PHASES)
        rows.append(row)
        print(f"iter {it + 1:2d} (n={n:3d}): " +
              "  ".join(f"{ph}={row[ph]:6.2f}s" for ph in PHASES) +
              f"  total={row['optimiser_total']:6.2f}s")

    import sklearn
    import pymoo
    summary = dict(
        campaign=args.campaign,
        checkpoint=str(Path(args.checkpoint).resolve()),
        host=platform.node(), cpu_count=os.cpu_count(),
        sklearn=sklearn.__version__, pymoo=pymoo.__version__,
        n_evaluations=int(n_total), n_iterations=int(n_iter),
        nsga_pop=cfg.nsga_pop, nsga_gen=cfg.nsga_gen, repeats=args.repeats,
        **{f"mean_{ph}_s": float(np.mean([r[ph] for r in rows])) for ph in PHASES},
        **{f"max_{ph}_s": float(np.max([r[ph] for r in rows])) for ph in PHASES},
        mean_optimiser_total_s=float(np.mean([r["optimiser_total"] for r in rows])),
        sum_optimiser_total_s=float(np.sum([r["optimiser_total"] for r in rows])),
    )
    return rows, summary


def write_tex(s: dict, path: str) -> None:
    tex = f"""% auto-generated by time_optimizer_phases.py -- {s['campaign']}
% host: {s['host']} ({s['cpu_count']} CPUs) | scikit-learn {s['sklearn']} | pymoo {s['pymoo']}
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Optimiser phases of {s['campaign']}, replayed offline from the
  checkpoint on the same host. Mean over {s['n_iterations']} iterations,
  {s['repeats']} timing repetition(s) each.}}
  \\label{{tab:phases-{s['campaign'].lower().replace(' ', '-')}}}
  \\begin{{tabular}}{{lrr}}
    \\toprule
    Phase & mean [s] & max [s] \\\\
    \\midrule
    GP fit, objectives & {s['mean_fit_obj_s']:.1f} & {s['max_fit_obj_s']:.1f} \\\\
    GP fit, constraints & {s['mean_fit_con_s']:.1f} & {s['max_fit_con_s']:.1f} \\\\
    NSGA-II on surrogate ({s['nsga_pop']}$\\times${s['nsga_gen']}) & {s['mean_nsga_s']:.1f} & {s['max_nsga_s']:.1f} \\\\
    Acquisition & {s['mean_acq_s']:.2f} & {s['max_acq_s']:.2f} \\\\
    Hypervolume & {s['mean_hv_s']:.3f} & {s['max_hv_s']:.3f} \\\\
    Checkpoint write & {s['mean_ckpt_s']:.2f} & {s['max_ckpt_s']:.2f} \\\\
    \\midrule
    Optimiser total per iteration & {s['mean_optimiser_total_s']:.1f} & \\\\
    Optimiser total per campaign & {s['sum_optimiser_total_s']:.0f} & \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w") as f:
        f.write(tex)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--n-init", type=int, required=True)
    ap.add_argument("--n-infill", type=int, default=6)
    ap.add_argument("--nsga-pop", type=int, default=60)
    ap.add_argument("--nsga-gen", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="phases")
    args = ap.parse_args()

    rows, summary = replay(args)
    with open(f"{args.out}_phases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"{args.out}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_tex(summary, f"{args.out}_table.tex")
    print(f"\n{summary['campaign']} on {summary['host']}: optimiser costs "
          f"{summary['mean_optimiser_total_s']:.1f} s per iteration on average, "
          f"{summary['sum_optimiser_total_s']:.0f} s over the campaign.")
    print(f"written: {args.out}_phases.csv, {args.out}_summary.json, {args.out}_table.tex")


if __name__ == "__main__":
    main()
