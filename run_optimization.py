"""
run_optimization.py  --  CLOUD / MANY-CORE EDITION
==================================================
Production driver: connect the REAL OpenMC (Open source Monte Carlo particle
transport code) evaluator (openmc_evaluator.py) to the dual-stage
active-learning optimizer (reactor_optimization.py) and run it.

    python run_optimization.py --smoke                    # tiny end-to-end test (~minutes)
    python run_optimization.py                            # full-fidelity session
    python run_optimization.py --resume out/optimization_checkpoint.json   # CONTINUE

ALWAYS run --smoke FIRST on a new machine. It exercises the entire chain
(build -> deplete -> extract -> surrogate -> NSGA-II -> infill -> save/plot)
with a handful of cheap, deliberately noisy evaluations, so you debug logic in
minutes instead of at full-fidelity OpenMC speed.

WHAT CHANGED vs. the laptop version (and NOTHING else changed -- the evaluator,
reactor model, and optimizer files are byte-identical):

  1. --threads N   (default: every core the machine reports)
     Sets the OMP_NUM_THREADS (OpenMP -- Open Multi-Processing -- thread count)
     environment variable BEFORE OpenMC is imported. This single variable
     governs BOTH kinds of transport in this pipeline:
       * openmc.Model.run() statepoint runs (the BOL -- Beginning Of Life --
         peaking tally), and
       * the in-memory transport that openmc.deplete.CoupledOperator drives
         through openmc.lib during depletion.
     That is why NO change is needed in openmc_evaluator.py or
     reactor_model.py: neither hardcodes a thread count, so the OpenMP runtime
     inherits this value everywhere. The imports of the OpenMC-dependent
     modules are therefore DEFERRED into main(), after the variable is set.

  2. --particles / --batches / --inactive
     Transport fidelity is now a command-line knob. The FULL-run default rises
     from 4000 particles x 60 batches to 20000 particles x 80 batches:
     on a ~64-core node you spend the extra speed on ~7.5x more active neutron
     histories per transport solve, i.e. ~sqrt(7.5) ~ 2.7x smaller Monte Carlo
     standard deviation on k. This directly attacks the known project risk that
     Monte Carlo statistical noise pollutes the surrogate and the Hypervolume
     (HV) convergence signal. (More particles per batch is ALSO what makes 64
     threads efficient: with only 4000 particles/batch, each thread would get
     ~60 particles and synchronization overhead would dominate.)

  3. Checkpoint metadata now records the transport fidelity, burnup schedule
     and thread count. On --resume the driver WARNS if the current fidelity
     differs from the checkpoint's -- the same philosophy as the existing
     k_target guard: never mix objective noise levels across accumulated
     sessions.

RESUME (accumulate across sessions)
-----------------------------------
Every run writes TWO files: the Pareto results (for plotting/reporting) AND a
full checkpoint (every evaluation) that a later run can continue from. With
--resume, the optimizer SKIPS the initial Latin-Hypercube DOE (Design Of
Experiments), retrains the surrogate on ALL previously-evaluated designs, and
adds `n_iter` more active-learning iterations on top. Run a batch, look at the
Hypervolume (HV) curve in the PNG, and if it is still climbing, --resume to add
more. Stop when HV plateaus -- THAT plateau, on your real OpenMC landscape, is
your true evaluation budget.

Keep --ktarget AND the fidelity flags THE SAME across resumed sessions; the
driver warns you if the checkpoint disagrees with the current settings.

RECOMMENDED PATTERN ON A PAID CLOUD MACHINE: run in CHUNKS instead of one
monolithic session, because the checkpoint is written at the END of a session:
    python run_optimization.py --iters 2 --out out          # 24 DOE + 12 infill
    python run_optimization.py --resume out/optimization_checkpoint.json --iters 3 --out out
    python run_optimization.py --resume out/optimization_checkpoint.json --iters 3 --out out
Total = 24 + 8x6 = 72 evaluations, with a durable checkpoint after every chunk.

Outputs (in --out, default current dir):
    optimization_results.json      Pareto designs + raw objectives + HV history
    optimization_checkpoint.json   FULL evaluation set (this is what --resume reads)
    optimization_openmc.png        objective space + convergence figure
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# ROUTE A default: a single FROZEN leakage-corrected EOC (End Of Cycle)
# target, the SAME for every design regardless of refl_thick. Measure ONCE
# (run measure_leakage_target.py on THIS machine) and paste the value here.
# The 4.55/4.05, pitch 1.26 reference on the OLD geometry gave ~1.085 -- the
# frozen 32-assembly geometry WILL give a different number, so re-measure
# before the first full session.
#
# ROUTE B (reflector thickness IS a real design variable): this constant is
# ignored -- pass --ktarget-table ktarget_vs_refl.json instead (the table
# sweep_ktarget.py writes). k_target is then interpolated per-design from
# design['refl_thick']. Never mix routes within one resumed session.
# ---------------------------------------------------------------------------
K_TARGET = 1.0556


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config to validate the whole chain end-to-end")
    ap.add_argument("--ktarget", type=float, default=K_TARGET,
                    help="ROUTE A: frozen leakage-corrected EOC target "
                         "(k_inf), one value for every design")
    ap.add_argument("--ktarget-table", default=None, metavar="PATH.json",
                    help="ROUTE B: path to the k_target-vs-refl_thick JSON "
                         "table written by sweep_ktarget.py. Overrides "
                         "--ktarget; k_target is interpolated per-design "
                         "from design['refl_thick'] inside the evaluator.")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--resume", metavar="CHECKPOINT.json", default=None,
                    help="continue from a checkpoint written by a previous run "
                         "(skips the initial DOE, retrains on all prior data, "
                         "and adds n_iter more active-learning iterations)")
    ap.add_argument("--checkpoint", default=None,
                    help="where to WRITE the resumable checkpoint "
                         "(default: optimization_checkpoint.json in --out)")
    ap.add_argument("--iters", type=int, default=None,
                    help="override the number of active-learning iterations for "
                         "THIS run (handy on --resume, e.g. --iters 3 to add 3 "
                         "more rounds of infill)")
    # ------------------------- NEW: many-core knobs ------------------------ #
    ap.add_argument("--threads", type=int, default=os.cpu_count(),
                    help="OpenMP (Open Multi-Processing) threads for EVERY "
                         "OpenMC transport solve (statepoint runs AND the "
                         "in-memory depletion transport). Default: all cores "
                         "the machine reports (os.cpu_count()).")
    ap.add_argument("--particles", type=int, default=None,
                    help="override particles per batch (full-run default 20000, "
                         "smoke default 800)")
    ap.add_argument("--batches", type=int, default=None,
                    help="override total batches (full-run default 80, smoke 30)")
    ap.add_argument("--inactive", type=int, default=None,
                    help="override inactive batches discarded for source "
                         "convergence (full-run default 20, smoke 10)")
    args = ap.parse_args()

    # ROUTE A (float) vs ROUTE B (per-design table) -- computed once, used by
    # the evaluator, the resume check, and both status prints below.
    # --ktarget-table wins if both are given.
    k_target_arg = args.ktarget_table or args.ktarget

    # ----------------------------------------------------------------------- #
    # Set the thread count BEFORE any OpenMC import. The OpenMP runtime reads
    # OMP_NUM_THREADS when the shared library initializes, so this must happen
    # before `import openmc` executes anywhere in the process. That is the
    # reason the heavy imports below live INSIDE main() instead of at the top
    # of the file.
    # ----------------------------------------------------------------------- #
    n_threads = max(1, int(args.threads or 1))
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from reactor_optimization import (example_reactor_problem, OptimizerConfig,
                                      ActiveLearningMOO)
    from openmc_evaluator import OpenMCEvaluator

    spec = example_reactor_problem()

    if args.smoke:
        # 4 + 1*2 = 6 real evaluations, coarse transport & burnup
        cfg = OptimizerConfig(n_init=4, n_iter=1, n_infill=2,
                              nsga_pop=20, nsga_gen=20, surrogate="gp", seed=1)
        transport = dict(particles=800, batches=30, inactive=10)
        burnup = [1.0, 4.0, 8.0, 8.0, 8.0]
        print(">>> SMOKE TEST <<<")
    else:
        # 24 + 8*6 = 72 real evaluations. Fidelity raised for a many-core node:
        # 20000 x (80-20) = 1.2M active histories/solve vs. 4000 x 40 = 160k on
        # the laptop profile -> ~2.7x lower Monte Carlo noise on k and F_dh.
        cfg = OptimizerConfig(n_init=24, n_iter=8, n_infill=6,
                              nsga_pop=60, nsga_gen=80, surrogate="gp", seed=1)
        transport = dict(particles=20000, batches=80, inactive=20)
        burnup = None   # use the evaluator's default fine-near-EOC schedule

    # command-line fidelity overrides (apply to smoke AND full profiles)
    if args.particles is not None:
        transport["particles"] = int(args.particles)
    if args.batches is not None:
        transport["batches"] = int(args.batches)
    if args.inactive is not None:
        transport["inactive"] = int(args.inactive)

    if args.iters is not None:
        cfg.n_iter = args.iters     # run exactly this many AL iterations now

    ev = OpenMCEvaluator(spec, k_target=k_target_arg, transport=transport,
                         **({"burnup_steps": burnup} if burnup else {}),
                         workdir="openmc_runs")

    opt = ActiveLearningMOO(spec, ev, cfg)
    # Ensure the output directory exists BEFORE anything tries to write into it
    # (results JSON, checkpoint, and the HV plot all land here). write_text does
    # not create missing parent folders, so without this the run crashes at the
    # very end -- after all the expensive OpenMC evaluations have already run.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    #   parents=True     create intermediate dirs too (like mkdir -p)
    #   exist_ok=True    do not error if the folder is already there
    ckpt_out = args.checkpoint or str(Path(args.out) / "optimization_checkpoint.json")

    if args.resume:
        prev_meta = json.loads(Path(args.resume).read_text()).get("meta", {})
        prev_kt = prev_meta.get("k_target")
        if prev_kt is not None:
            # Numeric vs numeric (both Route A): tolerate float noise. Any
            # other combination (a table PATH string, or Route A meeting
            # Route B): compare as text so a route switch is always caught.
            if isinstance(prev_kt, (int, float)) and isinstance(k_target_arg, (int, float)):
                kt_mismatch = abs(prev_kt - k_target_arg) > 1e-9
            else:
                kt_mismatch = str(prev_kt) != str(k_target_arg)
            if kt_mismatch:
                print(f"!! WARNING: checkpoint k_target={prev_kt!r} differs "
                      f"from current {k_target_arg!r}. Objectives across "
                      f"sessions will be inconsistent -- match --ktarget / "
                      f"--ktarget-table to the checkpoint's value.")
        prev_tr = prev_meta.get("transport")
        if prev_tr and any(int(prev_tr.get(k, -1)) != int(transport[k])
                           for k in ("particles", "batches", "inactive")):
            print(f"!! WARNING: checkpoint transport fidelity {prev_tr} differs "
                  f"from current {transport}. Mixing fidelities mixes Monte "
                  f"Carlo noise levels across sessions -- re-run with "
                  f"--particles {prev_tr.get('particles')} "
                  f"--batches {prev_tr.get('batches')} "
                  f"--inactive {prev_tr.get('inactive')} to match.")
        if bool(prev_meta.get("smoke")) != bool(args.smoke):
            print("!! WARNING: checkpoint smoke flag differs from this run "
                  "(smoke and full runs must never share a checkpoint).")
        n_loaded = opt.load_checkpoint(args.resume)
        added = cfg.n_iter * cfg.n_infill
        print(f"RESUME: loaded {n_loaded} prior real evaluations from "
              f"{args.resume}; adding {cfg.n_iter} x {cfg.n_infill} = {added} more "
              f"(target total {n_loaded + added}).")
    else:
        n_planned = cfg.n_init + cfg.n_iter * cfg.n_infill
        print(f"FRESH run: planned real evals = {n_planned} "
              f"({cfg.n_init} DOE + {cfg.n_iter} x {cfg.n_infill} infill).")

    print(f"hardware: {os.cpu_count()} CPUs visible | "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    print(f"transport: {transport['particles']} particles x "
          f"{transport['batches']} batches ({transport['inactive']} inactive)")
    kt_display = (f"{k_target_arg:.4f}" if isinstance(k_target_arg, (int, float))
                  else f"table:{k_target_arg}")
    print(f"specific power = {ev.spec_power:.2f} W/gHM | k_target = {kt_display}")

    res = opt.run(verbose=True)

    path = opt.save(str(Path(args.out) / "optimization_results.json"))
    print("saved ->", path)
    ckpt = opt.save_checkpoint(ckpt_out,
                               meta={"k_target": k_target_arg,
                                     "smoke": bool(args.smoke),
                                     "transport": dict(transport),
                                     "burnup_steps": (list(burnup) if burnup
                                                      else "evaluator-default"),
                                     "omp_threads": n_threads})
    print(f"checkpoint -> {ckpt}  ({res['n_real_evaluations']} evals total)")
    kt_flag = (f"--ktarget-table {k_target_arg}" if args.ktarget_table
               else f"--ktarget {k_target_arg:.4f}")
    print(f"   to add more later:  python run_optimization.py "
          f"--resume {ckpt} {kt_flag} "
          f"--particles {transport['particles']} "
          f"--batches {transport['batches']} "
          f"--inactive {transport['inactive']}"
          + (" --smoke" if args.smoke else ""))
    _plot(res, args.out)


def _plot(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allF = res["all_F"]
    pf = res["pareto_F"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(-allF[:, 0], allF[:, 1], s=18, c="lightgray",
                  label="all evaluations")
    if len(pf):
        ax[0].scatter(-pf[:, 0], pf[:, 1], s=42, c="crimson", zorder=3,
                      label="Pareto front")
    ax[0].set_xlabel("Cycle length [EFPD]  (maximise \u2192)")
    ax[0].set_ylabel("Power peaking factor  (\u2190 minimise)")
    ax[0].set_title("Objective space (OpenMC)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(range(len(res["hv_history"])), res["hv_history"], "o-", c="navy")
    ax[1].set_xlabel("Active-learning iteration (cumulative across resumes)")
    ax[1].set_ylabel("Hypervolume (feasible)")
    ax[1].set_title("Convergence"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    figpath = Path(outdir) / "optimization_openmc.png"
    fig.savefig(figpath, dpi=130)
    print("saved ->", figpath)


if __name__ == "__main__":
    main()
