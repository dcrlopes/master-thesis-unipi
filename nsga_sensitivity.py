#!/usr/bin/env python3
"""
nsga_sensitivity.py
===================
Empirical justification for the NSGA-II (Non-dominated Sorting Genetic
Algorithm II) population size and generation count used in the
surrogate-assisted active-learning loop.

WHY THIS SCRIPT EXISTS
----------------------
In run_optimization.py the full-run profile hardcodes

    OptimizerConfig(n_init=24, n_iter=8, n_infill=6,
                    nsga_pop=60, nsga_gen=80, surrogate="gp", seed=1)

while the OptimizerConfig dataclass in reactor_optimization.py declares
nsga_pop=80, nsga_gen=120, and the --smoke profile uses 20 and 20. Three
different pairs therefore exist in the repository and none of them carries a
recorded rationale. This script replaces "it was the default" with a
measurement.

WHAT IS MEASURED
----------------
NSGA-II here never calls OpenMC (Open source Monte Carlo particle transport
code). It searches the Gaussian Process (GP) surrogate, so one search costs
seconds. The quantity that actually matters downstream is not the surrogate
Pareto front itself but the n_infill designs the acquisition step selects and
sends to OpenMC. This script freezes a real campaign archive, fits the
surrogates ONCE, and then re-runs ONLY the surrogate search at several
(pop, gen) settings and several random seeds, comparing:

    1. the selected infill designs, against a baseline setting,
    2. the surrogate-predicted objectives of those selected designs,
    3. the feasibility state of the final NSGA-II population,
    4. the hypervolume of the surrogate front when feasible points exist,
    5. the minimum total constraint violation when they do not,
    6. wall-clock time of the search.

Because the surrogates are fitted once and shared by every setting, the only
thing that varies is the search itself. This is the controlled experiment.

THE ZERO-FEASIBLE REGIME
------------------------
Campaigns 4 and 5 ended with no feasible design. pymoo then reports
res.X = None and ActiveLearningMOO._least_infeasible_candidates falls back to
the ENTIRE final population sorted by constraint violation. Selecting the most
uncertain designs from all of it removes the exploitation stage and leaves pure
exploration. The --top-k flag reproduces the proposed restriction of the
candidate pool to the K least-infeasible members, so the effect of that change
can be quantified on frozen data before any new campaign is launched.

WHAT IS NOT MEASURED
--------------------
Nothing here is a truth evaluation. Every objective and constraint value used
is a surrogate prediction, so the results support statements about the
SEARCH being insensitive or sensitive to its settings. They do not support
statements about the physics.

USAGE
-----
Self-test first, on a synthetic archive built with the analytic evaluator. No
OpenMC and no checkpoint needed, and it exercises the whole chain in seconds:

    python nsga_sensitivity.py --self-test --out nsga_sens_selftest

Then the real study, on a frozen campaign checkpoint:

    python nsga_sensitivity.py \
        --checkpoint out/optimization_checkpoint.json \
        --settings 20x20,60x80,60x160,120x80,120x160,80x120 \
        --seeds 8 --out nsga_sens

Reproducing an EARLIER state of the campaign, for instance the archive as it
stood before the last infill iteration of six designs:

    python nsga_sensitivity.py --checkpoint out/optimization_checkpoint.json \
        --n-train 54 --out nsga_sens_it2

Flags:
    --checkpoint PATH   optimization_checkpoint.json written by a real campaign
    --settings LIST     comma-separated POPxGEN pairs, first one is the baseline
    --seeds N           independent NSGA-II seeds per setting (NSGA-II is
                        stochastic, one seed proves nothing)
    --base-seed N       first seed, subsequent seeds are base, base+1, ...
    --n-infill N        designs the acquisition selects, must match the campaign
    --n-train N         truncate the archive to its first N evaluations
    --top-k N           restrict the candidate pool to the N least-infeasible
                        members before the uncertainty ranking (0 disables,
                        which is the current repository behaviour)
    --match-tol F       fraction of each variable's range within which two
                        designs count as the same design
    --out DIR           output directory
    --no-fig            skip the figure

Outputs written to --out:
    nsga_sensitivity.json      every seed, every setting, full record
    nsga_sensitivity.csv       one row per setting, aggregated
    nsga_sensitivity.tex       booktabs table for the thesis
    nsga_sensitivity.pdf       two-panel figure for the thesis

DEPENDENCIES
------------
numpy, scikit-learn, pymoo, matplotlib. OpenMC is NOT imported.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# The optimizer module imports sklearn and pymoo only, never openmc, so this
# script runs on any machine that can train the surrogate.
from reactor_optimization import (ActiveLearningMOO, AnalyticEvaluator,
                                  Evaluator, OptimizerConfig,
                                  _SurrogateProblem, example_reactor_problem)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

# The discrete gadolinia ladder from reactor_model.GD_PIN_COUNTS, duplicated
# here because reactor_model imports openmc and this script must not.
GD_PIN_LADDER = [12, 16, 20, 24, 32, 40]


def snap_gd_pins(x) -> int:
    """Nearest ladder value, identical rule to reactor_model.snap_gd_pins."""
    x = float(x)
    return min(GD_PIN_LADDER, key=lambda n: (abs(n - x), n))


class FrozenEvaluator(Evaluator):
    """A truth evaluator that refuses to evaluate.

    ActiveLearningMOO requires an evaluator object, and load_checkpoint writes
    to evaluator.n_calls. Nothing in this study is allowed to spend an OpenMC
    evaluation, so any call is a bug and must fail loudly rather than silently
    burn compute.
    """

    def evaluate_one(self, design: dict) -> dict:
        raise RuntimeError(
            "FrozenEvaluator was called. This study must never run a truth "
            "evaluation; the archive is frozen.")


# ---------------------------------------------------------------------------
# archive loading
# ---------------------------------------------------------------------------
def load_archive(spec, checkpoint: str | None, n_train: int | None,
                 self_test: bool, self_test_n: int = 48):
    """Return a populated ActiveLearningMOO whose archive will not grow.

    With --self-test the archive is generated by AnalyticEvaluator, whose
    formulas are toy physics and exist only to exercise the pipeline. With a
    checkpoint it is the real campaign archive, loaded through the repository's
    own load_checkpoint so the minimise-space sign convention and the frozen
    hypervolume reference point are reproduced exactly.
    """
    if self_test:
        ev = AnalyticEvaluator(spec, noise=0.02, seed=1)
        opt = ActiveLearningMOO(spec, ev, OptimizerConfig(seed=1))
        X0 = spec.design_space.lhs(self_test_n, seed=1,
                                   accept=spec.exact_ok if spec.exact_constraints
                                   else None)
        F0, G0, raw0 = ev.evaluate(X0)
        opt._add(X0, F0, G0, raw0)
        opt._hv()          # freezes the reference point the same way run() does
        meta = {"source": "self-test analytic archive",
                "n_evaluations": int(len(opt.X))}
        return opt, meta

    if not checkpoint:
        raise SystemExit("give --checkpoint PATH or use --self-test")

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    opt = ActiveLearningMOO(spec, FrozenEvaluator(spec), OptimizerConfig(seed=1))
    try:
        n_loaded = opt.load_checkpoint(str(ckpt_path))
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\n\nThe checkpoint does not match the problem definition on "
            f"this branch. Check out the branch the campaign was run on, or "
            f"point --checkpoint at the matching campaign.")

    if n_train is not None and n_train < n_loaded:
        # Truncation reproduces the archive as it stood at an earlier
        # iteration. The frozen hypervolume reference point is kept, since in
        # a real resumed campaign it is frozen once and never recomputed.
        opt.X = opt.X[:n_train]
        opt.F = opt.F[:n_train]
        opt.G = opt.G[:n_train]
        opt.raw = opt.raw[:n_train]
        n_loaded = n_train

    raw_meta = json.loads(ckpt_path.read_text()).get("meta", {})
    meta = {"source": str(ckpt_path),
            "n_evaluations": int(len(opt.X)),
            "checkpoint_meta": raw_meta}
    return opt, meta


# ---------------------------------------------------------------------------
# one NSGA-II run at one setting and one seed
# ---------------------------------------------------------------------------
def run_one(spec, opt, obj_sur, con_sur, pop, gen, seed, n_infill,
            top_k, hv_ref):
    """Run the surrogate search once and apply the campaign's acquisition.

    The candidate extraction, the uncertainty score, the de-duplication rule
    and the random fallback are the ones in ActiveLearningMOO.run, called
    through the repository's own code where possible so this script cannot
    drift away from the campaign it is describing.
    """
    prob = _SurrogateProblem(spec, obj_sur, con_sur)
    algo = NSGA2(pop_size=pop, sampling=LHS())

    t0 = time.perf_counter()
    res = minimize(prob, algo, ("n_gen", gen), seed=seed, verbose=False)
    t_search = time.perf_counter() - t0

    # --- state of the final population ------------------------------------
    pop_obj = getattr(res, "pop", None)
    if pop_obj is not None and len(pop_obj):
        cv = np.asarray(pop_obj.get("CV"), dtype=float).ravel()
        n_feasible_pop = int(np.sum(cv <= 1e-9))
        v_min = float(np.min(cv))
        v_mean = float(np.mean(cv))
    else:
        n_feasible_pop, v_min, v_mean = 0, float("nan"), float("nan")

    had_feasible = res.X is not None

    # --- hypervolume of the surrogate front, only when it is defined -------
    hv_value = float("nan")
    n_front = 0
    if had_feasible and res.F is not None:
        front = np.atleast_2d(np.asarray(res.F, dtype=float))
        n_front = int(front.shape[0])
        if hv_ref is not None:
            keep = np.all(front < hv_ref, axis=1)
            if keep.any():
                hv_value = float(HV(ref_point=np.asarray(hv_ref,
                                                         dtype=float))(front[keep]))
            else:
                hv_value = 0.0

    # --- acquisition, reproduced exactly ----------------------------------
    t1 = time.perf_counter()
    cand = ActiveLearningMOO._least_infeasible_candidates(res)
    used_random_fallback = False
    if cand is None or cand.shape[0] == 0:
        cand = np.atleast_2d(
            spec.design_space.lhs(max(n_infill * 4, 32), seed=seed + 777))
        used_random_fallback = True

    n_cand_full = int(cand.shape[0])
    # Proposed Campaign 6 restriction. _least_infeasible_candidates already
    # returns the population sorted least-infeasible first in the zero-feasible
    # branch, so the head of the array is the correct slice.
    if top_k and not had_feasible and not used_random_fallback:
        cand = cand[:min(top_k, cand.shape[0])]
    n_cand_used = int(cand.shape[0])

    _, std = obj_sur.predict(cand)
    score = (std / (std.max(axis=0) + 1e-12)).sum(axis=1)
    order = np.argsort(-score)
    chosen = []
    for idx in order:
        x = cand[idx]
        if opt.X.size and np.min(np.linalg.norm(opt.X - x, axis=1)) < 1e-6:
            continue
        chosen.append(x)
        if len(chosen) >= n_infill:
            break
    if not chosen:
        chosen = list(spec.design_space.lhs(n_infill, seed=seed + 99))
        used_random_fallback = True
    Xsel = np.atleast_2d(np.array(chosen))
    t_acq = time.perf_counter() - t1

    # surrogate-predicted objectives and constraints of the selected designs
    f_sel, _ = obj_sur.predict(Xsel)
    g_sel = con_sur.predict(Xsel)[0] if con_sur is not None else None
    exact_cols = {name: [fn(spec.design_space.as_dict(x)) for x in Xsel]
                  for name, fn in spec.exact_constraints.items()}
    if g_sel is not None:
        for name, vals in exact_cols.items():
            g_sel[:, spec.constraint_names.index(name)] = vals
        v_sel = float(np.mean(np.sum(np.clip(g_sel, 0.0, None), axis=1)))
    else:
        v_sel = float("nan")

    return {
        "pop": pop, "gen": gen, "seed": int(seed),
        "n_surrogate_evals": int(pop * gen),
        "t_search_s": t_search, "t_acquisition_s": t_acq,
        "had_feasible": bool(had_feasible),
        "n_feasible_pop": n_feasible_pop,
        "n_front": n_front,
        "hv_surrogate": hv_value,
        "cv_min_pop": v_min, "cv_mean_pop": v_mean,
        "n_candidates_available": n_cand_full,
        "n_candidates_used": n_cand_used,
        "used_random_fallback": bool(used_random_fallback),
        "X_selected": Xsel.tolist(),
        "F_selected_pred": np.asarray(f_sel, dtype=float).tolist(),
        "violation_selected_mean": v_sel,
        "gd_pins_selected_snapped": [snap_gd_pins(x[-1]) for x in Xsel],
    }


# ---------------------------------------------------------------------------
# comparison of selected designs
# ---------------------------------------------------------------------------
def agreement(X_a, X_b, xl, xu, tol):
    """Fraction of designs in X_a that have a partner in X_b.

    Distance is measured in the unit hypercube obtained by dividing each
    variable by its own range, so a tolerance of 0.02 means two designs agree
    when every variable differs by less than two per cent of its allowed span.
    Matching is greedy and one-to-one, so a single design in X_b cannot absorb
    several designs from X_a.
    """
    span = np.asarray(xu, dtype=float) - np.asarray(xl, dtype=float)
    A = (np.atleast_2d(X_a) - xl) / span
    B = (np.atleast_2d(X_b) - xl) / span
    if A.size == 0 or B.size == 0:
        return 0.0
    taken = set()
    matched = 0
    for a in A:
        d = np.linalg.norm(B - a, axis=1) / np.sqrt(B.shape[1])
        for j in np.argsort(d):
            if j in taken:
                continue
            if d[j] <= tol:
                taken.add(int(j))
                matched += 1
            break
    return matched / len(A)


def objective_shift(F_a, F_b):
    """Mean distance between the two predicted objective clouds, per objective.

    Reported in minimise space, so column 0 is negative cycle length in
    Effective Full Power Days (EFPD) and column 1 is the radial enthalpy-rise
    hot channel factor F_dH.
    """
    A = np.atleast_2d(np.asarray(F_a, dtype=float))
    B = np.atleast_2d(np.asarray(F_b, dtype=float))
    if A.size == 0 or B.size == 0:
        return [float("nan")] * 2
    return list(np.abs(A.mean(axis=0) - B.mean(axis=0)))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_tex(path, rows, baseline_label, infeasible_regime, n_seeds):
    """booktabs table, one row per setting."""
    if infeasible_regime:
        col_head = (r"$V_{\min}$ & $\bar{V}_{\mathrm{sel}}$ & "
                    r"agreement & $t_{\mathrm{search}}$ [s]")
        def body(r):
            return (f"{r['cv_min_pop_mean']:.4f} & "
                    f"{r['violation_selected_mean_mean']:.4f} & "
                    f"{r['agreement_mean']:.2f} $\\pm$ {r['agreement_std']:.2f} & "
                    f"{r['t_search_s_mean']:.1f}")
    else:
        col_head = (r"HV & feasible & agreement & $t_{\mathrm{search}}$ [s]")
        def body(r):
            return (f"{r['hv_surrogate_mean']:.4g} & "
                    f"{r['n_feasible_pop_mean']:.0f} & "
                    f"{r['agreement_mean']:.2f} $\\pm$ {r['agreement_std']:.2f} & "
                    f"{r['t_search_s_mean']:.1f}")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity of the surrogate search to the NSGA-II "
        r"population size and generation count. Every setting searches the "
        r"same frozen Gaussian Process surrogates, fitted once on the complete "
        f"campaign archive. Values are means over {n_seeds} independent random "
        r"seeds. Agreement is the fraction of the selected infill designs that "
        f"coincide with those selected by the baseline setting {baseline_label} "
        r"at the same seed.}",
        r"\label{tab:nsga_sensitivity}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Population & Generations & " + col_head + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['pop']} & {r['gen']} & " + body(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(lines))


def make_figure(path, rows, infeasible_regime, baseline_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['pop']}$\\times${r['gen']}" for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))

    if infeasible_regime:
        y = [r["cv_min_pop_mean"] for r in rows]
        e = [r["cv_min_pop_std"] for r in rows]
        ax[0].set_ylabel("Minimum total constraint violation")
        ax[0].set_title("Closest approach to the feasible set")
    else:
        y = [r["hv_surrogate_mean"] for r in rows]
        e = [r["hv_surrogate_std"] for r in rows]
        ax[0].set_ylabel("Hypervolume of the surrogate front")
        ax[0].set_title("Surrogate front quality")
    ax[0].errorbar(x, y, yerr=e, fmt="o-", c="navy", capsize=3)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=30, ha="right")
    ax[0].set_xlabel("Population $\\times$ generations")
    ax[0].grid(alpha=0.3)

    ya = [r["agreement_mean"] for r in rows]
    ea = [r["agreement_std"] for r in rows]
    ax[1].errorbar(x, ya, yerr=ea, fmt="s-", c="crimson", capsize=3)
    ax[1].axhline(1.0, ls="--", c="gray", lw=0.8)
    ax[1].set_ylim(-0.05, 1.15)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=30, ha="right")
    ax[1].set_xlabel("Population $\\times$ generations")
    ax[1].set_ylabel("Fraction of infill designs shared")
    ax[1].set_title(f"Agreement with baseline {baseline_label}")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def parse_settings(text):
    out = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise SystemExit(f"bad setting {token!r}, expected POPxGEN")
        p, g = token.split("x", 1)
        out.append((int(p), int(g)))
    if not out:
        raise SystemExit("no settings given")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="optimization_checkpoint.json from a real campaign")
    ap.add_argument("--settings", default="20x20,60x80,60x160,80x120,120x80,120x160",
                    help="comma-separated POPxGEN pairs, first is the baseline")
    ap.add_argument("--baseline", default="60x80",
                    help="POPxGEN used as the reference for the agreement "
                         "metric. The default is the pair actually used by the "
                         "full-run profile of run_optimization.py, which is the "
                         "setting the thesis has to justify.")
    ap.add_argument("--seeds", type=int, default=8,
                    help="independent NSGA-II seeds per setting")
    ap.add_argument("--base-seed", type=int, default=1,
                    help="first seed, matching OptimizerConfig.seed of the "
                         "campaign being described")
    ap.add_argument("--n-infill", type=int, default=6,
                    help="designs the acquisition selects per iteration, must "
                         "match the campaign's cfg.n_infill")
    ap.add_argument("--n-train", type=int, default=None,
                    help="truncate the archive to its first N evaluations")
    ap.add_argument("--top-k", type=int, default=0,
                    help="restrict the candidate pool to the N least-infeasible "
                         "population members before the uncertainty ranking "
                         "(0 keeps the current repository behaviour)")
    ap.add_argument("--match-tol", type=float, default=0.02,
                    help="fraction of each variable's range within which two "
                         "designs count as identical")
    ap.add_argument("--out", default="nsga_sens", help="output directory")
    ap.add_argument("--self-test", action="store_true",
                    help="build a synthetic archive with the analytic "
                         "evaluator instead of loading a checkpoint")
    ap.add_argument("--no-fig", action="store_true", help="skip the figure")
    args = ap.parse_args()

    print(f"python   : {sys.executable}")
    print(f"host     : {platform.node()}  |  {os.cpu_count()} CPUs visible")

    settings = parse_settings(args.settings)
    baseline = parse_settings(args.baseline)[0] if args.baseline else settings[0]
    if baseline not in settings:
        settings.insert(0, baseline)
    baseline_label = f"{baseline[0]}x{baseline[1]}"

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    spec = example_reactor_problem()
    opt, archive_meta = load_archive(spec, args.checkpoint, args.n_train,
                                     args.self_test)
    print(f"archive  : {archive_meta['n_evaluations']} real evaluations from "
          f"{archive_meta['source']}")

    n_feas_archive = int(opt._feasible_mask().sum())
    print(f"           {n_feas_archive} of them feasible on all "
          f"{spec.n_constr} constraints")

    # ---- fit the surrogates ONCE, so only the search varies ---------------
    t0 = time.perf_counter()
    obj_sur = opt._new_surrogate().fit(opt.X, opt.F)
    con_sur = (opt._new_surrogate().fit(opt.X, opt.G) if spec.n_constr else None)
    print(f"surrogate: fitted once in {time.perf_counter() - t0:.1f} s and "
          f"shared by every setting")

    hv_ref = (opt._hv_ref_frozen.tolist()
              if opt._hv_ref_frozen is not None else None)
    if hv_ref is None:
        print("           no frozen hypervolume reference point in the "
              "checkpoint, hypervolume will be reported as not available")

    seeds = [args.base_seed + i for i in range(args.seeds)]

    # ---- run every setting at every seed ----------------------------------
    records = []
    for pop, gen in settings:
        for seed in seeds:
            r = run_one(spec, opt, obj_sur, con_sur, pop, gen, seed,
                        args.n_infill, args.top_k, hv_ref)
            records.append(r)
            print(f"  pop={pop:4d} gen={gen:4d} seed={seed:3d}  "
                  f"feasible={r['had_feasible']!s:5s}  "
                  f"CVmin={r['cv_min_pop']:.4f}  "
                  f"cand={r['n_candidates_used']}/{r['n_candidates_available']}  "
                  f"t={r['t_search_s']:.1f}s")

    # ---- aggregate --------------------------------------------------------
    by_seed = {(r["pop"], r["gen"], r["seed"]): r for r in records}
    for r in records:
        base = by_seed[(baseline[0], baseline[1], r["seed"])]
        r["agreement"] = agreement(r["X_selected"], base["X_selected"],
                                   spec.design_space.xl, spec.design_space.xu,
                                   args.match_tol)
        r["gd_pins_agreement"] = (
            len(set(r["gd_pins_selected_snapped"]) &
                set(base["gd_pins_selected_snapped"])) /
            max(len(set(base["gd_pins_selected_snapped"])), 1))
        shift = objective_shift(r["F_selected_pred"], base["F_selected_pred"])
        r["d_cycle_EFPD"] = abs(shift[0])
        r["d_peaking"] = abs(shift[1])

    infeasible_regime = not any(r["had_feasible"] for r in records)
    if infeasible_regime:
        print("\nZero-feasible regime: pymoo returned no feasible design at "
              "any setting, so the hypervolume of the surrogate front is "
              "undefined and the minimum total constraint violation is the "
              "primary metric.")

    agg_fields = ["cv_min_pop", "cv_mean_pop", "hv_surrogate",
                  "n_feasible_pop", "t_search_s", "t_acquisition_s",
                  "agreement", "gd_pins_agreement", "d_cycle_EFPD",
                  "d_peaking", "violation_selected_mean",
                  "n_candidates_used"]
    rows = []
    for pop, gen in settings:
        sub = [r for r in records if r["pop"] == pop and r["gen"] == gen]
        row = {"pop": pop, "gen": gen, "n_surrogate_evals": pop * gen,
               "n_seeds": len(sub), "is_baseline": (pop, gen) == baseline}
        for f in agg_fields:
            vals = np.array([r[f] for r in sub], dtype=float)
            with np.errstate(invalid="ignore"):
                row[f + "_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
                row[f + "_std"] = float(np.nanstd(vals)) if vals.size else float("nan")
        rows.append(row)

    # ---- write ------------------------------------------------------------
    json_path = outdir / "nsga_sensitivity.json"
    json_path.write_text(json.dumps({
        "archive": archive_meta,
        "n_feasible_in_archive": n_feas_archive,
        "baseline": baseline_label,
        "seeds": seeds,
        "n_infill": args.n_infill,
        "top_k": args.top_k,
        "match_tol": args.match_tol,
        "hv_ref": hv_ref,
        "infeasible_regime": infeasible_regime,
        "design_variables": spec.design_space.names,
        "constraint_names": spec.constraint_names,
        "aggregate": rows,
        "records": records,
    }, indent=2, default=float))

    csv_fields = (["pop", "gen", "n_surrogate_evals", "n_seeds", "is_baseline"] +
                  [f + s for f in agg_fields for s in ("_mean", "_std")])
    write_csv(outdir / "nsga_sensitivity.csv", rows, csv_fields)
    write_tex(outdir / "nsga_sensitivity.tex", rows, baseline_label,
              infeasible_regime, len(seeds))
    if not args.no_fig:
        make_figure(outdir / "nsga_sensitivity.pdf", rows, infeasible_regime,
                    baseline_label)

    # ---- console summary --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'pop':>5} {'gen':>5} {'CVmin':>9} {'HV':>10} {'agree':>7} "
          f"{'dEFPD':>9} {'dF_dH':>8} {'t[s]':>7}")
    for r in rows:
        hv = r["hv_surrogate_mean"]
        hv_s = "n/a" if not np.isfinite(hv) else f"{hv:.4g}"
        print(f"{r['pop']:5d} {r['gen']:5d} {r['cv_min_pop_mean']:9.4f} "
              f"{hv_s:>10} {r['agreement_mean']:7.2f} "
              f"{r['d_cycle_EFPD_mean']:9.1f} {r['d_peaking_mean']:8.4f} "
              f"{r['t_search_s_mean']:7.1f}")
    print("=" * 78)

    others = [r for r in rows if not r["is_baseline"]]
    if others:
        worst = min(others, key=lambda r: r["agreement_mean"])
        if worst["agreement_mean"] >= 0.999:
            verdict = ("Every setting selected the same infill designs as the "
                       f"baseline {baseline_label} at every seed. The loop's "
                       "output is insensitive to the search settings over the "
                       "range tested.")
        elif worst["agreement_mean"] >= 0.5:
            verdict = (f"The weakest agreement with the baseline was "
                       f"{worst['agreement_mean']:.2f} at "
                       f"{worst['pop']}x{worst['gen']}. The selection is "
                       "partially sensitive, so report the spread rather than "
                       "claiming insensitivity.")
        else:
            verdict = (f"Agreement fell to {worst['agreement_mean']:.2f} at "
                       f"{worst['pop']}x{worst['gen']}. The selection IS "
                       "sensitive to the search settings, which is a finding "
                       "in its own right and must be reported.")
        print("\n" + verdict)

    print(f"\nwritten -> {json_path}")
    print(f"written -> {outdir / 'nsga_sensitivity.csv'}")
    print(f"written -> {outdir / 'nsga_sensitivity.tex'}")
    if not args.no_fig:
        print(f"written -> {outdir / 'nsga_sensitivity.pdf'}")


if __name__ == "__main__":
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
