#!/usr/bin/env python3
"""
analyze_results.py -- Post-processing, diagnostics and richer visualisation for the
surrogate-assisted multi-objective optimization (MOO) results produced by
run_optimization.py (OpenMC evaluator).

Reads:
  * optimization_results.json      (Pareto set + hypervolume history)
  * optimization_checkpoint.json   (optional but recommended: all raw evaluations)

Produces (in --outdir):
  * fig1_pareto_annotated.png      objective space with censoring line, error bars,
                                   infeasible points and labelled Pareto designs
  * fig2_convergence.png           hypervolume (HV) history + per-iteration gain
  * fig3_parallel_coordinates.png  design variables + objectives, Pareto highlighted
  * fig4_variable_sensitivity.png  each design variable vs each objective
  * fig5_constraint_margins.png    constraint margins of the Pareto designs
  * analysis_report.txt            everything also printed to the console

Only numpy + matplotlib are required (both already in the openmc-env environment).

Example:
  python analyze_results.py \
      --results out/optimization_results.json \
      --checkpoint out/optimization_checkpoint.json \
      --outdir fig_analysis --doe-size 24 --sigma-k 220
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend: write PNG files, never open a window
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- palette
NAVY = "#0E2A47"
BLUE = "#5B8AC0"
GOLD = "#F2A900"
RED = "#C8102E"
GRAY = "#9AA5B1"
PARETO_COLORS = ["#C8102E", "#F2A900", "#2E7D32", "#6A1B9A", "#00838F", "#E65100"]

CONSTRAINTS = ["g_kmin", "g_kmax", "g_enr", "g_peak"]
CEIL_TOL = 1e-6  # EFPD tolerance used to detect the burnup-schedule ceiling


# ----------------------------------------------------------------------------- helpers
def load(results_path, checkpoint_path):
    with open(results_path) as f:
        res = json.load(f)
    ck = None
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            ck = json.load(f)
    variables = res["design_variables"]
    pareto = res["pareto_raw"]
    all_raw = ck["all_raw"] if ck else list(pareto)
    hv = res.get("hv_history", ck.get("hv_history", []) if ck else [])
    meta = (ck or {}).get("meta", {})
    return variables, pareto, all_raw, hv, meta


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def feasible_mask(rows):
    return np.array([all(r[g] <= 0.0 for g in CONSTRAINTS) for r in rows])


def ceiling_info(all_raw):
    """Censoring detection, two generations of checkpoints:

    NEW (adaptive-depletion evaluator): every raw record carries an explicit
    boolean 'censored' written by openmc_evaluator (True = the design was
    still above k_target at the --max-burnup cap, so its cycle length is a
    LOWER BOUND). Trust it when present.

    LEGACY (fixed 35.5 MWd/kg schedule): no flag; every design still critical
    at the schedule end reports the same truncated cycle length, so censoring
    is inferred from proximity to the maximum ('ceiling')."""
    cyc = col(all_raw, "cycle_length")
    if all_raw and all("censored" in r for r in all_raw):
        censored = np.array([bool(r["censored"]) for r in all_raw])
        ceiling = cyc[censored].min() if censored.any() else cyc.max()
        return ceiling, censored
    ceiling = cyc.max()
    censored = np.abs(cyc - ceiling) < CEIL_TOL
    return ceiling, censored


def sigma_cycle_est(row, sigma_k_pcm, k_target):
    """1-sigma statistical noise on the cycle length, propagated from the
    Monte Carlo k-effective uncertainty through the mean reactivity slope.
    Factor sqrt(2): the end-of-cycle crossing is interpolated between two
    independently noisy k values."""
    slope = (row["k_bol"] - k_target) / max(row["cycle_length"], 1.0)  # dk per EFPD
    if slope <= 0:
        return float("nan")
    return np.sqrt(2.0) * (sigma_k_pcm * 1e-5) / slope


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


# ----------------------------------------------------------------------------- figures
def fig_pareto(variables, pareto, all_raw, ceiling, censored, sigma_k, k_target, outdir, dpi):
    feas = feasible_mask(all_raw)
    cyc, pk = col(all_raw, "cycle_length"), col(all_raw, "peaking")

    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    ax.scatter(cyc[~feas], pk[~feas], marker="x", s=45, c=GRAY, alpha=0.6,
               label=f"infeasible ({int((~feas).sum())})")
    ax.scatter(cyc[feas & ~censored], pk[feas & ~censored], s=38, c=BLUE, alpha=0.75,
               edgecolors="none", label=f"feasible ({int((feas & ~censored).sum())})")
    ax.scatter(cyc[feas & censored], pk[feas & censored], s=48, facecolors="none",
               edgecolors=GOLD, linewidths=1.8,
               label=f"feasible, censored at schedule end ({int((feas & censored).sum())})")

    if censored.any():
        ax.axvline(ceiling, color=GOLD, ls="--", lw=1.5)
        ax.text(ceiling, ax.get_ylim()[1], f"  depletion cap\n  = {ceiling:.0f} EFPD "
                f"({int(censored.sum())}/{len(all_raw)} evals are lower bounds)",
                va="top", ha="left", fontsize=9, color="#8a6d00")

    for i, p in enumerate(pareto):
        c = PARETO_COLORS[i % len(PARETO_COLORS)]
        xerr = sigma_cycle_est(p, sigma_k, k_target) if sigma_k > 0 else None
        ax.errorbar(p["cycle_length"], p["peaking"], xerr=xerr, fmt="o", ms=11,
                    mfc=c, mec=NAVY, mew=1.2, ecolor=c, elinewidth=2, capsize=4, zorder=5)
        txt = (f"P{i+1}: e=({p['enrich_inner']:.2f}/{p['enrich_outer']:.2f})%  "
               f"Gd={p['gd_wt']:.2f}%\n      p={p['pitch']:.3f} cm  "
               f"refl={p['refl_thick']:.1f} cm")
        ax.annotate(txt, (p["cycle_length"], p["peaking"]),
                    textcoords="offset points", xytext=(10, -22 if i % 2 else 12),
                    fontsize=8, color=c, fontweight="bold")

    ax.set_xlabel("Cycle length [EFPD]  (maximise →)")
    ax.set_ylabel("Radial power peaking factor F$_{\\Delta h}$  (← minimise)")
    ax.set_title("Objective space — annotated (error bars: ±1σ statistical noise on cycle length)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(Path(outdir) / "fig1_pareto_annotated.png", dpi=dpi)
    plt.close(fig)


def fig_convergence(hv, outdir, dpi):
    hv = np.asarray(hv, dtype=float)
    it = np.arange(len(hv))
    gains = np.diff(hv)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    ax.plot(it, hv, "-o", color=NAVY, lw=2, ms=7, zorder=3, label="hypervolume (feasible)")
    ax2 = ax.twinx()
    ax2.bar(it[1:], gains, width=0.55, color=GOLD, alpha=0.55, label="gain per iteration")
    ax2.set_ylabel("ΔHV per iteration", color="#8a6d00")
    ax2.tick_params(axis="y", labelcolor="#8a6d00")

    if len(hv) >= 4:
        recent = (hv[-1] - hv[-4]) / hv[-4] * 100.0
        verdict = "still improving — NOT converged" if recent > 1.0 else "plateauing"
        ax.text(0.02, 0.97, f"gain over last 3 iterations: {recent:+.1f}%  →  {verdict}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="#fff8e1", ec=GOLD))
    ax.set_xlabel("Active-learning iteration (0 = after DOE/LHS)")
    ax.set_ylabel("Hypervolume (feasible)")
    ax.set_title("Convergence of the surrogate-assisted optimization")
    ax.grid(alpha=0.3)
    fig.savefig(Path(outdir) / "fig2_convergence.png", dpi=dpi)
    plt.close(fig)


def fig_parallel(variables, pareto, all_raw, outdir, dpi):
    axes_keys = variables + ["cycle_length", "peaking"]
    feas_rows = [r for r in all_raw if all(r[g] <= 0 for g in CONSTRAINTS)]
    data = np.array([[r[k] for k in axes_keys] for r in feas_rows])
    lo, hi = data.min(axis=0), data.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)

    def norm(rows):
        arr = np.array([[r[k] for k in axes_keys] for r in rows])
        return (arr - lo) / span

    fig, ax = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    x = np.arange(len(axes_keys))
    for y in norm(feas_rows):
        ax.plot(x, y, color=GRAY, alpha=0.30, lw=1)
    for i, p in enumerate(pareto):
        ax.plot(x, norm([p])[0], color=PARETO_COLORS[i % len(PARETO_COLORS)],
                lw=2.8, marker="o", ms=5, label=f"P{i+1}", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in axes_keys], fontsize=9)
    for xi, (l, h) in enumerate(zip(lo, hi)):
        ax.text(xi, -0.05, f"{l:.3g}", ha="center", va="top", fontsize=8, color=NAVY)
        ax.text(xi, 1.03, f"{h:.3g}", ha="center", va="bottom", fontsize=8, color=NAVY)
    ax.set_yticks([])
    ax.set_ylim(-0.12, 1.12)
    ax.set_title("Parallel coordinates — feasible evaluations (gray) and Pareto designs "
                 "(colored); axis extremes = observed min/max")
    ax.legend(loc="upper left", fontsize=9)
    for xi in x:
        ax.axvline(xi, color="k", alpha=0.15, lw=0.8)
    fig.savefig(Path(outdir) / "fig3_parallel_coordinates.png", dpi=dpi)
    plt.close(fig)


def fig_sensitivity(variables, pareto, all_raw, censored, outdir, dpi):
    feas = feasible_mask(all_raw)
    fig, axes = plt.subplots(2, len(variables), figsize=(3.0 * len(variables), 6.4),
                             sharey="row", constrained_layout=True)
    for j, v in enumerate(variables):
        xv = col(all_raw, v)
        for i, obj in enumerate(["cycle_length", "peaking"]):
            yv = col(all_raw, obj)
            a = axes[i, j]
            a.scatter(xv[feas & ~censored], yv[feas & ~censored], s=22, c=BLUE,
                      alpha=0.75, label="feasible" if (i == 0 and j == 0) else None)
            a.scatter(xv[feas & censored], yv[feas & censored], s=26, facecolors="none",
                      edgecolors=GOLD, lw=1.4,
                      label="censored" if (i == 0 and j == 0) else None)
            a.scatter(xv[~feas], yv[~feas], s=18, marker="x", c=GRAY, alpha=0.5,
                      label="infeasible" if (i == 0 and j == 0) else None)
            a.scatter([p[v] for p in pareto], [p[obj] for p in pareto], s=90, marker="*",
                      c=RED, edgecolors=NAVY, zorder=5,
                      label="Pareto" if (i == 0 and j == 0) else None)
            if i == 1:
                a.set_xlabel(v, fontsize=9)
            a.grid(alpha=0.25)
        axes[0, j].tick_params(labelbottom=False)
    axes[0, 0].set_ylabel("cycle_length [EFPD]")
    axes[1, 0].set_ylabel("peaking F$_{\\Delta h}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Design-variable sensitivity (columns) vs objectives (rows)", y=1.10)
    fig.savefig(Path(outdir) / "fig4_variable_sensitivity.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)


def fig_constraints(pareto, outdir, dpi):
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    ny = len(CONSTRAINTS)
    width = 0.8 / max(len(pareto), 1)
    for i, p in enumerate(pareto):
        vals = [p[g] for g in CONSTRAINTS]
        ypos = np.arange(ny) + (i - (len(pareto) - 1) / 2) * width
        ax.barh(ypos, vals, height=width * 0.9,
                color=PARETO_COLORS[i % len(PARETO_COLORS)], label=f"P{i+1}")
        for y, v in zip(ypos, vals):
            note = "  ~ACTIVE" if abs(v) < 0.005 else ""
            ax.text(min(v, 0) - 0.02, y, f"{v:.4f}{note}", va="center", ha="right",
                    fontsize=8)
    ax.axvline(0, color=RED, lw=1.5)
    ax.text(0.005, ny - 0.4, "g = 0 (boundary; g ≤ 0 feasible)", color=RED, fontsize=9)
    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels(CONSTRAINTS)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("constraint value g (symlog scale; more negative = more margin)")
    ax.set_title("Constraint margins of the Pareto designs")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(Path(outdir) / "fig5_constraint_margins.png", dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def report(variables, pareto, all_raw, hv, meta, ceiling, censored, args, out_lines):
    P = out_lines.append
    feas = feasible_mask(all_raw)
    n = len(all_raw)

    P("=" * 78)
    P("OPTIMIZATION RUN DIAGNOSTIC REPORT")
    P("=" * 78)
    if meta:
        tr = meta.get("transport", {})
        P(f"Transport settings : {tr.get('particles','?')} particles x "
          f"{tr.get('batches','?')} batches ({tr.get('inactive','?')} inactive), "
          f"{meta.get('omp_threads','?')} OpenMP threads")
        P(f"k_target source    : {meta.get('k_target','?')}   "
          f"burnup steps: {meta.get('burnup_steps','?')}")
    P(f"Real evaluations   : {n}   feasible: {int(feas.sum())}   "
      f"infeasible: {int((~feas).sum())}")
    for g in CONSTRAINTS:
        v = sum(1 for r in all_raw if r[g] > 0)
        if v:
            P(f"                     {g} violated in {v} evaluations")
    if args.doe_size and args.doe_size < n:
        d = args.doe_size
        fd = feasible_mask(all_raw[:d]).sum()
        fs = feasible_mask(all_raw[d:]).sum()
        P(f"Budget split       : DOE/LHS {d} evals ({int(fd)} feasible)  |  "
          f"active-learning {n-d} evals ({int(fs)} feasible)")
        cd, cs = censored[:d].sum(), censored[d:].sum()
        P(f"Censored split     : DOE {int(cd)}/{d}  |  active-learning {int(cs)}/{n-d}")

    P("-" * 78)
    P(f"CENSORING: {int(censored.sum())}/{n} evaluations "
      f"({100*censored.sum()/n:.0f}%) hit the depletion-schedule end at "
      f"{ceiling:.1f} EFPD.")
    P("Their true cycle lengths are LOWER BOUNDS, not measurements. Any Pareto")
    P("point sitting on this ceiling is schedule-limited, not physics-limited.")

    P("-" * 78)
    P("PARETO SET")
    hdr = (f"{'':4}{'e_in%':>7}{'e_out%':>7}{'Gd%':>6}{'pitch':>7}{'refl':>6}"
           f"{'EFPD':>8}{'±1σ':>6}{'GWd/tHM':>9}{'F_dh':>7}{'k_bol':>7}  flags")
    P(hdr)
    for i, p in enumerate(pareto):
        cens = abs(p["cycle_length"] - ceiling) < CEIL_TOL
        s_cyc = sigma_cycle_est(p, args.sigma_k, args.k_target)
        bu = p["cycle_length"] * args.specific_power / 1000.0
        active = [g for g in CONSTRAINTS if abs(p[g]) < 0.005]
        flags = ("CENSORED " if cens else "") + \
                (f"active:{','.join(active)}" if active else "")
        P(f"P{i+1:<3}{p['enrich_inner']:>7.2f}{p['enrich_outer']:>7.2f}"
          f"{p['gd_wt']:>6.2f}{p['pitch']:>7.3f}{p['refl_thick']:>6.1f}"
          f"{p['cycle_length']:>8.0f}{s_cyc:>6.0f}{bu:>9.1f}"
          f"{p['peaking']:>7.3f}{p['k_bol']:>7.3f}  {flags}")
    P(f"(burnup = EFPD x specific power {args.specific_power} W/gHM / 1000; "
      f"±1σ from σ_k = {args.sigma_k:.0f} pcm)")

    P("-" * 78)
    P("STATISTICAL DISTINGUISHABILITY OF PARETO POINTS")
    any_pair = False
    for i in range(len(pareto)):
        for j in range(i + 1, len(pareto)):
            a, b = pareto[i], pareto[j]
            d_c = abs(a["cycle_length"] - b["cycle_length"])
            d_p = abs(a["peaking"] - b["peaking"])
            s = max(sigma_cycle_est(a, args.sigma_k, args.k_target),
                    sigma_cycle_est(b, args.sigma_k, args.k_target))
            if d_c < 2 * s and d_p < args.peaking_noise:
                any_pair = True
                P(f"  P{i+1} vs P{j+1}: ΔEFPD={d_c:.0f} (<2σ={2*s:.0f}), "
                  f"ΔF_dh={d_p:.4f} (< assumed pin noise {args.peaking_noise}) "
                  f"-> statistically the SAME design")
    if not any_pair:
        P("  all Pareto points are separated by more than the noise estimates")
    P(f"Feasible peaking spans {col(all_raw,'peaking')[feas].min():.3f}"
      f"-{col(all_raw,'peaking')[feas].max():.3f}; the Pareto spread "
      f"({min(p['peaking'] for p in pareto):.3f}-{max(p['peaking'] for p in pareto):.3f})"
      f" is comparable to typical pin-power Monte Carlo noise at this particle count.")

    P("-" * 78)
    P("VARIABLE -> OBJECTIVE RANK CORRELATIONS (Spearman, feasible evals)")
    nc = feas & ~censored
    for v in variables:
        r_c = spearman(col(all_raw, v)[nc], col(all_raw, "cycle_length")[nc])
        r_p = spearman(col(all_raw, v)[feas], col(all_raw, "peaking")[feas])
        P(f"  {v:<14} cycle_length (non-censored): {r_c:+.2f}   "
          f"peaking: {r_p:+.2f}")

    P("-" * 78)
    P("HYPERVOLUME TREND")
    hv = np.asarray(hv, float)
    for i in range(1, len(hv)):
        P(f"  iter {i}: HV={hv[i]:8.1f}  gain={hv[i]-hv[i-1]:+7.1f}")
    if len(hv) >= 4:
        recent = (hv[-1] - hv[-4]) / hv[-4] * 100
        P(f"  gain over last 3 iterations: {recent:+.1f}%  "
          f"({'NOT converged - continue via --resume' if recent > 1 else 'plateau reached'})")
    P("=" * 78)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="optimization_results.json",
                    help="path to optimization_results.json")
    ap.add_argument("--checkpoint", default="optimization_checkpoint.json",
                    help="path to optimization_checkpoint.json (enables the "
                         "all-evaluations figures; optional)")
    ap.add_argument("--outdir", default="fig_analysis",
                    help="directory where figures and the report are written")
    ap.add_argument("--specific-power", type=float, default=9.98,
                    help="core specific power in W/gHM, used to convert EFPD to "
                         "burnup in GWd/tHM (default 9.98)")
    ap.add_argument("--sigma-k", type=float, default=220.0,
                    help="1-sigma Monte Carlo uncertainty on k-effective in pcm "
                         "(default 220, measured from run.log at 4000x60); "
                         "set 0 to disable error bars")
    ap.add_argument("--k-target", type=float, default=1.055,
                    help="representative end-of-cycle k target used only for the "
                         "reactivity-slope noise estimate (default 1.055)")
    ap.add_argument("--peaking-noise", type=float, default=0.02,
                    help="assumed 1-sigma pin-power/peaking noise used in the "
                         "distinguishability check (default 0.02; measure it "
                         "with seed replicates for a rigorous value)")
    ap.add_argument("--doe-size", type=int, default=None,
                    help="number of initial DOE/LHS evaluations (e.g. 24) to "
                         "split budget statistics between DOE and active learning")
    ap.add_argument("--dpi", type=int, default=160, help="figure resolution")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    variables, pareto, all_raw, hv, meta = load(args.results, args.checkpoint)
    have_ck = len(all_raw) > len(pareto)
    ceiling, censored = ceiling_info(all_raw)

    fig_pareto(variables, pareto, all_raw, ceiling, censored,
               args.sigma_k, args.k_target, outdir, args.dpi)
    if hv is not None and len(hv) > 1:
        fig_convergence(hv, outdir, args.dpi)
    if have_ck:
        fig_parallel(variables, pareto, all_raw, outdir, args.dpi)
        fig_sensitivity(variables, pareto, all_raw, censored, outdir, args.dpi)
    fig_constraints(pareto, outdir, args.dpi)

    lines = []
    report(variables, pareto, all_raw, hv, meta, ceiling, censored, args, lines)
    text = "\n".join(lines)
    print(text)
    (outdir / "analysis_report.txt").write_text(text)
    print(f"\nfigures + report written to: {outdir.resolve()}")
    if not have_ck:
        print("NOTE: checkpoint not found -> only Pareto-based figures were produced.")


if __name__ == "__main__":
    sys.exit(main())
