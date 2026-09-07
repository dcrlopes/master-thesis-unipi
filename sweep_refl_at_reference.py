#!/usr/bin/env python3
"""
sweep_refl_at_reference.py
==========================
Resolves the reflector-slope discrepancy between ktarget_table_c8.json and
sweep_ktarget_enrich.py.

THE DISCREPANCY
---------------
Both routines measure the same quantity, the Route B leakage factor

        LF_2D(t) = k_inf / k_eff_core(t),

as a function of reflector thickness, and both fit it well.

    ktarget_table_c8.json   fit_slope_per_cm -5.160e-04, about -47.6 pcm/cm,
                            chi2/dof 0.378, reference composition
                            enrich_inner 4.55, enrich_outer 4.05, gd_wt 0.
    sweep_ktarget_enrich.py -19.0 to -36.9 pcm/cm across six uniform
                            enrichments, weighted mean -25.1 +/- 4.0 pcm/cm,
                            chi2/dof 0.72 about that mean, so the slope is
                            enrichment-independent.

The two disagree by a factor of about 1.9, roughly six standard deviations,
and the gap is not explained by any of the following, each checked:

    gadolinia        both references carry gd_wt = 0.
    enforce_vessel   raises ValueError or does nothing, never alters the
                     model, so it cannot shift a slope.
    axial factor     a constant multiplier, worth 3 per cent, not a factor
                     of two.
    interaction      the enrichment by reflector term of the sweep is 0.1
                     standard deviations from zero, so the slope does not
                     depend on enrichment.

The one uncontrolled difference left is the composition itself. The table's
reference assembly is zoned within the assembly at 4.55 and 4.05 wt%, and
every sweep node is flat at a single enrichment. This script removes that
difference.

WHAT IT DOES
------------
It runs the sweep's own solve routine at the table's own reference design,
over the table's own reflector grid, and reports the slope three ways.

    1. This run's slope, in pcm per cm.
    2. The table's slope, read from fit_slope_per_cm.
    3. The sweep's weighted mean over its six enrichment rows.

It also compares the ABSOLUTE leakage factor node by node against the
table, which is the stronger test: two routines that agree on the level
but not on the slope have a different problem from two that agree on
neither.

    slope reproduces the table    the composition explains the gap, the
                                  table is correct for the geometry the
                                  campaign used, and the sweep must be
                                  quoted as a flat-enrichment study only.
    slope reproduces the sweep    the two routines disagree on the same
                                  physical problem. That is a defect and
                                  it must be found before either number is
                                  quoted.
    neither                       report both and quote no reflector slope.

COST
----
One assembly solve plus seven core solves per seed. At the sweep's own
fidelity on wks720 with 32 threads this is about 12 min for one seed. The
assembly solve is reflective, so k_inf does not depend on the reflector and
one solve serves all seven nodes.

The slope uncertainty is about 14 pcm/cm at one seed and 10 pcm/cm at two,
against a 23 pcm/cm gap to resolve. One seed separates the two hypotheses
at about 1.6 standard deviations, two seeds at about 2.3. Start with one.

Every solve is cached in <out>/runs.json with the fidelity in the key, so
an interrupted job resumes and a relaunch at a different particle count
never reuses an old solve.

USAGE (conda env openmc-env, from the repository root)
    python sweep_refl_at_reference.py --selftest
    python sweep_refl_at_reference.py --dry-run
    setsid nohup python -u sweep_refl_at_reference.py --threads 32 \
        --out kt_refl_ref > kt_refl_ref.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PITCH_FIXED = 1.26                 # cm, Campaign 8, never a variable here
SWEEP_MEAN_PCM_PER_CM = -25.1      # weighted mean of the six sweep rows
SWEEP_MEAN_SD_PCM_PER_CM = 4.0


# --------------------------------------------------------------------------
# Pure-python part, unit-tested by --selftest
# --------------------------------------------------------------------------

def line_fit(t, z, sd):
    """Weighted least squares z = a + b t.

    Returns (a, b, sd_b, chi2_per_dof). Weights are 1/sd**2. The intercept
    uncertainty is not needed and is not returned.
    """
    t = np.asarray(t, float)
    z = np.asarray(z, float)
    w = 1.0 / np.asarray(sd, float) ** 2
    A = np.vstack([np.ones_like(t), t]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    beta = cov @ (A.T @ W @ z)
    r = z - A @ beta
    dof = len(z) - 2
    chi2 = float((w * r ** 2).sum())
    return float(beta[0]), float(beta[1]), math.sqrt(float(cov[1, 1])), \
        (chi2 / dof if dof > 0 else float("nan"))


def slope_in_pcm(b, lf_ref):
    """dLF/dt converted to pcm per cm with the residual definition used by
    validate_ktarget_burnup.py and sweep_ktarget_enrich.py,
    r = 1e5 (LF - LF_ref) / LF_ref."""
    return 1e5 * b / lf_ref


def verdict(this_pcm, this_sd, table_pcm, sweep_pcm, sweep_sd):
    """Name which hypothesis this run supports, in standard deviations."""
    n_table = abs(this_pcm - table_pcm) / this_sd if this_sd > 0 else float("inf")
    n_sweep = abs(this_pcm - sweep_pcm) / math.sqrt(this_sd ** 2 + sweep_sd ** 2)
    if n_table < 2.0 <= n_sweep:
        tag = "TABLE: the composition explains the gap"
    elif n_sweep < 2.0 <= n_table:
        tag = "SWEEP: the two routines disagree, this is a defect"
    elif n_table < 2.0 and n_sweep < 2.0:
        tag = "UNRESOLVED: consistent with both, add seeds"
    else:
        tag = "NEITHER: inconsistent with both, do not quote a slope"
    return tag, n_table, n_sweep


def selftest():
    t = np.array([2.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.66])
    z = 1.0535 - 5.0e-4 * (t - 2.0)
    sd = np.full_like(t, 1.0e-5)
    a, b, sd_b, chi2 = line_fit(t, z, sd)
    assert abs(b + 5.0e-4) < 1e-12, b
    assert chi2 < 1e-6, chi2
    assert abs(slope_in_pcm(-5.0e-4, 1.0535) + 47.4608) < 1e-3

    tag, nt, ns = verdict(-47.0, 5.0, -47.6, -25.1, 4.0)
    assert tag.startswith("TABLE"), tag
    tag, nt, ns = verdict(-25.5, 5.0, -47.6, -25.1, 4.0)
    assert tag.startswith("SWEEP"), tag
    tag, nt, ns = verdict(-36.0, 20.0, -47.6, -25.1, 4.0)
    assert tag.startswith("UNRESOLVED"), tag
    tag, nt, ns = verdict(-90.0, 5.0, -47.6, -25.1, 4.0)
    assert tag.startswith("NEITHER"), tag
    print("selftest OK")


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

def preflight(args, table_path):
    print("=" * 74)
    print(" PREFLIGHT")
    print("=" * 74)
    print(f"  host        : {socket.gethostname()}")
    print(f"  cwd         : {os.getcwd()}")
    env = os.environ.get("CONDA_DEFAULT_ENV", "NONE")
    print(f"  conda env   : {env}")
    if env != "openmc-env":
        sys.exit("FAIL: wrong conda environment. Run: conda activate openmc-env")
    try:
        import openmc
    except Exception as exc:
        sys.exit(f"FAIL: openmc not importable ({exc}). You are probably in (base).")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  openmc      : {openmc.__version__}")
    if openmc.__version__ != "0.15.3":
        sys.exit("FAIL: openmc is not 0.15.3, this is not the campaign environment")
    try:
        branch = subprocess.run(["git", "branch", "--show-current"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        branch = "unknown"
    print(f"  branch      : {branch}")
    if branch != "main":
        sys.exit("FAIL: wrong branch. Run: git checkout main")
    xs = os.environ.get("OPENMC_CROSS_SECTIONS", "")
    if not xs or not Path(xs).is_file():
        sys.exit("FAIL: OPENMC_CROSS_SECTIONS is not set or the file is missing")
    print(f"  XS          : {xs}")
    print("  chain       : not required, this study has no depletion")
    for f in (table_path, "reactor_model.py", "core_geometry.py"):
        if not Path(f).exists():
            sys.exit(f"FAIL: missing {f}")
    print(f"  threads     : {args.threads or 'openmc default'} of {os.cpu_count()}")
    busy = subprocess.run(
        ["pgrep", "-af", r"python.*(run_optimization|confirm3d\.py|boron_worth|"
                         r"validate_ktarget|sweep_ktarget)"],
        capture_output=True, text=True).stdout.strip()
    busy = "\n".join(l for l in busy.splitlines()
                     if "sweep_refl_at_reference" not in l)
    if busy:
        print("  RUNNING JOBS:")
        print(busy)
        sys.exit("FAIL: a simulation is already running. Wait or stop it first.")
    print("  no simulation currently running")
    print("preflight OK\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ktarget-table", default="ktarget_table_c8.json",
                    help="supplies the reference design, the reflector grid, "
                         "the axial factor and the slope to be tested")
    ap.add_argument("--refl", default=None,
                    help="comma-separated override of the reflector grid, in cm")
    ap.add_argument("--asm-particles", type=int, default=40000)
    ap.add_argument("--asm-batches", type=int, default=120)
    ap.add_argument("--asm-inactive", type=int, default=30)
    ap.add_argument("--core-particles", type=int, default=60000)
    ap.add_argument("--core-batches", type=int, default=170)
    ap.add_argument("--core-inactive", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1,
                    help="replicates per node. 1 gives about 14 pcm/cm on the "
                         "slope, 2 gives about 10")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="kt_refl_ref")
    ap.add_argument("--fast", action="store_true",
                    help="quarter particles, half batches, for a smoke run only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        return selftest()

    tab = json.loads(Path(args.ktarget_table).read_text())
    ref = tab.get("design", {})
    e_in = float(ref["enrich_inner"])
    e_out = float(ref["enrich_outer"])
    gd_wt = float(ref.get("gd_wt", 0.0))
    lax = float(tab.get("axial_leakage_factor", 1.0))
    refl = ([float(s) for s in args.refl.split(",")] if args.refl
            else [float(x) for x in tab["refl_thick_cm"]])

    # The table stores k_target = LF_2D * L_ax unless the 2D fit is present.
    if "k_target_2d_raw" in tab:
        tab_lf = np.asarray(tab["k_target_2d_raw"], float)
    elif "k_target_2d_fit" in tab:
        tab_lf = np.asarray(tab["k_target_2d_fit"], float)
    else:
        tab_lf = np.asarray(tab["k_target"], float) / lax
    tab_slope_k = float(tab["fit_slope_per_cm"])
    tab_slope_pcm = slope_in_pcm(tab_slope_k / lax, float(tab_lf.mean()))

    if not args.dry_run:
        preflight(args, args.ktarget_table)
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    asm_tr = dict(particles=args.asm_particles, batches=args.asm_batches,
                  inactive=args.asm_inactive)
    core_tr = dict(particles=args.core_particles, batches=args.core_batches,
                   inactive=args.core_inactive)
    if args.fast:
        for tr in (asm_tr, core_tr):
            tr["particles"] = max(tr["particles"] // 4, 500)
            tr["batches"] = max(tr["batches"] // 2, 20)
            tr["inactive"] = max(tr["inactive"] // 2, 5)

    n_solves = args.seeds * (1 + len(refl))
    print("=" * 74)
    print("Reflector slope at the k_target table's own reference composition")
    print("=" * 74)
    print(f"reference design : enrich_inner {e_in} wt%, enrich_outer {e_out} wt%, "
          f"gd_wt {gd_wt} wt%")
    print(f"pitch            : {PITCH_FIXED} cm, fixed")
    print(f"reflector nodes  : {refl}")
    print(f"assembly         : {asm_tr}")
    print(f"core             : {core_tr}")
    print(f"seeds            : {args.seeds}")
    print(f"axial factor     : {lax} (constant, divides out of the slope)")
    print(f"table slope      : {tab_slope_pcm:+.1f} pcm/cm")
    print(f"sweep mean slope : {SWEEP_MEAN_PCM_PER_CM:+.1f} +/- "
          f"{SWEEP_MEAN_SD_PCM_PER_CM:.1f} pcm/cm")
    print(f"{n_solves} solves in total")
    print("=" * 74)
    if args.dry_run:
        return 0

    import openmc
    import reactor_model as rm
    op, geo = rm.Operating(), rm.Geometry17x17()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    store = outdir / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}

    def save():
        store.write_text(json.dumps(done, indent=1))

    def fid(tr):
        return f"{tr['particles']}x{tr['batches']}x{tr['inactive']}"

    def design(t):
        """The table's reference assembly, at reflector thickness t."""
        return {"enrich_inner": e_in, "enrich_outer": e_out, "gd_wt": gd_wt,
                "pitch": PITCH_FIXED, "refl_thick": t, "gd_pins": 12}

    def run_asm(seed):
        key = f"asm_ref_s{seed}_{fid(asm_tr)}"
        if key in done:
            return done[key]
        m, _fc, _lat = rm.make_assembly_model(design(refl[0]), op, geo,
                                              bc="reflective", **asm_tr)
        m.settings.seed = seed
        t0 = time.time()
        sp = m.run(cwd=str(outdir / key), output=False, threads=args.threads)
        with openmc.StatePoint(sp) as s:
            done[key] = dict(k=float(s.keff.nominal_value),
                             sd=float(s.keff.std_dev), wall_s=time.time() - t0)
        save()
        return done[key]

    def run_core(t, seed):
        key = f"core_ref_t{t:g}_s{seed}_{fid(core_tr)}"
        if key in done:
            return done[key]
        m, _fc = rm.make_core_model(design(t), op, geo, refl_thick=t,
                                    enforce_vessel=False, **core_tr)
        m.settings.seed = seed
        t0 = time.time()
        sp = m.run(cwd=str(outdir / key), output=False, threads=args.threads)
        with openmc.StatePoint(sp) as s:
            done[key] = dict(k=float(s.keff.nominal_value),
                             sd=float(s.keff.std_dev), wall_s=time.time() - t0)
        save()
        return done[key]

    def avg(vals):
        k = np.array([v["k"] for v in vals])
        sd = np.array([v["sd"] for v in vals])
        return float(k.mean()), float(math.sqrt((sd ** 2).sum()) / len(vals))

    seeds = list(range(1, args.seeds + 1))
    ki, ki_sd = avg([run_asm(s) for s in seeds])
    print(f"\nk_inf at the reference composition = {ki:.5f} +/- {ki_sd:.5f}")
    print(f"table k_inf_assembly               = {tab.get('k_inf_assembly')}\n")

    print(f"{'refl[cm]':>9} {'k_core':>9} {'sd':>8} {'LF_2D':>9} "
          f"{'table LF':>9} {'diff[pcm]':>10}")
    lf, lf_sd = [], []
    for j, t in enumerate(refl):
        ke, ke_sd = avg([run_core(t, s) for s in seeds])
        v = ki / ke
        # k_inf noise is common to every node and cancels in the slope, so
        # only the core solve contributes to the per-node weight.
        s_v = v * ke_sd / ke
        lf.append(v)
        lf_sd.append(s_v)
        ref_v = float(tab_lf[j]) if j < len(tab_lf) else float("nan")
        d = 1e5 * (v - ref_v) / ref_v if ref_v == ref_v else float("nan")
        print(f"{t:>9.2f} {ke:>9.5f} {ke_sd:>8.5f} {v:>9.5f} "
              f"{ref_v:>9.5f} {d:>+10.0f}")

    a, b, sd_b, chi2 = line_fit(refl, lf, lf_sd)
    lf_mean = float(np.mean(lf))
    this_pcm = slope_in_pcm(b, lf_mean)
    this_sd = abs(slope_in_pcm(sd_b, lf_mean))
    span = (refl[-1] - refl[0]) * this_pcm

    tag, n_table, n_sweep = verdict(this_pcm, this_sd, tab_slope_pcm,
                                    SWEEP_MEAN_PCM_PER_CM,
                                    SWEEP_MEAN_SD_PCM_PER_CM)

    L = []
    P = L.append
    P("=" * 74)
    P("RESULT")
    P("=" * 74)
    P(f"  this run    : {this_pcm:+.1f} +/- {this_sd:.1f} pcm/cm, "
      f"chi2/dof {chi2:.2f}")
    P(f"  table       : {tab_slope_pcm:+.1f} pcm/cm   "
      f"({n_table:.1f} sigma from this run)")
    P(f"  sweep mean  : {SWEEP_MEAN_PCM_PER_CM:+.1f} +/- "
      f"{SWEEP_MEAN_SD_PCM_PER_CM:.1f} pcm/cm   "
      f"({n_sweep:.1f} sigma from this run)")
    P(f"  range {refl[0]:.2f} to {refl[-1]:.2f} cm : {span:+.0f} pcm")
    P("")
    P(f"  VERDICT: {tag}")
    P("")
    P("  A chi2/dof far above one means the leakage factor is not linear in")
    P("  reflector thickness at this composition, in which case no single")
    P("  slope should be quoted and the table's linear fit is the wrong form.")
    txt = "\n".join(L)
    print(txt)

    (outdir / "report.txt").write_text(txt + "\n", encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(dict(
        reference=dict(enrich_inner=e_in, enrich_outer=e_out, gd_wt=gd_wt,
                       pitch=PITCH_FIXED),
        refl_thick_cm=refl, k_inf=ki, k_inf_sd=ki_sd,
        lf_2d=lf, lf_2d_sd=lf_sd, table_lf_2d=tab_lf.tolist(),
        slope_pcm_per_cm=this_pcm, slope_sd_pcm_per_cm=this_sd,
        chi2_per_dof=chi2, table_slope_pcm_per_cm=tab_slope_pcm,
        sweep_mean_pcm_per_cm=SWEEP_MEAN_PCM_PER_CM, verdict=tag,
        sigma_from_table=n_table, sigma_from_sweep=n_sweep,
        fidelity=dict(assembly=asm_tr, core=core_tr, seeds=args.seeds),
    ), indent=1), encoding="utf-8")
    print(f"\nwrote {outdir}/report.txt and {outdir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
