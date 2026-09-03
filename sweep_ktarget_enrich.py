#!/usr/bin/env python3
"""
sweep_ktarget_enrich.py
=======================
Tabulates the Route B leakage factor over ENRICHMENT and REFLECTOR
THICKNESS at the fixed Campaign 8 pitch,

        LF_2D(e, t_refl) = k_inf(e) / k_eff_core(e, t_refl),

on the single-enrichment, gadolinia-free composition, at beginning of
life, with the same make_assembly_model and make_core_model the
Campaign 8 calibration table used. It is the table a follow-on campaign
would query instead of ktarget_table_c8.json, and it is an INDEPENDENT
test of the composition dependence measured by
validate_ktarget_burnup.py on the real Campaign 8 designs.

WHAT IT REPORTS
---------------
  1. The grid of LF_2D and its axial-corrected form k_target = LF_2D * L_ax,
     with L_ax read from ktarget_table_c8.json (schema 3).
  2. A planar fit  LF_2D = a + b * e + c * t_refl  and the enrichment
     slope b in pcm per weight per cent, defined exactly as the
     residual of validate_ktarget_burnup.py, so it compares directly
     with the -62.5 pcm/wt% measured there on gadolinia-bearing designs.
     Agreement means the effect is compositional and the gadolinia is
     irrelevant, as the two-regressor test indicated.
  3. The value the plane predicts at the reference composition of the
     existing table, against that table's own row, as a consistency check.

WHAT IT DOES NOT DO
-------------------
It does NOT retrofit Campaign 8. The target enters the end-of-cycle
criterion, so a different target changes every archived cycle length and
therefore the infill points the optimiser chose. The output is the table
of the follow-on formulation and the measurement behind the
recommendation, nothing else.

Every OpenMC solve is checkpointed in <out>/runs.json, so an interrupted
job resumes where it stopped. The cache key carries the fidelity, so a
relaunch at a different particle count never mixes with old solves.

COST on wks720, 32 threads, defaults below: 6 enrichments x (1 assembly
+ 7 core solves) = 48 solves, about 1 h.

USAGE (conda env openmc-env, from the repository root)
    python sweep_ktarget_enrich.py --selftest
    python sweep_ktarget_enrich.py --dry-run
    setsid nohup python -u sweep_ktarget_enrich.py --threads 32 \\
        --out kt_enrich > kt_enrich.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

PITCH_FIXED = 1.26          # cm, Campaign 8
REFL_DEFAULT = "2.0,2.6,3.2,3.8,4.4,5.0,5.66"     # the C8 table nodes
ENRICH_DEFAULT = "3.0,5.0,7.0,9.0,11.0,13.0"      # spans the C8 box


# --------------------------------------------------------------------------
# Pure-python part, unit-tested by --selftest
# --------------------------------------------------------------------------

def plane_fit(e, t, z):
    """Least squares z = a + b e + c t. Returns (a, b, c), covariance,
    residual rms, dof."""
    e = np.asarray(e, float); t = np.asarray(t, float); z = np.asarray(z, float)
    A = np.vstack([np.ones_like(e), e, t]).T
    XtXi = np.linalg.inv(A.T @ A)
    beta = XtXi @ (A.T @ z)
    r = z - A @ beta
    dof = len(z) - 3
    s2 = float((r ** 2).sum() / dof) if dof > 0 else float("nan")
    return beta, s2 * XtXi, math.sqrt(s2) if dof > 0 else float("nan"), dof, r


def slope_in_pcm(b, lf_ref):
    """dLF/de converted to pcm per wt% with the residual definition of
    validate_ktarget_burnup.py, r = 1e5 (LF - LF_ref) / LF_ref."""
    return 1e5 * b / lf_ref


def selftest():
    e = np.array([3, 3, 7, 7, 11, 11], float)
    t = np.array([2, 5, 2, 5, 2, 5], float)
    z = 1.06 - 0.0007 * e - 0.0005 * t
    beta, cov, rms, dof, r = plane_fit(e, t, z)
    assert abs(beta[0] - 1.06) < 1e-12 and abs(beta[1] + 0.0007) < 1e-12 \
        and abs(beta[2] + 0.0005) < 1e-12, beta
    assert rms < 1e-12 and dof == 3
    assert abs(slope_in_pcm(-0.00065, 1.05) + 61.9047619) < 1e-6
    print("selftest OK")


# --------------------------------------------------------------------------
# OpenMC part
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enrich", default=ENRICH_DEFAULT,
                    help="comma-separated enrichment nodes, wt% U-235")
    ap.add_argument("--refl", default=REFL_DEFAULT,
                    help="comma-separated reflector thickness nodes, cm")
    ap.add_argument("--ktarget-table", default="ktarget_table_c8.json",
                    help="schema-3 table: supplies L_ax and the reference "
                         "row for the consistency check")
    ap.add_argument("--asm-particles", type=int, default=40000)
    ap.add_argument("--asm-batches", type=int, default=120)
    ap.add_argument("--asm-inactive", type=int, default=30)
    ap.add_argument("--core-particles", type=int, default=60000)
    ap.add_argument("--core-batches", type=int, default=170)
    ap.add_argument("--core-inactive", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1,
                    help="independent seeds per node (1 is enough for the "
                         "plane, the residual rms is the noise estimate)")
    ap.add_argument("--fast", action="store_true",
                    help="quarter fidelity, quick look only")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="kt_enrich")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    enr = [float(s) for s in args.enrich.split(",")]
    refl = [float(s) for s in args.refl.split(",")]
    asm_tr = dict(particles=args.asm_particles, batches=args.asm_batches,
                  inactive=args.asm_inactive)
    core_tr = dict(particles=args.core_particles, batches=args.core_batches,
                   inactive=args.core_inactive)
    if args.fast:
        for tr in (asm_tr, core_tr):
            tr["particles"] = max(tr["particles"] // 4, 500)
            tr["batches"] = max(tr["batches"] // 2, 20)
            tr["inactive"] = max(tr["inactive"] // 2, 5)

    tab = json.loads(Path(args.ktarget_table).read_text())
    lax = float(tab.get("axial_leakage_factor", 1.0))
    ref_design = tab.get("design", {})
    ref_e = 0.5 * (float(ref_design.get("enrich_inner", 4.55))
                   + float(ref_design.get("enrich_outer", 4.05)))
    ref_refl = np.asarray(tab["refl_thick_cm"], float)
    ref_k2d = np.asarray(tab.get("k_target_2d_fit", tab["k_target"]), float)
    if "k_target_2d_fit" not in tab:
        ref_k2d = ref_k2d / lax

    n_solves = len(enr) * (1 + len(refl)) * args.seeds
    print("=" * 78)
    print("Route B leakage factor over enrichment and reflector thickness")
    print(f"pitch {PITCH_FIXED} cm fixed, gadolinia 0, single enrichment")
    print(f"enrichment nodes {enr}")
    print(f"reflector nodes  {refl}")
    print(f"assembly {asm_tr}  core {core_tr}  seeds {args.seeds}")
    print(f"axial factor {lax} and reference composition {ref_e:.2f} wt% "
          f"from {args.ktarget_table}")
    print(f"{n_solves} solves in total")
    print("=" * 78)
    if args.dry_run:
        return

    import openmc
    import reactor_model as rm
    op, geo = rm.Operating(), rm.Geometry17x17()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    store = outdir / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}

    def save():
        store.write_text(json.dumps(done, indent=1))

    def fid(tr):
        return f"{tr['particles']}x{tr['batches']}x{tr['inactive']}"

    def design(e, t):
        return {"enrich_inner": e, "enrich_outer": e, "gd_wt": 0.0,
                "pitch": PITCH_FIXED, "refl_thick": t, "gd_pins": 12}

    def run_asm(e, seed):
        key = f"asm_e{e:g}_s{seed}_{fid(asm_tr)}"
        if key in done:
            return done[key]
        m, _fc, _lat = rm.make_assembly_model(design(e, refl[0]), op, geo,
                                              bc="reflective", **asm_tr)
        m.settings.seed = seed
        t0 = time.time()
        sp = m.run(cwd=str(outdir / key), output=False, threads=args.threads)
        with openmc.StatePoint(sp) as s:
            done[key] = dict(k=float(s.keff.nominal_value),
                             sd=float(s.keff.std_dev), wall_s=time.time() - t0)
        save()
        return done[key]

    def run_core(e, t, seed):
        key = f"core_e{e:g}_t{t:g}_s{seed}_{fid(core_tr)}"
        if key in done:
            return done[key]
        m, _fc = rm.make_core_model(design(e, t), op, geo, refl_thick=t,
                                    **core_tr)
        m.settings.seed = seed
        t0 = time.time()
        sp = m.run(cwd=str(outdir / key), output=False, threads=args.threads)
        with openmc.StatePoint(sp) as s:
            done[key] = dict(k=float(s.keff.nominal_value),
                             sd=float(s.keff.std_dev), wall_s=time.time() - t0)
        save()
        return done[key]

    def avg(vals):
        k = np.array([v["k"] for v in vals]); sd = np.array([v["sd"] for v in vals])
        return float(k.mean()), float(math.sqrt((sd ** 2).sum()) / len(vals))

    seeds = list(range(1, args.seeds + 1))
    kinf, Z, SD, rows = [], [], [], []
    for e in enr:
        ki, ki_sd = avg([run_asm(e, s) for s in seeds])
        kinf.append(ki)
        print(f"\nenrichment {e:5.2f} wt%   k_inf = {ki:.5f} +/- {ki_sd:.5f}")
        print(f"{'refl[cm]':>9} {'k_eff':>9} {'sd':>8} {'LF_2D':>9} {'k_target':>9}")
        zrow, sdrow = [], []
        for t in refl:
            ke, ke_sd = avg([run_core(e, t, s) for s in seeds])
            lf = ki / ke
            lf_sd = lf * math.sqrt((ki_sd / ki) ** 2 + (ke_sd / ke) ** 2)
            zrow.append(lf); sdrow.append(lf_sd)
            rows.append((e, t, lf, lf_sd))
            print(f"{t:>9.2f} {ke:>9.5f} {ke_sd:>8.5f} {lf:>9.5f} {lf*lax:>9.5f}")
        if any(b > a + 3e-3 for a, b in zip(zrow, zrow[1:])):
            print("   WARNING: LF not monotone in refl_thick at this enrichment")
        Z.append(zrow); SD.append(sdrow)

    # ---- planar fit and the comparison that matters ---------------------- #
    E = np.array([r[0] for r in rows]); T = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows]); LSD = np.array([r[3] for r in rows])
    beta, cov, rms, dof, resid = plane_fit(E, T, L)
    a, b, c = (float(x) for x in beta)
    sb = math.sqrt(cov[1, 1]); sc = math.sqrt(cov[2, 2])
    lf_ref = a + b * ref_e + c * float(ref_refl.mean())
    slope_pcm = slope_in_pcm(b, lf_ref); slope_pcm_sd = slope_in_pcm(sb, lf_ref)
    refl_pcm = slope_in_pcm(c, lf_ref)
    mean_sd_pcm = 1e5 * float(LSD.mean()) / lf_ref

    print("\n" + "=" * 78)
    print("PLANAR FIT  LF_2D = a + b * e + c * t_refl")
    print(f"  a = {a:.6f}   b = {b:+.6f} ({sb:.6f}) per wt%   "
          f"c = {c:+.6f} ({sc:.6f}) per cm")
    print(f"  residual rms {1e5*rms/lf_ref:.0f} pcm on {dof} dof, "
          f"mean solve uncertainty {mean_sd_pcm:.0f} pcm")
    print(f"  enrichment slope : {slope_pcm:+.1f} +/- {slope_pcm_sd:.1f} pcm/wt%")
    print(f"  reflector slope  : {refl_pcm:+.1f} pcm/cm  "
          f"(C8 table fit: {1e5*tab.get('fit_slope_per_cm', float('nan'))/lf_ref:+.1f})")
    print(f"  compare with validate_ktarget_burnup.py on the real designs: "
          f"-62.5 +/- 5.0 pcm/wt%")
    # consistency with the existing table at its own composition
    print("\nCONSISTENCY at the reference composition of the existing table")
    print(f"{'refl[cm]':>9} {'C8 table':>10} {'plane':>10} {'diff pcm':>9}")
    for t, k2d in zip(ref_refl, ref_k2d):
        pred = a + b * ref_e + c * float(t)
        print(f"{float(t):>9.2f} {k2d:>10.6f} {pred:>10.6f} "
              f"{1e5*(pred-k2d)/k2d:>+9.1f}")
    print("  (the existing table used two rings, 4.55 and 4.05 wt%, so a "
          "few tens of pcm of offset is expected)")

    table = {
        "schema": 4,
        "pitch_cm_fixed": PITCH_FIXED,
        "enrich_wt": enr,
        "refl_thick_cm": refl,
        "k_target_2d": Z,                 # [i_enrich][j_refl], pre-axial
        "k_target": (np.asarray(Z) * lax).tolist(),
        "k_target_sd": SD,
        "k_inf_assembly": kinf,
        "axial_leakage_factor": lax,
        "plane_fit": dict(a=a, b_per_wt=b, c_per_cm=c, b_sd=sb, c_sd=sc,
                          rms=rms, dof=dof,
                          enrich_slope_pcm_per_wt=slope_pcm,
                          enrich_slope_sd_pcm_per_wt=slope_pcm_sd),
        "design": {"gd_wt": 0.0, "single_enrichment": True,
                   "pitch": PITCH_FIXED},
        "transport": dict(assembly=asm_tr, core=core_tr, seeds=args.seeds),
        "geometry": "v2-envelope (no fuel clipping; corners+annulus reflector)",
        "note": "Follow-on Route B table over (enrichment, refl_thick) at the "
                "Campaign 8 pitch. NOT used by Campaign 8. Bilinear "
                "interpolation on (enrich_wt, refl_thick_cm), clamped.",
        "provenance": "sweep_ktarget_enrich.py, " + time.strftime("%Y-%m-%d"),
    }
    out = outdir / "ktarget_table_c8_enrich.json"
    out.write_text(json.dumps(table, indent=1))

    tex = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
           r"  \caption[Leakage factor over enrichment and reflector "
           r"thickness]{Route B leakage factor over enrichment and "
           r"reflector thickness at beginning of life.}",
           r"  \label{tab:kt-enrich}",
           r"  \begin{tabular}{c" + "c" * len(refl) + "}", r"    \toprule",
           r"    $e$ & " + " & ".join(f"{t:g}" for t in refl) + r" \\",
           r"    [wt\%] & " + " & ".join(["[cm]"] * len(refl)) + r" \\",
           r"    \midrule"]
    for e, zrow in zip(enr, Z):
        tex.append(f"    {e:g} & " + " & ".join(f"{z:.4f}" for z in zrow) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (outdir / "summary_table.tex").write_text("\n".join(tex) + "\n")
    print(f"\nwrote {out} and {outdir/'summary_table.tex'}")


if __name__ == "__main__":
    main()
