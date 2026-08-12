#!/usr/bin/env python3
"""
rescore_archive_core.py -- Solution 1 of the Campaign-4 plan: re-evaluate the
WHOLE feasible Campaign-3 archive at core level (BOL transport only) and
answer directly: "did the single-assembly filter discard designs that are
better at core level?"

WHAT IT DOES
  Stage SCREEN : every feasible archive design x N_screen seeds (default 3)
                 at moderate settings. Cheap (~2 min/run).
  Stage RESOLVE: the designs within a noise margin of the screened core
                 minimum, re-run to N_resolve seeds (default 8) at
                 conservative settings (long inactive tail -- the finite core
                 has a real source transient; one screened design in the
                 first study converged only at batch 94 against a 40-batch
                 cutoff).
  REPORT       : (i) core-level ranking with CIs, (ii) Spearman rank
                 correlation between assembly and core peaking -- THE
                 proxy-fidelity number for the thesis, (iii) the core-level
                 Pareto front (EFPD from the checkpoint x measured core
                 F_dH), (iv) the "discarded gems": designs far down the
                 assembly ranking that rise to the top at core level,
                 (v) g_peak(core) status of every design, (vi) the
                 discrepancy fit F_core/F_asm ~ a + b*refl + c*pitch,
                 the bridge model of the multi-fidelity plan.

Every run is cached in <out>/runs.json keyed by (design, seed, settings), so
the script is fully resumable and re-running it costs nothing.

USAGE
  conda activate openmc-env
  setsid nohup python -u rescore_archive_core.py \
      --checkpoint out_c3_atf75/optimization_checkpoint.json \
      --threads 64 --out core_rescore > core_rescore.log 2>&1 < /dev/null &
"""
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import numpy as np
import openmc

import reactor_model as rm

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--screen-seeds", type=int, default=3)
ap.add_argument("--screen-particles", type=int, default=100000)
ap.add_argument("--screen-batches", type=int, default=170)
ap.add_argument("--screen-inactive", type=int, default=60)
ap.add_argument("--resolve-seeds", type=int, default=8)
ap.add_argument("--resolve-particles", type=int, default=100000)
ap.add_argument("--resolve-batches", type=int, default=230)
ap.add_argument("--resolve-inactive", type=int, default=120)
ap.add_argument("--margin-k", type=float, default=4.0,
                help="resolve every design with screen mean <= min + k*sem")
ap.add_argument("--gpeak", type=float, default=2.0,
                help="the peaking limit, applied here at CORE level")
ap.add_argument("--stage", choices=["screen", "resolve", "auto"],
                default="auto")
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--out", default="core_rescore")
args = ap.parse_args()

if args.threads:
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

ck = json.loads(Path(args.checkpoint).read_text())
raw, cn, dv = ck["all_raw"], ck.get("constraint_names", []), ck["design_variables"]
feas = [(i, r) for i, r in enumerate(raw)
        if all(float(r.get(c, 0.0)) <= 1e-9 for c in cn)]
print(f"{len(feas)} feasible designs of {len(raw)} in the archive")

outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
store_p = outdir / "runs.json"
store = json.loads(store_p.read_text()) if store_p.exists() else {}
geo, op = rm.Geometry17x17(), rm.Operating()
NL = geo.lattice
try:
    import core_geometry as cg
    nx = ny = cg.CORE_MAP_32.shape[0]
except Exception:
    nx = ny = 6


def one_core_run(design, seed, case, particles, batches, inactive):
    m = rm.make_core_model(design, op, geo, particles=particles,
                           batches=batches, inactive=inactive)
    model = m[0] if isinstance(m, tuple) else m
    model.settings.seed = seed
    pitch = design.get("pitch", 1.26)
    half = nx * NL * pitch / 2.0
    pin = openmc.RegularMesh(); pin.dimension = (nx * NL, ny * NL)
    pin.lower_left = (-half, -half); pin.upper_right = (half, half)
    t1 = openmc.Tally(name="core_pin"); t1.filters = [openmc.MeshFilter(pin)]
    t1.scores = ["fission"]
    model.tallies = openmc.Tallies([t1])
    t0 = time.time()
    sp_path = model.run(cwd=str(case), output=False)
    wall = time.time() - t0
    with openmc.StatePoint(sp_path) as sp:
        keff = float(sp.keff.nominal_value)
        vp = sp.get_tally(name="core_pin").get_values(
            scores=["fission"]).reshape(ny * NL, nx * NL)
        H = np.asarray(getattr(sp, "entropy", []), dtype=float)
    fp = np.ma.masked_equal(vp, 0.0)
    fdh = float((fp / fp.mean()).max())
    conv = None
    if H.size:
        tail = H[inactive + (len(H) - inactive) // 2:]
        mu, sd = tail.mean(), tail.std(ddof=1)
        Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
        Hs[0], Hs[-1] = H[0], H[-1]
        bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
        conv = int(bad[-1]) + 2 if len(bad) else 1
    return dict(keff=keff, fdh_core=fdh, entropy_conv_batch=conv, wall_s=wall)


def get_runs(idx, design, seeds, particles, batches, inactive, tag):
    """Return the list of run dicts for this design at these settings,
    computing (and caching) any that are missing."""
    out = []
    for s in range(1, seeds + 1):
        key = f"d{idx}_s{s}_{tag}"
        if key not in store:
            case = outdir / f"design{idx}" / tag / f"seed{s}"
            case.mkdir(parents=True, exist_ok=True)
            store[key] = one_core_run(design, s, case,
                                      particles, batches, inactive)
            store_p.write_text(json.dumps(store, indent=1))
            r = store[key]
            flag = ("  !! entropy conv b%s > inactive %d"
                    % (r["entropy_conv_batch"], inactive)
                    if r["entropy_conv_batch"] and
                    r["entropy_conv_batch"] > inactive else "")
            print(f"    d{idx} s{s} [{tag}]: F_dH={r['fdh_core']:.4f} "
                  f"k_eff={r['keff']:.5f} ({r['wall_s']:.0f}s){flag}",
                  flush=True)
        out.append(store[key])
    return out


scr_tag = f"{args.screen_particles}x{args.screen_batches}i{args.screen_inactive}"
res_tag = f"{args.resolve_particles}x{args.resolve_batches}i{args.resolve_inactive}"

# ------------------------------ SCREEN ------------------------------------ #
screen = {}
if args.stage in ("screen", "auto"):
    print("=" * 78)
    print(f"STAGE SCREEN: {len(feas)} designs x {args.screen_seeds} seeds "
          f"@ {scr_tag}")
    print("=" * 78)
    for idx, rec in feas:
        design = {k: float(rec[k]) for k in dv}
        runs = get_runs(idx, design, args.screen_seeds,
                        args.screen_particles, args.screen_batches,
                        args.screen_inactive, scr_tag)
        v = [r["fdh_core"] for r in runs]
        screen[idx] = dict(mean=st.mean(v),
                           sem=(st.stdev(v) / math.sqrt(len(v))
                                if len(v) > 1 else float("nan")),
                           keff=st.mean(r["keff"] for r in runs))
        print(f"  d{idx:2d}: core F_dH {screen[idx]['mean']:.4f} "
              f"+/- {screen[idx]['sem']:.4f} (sem)   "
              f"assembly {rec['peaking']:.4f}   EFPD {rec['cycle_length']:.0f}")

# ------------------------------ RESOLVE ----------------------------------- #
resolved = {}
if args.stage in ("resolve", "auto") and screen:
    best = min(s["mean"] for s in screen.values())
    sems = [s["sem"] for s in screen.values() if not math.isnan(s["sem"])]
    pooled = st.median(sems)
    thr = best + args.margin_k * pooled
    picks = sorted([i for i, s in screen.items() if s["mean"] <= thr],
                   key=lambda i: screen[i]["mean"])
    print("=" * 78)
    print(f"STAGE RESOLVE: {len(picks)} designs within "
          f"{args.margin_k} x {pooled:.4f} of the minimum ({thr:.4f}) "
          f"-> {args.resolve_seeds} seeds @ {res_tag}")
    print("=" * 78)
    for idx in picks:
        rec = dict(feas)[idx]
        design = {k: float(rec[k]) for k in dv}
        runs = get_runs(idx, design, args.resolve_seeds,
                        args.resolve_particles, args.resolve_batches,
                        args.resolve_inactive, res_tag)
        v = [r["fdh_core"] for r in runs]
        resolved[idx] = dict(mean=st.mean(v),
                             sem=st.stdev(v) / math.sqrt(len(v)),
                             sd=st.stdev(v))

# ------------------------------ REPORT ------------------------------------ #
if screen:
    recs = dict(feas)
    idxs = sorted(screen, key=lambda i: screen[i]["mean"])
    asm_rank = {i: r for r, i in enumerate(
        sorted(screen, key=lambda i: recs[i]["peaking"]), 1)}
    core_rank = {i: r for r, i in enumerate(idxs, 1)}

    # Spearman rank correlation, assembly vs core
    n = len(idxs)
    dsum = sum((asm_rank[i] - core_rank[i]) ** 2 for i in idxs)
    rho = 1 - 6 * dsum / (n * (n * n - 1))

    print("\n" + "=" * 78)
    print("CORE-LEVEL RANKING (screen means; * = resolved at high settings)")
    print("=" * 78)
    print(f"{'core':>4s} {'asm':>4s} {'idx':>4s} {'F_core':>8s} {'+/-sem':>7s}"
          f" {'F_asm':>7s} {'EFPD':>6s}  {'g_peak(core)':>12s}")
    for i in idxs:
        m = resolved.get(i, screen[i])
        mark = "*" if i in resolved else " "
        g = m["mean"] - args.gpeak
        print(f"{core_rank[i]:>4d} {asm_rank[i]:>4d} {i:>4d} "
              f"{m['mean']:8.4f} {m['sem']:7.4f} {recs[i]['peaking']:7.4f} "
              f"{recs[i]['cycle_length']:6.0f}  {g:+11.4f}{mark}")
    print(f"\nSpearman rank correlation (assembly vs core peaking): "
          f"rho = {rho:+.3f}   (n={n})")
    print("   rho ~ +1: the proxy ranked correctly; rho ~ 0: the assembly")
    print("   filter was uninformative about core peaking; negative: inverted.")

    # discarded gems: big rank gains
    gems = sorted(idxs, key=lambda i: asm_rank[i] - core_rank[i],
                  reverse=True)[:5]
    print("\nlargest rank GAINS assembly -> core (the 'discarded gems' check):")
    for i in gems:
        print(f"  d{i}: assembly #{asm_rank[i]} -> core #{core_rank[i]} "
              f"(gain {asm_rank[i]-core_rank[i]:+d})  "
              f"EFPD={recs[i]['cycle_length']:.0f}")

    # core-level Pareto front (EFPD maximise, core F_dH minimise)
    pts = [(i, recs[i]["cycle_length"],
            (resolved.get(i, screen[i]))["mean"]) for i in idxs]
    front = [a for a in pts if not any(
        (b[1] >= a[1] and b[2] <= a[2] and (b[1] > a[1] or b[2] < a[2]))
        for b in pts)]
    print("\nCORE-LEVEL PARETO FRONT (EFPD x measured core F_dH):")
    for i, e, f in sorted(front, key=lambda t: -t[1]):
        m = resolved.get(i, screen[i])
        print(f"  d{i}: EFPD={e:6.0f}  core F_dH={f:.4f} +/- {m['sem']:.4f}"
              f"   {'PASSES' if f <= args.gpeak else 'violates'} "
              f"F<= {args.gpeak}")

    # discrepancy (bridge) fit: ratio ~ a + b*refl + c*pitch
    y = np.array([(resolved.get(i, screen[i]))["mean"] / recs[i]["peaking"]
                  for i in idxs])
    R = np.array([float(recs[i]["refl_thick"]) for i in idxs])
    P = np.array([float(recs[i]["pitch"]) for i in idxs])
    A = np.c_[np.ones(n), R, P]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\nBRIDGE FIT: F_core/F_asm = {b[0]:.3f} {b[1]:+.4f}*refl_thick "
          f"{b[2]:+.3f}*pitch    R^2={r2:.3f}")
    print("   (the multi-fidelity correction model; residual scatter = what a")
    print("    Kennedy-O'Hagan discrepancy GP would still have to learn)")

    # CSV
    csv = outdir / "core_rescore.csv"
    lines = ["idx,asm_rank,core_rank,fdh_asm,fdh_core,fdh_core_sem,"
             "resolved,EFPD,refl_thick,pitch,ratio"]
    for i in idxs:
        m = resolved.get(i, screen[i])
        lines.append(f"{i},{asm_rank[i]},{core_rank[i]},"
                     f"{recs[i]['peaking']},{m['mean']},{m['sem']},"
                     f"{int(i in resolved)},{recs[i]['cycle_length']},"
                     f"{recs[i]['refl_thick']},{recs[i]['pitch']},"
                     f"{m['mean']/recs[i]['peaking']}")
    csv.write_text("\n".join(lines) + "\n")
    print(f"\nCSV: {csv}    raw (resumable): {store_p}")
