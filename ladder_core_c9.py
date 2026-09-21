#!/usr/bin/env python3
"""
ladder_core_c9.py -- BOL core-level fidelity ladder of F_dH on the Campaign 9 front.

The fidelity ladder in the thesis (tab:fid-ladder) measures the ASSEMBLY F_dH,
which was the objective only in Campaigns 3 and 5. From Campaign 4 onwards
F_dH and the k window come from the CORE solve at 100000 x 170 (60 inactive).
This script measures that estimator: same core model, same zoned loading map,
same pin mesh and masking as OpenMCEvaluator._bol_core_peaking, at several
particle counts and several seeds per design. BOL only, no depletion.

Design of the study (paired): every design is solved on every rung with the
same number of seeds, so noise and bias are compared on identical designs.
Seeds are the evaluator's CRC32 design hash with a per-replica salt, so the
noise is independent across designs and across replicas.

Analysis, per design:
  F(N) = F_inf + a / sqrt(N)    weighted least squares over the rungs,
                                N = particles x active batches
  bias at a rung = mean F at that rung - F_inf   (absolute, not relative
                   to the top rung, which is itself biased)
Across designs: pooled seed-to-seed s.d. per rung and its fitted exponent
(-0.5 expected); and, at the campaign rung, which front pairs are closer than
twice their combined single-seed s.d. (not resolved by one campaign solve).

USAGE (wks720; never run the transport modes locally)
  python ladder_core_c9.py --estimate --threads 64
  nohup python -u ladder_core_c9.py --threads 64 > ladder_core_c9.log 2>&1 &
  python ladder_core_c9.py --analyse          # post-processing only, no OpenMC
  python ladder_core_c9.py --selftest         # checks the fit, no OpenMC
Outputs: <out>/runs.json (cache, resumable), <out>/summary.json
"""
import argparse
import json
import math
import os
import statistics as st
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
ap.add_argument("--manifest", default="c9_post/c9_front.json",
                help="design indices default to its 'front' list")
ap.add_argument("--designs", type=int, nargs="*", default=None)
ap.add_argument("--particles", type=int, nargs="+", default=[25000, 100000, 400000])
ap.add_argument("--campaign-particles", type=int, default=100000)
ap.add_argument("--batches", type=int, default=170)
ap.add_argument("--inactive", type=int, default=60)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--threads", type=int, default=64)
ap.add_argument("--out", default="ladder_core_c9")
ap.add_argument("--estimate", action="store_true", help="time one campaign-rung solve, extrapolate, exit")
ap.add_argument("--analyse", action="store_true", help="analyse runs.json only, no transport")
ap.add_argument("--selftest", action="store_true")
args = ap.parse_args()

KEYS = ("enrich_inner", "enrich_outer", "gd_wt", "pitch", "refl_thick", "gd_pins")  # as confirm3d.py
N_ACT = lambda p: p * (args.batches - args.inactive)


# --------------------------------------------------------------------------- #
#  Analysis (pure Python, runs anywhere)                                      #
# --------------------------------------------------------------------------- #
def fit_inf(Ns, means, sems):
    """Weighted LSQ of F = F_inf + a x, x = N^-1/2. Returns F_inf, a, sd(F_inf)."""
    xs = [n ** -0.5 for n in Ns]
    ws = [1.0 / max(s, 1e-12) ** 2 for s in sems]
    W = sum(ws); X = sum(w * x for w, x in zip(ws, xs)) / W; Y = sum(w * y for w, y in zip(ws, means)) / W
    Sxx = sum(w * (x - X) ** 2 for w, x in zip(ws, xs))
    a = sum(w * (x - X) * (y - Y) for w, x, y in zip(ws, xs, means)) / Sxx
    f_inf = Y - a * X
    return f_inf, a, math.sqrt(1.0 / W + X * X / Sxx)


def analyse(runs, designs, rungs):
    per, pooled = {}, {}
    for d in designs:
        rows = []
        for p in rungs:
            r = [runs[k] for k in runs if k.startswith(f"d{d}_p{p}_s")]
            if len(r) < 2:
                continue
            f = [x["fdh"] for x in r]
            rows.append(dict(particles=p, n=len(f), mean=st.mean(f), sd=st.stdev(f),
                             sem=st.stdev(f) / len(f) ** 0.5,
                             rel_err_hot=st.mean(x["rel_err"] for x in r),
                             n_tied=st.mean(x["n_tied"] for x in r),
                             keff=st.mean(x["keff"] for x in r), keff_sd_seeds=st.stdev(x["keff"] for x in r),
                             keff_sigma=st.mean(x["keff_sd"] for x in r),
                             wall_s=st.mean(x["wall_s"] for x in r)))
        if len(rows) >= 2:
            f_inf, a, sd_inf = fit_inf([N_ACT(r["particles"]) for r in rows],
                                       [r["mean"] for r in rows], [r["sem"] for r in rows])
            for r in rows:
                r["bias"] = r["mean"] - f_inf
            per[d] = dict(rungs=rows, F_inf=f_inf, F_inf_sd=sd_inf, a=a)
    for p in rungs:
        sds = [(r["n"] - 1, r["sd"]) for v in per.values() for r in v["rungs"] if r["particles"] == p]
        if sds:
            pooled[p] = math.sqrt(sum(k * s * s for k, s in sds) / sum(k for k, _ in sds))
    slope = None
    if len(pooled) >= 2:
        lx = [math.log(N_ACT(p)) for p in pooled]; ly = [math.log(s) for s in pooled.values()]
        mx, my = st.mean(lx), st.mean(ly)
        slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)
    # pairs of designs not resolved by one campaign-rung solve
    cp, unresolved = args.campaign_particles, []
    sd_c = {d: next((r["sd"] for r in v["rungs"] if r["particles"] == cp), None) for d, v in per.items()}
    ds = [d for d in per if sd_c[d] is not None]
    for i, d1 in enumerate(ds):
        for d2 in ds[i + 1:]:
            gap = abs(per[d1]["F_inf"] - per[d2]["F_inf"])
            if gap < 2 * math.hypot(sd_c[d1], sd_c[d2]):
                unresolved.append([d1, d2, round(gap, 5)])
    return dict(per_design=per, pooled_sd=pooled, sd_exponent=slope,
                campaign_particles=cp, unresolved_pairs=unresolved)


def report(s):
    print(f"\n{'design':>6} {'particles':>9} {'mean F':>8} {'sd':>7} {'bias':>8} {'hot err':>7} "
          f"{'n_tied':>6} {'k sd':>7} {'wall s':>7}")
    for d, v in s["per_design"].items():
        for r in v["rungs"]:
            print(f"{d:>6} {r['particles']:>9} {r['mean']:8.4f} {r['sd']:7.4f} {r['bias']:+8.4f} "
                  f"{r['rel_err_hot']*100:6.2f}% {r['n_tied']:6.1f} {r['keff_sd_seeds']*1e5:5.0f}pcm {r['wall_s']:7.1f}")
        print(f"{d:>6} F_inf = {v['F_inf']:.4f} +- {v['F_inf_sd']:.4f}")
    print("pooled seed-to-seed sd:", {p: round(x, 5) for p, x in s["pooled_sd"].items()},
          f" exponent {s['sd_exponent']:.3f} (expected -0.5)" if s["sd_exponent"] is not None else "")
    print(f"front pairs not resolved by one {s['campaign_particles']}-particle solve:",
          s["unresolved_pairs"] or "none")


def selftest():
    import random
    random.seed(1)
    rungs, runs = [25000, 100000, 400000], {}
    for d, (finf, a) in {1: (1.60, 400.0), 2: (1.62, 300.0)}.items():
        for p in rungs:
            for s in range(1, 9):
                sd = 0.004 * math.sqrt(1e5 / p)
                runs[f"d{d}_p{p}_s{s}"] = dict(fdh=finf + a / math.sqrt(N_ACT(p)) + random.gauss(0, sd),
                                               rel_err=0.01, n_tied=3, keff=1.1, keff_sd=3e-4, wall_s=1.0)
    s = analyse(runs, [1, 2], rungs)
    for d, finf in ((1, 1.60), (2, 1.62)):
        got = s["per_design"][d]["F_inf"]
        assert abs(got - finf) < 4 * s["per_design"][d]["F_inf_sd"] + 1e-3, (d, got)
    assert abs(s["sd_exponent"] + 0.5) < 0.15, s["sd_exponent"]
    print("selftest OK: F_inf recovered for both designs, sd exponent", round(s["sd_exponent"], 3))


if args.selftest:
    selftest(); raise SystemExit(0)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
store = out / "runs.json"
runs = json.loads(store.read_text()) if store.exists() else {}
ck = json.loads(Path(args.checkpoint).read_text())
designs = args.designs or json.loads(Path(args.manifest).read_text())["front"]

if args.analyse:
    s = analyse(runs, designs, args.particles)
    report(s)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    raise SystemExit(0)

# --------------------------------------------------------------------------- #
#  Transport (wks720 only)                                                     #
# --------------------------------------------------------------------------- #
os.environ["OMP_NUM_THREADS"] = str(args.threads)
import numpy as np
import openmc
import core_geometry as cg
import reactor_model as rm
import zoning as zn
from openmc_evaluator import _design_seed

geo, op = rm.Geometry17x17(), rm.Operating()      # C9 evaluator default: 1000 ppm


def one_run(design, particles, seed, case):
    """Mirror OpenMCEvaluator._bol_core_peaking, with the seed and fidelity set here."""
    m = rm.make_core_model(design, op, geo, design_map=zn.evaluator_design_map(design),
                           particles=particles, batches=args.batches, inactive=args.inactive)
    model = m[0] if isinstance(m, tuple) else m
    model.settings.seed = seed
    N, nx = geo.lattice, cg.CORE_MAP_32.shape[0]
    half = nx * N * design.get("pitch", 1.26) / 2.0
    mesh = openmc.RegularMesh()
    mesh.dimension = (nx * N, nx * N)
    mesh.lower_left, mesh.upper_right = (-half, -half), (half, half)
    t = openmc.Tally(name="core_pin_fission")
    t.filters, t.scores = [openmc.MeshFilter(mesh)], ["fission"]
    model.tallies = openmc.Tallies([t])
    case.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sp_path = model.run(cwd=str(case), output=False, threads=args.threads)
    wall = time.time() - t0
    with openmc.StatePoint(sp_path) as sp:
        tal = sp.get_tally(name="core_pin_fission")
        v = tal.get_values(scores=["fission"]).reshape(nx * N, nx * N)
        e = tal.get_values(scores=["fission"], value="std_dev").reshape(nx * N, nx * N)
        k = sp.keff
    f = np.ma.masked_equal(v, 0.0)
    q = f / f.mean()
    i = np.unravel_index(np.ma.argmax(q), q.shape)
    return dict(fdh=float(q.max()), rel_err=float(e[i] / v[i]),
                n_tied=int((v >= v[i] - e[i]).sum()), keff=float(k.nominal_value),
                keff_sd=float(k.std_dev), wall_s=wall, seed=seed)


def design_of(idx):
    r = ck["all_raw"][idx]
    return {k: float(r[k]) for k in KEYS}


if args.estimate:
    d = design_of(designs[0])
    r = one_run(d, args.campaign_particles, _design_seed(d, salt="ladder1"), out / "estimate")
    total = r["wall_s"] * sum(args.particles) / args.campaign_particles * args.seeds * len(designs)
    print(f"design {designs[0]}: {args.campaign_particles} particles in {r['wall_s']:.0f} s, "
          f"F_dH {r['fdh']:.4f}, keff {r['keff']:.5f}")
    print(f"full ladder: {len(designs)} designs x {args.seeds} seeds x {args.particles} "
          f"~ {total/3600:.1f} h (linear in particles)")
    raise SystemExit(0)

for idx in designs:
    d = design_of(idx)
    for p in args.particles:
        for s in range(1, args.seeds + 1):
            key = f"d{idx}_p{p}_s{s}"
            if key in runs:
                continue
            runs[key] = one_run(d, p, _design_seed(d, salt=f"ladder{s}"), out / f"d{idx}" / f"p{p}" / f"s{s}")
            store.write_text(json.dumps(runs, indent=1))
            r = runs[key]
            print(f"d{idx} p{p} s{s}: F {r['fdh']:.4f}  hot err {r['rel_err']*100:.2f}%  "
                  f"k {r['keff']:.5f}  {r['wall_s']:.0f} s", flush=True)

s = analyse(runs, designs, args.particles)
report(s)
(out / "summary.json").write_text(json.dumps(s, indent=1))
