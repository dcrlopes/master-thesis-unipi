#!/usr/bin/env python3
"""
rank_front_peaking.py -- find the true minimum-F_dH design beyond Monte Carlo
(MC) noise, using a statistically defensible candidate set.

WHY THE PARETO FRONT IS THE WRONG CANDIDATE SET
-----------------------------------------------
Under an ATF (Accident Tolerant Fuel) discharge burnup limit, EVERY design
whose end-of-cycle burnup exceeds the cap is truncated to the SAME cycle
length. In Campaign 2 that is 36 of 44 feasible designs at 75 MWd/kgHM. Once
the cycle-length objective is constant across them, the winner is simply
whichever has the lowest F_dH -- and a design that was DOMINATED in the
uncapped space (shorter cycle, low peaking) becomes fully competitive.

Worse, the stored F_dH values are single-seed draws with a measured
seed-to-seed standard deviation of ~0.018. A design whose stored value sits
up to ~4 sd ABOVE the observed minimum may still hold the true minimum.

So the candidate set is defined by a NOISE MARGIN, not by front membership:

    candidates = feasible designs with  stored F_dH <= min(stored) + k * sd

with k = 4 by default. Restricting to "the front", or to the 3 best, would
silently discard designs that could win.

TWO-STAGE SCREENING
-------------------
Evaluating every candidate at top fidelity is wasteful. Instead:

  STAGE A (screen)  -- all candidates, moderate fidelity, few seeds.
      Eliminate design i when its lower confidence bound exceeds the best
      design's upper bound (Gupta-style subset selection): if
          mean_i - z*SEM_i  >  min_j (mean_j + z*SEM_j)
      then i cannot be the winner and is dropped.

  STAGE B (resolve) -- survivors only, top fidelity, more seeds, full
      pairwise Welch tests with a Bonferroni-corrected threshold.

BIAS, NOT ONLY NOISE
--------------------
F_dH is a MAXIMUM over 264 noisy mesh cells, so it is biased UPWARD, and the
bias grows the FLATTER the true power distribution is -- more near-tied pins
means more chances for an upward fluctuation. The flattest designs, the ones
that should win, are penalised most. Raising fidelity therefore does not only
shrink error bars: it can shift the ranking systematically. `n_tied` (pins
within 1 sigma of the maximum) is reported as the direct fingerprint.

Only BOL (Beginning of Life) transport runs -- NO depletion. Every run is
checkpointed, so an interrupted job resumes.

USAGE
  conda activate openmc-env
  python rank_front_peaking.py --checkpoint <ckpt> --estimate
  nohup python rank_front_peaking.py --checkpoint <ckpt> \
      --threads 64 --out rank_front > rank.log 2>&1 &
"""
import argparse
import json
import statistics as st
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import openmc

import reactor_model as rm

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--prior-sd", type=float, default=0.0182,
                help="seed-to-seed sd of F_dH at campaign fidelity, from "
                     "peaking_noise_test.py (default: measured 0.0182)")
ap.add_argument("--margin-k", type=float, default=4.0,
                help="candidates are within k*prior_sd of the lowest stored "
                     "F_dH. k=4 is conservative; k=3 is tighter and cheaper")
ap.add_argument("--max-candidates", type=int, default=16,
                help="hard cap on the candidate set, lowest stored F_dH first")
ap.add_argument("--screen-particles", type=int, default=16000)
ap.add_argument("--screen-seeds", type=int, default=3)
ap.add_argument("--particles", type=int, default=64000,
                help="top-fidelity particle count for the survivors")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--batches", type=int, default=120)
ap.add_argument("--inactive", type=int, default=30)
ap.add_argument("--z-screen", type=float, default=2.0,
                help="confidence multiplier for elimination; 2.0 is ~95 pct. "
                     "Raise to 2.5-3 to eliminate more cautiously")
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--out", default="rank_front")
ap.add_argument("--estimate", action="store_true",
                help="select candidates, time one run, extrapolate, exit")
args = ap.parse_args()

if args.threads:
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

ck = json.loads(Path(args.checkpoint).read_text())
raw, con, dv = ck["all_raw"], ck.get("constraint_names", []), ck["design_variables"]


def feasible(r, tol=1e-9):
    return all(float(r.get(c, 0.0)) <= tol for c in con)


feas = [r for r in raw if feasible(r)]
lo = min(r["peaking"] for r in feas)
thresh = lo + args.margin_k * args.prior_sd
cands = sorted((r for r in feas if r["peaking"] <= thresh),
               key=lambda r: r["peaking"])[:args.max_candidates]

print("=" * 76)
print(f"feasible designs            : {len(feas)} of {len(raw)}")
print(f"lowest stored F_dH          : {lo:.4f}")
print(f"noise margin                : {args.margin_k} x {args.prior_sd} "
      f"= {args.margin_k * args.prior_sd:.4f}")
print(f"candidate threshold         : F_dH <= {thresh:.4f}")
print(f"CANDIDATES                  : {len(cands)}")
print("=" * 76)
for i, r in enumerate(cands):
    print(f"  cand{i:<2d} stored F_dH={r['peaking']:.4f}  EFPD={r['cycle_length']:7.0f}  "
          f"bu={r['bu_eoc_mwd_kg']:6.2f}  "
          + " ".join(f"{k}={float(r[k]):.3f}" for k in dv))

outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)
store = outdir / "runs.json"
done = json.loads(store.read_text()) if store.exists() else {}
geo, op = rm.Geometry17x17(), rm.Operating()


def one_run(design, seed, particles, case_dir):
    """Mirror _bol_peaking(), but keep the uncertainty information."""
    model, _c, _l = rm.make_assembly_model(
        design, op, geo, bc="reflective",
        particles=particles, batches=args.batches, inactive=args.inactive)
    model.settings.seed = seed      # _settings() hard-codes seed=1: the whole
                                    # campaign used ONE random stream
    N = geo.lattice
    half = N * design.get("pitch", 1.26) / 2.0
    mesh = openmc.RegularMesh()
    mesh.dimension = (N, N)
    mesh.lower_left = (-half, -half)
    mesh.upper_right = (half, half)
    t = openmc.Tally(name="pin_fission")
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ["fission"]
    model.tallies = openmc.Tallies([t])

    t0 = time.time()
    sp_path = model.run(cwd=str(case_dir), output=False)
    wall = time.time() - t0
    with openmc.StatePoint(sp_path) as sp:
        tal = sp.get_tally(name="pin_fission")
        val = tal.get_values(scores=["fission"]).reshape((N, N))
        sdv = tal.get_values(scores=["fission"], value="std_dev").reshape((N, N))
        keff = float(sp.keff.nominal_value)
    fm = np.ma.masked_equal(val, 0.0)
    norm = fm / fm.mean()
    idx = np.unravel_index(np.ma.argmax(norm), norm.shape)
    n_tied = int((np.ma.masked_equal(val, 0.0).filled(0)
                  >= val[idx] - sdv[idx]).sum())
    return dict(fdh=float(norm.max()), rel_err=float(sdv[idx] / val[idx]),
                keff=keff, hot=[int(idx[0]), int(idx[1])],
                n_tied=n_tied, wall_s=wall)


def batch(idxs, particles, nseeds, tag):
    """Run a set of candidates at one fidelity; return {i: (mean, sd, sem)}."""
    res = {}
    for i in idxs:
        design = {k: float(cands[i][k]) for k in dv}
        vals = []
        for s in range(1, nseeds + 1):
            key = f"c{i}_p{particles}_s{s}"
            if key not in done:
                case = outdir / f"cand{i}" / f"p{particles}" / f"seed{s}"
                case.mkdir(parents=True, exist_ok=True)
                done[key] = one_run(design, s, particles, case)
                store.write_text(json.dumps(done, indent=1))
            vals.append(done[key]["fdh"])
        m = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        sem = sd / len(vals) ** 0.5
        res[i] = (m, sd, sem)
        r1 = done[f"c{i}_p{particles}_s1"]
        hots = {tuple(done[f'c{i}_p{particles}_s{s}']["hot"])
                for s in range(1, nseeds + 1)}
        print(f"  [{tag}] cand{i:<2d} mean {m:.4f}  sd {sd:.4f}  sem {sem:.4f}  "
              f"rel.err {r1['rel_err'] * 100:4.2f}%  {r1['n_tied']:2d} tied  "
              f"{len(hots)} hot pins  ({r1['wall_s']:.0f} s)")
    return res


# --------------------------------------------------------------------------- #
if args.estimate:
    d = {k: float(cands[0][k]) for k in dv}
    case = outdir / "estimate"
    case.mkdir(parents=True, exist_ok=True)
    r = one_run(d, 1, args.screen_particles, case)
    per_hist = r["wall_s"] / (args.screen_particles * args.batches)
    hA = (per_hist * len(cands) * args.screen_seeds
          * args.screen_particles * args.batches / 3600)
    hB = per_hist * 4 * args.seeds * args.particles * args.batches / 3600
    print(f"\none run at {args.screen_particles} particles: {r['wall_s']:.0f} s "
          f"(F_dH={r['fdh']:.4f}, {r['n_tied']} pins tied with the max)")
    print(f"Stage A screen : {len(cands)} candidates x {args.screen_seeds} seeds "
          f"@ {args.screen_particles}  ->  ~{hA:.1f} h")
    print(f"Stage B resolve: ~4 survivors x {args.seeds} seeds "
          f"@ {args.particles}  ->  ~{hB:.1f} h")
    print(f"TOTAL ~{hA + hB:.1f} h  (survivor count is a guess; fewer = faster)")
    raise SystemExit(0)

# ---------------------------- STAGE A -------------------------------------- #
print("\n" + "=" * 76)
print(f"STAGE A -- screening {len(cands)} candidates at "
      f"{args.screen_particles} particles x {args.screen_seeds} seeds")
print("=" * 76)
A = batch(range(len(cands)), args.screen_particles, args.screen_seeds, "screen")

z = args.z_screen
best_ub = min(m + z * sem for m, _, sem in A.values())
surv = [i for i, (m, _, sem) in A.items() if m - z * sem <= best_ub]
print("-" * 76)
print(f"elimination bound: a candidate survives if mean - {z}*SEM <= {best_ub:.4f}")
for i, (m, _, sem) in sorted(A.items(), key=lambda kv: kv[1][0]):
    print(f"  cand{i:<2d} {m:.4f} [{m - z * sem:.4f}, {m + z * sem:.4f}]  "
          f"{'SURVIVES' if i in surv else 'eliminated'}")
print(f"survivors: {len(surv)} of {len(cands)}")

# ---------------------------- STAGE B -------------------------------------- #
print("\n" + "=" * 76)
print(f"STAGE B -- resolving {len(surv)} survivors at "
      f"{args.particles} particles x {args.seeds} seeds")
print("=" * 76)
B = batch(surv, args.particles, args.seeds, "final")

order = sorted(B, key=lambda i: B[i][0])
print("-" * 76)
for i in order:
    m, sd, sem = B[i]
    print(f"  cand{i:<2d} stored {cands[i]['peaking']:.4f} -> {m:.4f} "
          f"+/- {1.96 * sem:.4f} (95% CI)   shift {m - cands[i]['peaking']:+.4f}")
print("-" * 76)
print("ranking (best first): " + " < ".join(f"cand{i}" for i in order))

pairs = list(combinations(order, 2))
tcrit = 3.2                      # ~2-sided, few d.o.f., Bonferroni-adjusted
resolved = True
for a, b in pairs:
    ma, _, sa = B[a]
    mb, _, sb = B[b]
    sem = (sa ** 2 + sb ** 2) ** 0.5
    t = abs(ma - mb) / sem if sem else float("inf")
    if not t > tcrit and (a, b) == (order[0], order[1]):
        resolved = False
    print(f"  cand{a} vs cand{b}: d={ma - mb:+.4f} SEM={sem:.4f} |t|={t:.2f} "
          f"-> {'SEPARATED' if t > tcrit else 'not separated'}")

print("-" * 76)
w = order[0]
if resolved:
    print(f"WINNER: cand{w}, separated from the runner-up.")
    print("  " + "  ".join(f"{k}={float(cands[w][k]):.4f}" for k in dv))
    print(f"  stored F_dH {cands[w]['peaking']:.4f} -> resolved {B[w][0]:.4f}")
    if w != 0:
        print("  NOTE: the campaign's apparent winner (cand0) is NOT the true "
              "winner.\n  The single-seed ranking was a noise artifact.")
else:
    print(f"NOT RESOLVED: cand{order[0]} and cand{order[1]} remain tied.\n"
          f"Raise --particles, or report them as statistically equivalent.")

csv = outdir / "ranking.csv"
with open(csv, "w") as fh:
    fh.write("cand,particles,seed,fdh,rel_err,n_tied,keff,wall_s\n")
    for k, v in sorted(done.items()):
        if not k.startswith("c"):
            continue
        c = k[1:].split("_p")[0]
        p = k.split("_p")[1].split("_s")[0]
        s = k.split("_s")[1]
        fh.write(f"{c},{p},{s},{v['fdh']},{v['rel_err']},{v['n_tied']},"
                 f"{v['keff']},{v['wall_s']}\n")
print(f"\nCSV: {csv}\nraw (resumable): {store}")
