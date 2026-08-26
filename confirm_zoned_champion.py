#!/usr/bin/env python3
"""
confirm_zoned_champion.py -- Stage 3 of the zoned-loading study: put a real
cycle length on the zoned champion, closing the loop on the Stage 2
linear-reactivity estimate.

METHOD (and its stated assumptions)
-----------------------------------
1. The three ring variants (centre, middle, periphery enrichment scalings
   of the champion) are each depleted with the UNMODIFIED campaign
   machinery, OpenMCEvaluator._cycle_length: same Route B k_target
   interpolation, same adaptive burnup schedule, same transport settings
   (taken from the checkpoint meta unless overridden). Cost: about three
   times one campaign depletion, read the expectation printed at start.
2. Several independent zoned core Beginning of Life (BOL) solves (default
   eight seeds, about two minutes each) give the fission-power share s_z of
   each ring, seed-averaged, plus a mean, standard deviation and 95 percent
   Student confidence interval on the zoned peaking and the zoned core
   k-effective, with a PASS, INCONCLUSIVE or FAIL verdict against the
   peaking limit in the same convention as the idx 54 confirmation.
3. The zoned core reactivity trajectory is estimated as the power-weighted
   mix of the ring k-infinity histories, each ring burning at its relative
   specific power p_z = s_z / (n_z / 32):

       K(B) = sum_z s_z k_z(p_z B)

     K(B)  estimated core-equivalent k-infinity at core-average burnup B
     s_z   BOL fission-power share of ring z (dimensionless)
     k_z   ring-z assembly k-infinity history versus its own burnup
     p_z   relative specific power of ring z (dimensionless)
     B     core-average burnup [MWd/kgHM]

   End of cycle is the burnup where K(B) falls to the champion's own
   Route B k_target, and EFPD (Effective Full Power Days) follows from
   EFPD = B x 1000 / q_spec with q_spec the specific power [W/gHM].

   STATED ASSUMPTIONS, to be written next to the result: the power shares
   are frozen at their BOL values (gadolinia burnout and depletion will
   drift them), and ring k histories are parameterised by burnup alone.
   The gold standard, a full-core depletion, is future work.

USAGE
-----
  setsid nohup python -u confirm_zoned_champion.py \\
      --checkpoint openmc_runs_c5/out/optimization_checkpoint.json \\
      --idx 44 --m-center 0.85 --m-periphery 1.075 \\
      --ktarget-table ktarget_table.json \\
      --threads 64 --out zoned_confirm > zoned_confirm.log 2>&1 < /dev/null &

Flags
  --checkpoint      Block 2 checkpoint json (designs, meta, uniform EFPD)
  --idx             archive index of the champion
  --m-center        centre multiplier from the Stage 2 winner
  --m-periphery     periphery multiplier from the Stage 2 winner
  --ktarget-table   Route B k_target table json (default ktarget_table.json),
                    or --ktarget FLOAT for a frozen Route A value
  --particles/--batches/--inactive
                    depletion transport override (default: checkpoint meta)
  --core-particles/--core-batches/--core-inactive
                    zoned core BOL solve settings (defaults 100000/170/60)
  --core-seeds      independent seeds for the zoned core solve (default 8,
                    about 2 minutes each). Gives the interval and verdict
                    on the zoned peaking, and seed-averaged ring shares
  --shares-json     optional path to a cached core_bol_solve result dict to
                    skip the BOL solve (e.g. from zoned_refine runs.json),
                    single solve, no interval
  --threads         OMP_NUM_THREADS
  --out             working directory (default zoned_confirm)

The ring depletions are resumable: an existing finished ring (marker file
ring_summary.json in its case directory) is reused, so a wks720 reboot
costs only the interrupted ring.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import reactor_model as rm
import zoning as zn

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--idx", type=int, required=True)
ap.add_argument("--m-center", type=float, required=True)
ap.add_argument("--m-periphery", type=float, required=True)
ap.add_argument("--ktarget-table", default="ktarget_table.json")
ap.add_argument("--ktarget", type=float, default=None)
ap.add_argument("--particles", type=int, default=None)
ap.add_argument("--batches", type=int, default=None)
ap.add_argument("--inactive", type=int, default=None)
ap.add_argument("--core-particles", type=int, default=100000)
ap.add_argument("--core-batches", type=int, default=170)
ap.add_argument("--core-inactive", type=int, default=60)
ap.add_argument("--core-seeds", type=int, default=8)
ap.add_argument("--shares-json", default=None)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--out", default="zoned_confirm")
args = ap.parse_args()

zn.set_threads(args.threads)
outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)

dv, cn, raw, meta = zn.load_archive(args.checkpoint)
r = raw[args.idx]
design = zn.design_of(r, dv)
efpd_uniform = float(r.get("cycle_length", np.nan))
t_dep_uniform = float(r.get("t_deplete_s", np.nan))

# transport defaults come from the campaign meta so the ring depletions are
# noise-compatible with the archive value they are compared against
transport = dict(meta.get("transport",
                          dict(particles=20000, batches=80, inactive=20)))
for k in ("particles", "batches", "inactive"):
    v = getattr(args, k)
    if v is not None:
        transport[k] = int(v)

rmap = zn.ring_map()
counts = zn.ring_counts(rmap)
m_c, m_m, m_p = zn.balanced_multipliers(args.m_center, args.m_periphery,
                                        counts)
zdes = zn.zone_designs(design, m_c, m_m, m_p)

print("=" * 78)
print(f"STAGE 3 CONFIRMATION, champion idx {args.idx}")
print("  " + " ".join(f"{n}={design[n]:.4g}" for n in dv))
print(f"  multipliers C/M/P = {m_c:.4f}/{m_m:.4f}/{m_p:.4f} "
      f"(counts {counts}, core average preserved)")
print(f"  transport {transport}, uniform archive EFPD = "
      f"{efpd_uniform:.0f}")
if np.isfinite(t_dep_uniform):
    print(f"  EXPECTED WALL TIME: about 3 x {t_dep_uniform/60:.0f} min = "
          f"{3*t_dep_uniform/3600:.1f} h for the ring depletions, plus one "
          f"core solve")
print("=" * 78, flush=True)

# --------------------------------------------------------------------------- #
# The evaluator, exactly as the campaign builds it                            #
# --------------------------------------------------------------------------- #
from reactor_optimization import example_reactor_problem
from openmc_evaluator import OpenMCEvaluator

spec = example_reactor_problem()
k_target_arg = args.ktarget if args.ktarget is not None else \
    args.ktarget_table
ev = OpenMCEvaluator(spec, k_target=k_target_arg, transport=transport,
                     workdir=str(outdir),
                     core_particles=args.core_particles,
                     core_batches=args.core_batches,
                     core_inactive=args.core_inactive)
k_target = ev._k_target_for(design)
q_spec = ev.spec_power
print(f"Route target for this design: k_target = {k_target:.5f}, "
      f"q_spec = {q_spec:.2f} W/gHM\n", flush=True)

# --------------------------------------------------------------------------- #
# Ring depletions (resumable through per-ring marker files)                   #
# --------------------------------------------------------------------------- #
ring = {}
for name in zn.RING_NAMES:
    case = outdir / f"ring_{name}"
    marker = case / "ring_summary.json"
    if marker.exists():
        ring[name] = json.loads(marker.read_text())
        print(f"ring {name}: reusing finished depletion "
              f"(EFPD {ring[name]['efpd']:.0f})", flush=True)
        continue
    case.mkdir(parents=True, exist_ok=True)
    print(f"ring {name}: depleting variant "
          f"(e_in {zdes[name]['enrich_inner']:.3f}, "
          f"e_out {zdes[name]['enrich_outer']:.3f}) ...", flush=True)
    efpd, k_bol, kt_used, censored, bu_eoc, n_solves = \
        ev._cycle_length(zdes[name], case)
    ring[name] = dict(efpd=float(efpd), k_bol=float(k_bol),
                      k_target_used=float(kt_used),
                      censored=bool(censored), bu_eoc=float(bu_eoc),
                      n_solves=int(n_solves), mult=zdes[name]["zone_mult"])
    marker.write_text(json.dumps(ring[name], indent=2))
    print(f"ring {name}: EFPD(as-if-whole-core) = {efpd:.0f} "
          f"k_bol = {k_bol:.4f} censored = {censored}", flush=True)

hist = {name: zn.read_k_history(outdir / f"ring_{name}", q_spec)
        for name in zn.RING_NAMES}

# --------------------------------------------------------------------------- #
# Ring power shares from the zoned core BOL solve                             #
# --------------------------------------------------------------------------- #
if args.shares_json:
    core = json.loads(Path(args.shares_json).read_text())
    fdh_list, k_list = [core["fdh_core"]], [core["keff"]]
    print(f"shares from {args.shares_json} (single solve, no interval)")
else:
    dmap = zn.design_map_for(rmap, zdes)
    seed_p = outdir / "core_zoned_seeds.json"
    seeds = json.loads(seed_p.read_text()) if seed_p.exists() else {}
    for nn in range(1, args.core_seeds + 1):
        key = f"s{nn}"
        if key not in seeds:
            seeds[key] = zn.core_bol_solve(
                design, dmap, rm.Operating(), rm.Geometry17x17(),
                particles=args.core_particles, batches=args.core_batches,
                inactive=args.core_inactive, seed=nn,
                case=outdir / "core_bol_zoned" / f"seed{nn}")
            seed_p.write_text(json.dumps(seeds, indent=1))
        rr = seeds[key]
        print(f"  core seed {nn}: F={rr['fdh_core']:.4f} "
              f"k={rr['keff']:.5f} ({rr['wall_s']:.0f}s)", flush=True)
    ns = args.core_seeds
    fdh_list = [seeds[f"s{n}"]["fdh_core"] for n in range(1, ns + 1)]
    k_list = [seeds[f"s{n}"]["keff"] for n in range(1, ns + 1)]
    sh_mean = [sum(seeds[f"s{n}"]["ring_shares"][z]
                   for n in range(1, ns + 1)) / ns for z in range(3)]
    core = dict(fdh_core=sum(fdh_list) / ns, keff=sum(k_list) / ns,
                ring_shares=sh_mean)

fm, fsd, fsem, fhalf = zn.t_ci(fdh_list)
km_c, ksd_c, _, khalf_c = zn.t_ci(k_list)
verdict = "single solve, no verdict"
if len(fdh_list) > 1:
    verdict = ("PASS" if fm + fhalf <= 2.0 else
               "INCONCLUSIVE (CI straddles the limit)"
               if fm - fhalf <= 2.0 else "FAIL")
    print(f"\nzoned core F: mean {fm:.4f} sd {fsd:.4f} "
          f"95% CI [{fm - fhalf:.4f}, {fm + fhalf:.4f}]  "
          f"->  g_peak (2.0): {verdict}")
    print(f"zoned core k: mean {km_c:.5f} sd {ksd_c:.5f}")
s = core["ring_shares"]
n_tot = sum(counts)
p = [s[z] / (counts[z] / n_tot) for z in range(3)]
print(f"zoned core BOL: F = {core['fdh_core']:.4f} "
      f"k = {core['keff']:.5f}")
print(f"shares  C/M/P = {s[0]:.4f}/{s[1]:.4f}/{s[2]:.4f}")
print(f"rel. power p_z = {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}\n")

# --------------------------------------------------------------------------- #
# Power-weighted combination and the zoned end of cycle                       #
# --------------------------------------------------------------------------- #
def K(B):
    tot = 0.0
    for z, name in enumerate(zn.RING_NAMES):
        bu, k = hist[name]
        tot += s[z] * float(np.interp(p[z] * B, bu, k))
    return tot


b_max = min(hist[name][0][-1] / p[z]
            for z, name in enumerate(zn.RING_NAMES))
censored = any(ring[n]["censored"] for n in zn.RING_NAMES)
grid = np.linspace(0.0, b_max, 2000)
Kv = np.array([K(b) for b in grid])
below = np.where(Kv <= k_target)[0]
if len(below) == 0:
    B_eoc = b_max
    censored = True
    note = ("K(B) never reached k_target inside the depleted range: the "
            "zoned EFPD below is a LOWER BOUND (extend max_burnup or the "
            "ring histories).")
else:
    i = below[0]
    if i == 0:
        B_eoc, note = 0.0, "K(B) starts below k_target: check inputs."
    else:
        f = (k_target - Kv[i - 1]) / (Kv[i] - Kv[i - 1])
        B_eoc = float(grid[i - 1] + f * (grid[i] - grid[i - 1]))
        note = ""
efpd_zoned = B_eoc * 1000.0 / q_spec

out = dict(idx=args.idx, design=design,
           multipliers=dict(C=m_c, M=m_m, P=m_p),
           ring=ring, shares=dict(zip(zn.RING_NAMES, s)),
           rel_power=dict(zip(zn.RING_NAMES, p)),
           F_zoned_bol=core["fdh_core"], k_zoned_bol=core["keff"],
           F_zoned_sd=fsd, F_zoned_ci95_half=fhalf,
           n_core_seeds=len(fdh_list), verdict_peak=verdict,
           k_target=k_target, B_eoc_mwd_kg=B_eoc,
           efpd_zoned=efpd_zoned, efpd_uniform=efpd_uniform,
           dEFPD=efpd_zoned - efpd_uniform, censored=censored, note=note,
           assumptions="BOL-frozen power shares, burnup-parameterised ring "
                       "k histories, power-weighted k mix against the "
                       "Route B target")
(outdir / f"confirm_idx{args.idx}.json").write_text(
    json.dumps(out, indent=2))

print("=" * 78)
print("ZONED CYCLE LENGTH (power-weighted linear-reactivity combination)")
print("=" * 78)
print(f"  B_eoc = {B_eoc:.2f} MWd/kgHM  ->  EFPD_zoned = {efpd_zoned:.0f}"
      f"{'  (CENSORED lower bound)' if censored else ''}")
print(f"  EFPD_uniform (archive) = {efpd_uniform:.0f}   "
      f"dEFPD = {efpd_zoned - efpd_uniform:+.0f}")
print(f"  F_zoned(BOL) = {core['fdh_core']:.4f}   "
      f"k_zoned(BOL) = {core['keff']:.5f}")
if note:
    print(f"  NOTE: {note}")
print(f"  summary: {outdir / f'confirm_idx{args.idx}.json'}")

# --------------------------------------------------------------------------- #
# Figure: ring histories, the mix, the target, the end of cycle               #
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 4.8))
for z, name in enumerate(zn.RING_NAMES):
    bu, k = hist[name]
    ax.plot(bu / p[z], k, lw=1.1, alpha=0.7,
            label=f"ring {name} (m={ring[name]['mult']:.3f}, "
                  f"p={p[z]:.2f}), vs core-average B")
ax.plot(grid, Kv, "k-", lw=2.0, label="power-weighted mix K(B)")
ax.axhline(k_target, color="crimson", ls="--", lw=1.2,
           label=f"k_target = {k_target:.4f}")
if not censored:
    ax.plot(B_eoc, k_target, "r*", ms=14,
            label=f"EOC: {B_eoc:.1f} MWd/kg = {efpd_zoned:.0f} EFPD")
ax.set_xlabel("core-average burnup B [MWd/kgHM]")
ax.set_ylabel("k-infinity")
ax.set_title(f"idx {args.idx}, zoned "
             f"{m_c:.2f}/{m_m:.2f}/{m_p:.2f}")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig_p = outdir / f"fig_zoned_confirm_idx{args.idx}.png"
fig.savefig(fig_p, dpi=200)
print(f"  figure : {fig_p}")
