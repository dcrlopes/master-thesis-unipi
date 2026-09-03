#!/usr/bin/env python3
"""
rescore_zoned_core.py -- Stage T (transfer test) of the zoned-loading study.

THE QUESTION IT ANSWERS
-----------------------
Campaigns 1 to 5 optimise under UNIFORM loading (32 identical assemblies).
Zoned loading changes the assembly-power tilt, so the Pareto ranking found
under uniform loading might not survive. This script measures exactly that,
with the same discipline the Campaign 3 core rescore applied one level
down: every archive design is re-solved ONCE at core Beginning of Life
(BOL) under ONE fixed three-ring enrichment map, and the uniform-vs-zoned
rank agreement is quantified (Spearman rho overall and inside the best
designs, plus the correlation of rank shifts with refl_thick and pitch,
where the bias is expected to live).

Cost: one ~2 minute core solve per archive design at campaign settings.
Fully resumable (runs.json cache): safe against wks720 reboots.

DEFAULT MAP
-----------
m_C = 0.85, m_P = 1.075, m_M solved from the fissile balance (= 0.95 for
the 4/12/16 ring counts), preserving the core-average enrichment exactly.

USAGE (after Campaign 5 Block 2 has written its checkpoint)
-----------------------------------------------------------
  conda activate openmc-env
  setsid nohup python -u rescore_zoned_core.py \\
      --checkpoint openmc_runs_c5/out/optimization_checkpoint.json \\
      --threads 64 --out zoned_rescore > zoned_rescore.log 2>&1 < /dev/null &

  setsid    detach from the terminal session (survives ssh disconnect)
  nohup     ignore the hangup signal for the same reason
  -u        unbuffered python output, so the log updates live
  &         run in the background

Flags
  --checkpoint      path to the Block 2 optimization_checkpoint.json
  --m-center        centre-ring enrichment multiplier (default 0.85)
  --m-periphery     periphery multiplier (default 1.075, middle is solved)
  --seeds           independent transport seeds per design (default 1)
  --particles       particles per batch of the core solve (default 100000,
                    the campaign core setting)
  --batches         total batches (default 170, campaign setting)
  --inactive        discarded source-convergence batches (default 60)
  --threads         OMP_NUM_THREADS for OpenMC (default: environment)
  --feasible-only   rescore only strictly feasible designs (default: ALL,
                    because Campaign 4 taught that the feasible set can be
                    empty while the near-feasible designs are the story)
  --top             size of the "best designs" subset for the top-rank
                    Spearman (default 15)
  --out             output directory (default zoned_rescore)
  --analyze-only    skip solving, rebuild statistics and figure from cache
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import reactor_model as rm
import zoning as zn

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--m-center", type=float, default=0.85)
ap.add_argument("--m-periphery", type=float, default=1.075)
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--particles", type=int, default=100000)
ap.add_argument("--batches", type=int, default=170)
ap.add_argument("--inactive", type=int, default=60)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--feasible-only", action="store_true")
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--out", default="zoned_rescore")
ap.add_argument("--analyze-only", action="store_true")
args = ap.parse_args()

zn.set_threads(args.threads)
outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)
store_p = outdir / "runs.json"
store = json.loads(store_p.read_text()) if store_p.exists() else {}

dv, cn, raw, meta = zn.load_archive(args.checkpoint)
sel = [(i, r) for i, r in enumerate(raw)
       if (not args.feasible_only) or zn.is_feasible(r, cn)]

rmap = zn.ring_map()
counts = zn.ring_counts(rmap)
m_c, m_m, m_p = zn.balanced_multipliers(args.m_center, args.m_periphery,
                                        counts)
geo, op = rm.Geometry17x17(), rm.Operating()
tag = (f"mc{m_c:.3f}_mp{m_p:.3f}_"
       f"{args.particles}x{args.batches}i{args.inactive}")

print("=" * 78)
print(f"ZONED TRANSFER TEST  |  rings C/M/P = {counts}  "
      f"multipliers = {m_c:.4f} / {m_m:.4f} / {m_p:.4f} "
      f"(core-average preserved)")
sel_label = ("feasible only" if args.feasible_only
             else "all designs, the Campaign 4 lesson")
print(f"{len(sel)} designs of {len(raw)} in the archive "
      f"({sel_label})  |  tag {tag}")
print("=" * 78, flush=True)

# --------------------------------------------------------------------------- #
# Solve loop (cached, resume-safe)                                            #
# --------------------------------------------------------------------------- #
rows = []
for idx, r in sel:
    design = zn.design_of(r, dv)
    zdes = zn.zone_designs(design, m_c, m_m, m_p)
    dmap = zn.design_map_for(rmap, zdes)
    max_e = zn.max_zoned_enrichment(design, m_p)

    runs = []
    for s in range(1, args.seeds + 1):
        key = f"d{idx}_s{s}_{tag}"
        if key not in store and not args.analyze_only:
            case = outdir / f"design{idx}" / f"seed{s}"
            store[key] = zn.core_bol_solve(
                design, dmap, op, geo, particles=args.particles,
                batches=args.batches, inactive=args.inactive,
                seed=s, case=case)
            store_p.write_text(json.dumps(store, indent=1))
            rr = store[key]
            warn = ""
            if rr["entropy_conv_batch"] and \
                    rr["entropy_conv_batch"] > args.inactive:
                warn = (f"  !! entropy converged at batch "
                        f"{rr['entropy_conv_batch']} > inactive "
                        f"{args.inactive}")
            print(f"  d{idx:>3} s{s}: F_zoned={rr['fdh_core']:.4f} "
                  f"k_zoned={rr['keff']:.5f} "
                  f"shares C/M/P={rr['ring_shares'][0]:.3f}/"
                  f"{rr['ring_shares'][1]:.3f}/{rr['ring_shares'][2]:.3f} "
                  f"({rr['wall_s']:.0f}s){warn}", flush=True)
        if key in store:
            runs.append(store[key])
    if not runs:
        continue

    f_z = float(np.mean([x["fdh_core"] for x in runs]))
    k_z = float(np.mean([x["keff"] for x in runs]))
    sh = np.mean([x["ring_shares"] for x in runs], axis=0)
    f_u = float(r.get("peaking", np.nan))
    k_u = r.get("keff_core_bol", None)
    k_u = float(k_u) if k_u is not None else np.nan
    rows.append(dict(
        idx=idx, **{n: design[n] for n in dv},
        F_asm=float(r.get("peaking_asm", np.nan)),
        F_uniform=f_u, F_zoned=f_z, dF=f_z - f_u,
        k_core_uniform=k_u, k_core_zoned=k_z,
        dk_pcm=(k_z - k_u) * 1e5 if np.isfinite(k_u) else np.nan,
        share_C=float(sh[0]), share_M=float(sh[1]), share_P=float(sh[2]),
        max_enr_zoned=max_e,
        leu_screen_ok=max_e < zn.LEU_CAP_SCREEN,
        leu20_ok=max_e < zn.LEU_CAP_PHYSICAL,
        seeds=len(runs)))

if not rows:
    raise SystemExit("nothing rescored yet and --analyze-only given, or the "
                     "archive selection is empty.")

csv_p = outdir / "zoned_rescore.csv"
with open(csv_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {csv_p} ({len(rows)} designs)")

# --------------------------------------------------------------------------- #
# Transfer statistics                                                         #
# --------------------------------------------------------------------------- #
Fu = [x["F_uniform"] for x in rows]
Fz = [x["F_zoned"] for x in rows]
rho_all = zn.spearman(Fu, Fz)
order_u = sorted(range(len(rows)), key=lambda i: Fu[i])
top = order_u[:min(args.top, len(rows))]
rho_top = zn.spearman([Fu[i] for i in top], [Fz[i] for i in top])

rk_u, rk_z = zn._ranks(Fu), zn._ranks(Fz)
shift = [rk_z[i] - rk_u[i] for i in range(len(rows))]
r_refl = zn.pearson(shift, [x["refl_thick"] for x in rows])
r_pitch = zn.pearson(shift, [x["pitch"] for x in rows])
dF = [x["dF"] for x in rows]

summary = dict(
    n=len(rows), multipliers=dict(C=m_c, M=m_m, P=m_p),
    spearman_all=rho_all, spearman_top=rho_top, top_size=len(top),
    dF_mean=float(np.mean(dF)), dF_sd=float(np.std(dF, ddof=1)),
    dk_pcm_mean=float(np.nanmean([x["dk_pcm"] for x in rows])),
    rankshift_vs_refl_thick=r_refl, rankshift_vs_pitch=r_pitch,
    n_under_2p0_uniform=int(sum(1 for v in Fu if v <= 2.0)),
    n_under_2p0_zoned=int(sum(1 for v in Fz if v <= 2.0)),
    n_leu_screen_violations=int(sum(1 for x in rows
                                    if not x["leu_screen_ok"])),
    noise_note="per-design sigma_F is about 0.018 at these transport "
               "settings (Campaign 3 measurement): treat rank swaps of "
               "designs closer than about 0.04 in F as ties.")
(outdir / "transfer_summary.json").write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 78)
print("TRANSFER STATISTICS (uniform -> zoned)")
print("=" * 78)
print(f"Spearman rho, all {len(rows)} designs        : {rho_all:+.3f}")
print(f"Spearman rho, best {len(top)} by uniform F   : {rho_top:+.3f}")
print(f"mean dF (zoned - uniform)                    : "
      f"{summary['dF_mean']:+.3f} +/- {summary['dF_sd']:.3f}")
print(f"mean dk (zoned - uniform)                    : "
      f"{summary['dk_pcm_mean']:+.0f} pcm")
print(f"designs with F <= 2.0    uniform -> zoned    : "
      f"{summary['n_under_2p0_uniform']} -> {summary['n_under_2p0_zoned']}")
print(f"rank shift vs refl_thick / pitch (Pearson)   : "
      f"{r_refl:+.3f} / {r_pitch:+.3f}")
print(f"LEU screen (19.75 wt%) violations under map  : "
      f"{summary['n_leu_screen_violations']}")
print("\nTop of the front, uniform rank -> zoned rank:")
print(f"{'idx':>5} {'F_uni':>7} {'F_zon':>7} {'rk_u':>5} {'rk_z':>5} "
      f"{'refl':>6} {'pitch':>6}")
for i in top[:10]:
    x = rows[i]
    print(f"{x['idx']:>5} {x['F_uniform']:7.3f} {x['F_zoned']:7.3f} "
          f"{rk_u[i]:>5} {rk_z[i]:>5} {x['refl_thick']:6.2f} "
          f"{x['pitch']:6.3f}")
print("\nDECISION GUIDE (thresholds are guidance, argue them in the text): "
      "rho_top >= 0.8 with the best 3 to 5 designs preserved supports "
      "treating the uniform campaign as a validated proxy and applying "
      "zoning at candidate selection. A scrambled top is itself a thesis "
      "finding: the loading pattern must join the design space (scoped as "
      "future work).")

# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
refl = [x["refl_thick"] for x in rows]
sc = a1.scatter(Fu, Fz, c=refl, cmap="viridis", s=28)
lo = min(min(Fu), min(Fz)) - 0.05
hi = max(max(Fu), max(Fz)) + 0.05
a1.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change")
a1.axvline(2.0, color="crimson", ls=":", lw=1)
a1.axhline(2.0, color="crimson", ls=":", lw=1)
a1.set_xlabel(r"$F_{\Delta H}$ core, uniform loading")
a1.set_ylabel(r"$F_{\Delta H}$ core, zoned "
              f"({m_c:.2f}/{m_m:.2f}/{m_p:.2f})")
a1.legend(fontsize=8)
fig.colorbar(sc, ax=a1, label="refl_thick [cm]")

a2.scatter(rk_u, rk_z, c=refl, cmap="viridis", s=28)
a2.plot([0, len(rows)], [0, len(rows)], "k--", lw=1)
a2.set_xlabel("rank, uniform")
a2.set_ylabel("rank, zoned")
a2.set_title(f"Spearman: all {rho_all:+.3f}, top {len(top)} {rho_top:+.3f}",
             fontsize=10)
fig.suptitle("Uniform-to-zoned transfer of the core radial peaking")
fig.tight_layout()
fig_p = outdir / "fig_zoned_transfer.png"
fig.savefig(fig_p, dpi=200)
print(f"\nfigure  : {fig_p}")
print(f"summary : {outdir / 'transfer_summary.json'}")
