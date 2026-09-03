#!/usr/bin/env python3
"""
refine_zoning.py -- Stage 2 of the zoned-loading study: refine the ring
multipliers of SELECTED champions with cheap core Beginning of Life (BOL)
solves only.

WHAT IT DOES
------------
For each requested archive design it scans a small (m_C, m_P) grid, with
the middle multiplier solved from the fissile balance each time, and
reports for every grid point:

  F_zoned      core radial enthalpy-rise hot channel factor (measured)
  k_zoned      core k-effective at BOL (measured, free from the same solve)
  dk_pcm       reactivity change against the (1, 1, 1) uniform anchor,
               which is RUN by this script with the same settings and seed
               policy, so the comparison is internally consistent
  dEFPD_est    estimated cycle-length price in Effective Full Power Days,
               dEFPD = dk [pcm] / |slope| [pcm per MWd/kg] x 1000 / q_spec,
               a linear-reactivity-model estimate. The slope is extracted
               from the champion's OWN uniform depletion history when
               --case-dir is given, else taken from --slope-pcm-per-mwdkg.

The zoning search costs about 2 minutes per grid point, so a 5 x 5 grid is
under one hour per champion. Fully resumable (runs.json cache).

USAGE (after rescore_zoned_core.py has identified the champions)
----------------------------------------------------------------
  python refine_zoning.py --checkpoint out_c5/optimization_checkpoint.json --list

  setsid nohup python -u refine_zoning.py \\
      --checkpoint out_c5/optimization_checkpoint.json \\
      --idx 7 --idx 35 --idx 58 --idx 54 --case-root out_c5 \\
      --threads 64 --out zoned_refine > zoned_refine.log 2>&1 < /dev/null &

Flags
  --checkpoint            Block 2 checkpoint json
  --list                  print the archive (idx, objectives, constraints)
                          and exit, to pick champions
  --idx                   archive index to refine (repeatable, processed
                          sequentially)
  --grid-mc               comma list of centre multipliers
                          (default 1.00,0.95,0.90,0.85,0.80)
  --grid-mp               comma list of periphery multipliers
                          (default 1.000,1.025,1.050,1.075,1.100)
  --case-root             the campaign workdir (e.g. out_c5). The per
                          champion case directory is derived as
                          case_root/case_NNNN, so each --idx gets the
                          reactivity slope from its OWN depletion history
  --case-dir              one explicit case directory (single --idx use,
                          overrides --case-root)
  --slope-pcm-per-mwdkg   manual late-cycle reactivity slope, used when no
                          case directory is available (magnitude, pcm per
                          MWd/kgHM)
  --leu-cap               enrichment screen for flagging (default 19.75)
  --particles/--batches/--inactive   core solve settings
                          (defaults 100000 / 170 / 60, campaign values)
  --threads               OMP_NUM_THREADS for OpenMC
  --out                   output directory (default zoned_refine)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import reactor_model as rm
import zoning as zn

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--list", action="store_true")
ap.add_argument("--idx", type=int, action="append", default=[])
ap.add_argument("--grid-mc", default="1.00,0.95,0.90,0.85,0.80")
ap.add_argument("--grid-mp", default="1.000,1.025,1.050,1.075,1.100")
ap.add_argument("--case-root", default=None)
ap.add_argument("--case-dir", default=None)
ap.add_argument("--slope-pcm-per-mwdkg", type=float, default=None)
ap.add_argument("--leu-cap", type=float, default=zn.LEU_CAP_SCREEN)
ap.add_argument("--particles", type=int, default=100000)
ap.add_argument("--batches", type=int, default=170)
ap.add_argument("--inactive", type=int, default=60)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--out", default="zoned_refine")
args = ap.parse_args()

zn.set_threads(args.threads)
dv, cn, raw, meta = zn.load_archive(args.checkpoint)

if args.list:
    print(f"{'idx':>5} {'EFPD':>8} {'F_core':>7} {'k_inf':>7} {'k_core':>7} "
          f"{'feas':>5}  design")
    for i, r in enumerate(raw):
        print(f"{i:>5} {float(r.get('cycle_length', np.nan)):8.0f} "
              f"{float(r.get('peaking', np.nan)):7.3f} "
              f"{float(r.get('k_bol', np.nan)):7.4f} "
              f"{float(r.get('keff_core_bol', np.nan)):7.4f} "
              f"{'yes' if zn.is_feasible(r, cn) else '  no':>5}  "
              + " ".join(f"{n}={float(r[n]):.4g}" for n in dv))
    raise SystemExit(0)

if not args.idx:
    raise SystemExit("give at least one --idx (use --list to browse the "
                     "archive first).")

outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)
store_p = outdir / "runs.json"
store = json.loads(store_p.read_text()) if store_p.exists() else {}

geo, op = rm.Geometry17x17(), rm.Operating()
rmap = zn.ring_map()
counts = zn.ring_counts(rmap)
q_spec = rm.core_specific_power_w_per_g(op, geo)      # W per gram heavy metal
grid_mc = [float(x) for x in args.grid_mc.split(",")]
grid_mp = [float(x) for x in args.grid_mp.split(",")]

# ---- reactivity slope for the cycle-length price, resolved per champion -- #
def slope_for(idx):
    cd = args.case_dir
    if cd is None and args.case_root:
        cd = str(Path(args.case_root) / f"case_{idx:04d}")
    if cd:
        try:
            bu, k = zn.read_k_history(cd, q_spec)
            s = abs(zn.late_slope_pcm_per_mwdkg(bu, k))
            print(f"slope for idx {idx} from {cd}: {s:.0f} pcm per MWd/kgHM "
                  f"({len(bu)} points to {bu[-1]:.1f} MWd/kg)")
            return s
        except Exception as e:
            print(f"slope extraction failed for idx {idx} ({e}), falling "
                  f"back to --slope-pcm-per-mwdkg")
    if args.slope_pcm_per_mwdkg is None:
        print(f"NOTE idx {idx}: no slope available, dEFPD_est will be "
              f"omitted for this champion.")
    return args.slope_pcm_per_mwdkg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for idx in args.idx:
    r = raw[idx]
    design = zn.design_of(r, dv)
    slope = slope_for(idx)
    tagset = f"{args.particles}x{args.batches}i{args.inactive}"
    print("=" * 78)
    print(f"CHAMPION idx {idx}: "
          + " ".join(f"{n}={design[n]:.4g}" for n in dv))
    print(f"archive uniform: F_core={float(r.get('peaking', np.nan)):.4f} "
          f"k_core={float(r.get('keff_core_bol', np.nan)):.5f} "
          f"EFPD={float(r.get('cycle_length', np.nan)):.0f}")
    print("=" * 78, flush=True)

    rows = []
    # (1.0, 1.0) anchor is guaranteed to be in the scan.
    pts = [(1.0, 1.0)] + [(mc, mp) for mc in grid_mc for mp in grid_mp
                          if not (mc == 1.0 and mp == 1.0)]
    for mc, mp in pts:
        try:
            m_c, m_m, m_p = zn.balanced_multipliers(mc, mp, counts)
        except ValueError as e:
            print(f"  skip m_C={mc} m_P={mp}: {e}")
            continue
        key = f"d{idx}_mc{m_c:.4f}_mp{m_p:.4f}_{tagset}"
        if key not in store:
            dmap = zn.design_map_for(
                rmap, zn.zone_designs(design, m_c, m_m, m_p))
            case = outdir / f"design{idx}" / f"mc{m_c:.3f}_mp{m_p:.3f}"
            store[key] = zn.core_bol_solve(
                design, dmap, op, geo, particles=args.particles,
                batches=args.batches, inactive=args.inactive,
                seed=1, case=case)
            store_p.write_text(json.dumps(store, indent=1))
            rr = store[key]
            print(f"  m_C={m_c:.3f} m_M={m_m:.3f} m_P={m_p:.3f}: "
                  f"F={rr['fdh_core']:.4f} k={rr['keff']:.5f} "
                  f"({rr['wall_s']:.0f}s)", flush=True)
        rr = store[key]
        max_e = zn.max_zoned_enrichment(design, m_p)
        rows.append(dict(idx=idx, m_C=m_c, m_M=m_m, m_P=m_p,
                         F_zoned=rr["fdh_core"], k_zoned=rr["keff"],
                         max_enr=max_e, leu_ok=max_e < args.leu_cap,
                         share_C=rr["ring_shares"][0],
                         share_M=rr["ring_shares"][1],
                         share_P=rr["ring_shares"][2]))

    anchor = next(x for x in rows if x["m_C"] == 1.0 and x["m_P"] == 1.0)
    for x in rows:
        x["dk_pcm"] = (x["k_zoned"] - anchor["k_zoned"]) * 1e5
        x["dF"] = x["F_zoned"] - anchor["F_zoned"]
        x["dEFPD_est"] = (x["dk_pcm"] / slope * 1000.0 / q_spec
                          if slope else float("nan"))

    csv_p = outdir / f"zoning_grid_idx{idx}.csv"
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [x for x in rows if x["leu_ok"]]
    best = min(ok or rows, key=lambda x: x["F_zoned"])
    print("\nBEST GRID POINT (lowest F among LEU-screen-passing points):")
    print(f"  m_C/m_M/m_P = {best['m_C']:.3f}/{best['m_M']:.3f}/"
          f"{best['m_P']:.3f}")
    print(f"  F_zoned = {best['F_zoned']:.4f}  "
          f"(anchor {anchor['F_zoned']:.4f}, dF = {best['dF']:+.4f})")
    print(f"  k_zoned = {best['k_zoned']:.5f}  (dk = {best['dk_pcm']:+.0f} "
          f"pcm)")
    if slope:
        print(f"  estimated cycle price = {best['dEFPD_est']:+.0f} EFPD "
              f"(linear-reactivity estimate, confirm with "
              f"confirm_zoned_champion.py)")
    print(f"  wrote {csv_p}")

    # ---- heatmap figure -------------------------------------------------- #
    mcs = sorted({x["m_C"] for x in rows}, reverse=True)
    mps = sorted({x["m_P"] for x in rows})
    Z = np.full((len(mcs), len(mps)), np.nan)
    for x in rows:
        Z[mcs.index(x["m_C"]), mps.index(x["m_P"])] = x["F_zoned"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(Z, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(mps)), [f"{v:.3f}" for v in mps])
    ax.set_yticks(range(len(mcs)), [f"{v:.2f}" for v in mcs])
    ax.set_xlabel("m_P (periphery multiplier)")
    ax.set_ylabel("m_C (centre multiplier)")
    for x in rows:
        i, j = mcs.index(x["m_C"]), mps.index(x["m_P"])
        ax.text(j, i, f"{x['F_zoned']:.2f}\n{x['dk_pcm']:+.0f}",
                ha="center", va="center", fontsize=7,
                color="w" if x["F_zoned"] > np.nanmean(Z) else "k")
    ax.set_title(f"idx {idx}: zoned core F (top) and dk [pcm] (bottom)")
    fig.colorbar(im, label=r"$F_{\Delta H}$ core, zoned")
    fig.tight_layout()
    fig_p = outdir / f"fig_zoning_grid_idx{idx}.png"
    fig.savefig(fig_p, dpi=200)
    print(f"  figure {fig_p}\n")
