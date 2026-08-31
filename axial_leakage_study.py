#!/usr/bin/env python3
"""axial_leakage_study.py -- measure what the 2D core omits: axial leakage,
on the designs that matter, and express it in the two places the campaigns
use an eigenvalue.

FOR EACH DESIGN
    2D ARO     infinitely tall core, unrodded (the campaign quantity)
    3D ARO     finite active height, axial reflectors, vacuum ends
    2D ALLRE   sixteen regulating CRAs in, infinitely tall
    3D ALLRE   sixteen regulating CRAs in, finite height
each with --seeds independent seeds, same seeds across states so the
differences are paired.

WHAT IS DERIVED
    dk_axial(ARO)   = k_2D - k_3D, the axial leakage worth, unrodded
    dk_axial(ALLRE) = the same with the regulating banks in
    L_ax            = k_2D / k_3D, the axial leakage FACTOR. Route B builds
                      k_target = k_inf / k_eff(2D core); the 3D-consistent
                      target is k_target_3D = k_target * L_ax, which is how
                      the cycle length can be corrected without re-running
                      depletion.
    margin_3D       = subcriticality under ALLRE in the 3D core, in pcm of
                      reactivity: the controllability margin of the real
                      geometry.
    k_2D_equiv      = the 2D eigenvalue that corresponds to a 3D bound,
                      k_2D = k_3D * L_ax, so a 3D-derived k_max_ctrl can be
                      enforced on the 2D evaluator.

OUTPUT
    <out>/axial_<absorber>.json in the rod_bank_worth screen layout (mode
    "screen", states {"ALLRE": {...}}) built from the 3D numbers, so
    derive_kmax.py reads it unchanged and returns the 3D-basis k_max_ctrl.
    Extra keys carry the 2D numbers and the leakage factors.

RUN (wks720, conda env openmc-env, no Docker)
    conda activate openmc-env && python -c "import numpy, openmc; print('env ok')" \\
      && python -u axial_leakage_study.py \\
           --checkpoint out_c6/optimization_checkpoint.json \\
           --idx 31,1,22,69,59,86 --m-center 0.72 --m-periphery 1.15 \\
           --h-active 120 --axial-refl 15 --seeds 2 --threads 32 \\
           --out axial_c6 2>&1 | tee axial_c6.log

Requires apply_core3d.py to have been applied.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--idx", required=True, help="comma list of archive indices")
ap.add_argument("--m-center", type=float, required=True)
ap.add_argument("--m-periphery", type=float, required=True)
ap.add_argument("--absorber", choices=["B4C", "AIC"], default="B4C")
ap.add_argument("--h-active", type=float, default=None,
                help="active height in cm (default: Geometry17x17."
                     "active_height, 120)")
ap.add_argument("--axial-refl", type=float, default=15.0,
                help="borated-water axial reflector above and below, cm "
                     "(default 15). An ASSUMPTION: no open source gives "
                     "the LABGENE axial reflector.")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--particles", type=int, default=150000)
ap.add_argument("--batches", type=int, default=200)
ap.add_argument("--inactive", type=int, default=80,
                help="more than the 2D 60: the axial source shape must "
                     "converge too")
ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--skip-2d", action="store_true",
                help="use the archived 2D k_core instead of re-solving "
                     "(unpaired seeds, cheaper)")
ap.add_argument("--out", default="axial_study")
args = ap.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.threads)
import numpy as np                                      # noqa: E402
import reactor_model as rm                              # noqa: E402
import zoning as zn                                     # noqa: E402
from openmc_evaluator import _design_seed               # noqa: E402

RE = (sorted(zn.RE_BANK_POSITIONS) if hasattr(zn, "RE_BANK_POSITIONS")
      else [(r, c) for r in (1, 2, 3, 4) for c in (1, 2, 3, 4)])
assert len(RE) == 16, RE

ck = json.loads(Path(args.checkpoint).read_text())
dv = ck["design_variables"]
geo, op = rm.Geometry17x17(), rm.Operating()
h_act = args.h_active if args.h_active is not None else geo.active_height
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rmap = zn.ring_map()
nC, nM, nP = zn.ring_counts(rmap)
m_m = (32 - nC * args.m_center - nP * args.m_periphery) / nM
print(f"multipliers C/M/P = {args.m_center:.3f}/{m_m:.4f}/"
      f"{args.m_periphery:.3f}   h_active = {h_act:.1f} cm   "
      f"axial reflector = {args.axial_refl:.1f} cm water each end")


def rho(k):
    return 1e5 * (1.0 - 1.0 / k)


def solve(design, dmap, seed, case, rodded, three_d):
    return zn.core_bol_solve(
        design, dmap, op, geo, particles=args.particles,
        batches=args.batches, inactive=args.inactive, seed=seed, case=case,
        rodded_map=((set(RE), args.absorber) if rodded else None),
        h_active=(h_act if three_d else None),
        axial_refl_cm=(args.axial_refl if three_d else 0.0))


results = dict(mode_note="3D screen for derive_kmax; 2D in extras",
               absorber=args.absorber, h_active_cm=h_act,
               axial_refl_cm=args.axial_refl, margin_pcm=1000.0,
               multipliers=dict(C=args.m_center, M=m_m, P=args.m_periphery),
               states=[])

hdr = (f"{'idx':>4} {'k2D':>8} {'k3D':>8} {'dk_ax':>7} {'L_ax':>7} "
       f"{'k2D_RE':>8} {'k3D_RE':>8} {'dk_axRE':>8} {'marg3D':>7}")
print(hdr)
for idx in [int(x) for x in args.idx.split(",")]:
    d = {k: float(ck["all_raw"][idx][k]) for k in dv}
    zdes = zn.zone_designs(d, args.m_center, m_m, args.m_periphery)
    dmap = zn.design_map_for(rmap, zdes)
    s0 = _design_seed(d, salt="axial")
    acc = {"2D_ARO": [], "3D_ARO": [], "2D_RE": [], "3D_RE": []}
    F3 = []
    for i in range(args.seeds):
        seed = s0 + 7919 * i
        if args.skip_2d:
            acc["2D_ARO"].append(float(ck["all_raw"][idx]["keff_core_bol"]))
        else:
            acc["2D_ARO"].append(solve(d, dmap, seed,
                                       out / f"i{idx}_2D_ARO_s{i}",
                                       False, False)["keff"])
            acc["2D_RE"].append(solve(d, dmap, seed,
                                      out / f"i{idx}_2D_RE_s{i}",
                                      True, False)["keff"])
        r3 = solve(d, dmap, seed, out / f"i{idx}_3D_ARO_s{i}", False, True)
        acc["3D_ARO"].append(r3["keff"])
        F3.append(r3["fdh_core"])
        acc["3D_RE"].append(solve(d, dmap, seed, out / f"i{idx}_3D_RE_s{i}",
                                  True, True)["keff"])
    m = {k: float(np.mean(v)) for k, v in acc.items() if v}
    sd = {k: (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
          for k, v in acc.items() if v}
    k2, k3 = m["2D_ARO"], m["3D_ARO"]
    L = k2 / k3
    dk = 1e5 * (k2 - k3)
    k3re = m["3D_RE"]
    marg3 = -rho(k3re)
    k2re = m.get("2D_RE")
    dkre = 1e5 * (k2re - k3re) if k2re else float("nan")
    print(f"{idx:>4} {k2:8.5f} {k3:8.5f} {dk:7.0f} {L:7.4f} "
          f"{(k2re if k2re else float('nan')):8.5f} {k3re:8.5f} "
          f"{dkre:8.0f} {marg3:7.0f}  "
          f"{'ok' if marg3 >= 1000 else 'NO'}  F3D={float(np.mean(F3)):.3f}",
          flush=True)
    results["states"].append(dict(
        mode="screen", idx=idx, k0=k3, k0_sd=sd["3D_ARO"],
        excess_pcm=rho(k3), F0=float(np.mean(F3)),
        states={"ALLRE": dict(k=k3re, k_sd=sd["3D_RE"],
                              worth_pcm=1e5 * (1.0 / k3re - 1.0 / k3),
                              margin_pcm=marg3, ok=marg3 >= 1000.0)},
        extras=dict(k_2d=k2, k_2d_sd=sd["2D_ARO"], k_2d_allre=k2re,
                    dk_axial_pcm=dk, dk_axial_allre_pcm=dkre,
                    L_ax=L, k_target_factor=L,
                    margin_2d_pcm=(-rho(k2re) if k2re else None))))

path = out / f"axial_{args.absorber}.json"
path.write_text(json.dumps(results, indent=2, default=float))
Ls = [s["extras"]["L_ax"] for s in results["states"]]
print(f"\nwrote {path}")
print(f"axial leakage factor L_ax: min {min(Ls):.4f}  max {max(Ls):.4f}  "
      f"mean {float(np.mean(Ls)):.4f}")
print("next: python derive_kmax.py", path, "--margin 1000   "
      "(3D-basis k_max_ctrl); enforce on the 2D evaluator as "
      "k_max_2D = k_max_ctrl_3D * L_ax")
