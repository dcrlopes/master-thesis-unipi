#!/usr/bin/env python3
"""
tier1_coefficients.py -- reactivity coefficients of the champion lattice.

Solves the champion assembly at perturbed operating states and reports:

  MTC     moderator temperature coefficient, pcm per K, from mod_T +/- 30 K
          WITH the corresponding water density change at 15.5 MPa
  FTC     fuel temperature (Doppler) coefficient, pcm per K, 900 -> 1200 K
  swing   cold-zero-power to hot reactivity difference, pcm
  boron   worth of the 1000 ppm soluble boron the Operating defaults carry,
          pcm total and pcm per ppm

The boron state exists because Operating.boron_ppm defaults to 1000 and no
campaign code overrides it, so every archived result was computed WITH
soluble boron despite the soluble-boron-free design basis. This script
measures what that assumption is worth.

Assembly level, reflective: leakage feedback is not captured, so the MTC
here is the lattice component only. Water densities are IAPWS-97 values at
15.5 MPa (VERIFY): 550 K 0.767, 580 K 0.712, 610 K 0.641 g/cm3, and
0.9965 at 300 K, 0.1 MPa for the cold state.

Usage:
  python tier1_coefficients.py --checkpoint out_c5/optimization_checkpoint.json \
      --idx 7 --seeds 3 --threads 64 --out tier1

Flags: --idx archive index; --seeds independent transport seeds per state;
--particles/--batches/--inactive override the checkpoint transport block;
--out output directory.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "64")

import numpy as np
import openmc
import reactor_model as rm
from openmc_evaluator import _design_seed

RHO = {550.0: 0.767, 580.0: 0.712, 610.0: 0.641, 300.0: 0.9965}

_orig_make_water = rm.make_water
_STATE_RHO = {"v": None}


def _patched_make_water(boron_ppm, T):
    w = _orig_make_water(boron_ppm, T)
    if _STATE_RHO["v"] is not None:
        w.set_density("g/cm3", _STATE_RHO["v"])
    return w


rm.make_water = _patched_make_water


def solve(design, op, transport, seed):
    model, _fc, _lat = rm.make_assembly_model(
        design, op, rm.Geometry17x17(), bc="reflective", pin_tally=False,
        **transport)
    model.settings.seed = seed
    model.settings.temperature = {"method": "interpolation"}
    sp = model.run(output=False)
    with openmc.StatePoint(sp) as s:
        k = float(s.keff.nominal_value)
    for f in Path(".").glob("statepoint.*.h5"):
        f.unlink()
    for f in ("summary.h5", "tallies.out"):
        Path(f).unlink(missing_ok=True)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--particles", type=int, default=None)
    ap.add_argument("--batches", type=int, default=None)
    ap.add_argument("--inactive", type=int, default=None)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", default="tier1")
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    ck = json.loads(Path(args.checkpoint).read_text())
    dv = ck["design_variables"]
    design = {k: float(ck["all_raw"][args.idx][k]) for k in dv}
    transport = dict(ck.get("meta", {}).get("transport",
                     dict(particles=16000, batches=120, inactive=30)))
    for key in ("particles", "batches", "inactive"):
        v = getattr(args, key)
        if v is not None:
            transport[key] = v
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_seed = _design_seed(design)

    # name: (fuel_T, mod_T, boron_ppm, water_density)
    states = {
        "nominal":  (900.0, 580.0, 1000.0, RHO[580.0]),
        "mod_m30":  (900.0, 550.0, 1000.0, RHO[550.0]),
        "mod_p30":  (900.0, 610.0, 1000.0, RHO[610.0]),
        "fuel_1200": (1200.0, 580.0, 1000.0, RHO[580.0]),
        "cold":     (300.0, 300.0, 1000.0, RHO[300.0]),
        "boron0":   (900.0, 580.0, 0.0, RHO[580.0]),
    }
    res = {}
    cwd = os.getcwd()
    work = out / "runs"
    work.mkdir(exist_ok=True)
    os.chdir(work)
    try:
        for name, (fT, mT, ppm, rho) in states.items():
            _STATE_RHO["v"] = rho
            op = rm.Operating(fuel_T=fT, clad_T=min(600.0, mT + 20.0),
                              mod_T=mT, boron_ppm=ppm)
            ks = [solve(design, op, transport, base_seed + 1000 + i)
                  for i in range(args.seeds)]
            m = float(np.mean(ks))
            sd = float(np.std(ks, ddof=1)) if len(ks) > 1 else float("nan")
            res[name] = dict(k=m, sd=sd, ks=ks, fuel_T=fT, mod_T=mT,
                             boron_ppm=ppm, rho=rho)
            print(f"{name:>10}: k = {m:.5f} +/- {sd:.5f}  "
                  f"(fuel {fT:.0f} K, mod {mT:.0f} K, rho {rho:.4f}, "
                  f"boron {ppm:.0f} ppm)", flush=True)
    finally:
        os.chdir(cwd)
        _STATE_RHO["v"] = None

    def pcm(a, b):
        return (res[a]["k"] - res[b]["k"]) * 1e5

    summary = dict(
        design_idx=args.idx, transport=transport, seeds=args.seeds,
        states=res,
        MTC_pcm_per_K=pcm("mod_p30", "mod_m30") / 60.0,
        FTC_pcm_per_K=pcm("fuel_1200", "nominal") / 300.0,
        swing_cold_minus_hot_pcm=pcm("cold", "nominal"),
        boron_worth_pcm=pcm("boron0", "nominal"),
        boron_worth_pcm_per_ppm=pcm("boron0", "nominal") / 1000.0,
    )
    print(f"\nMTC   = {summary['MTC_pcm_per_K']:+8.2f} pcm/K "
          f"(mod 550 -> 610 K with density)")
    print(f"FTC   = {summary['FTC_pcm_per_K']:+8.2f} pcm/K (fuel 900 -> 1200 K)")
    print(f"swing = {summary['swing_cold_minus_hot_pcm']:+8.0f} pcm "
          f"(cold zero power minus hot)")
    print(f"boron = {summary['boron_worth_pcm']:+8.0f} pcm total, "
          f"{summary['boron_worth_pcm_per_ppm']:+.2f} pcm/ppm")
    (out / f"tier1_idx{args.idx}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nwrote {out}/tier1_idx{args.idx}.json")


if __name__ == "__main__":
    main()
