#!/usr/bin/env python3
"""
confirm3d.py -- three-dimensional confirmation of Campaign 8 candidates with
the assembly hardware stack, the core barrel, the downcomer and the vessel
wall (hardware3d.py), against the campaign's own two-dimensional solve.

Per design and per rod state (ARO, ARI = RE1 to RE4, RE12 = RE1 + RE2):
  2D      the campaign solve, zoning.core_bol_solve, for closure
  3D-hw   hardware3d.build_model_3d_hw with the same lattice
and from them
  L_ax_hw = k2D / k3D_hw          the axial factor with hardware and barrel
  F_2D, F_3D                      radial peaking, the 3D one axially integrated
  margins                         -rho(k) for the rodded states, in pcm

Resumable: finished solves are read from <out>/runs.json.

USAGE
  python confirm3d.py --selftest
  python confirm3d.py --checkpoint out_c8/optimization_checkpoint.json --designs 47 53 --dry-run
  python confirm3d.py --checkpoint ... --designs 47 --plot          # geometry slices, no transport
  python confirm3d.py --checkpoint ... --designs 47 --smoke         # 5 min wiring test
  setsid nohup python -u confirm3d.py --checkpoint out_c8/optimization_checkpoint.json \\
      --designs 47 23 21 53 31 --states ARO ARI RE12 --seeds 2 --threads 32 \\
      --out confirm3d_c8 > confirm3d_c8.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import hardware3d as hw


def rho_pcm(k):
    return 1.0e5 * (k - 1.0) / k


def seed_for(design, salt):
    key = json.dumps(design, sort_keys=True) + salt
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def design_from(ckpt, idx):
    r = ckpt["all_raw"][idx]
    return {k: float(r[k]) for k in ("enrich_inner", "enrich_outer", "gd_wt",
                                      "pitch", "refl_thick", "gd_pins")}


def selftest():
    print("selftest (no OpenMC):")
    class Geo: fuel_or=0.4058; clad_ir=0.4140; clad_or=0.4750; gt_ir=0.5715; gt_or=0.6121; lattice=17
    spec = hw.HardwareSpec()
    z = hw.elevations(spec)
    order = ["water_below", "bottom_nozzle", "lower_cap", "fuel", "plenum", "upper_cap",
             "upper_gap", "top_nozzle", "water_above"]
    for a, b in zip(order[:-1], order[1:]):
        assert abs(z[a][1] - z[b][0]) < 1e-9, (a, b)
    assert all(g1 <= z["fuel"][1] or g0 >= z["plenum"][0] for g0, g1, _ in z["grids"]), "grid straddles"
    assert sum(1 for g0, g1, _ in z["grids"] if g0 >= z["plenum"][0]) == 1, "exactly one grid in the plenum"
    # benchmark identities from surfaces.py: parked AIC spans plenum+cap+gap exactly,
    # inserted B4C ends at the fuel top, the strap inner box equals twice the guide-tube OR
    assert abs((z["parked_aic"][1] - z["parked_aic"][0]) - 23.176) < 1e-9
    assert abs(z["rod_b4c"][1] - z["fuel"][1]) < 1e-9 and abs(z["rod_b4c"][0] - z["rod_b4c"][1] + 77.48) < 0.01
    assert abs(spec.grid_box_in - 2 * 0.6121) < 1e-9
    vf = hw.volume_fractions(Geo(), 1.26, spec)
    assert all(abs(sum(d.values()) - 1) < 1e-9 for k, d in vf.items() if not k.startswith("_"))
    total = z["water_above"][1] - z["water_below"][0]
    print(f"  stack tiles {z['water_below'][0]:.3f} to {z['water_above'][1]:.3f} cm ({total:.3f} cm), "
          f"4 grids, one in the plenum, fractions sum to one")
    print(f"  benchmark identities hold: parked AIC = plenum+cap+gap = 23.176 cm, B4C 77.48 cm to the fuel top, "
          f"strap box 1.2242 = 2 x gt_or, spring bore fraction {vf['_areas']['spring_frac_of_rod_bore']:.4f}")
    print("selftest OK")
    return 0


def run_solve(model, case: Path, geo, pitch, inactive):
    """Same mesh tally and peaking rule as zoning.core_bol_solve."""
    import numpy as np, openmc
    import zoning as zn
    NL = geo.lattice
    rmap = zn.ring_map(); ny, nx = rmap.shape
    half = nx * NL * pitch / 2.0
    mesh = openmc.RegularMesh(); mesh.dimension = (nx * NL, ny * NL)
    mesh.lower_left = (-half, -half); mesh.upper_right = (half, half)
    t = openmc.Tally(name="core_pin_fission"); t.filters = [openmc.MeshFilter(mesh)]; t.scores = ["fission"]
    model.tallies = openmc.Tallies([t])
    case.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sp_path = model.run(cwd=str(case), output=False)
    wall = time.time() - t0
    with openmc.StatePoint(sp_path) as sp:
        keff = float(sp.keff.nominal_value); sd = float(sp.keff.std_dev)
        v = sp.get_tally(name="core_pin_fission").get_values(scores=["fission"]).reshape(ny * NL, nx * NL)
        H = np.asarray(getattr(sp, "entropy", []), dtype=float)
    f = np.ma.masked_equal(v, 0.0)
    fdh = float((f / f.mean()).max())
    conv = None
    if H.size:
        tail = H[inactive + (len(H) - inactive) // 2:]
        mu, s = float(tail.mean()), float(tail.std(ddof=1))
        Hs = np.convolve(H, np.ones(3) / 3.0, mode="same"); Hs[0], Hs[-1] = H[0], H[-1]
        bad = np.where(~((Hs >= mu - 3 * s) & (Hs <= mu + 3 * s)))[0]
        conv = int(bad[-1]) + 2 if len(bad) else 1
    return dict(keff=keff, sd=sd, fdh=fdh, entropy_conv=conv, wall_s=round(wall, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint"); ap.add_argument("--designs", type=int, nargs="*", default=[])
    ap.add_argument("--states", nargs="*", default=["ARO", "ARI", "RE12"])
    ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--particles", type=int, default=150000); ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--inactive", type=int, default=80,
                    help="raised from 60: the axial source shape needs more inactive batches")
    ap.add_argument("--m-center", type=float, default=0.72); ap.add_argument("--m-periphery", type=float, default=1.15)
    ap.add_argument("--barrel", type=float, default=5.08); ap.add_argument("--vessel-wall", type=float, default=10.0)
    ap.add_argument("--no-vessel-wall", action="store_true")
    ap.add_argument("--water-below", type=float, default=15.0); ap.add_argument("--water-above", type=float, default=15.0)
    ap.add_argument("--refl-steel-vol", type=float, default=None,
                    help="heavy-reflector steel fraction, campaign 0.90 if omitted, benchmark 0.956")
    ap.add_argument("--cr-abs-radius", type=float, default=None,
                    help="absorber radius, campaign 0.4331 if omitted, benchmark B4C 0.4229")
    ap.add_argument("--rod-stack", choices=["benchmark", "full-b4c"], default="benchmark")
    ap.add_argument("--no-parked-rods", action="store_true")
    ap.add_argument("--refl-override", type=float, default=None,
                    help="replace the design's reflector thickness, e.g. to test a 3.7 cm downcomer")
    ap.add_argument("--out", default="confirm3d_c8")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--plot", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    print(f"python : {sys.version.split()[0]}   cwd: {os.getcwd()}")
    spec = hw.HardwareSpec(barrel=a.barrel, vessel_wall=a.vessel_wall, include_vessel_wall=not a.no_vessel_wall,
                           water_below=a.water_below, water_above=a.water_above,
                           refl_steel_vol=a.refl_steel_vol, cr_abs_radius=a.cr_abs_radius,
                           rod_stack=a.rod_stack, model_parked_rods=not a.no_parked_rods)
    if not a.checkpoint or not a.designs:
        print("FAIL: --checkpoint and --designs required"); return 2
    ckpt = json.loads(Path(a.checkpoint).read_text())
    try:
        import openmc, numpy  # noqa
        import reactor_model as rm, zoning as zn
        print(f"openmc : {openmc.__version__}   XS: {os.environ.get('OPENMC_CROSS_SECTIONS')}")
        _ = zn.RE12_POSITIONS
    except Exception as e:
        if a.dry_run:
            class Geo: fuel_or=0.4058; clad_ir=0.4140; clad_or=0.4750; gt_ir=0.5715; gt_or=0.6121; lattice=17
            for idx in a.designs:
                d = design_from(ckpt, idx)
                if a.refl_override is not None: d["refl_thick"] = a.refl_override
                print(f"\n=== design {idx} ===\n" + hw.describe(d, Geo(), spec))
            return 0
        print(f"FAIL: {e}. Activate openmc-env on the campaign8 branch."); return 2
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    geo, op = rm.Geometry17x17(), rm.Operating()
    rmap = zn.ring_map(); nC, nM, nP = zn.ring_counts(rmap)
    m_m = (32 - nC * a.m_center - nP * a.m_periphery) / nM
    rodded = {"ARO": None, "ARI": (set(zn.RE_BANK_POSITIONS), "B4C"), "RE12": (set(zn.RE12_POSITIONS), "B4C")}
    fid = dict(particles=a.particles, batches=a.batches, inactive=a.inactive)
    if a.smoke:
        fid = dict(particles=5000, batches=40, inactive=15); a.seeds = 1; a.states = ["ARO"]
        print("SMOKE: 5000 x 40 (15 inactive), ARO only, one seed")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else {}
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}

    for idx in a.designs:
        d = design_from(ckpt, idx)
        if a.refl_override is not None:
            d["refl_thick"] = a.refl_override
        print(f"\n=== design {idx} ===\n" + hw.describe(d, geo, spec))
        if a.dry_run:
            continue
        zdes = zn.zone_designs(d, a.m_center, m_m, a.m_periphery)
        dmap = zn.design_map_for(rmap, zdes)
        if a.plot:
            model, info = hw.build_model_3d_hw(d, op, geo, design_map=dmap, rodded_map=rodded["ARI"], spec=spec)
            pdir = out / f"d{idx}" / "plots"; pdir.mkdir(parents=True, exist_ok=True)
            w = 2 * info["r_vessel_out"] + 2; hgt = info["z_top"] - info["z_bottom"] + 2
            plots = []
            zmid = 0.5 * (info["z_bottom"] + info["z_top"])
            views = (("xz", (w, hgt), (0.0, 0.0, zmid), "elevation"),
                     ("xy", (w, w), (0.0, 0.0, 0.0), "plan_fuel"),
                     ("xy", (w, w), (0.0, 0.0, 0.5 * (spec.h_active + spec.plenum)), "plan_plenum"))
            for basis, width, origin, name in views:
                pl = openmc.Plot()
                pl.basis, pl.origin, pl.width = basis, origin, width
                pl.pixels = (1800, max(200, int(1800 * width[1] / width[0])))
                pl.color_by = "material"
                # OpenMC runs with cwd = pdir, so the filename must be the bare
                # name. A path relative to the repository root would be resolved
                # INSIDE pdir and the run aborts with "Directory does not exist".
                pl.filename = name
                plots.append(pl)
            model.plots = openmc.Plots(plots)
            model.plot_geometry(cwd=str(pdir))
            made = sorted(q.name for q in pdir.glob("*.png"))
            print(f"  plots -> {pdir}/ : {', '.join(made) if made else 'NONE WRITTEN'}")
            print("  check by eye: three strap bands in the fuel and one in the plenum, the parked")
            print("  AIC band just above the fuel, and fuel / reflector / barrel / downcomer / vessel")
            continue
        res = summary.get(str(idx), {"design": d, "spec": dataclass_dict(spec)})
        for st in a.states:
            for mode in ("2D", "3Dhw"):
                ks, fs = [], []
                for s in range(a.seeds):
                    key = f"{idx}|{st}|{mode}|{s}|{a.refl_override}|{a.refl_steel_vol}|{a.cr_abs_radius}|{a.rod_stack}|{a.no_parked_rods}"
                    if key in cache:
                        rec = cache[key]
                    else:
                        case = out / f"d{idx}" / f"{st}_{mode}_s{s}"
                        seed = seed_for(d, f"{st}{mode}{s}")
                        if mode == "2D":
                            with hw._patched(rm, spec):
                                r = zn.core_bol_solve(d, dmap, op, geo, seed=seed, case=case, rodded_map=rodded[st], **fid)
                            rec = dict(keff=r["keff"], sd=r["keff_sd"], fdh=r["fdh_core"],
                                       entropy_conv=r.get("entropy_conv_batch"), wall_s=r.get("wall_s"))
                        else:
                            model, info = hw.build_model_3d_hw(d, op, geo, design_map=dmap, rodded_map=rodded[st],
                                                               spec=spec, seed=seed, **fid)
                            rec = run_solve(model, case, geo, d["pitch"], fid["inactive"])
                            rec["downcomer_cm"] = info["downcomer"]
                        cache[key] = rec
                        (out / "runs.json").write_text(json.dumps(cache, indent=1))
                    print(f"  d{idx} {st:4s} {mode:4s} seed {s}: k = {rec['keff']:.5f} +/- {rec['sd']:.5f} "
                          f"F = {rec['fdh']:.3f} entropy_conv {rec.get('entropy_conv')} ({rec.get('wall_s')} s)", flush=True)
                    ks.append(rec["keff"]); fs.append(rec["fdh"])
                res[f"{st}_{mode}"] = dict(k=sum(ks) / len(ks), F=sum(fs) / len(fs), n=len(ks))
            k2, k3 = res[f"{st}_2D"]["k"], res[f"{st}_3Dhw"]["k"]
            res[f"{st}_Lax_hw"] = k2 / k3
            res[f"{st}_dk_pcm"] = rho_pcm(k2) - rho_pcm(k3)
            if st != "ARO":
                res[f"{st}_margin2D_pcm"] = -rho_pcm(k2); res[f"{st}_margin3D_pcm"] = -rho_pcm(k3)
        summary[str(idx)] = res
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        print(f"design {idx}: L_ax_hw(ARO) {res['ARO_Lax_hw']:.4f}  F2D {res['ARO_2D']['F']:.3f}  F3D {res['ARO_3Dhw']['F']:.3f}"
              + (f"  ARI margin 2D/3D {res['ARI_margin2D_pcm']:.0f}/{res['ARI_margin3D_pcm']:.0f} pcm" if "ARI" in a.states else ""))
    if not a.dry_run and not a.plot:
        print(f"wrote {out}/summary.json")
    return 0


def dataclass_dict(spec):
    import dataclasses
    return dataclasses.asdict(spec)


if __name__ == "__main__":
    raise SystemExit(main())
