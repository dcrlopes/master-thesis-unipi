#!/usr/bin/env python3
"""
make_bg_pin_power_c9.py -- beginning-of-life pin fission maps of the Campaign 9
champion C9-47 for Figures fig:bg-pin-asm and fig:bg-pin-core of the
Theoretical Background chapter.

Two transport solves, both with the campaign builders and the campaign
peaking extraction (mesh tally of the fission score, zero-fission bins masked,
normalised to the mean of the fuelled pins):
  1. the 17x17 assembly under reflective boundaries (make_assembly_model),
  2. the 2D core of 32 assemblies with the frozen enrichment zoning of the
     evaluator (zoning.core_bol_solve, i.e. make_core_model + design map).
Fresh fuel, all rods out, 1000 ppm, as in every campaign solve.

Run in the OpenMC environment (WSL):
  python make_bg_pin_power_c9.py --out pinmaps_c9 [--threads 8]
Writes <out>/pinmaps_c9.npz (asm, core maps, the Gd and guide-tube masks,
k values) and <out>/pinmaps_c9.json (scalars). Plotting is done by
plot_bg_pin_power_c9.py, which needs only numpy and matplotlib.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import openmc

import reactor_model as rm
import zoning as zn

CASE_ID = 47


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--out", default="pinmaps_c9")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--asm", default="100000,150,50")
    ap.add_argument("--core", default="150000,200,80")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(a.threads)

    dv, _cn, raw, _meta = zn.load_archive(a.checkpoint)
    rec = raw[CASE_ID]
    design = {k: float(rec[k]) for k in ("enrich_inner", "enrich_outer", "gd_wt",
                                         "pitch", "refl_thick", "gd_pins")}
    print("C9-47 design:", design)

    op, geo = rm.Operating(), rm.Geometry17x17()
    N = geo.lattice
    pitch = design.get("pitch", 1.26)

    # ---- 1. assembly, reflective --------------------------------------------
    p, b, i = (int(x) for x in a.asm.split(","))
    model, _fc, _lat = rm.make_assembly_model(design, op, geo, bc="reflective",
                                              particles=p, batches=b, inactive=i)
    model.settings.seed = 1
    half = N * pitch / 2.0
    mesh = openmc.RegularMesh()
    mesh.dimension = (N, N)
    mesh.lower_left = (-half, -half)
    mesh.upper_right = (half, half)
    t = openmc.Tally(name="pin_fission")
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ["fission"]
    model.tallies = openmc.Tallies([t])
    sp_path = model.run(cwd=str(out / "asm"), output=False, threads=a.threads)
    with openmc.StatePoint(sp_path) as sp:
        tal = sp.get_tally(name="pin_fission")
        asm = tal.get_values(scores=["fission"]).reshape(N, N)
        asm_rel = tal.get_values(scores=["fission"], value="rel_err").reshape(N, N)
        k_asm, k_asm_sd = float(sp.keff.nominal_value), float(sp.keff.std_dev)
    fm = np.ma.masked_equal(asm, 0.0)
    asm_norm = (fm / fm.mean()).filled(0.0)
    fdh_asm = float(asm_norm.max())
    print(f"assembly k_inf {k_asm:.5f} +- {k_asm_sd:.5f}  F_dh {fdh_asm:.4f}")

    # ---- 2. core, zoned ------------------------------------------------------
    p, b, i = (int(x) for x in a.core.split(","))
    # core_bol_solve does not return the map, so repeat its tally here with
    # the same builder call and the same extraction.
    m = rm.make_core_model(design, op, geo,
                           design_map=zn.evaluator_design_map(design),
                           particles=p, batches=b, inactive=i)
    cmodel = m[0] if isinstance(m, tuple) else m
    cmodel.settings.seed = 1
    rmap = zn.ring_map()
    ny, nx = rmap.shape
    halfc = nx * N * pitch / 2.0
    cmesh = openmc.RegularMesh()
    cmesh.dimension = (nx * N, ny * N)
    cmesh.lower_left = (-halfc, -halfc)
    cmesh.upper_right = (halfc, halfc)
    ct = openmc.Tally(name="core_pin_fission")
    ct.filters = [openmc.MeshFilter(cmesh)]
    ct.scores = ["fission"]
    cmodel.tallies = openmc.Tallies([ct])
    sp_path = cmodel.run(cwd=str(out / "core"), output=False, threads=a.threads)
    with openmc.StatePoint(sp_path) as sp:
        tal = sp.get_tally(name="core_pin_fission")
        core = tal.get_values(scores=["fission"]).reshape(ny * N, nx * N)
        core_rel = tal.get_values(scores=["fission"], value="rel_err").reshape(ny * N, nx * N)
        k_core, k_core_sd = float(sp.keff.nominal_value), float(sp.keff.std_dev)
    fc = np.ma.masked_equal(core, 0.0)
    core_norm = (fc / fc.mean()).filled(0.0)
    fdh_core = float(core_norm.max())
    print(f"core k_eff {k_core:.5f} +- {k_core_sd:.5f}  F_dh {fdh_core:.4f}")

    # lattice masks in lattice (row from top, column from left) indexing
    gd = np.zeros((N, N), bool)
    for (r, c) in rm.gd_pattern(design.get("gd_pins", 12)):
        gd[r, c] = True
    gt = np.zeros((N, N), bool)
    for (r, c) in rm.GUIDE_TUBE_POSITIONS:
        gt[r, c] = True

    np.savez(out / "pinmaps_c9.npz", asm=asm_norm, asm_rel=asm_rel,
             core=core_norm, core_rel=core_rel, gd=gd, gt=gt, rmap=rmap,
             core_map=np.asarray(rmap >= 0))
    scal = dict(case=CASE_ID, design=design, gd_pins_used=rm.snap_gd_pins(design.get("gd_pins", 12)),
                asm_settings=a.asm, core_settings=a.core,
                k_asm=k_asm, k_asm_sd=k_asm_sd, fdh_asm=fdh_asm,
                k_core=k_core, k_core_sd=k_core_sd, fdh_core=fdh_core,
                archive_peaking=rec.get("peaking"), archive_peaking_asm=rec.get("peaking_asm"))
    (out / "pinmaps_c9.json").write_text(json.dumps(scal, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
