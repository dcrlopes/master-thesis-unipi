#!/usr/bin/env python3
"""
apply_zoned_core.py -- add PER-POSITION design overrides to
reactor_model.make_core_model, enabling zoned core loadings.

WHAT IT CHANGES (one file, four anchored edits)
-----------------------------------------------
make_core_model gains an optional argument

    design_map : dict | None
        {(row, col): design_dict} overrides for individual fuel positions of
        the core map. Each DISTINCT override builds its own materials and
        assembly universe THROUGH THE SAME builders as the base assembly
        (build_materials and build_assembly_universe), so the pin layout, the
        gadolinia pattern and the zone-enrichment derate of the gadolinia
        rods stay single-sourced. The lattice pitch is ALWAYS the base
        design's pitch: zoning changes enrichments, never geometry.

With design_map=None (the default) the produced model is BIT-IDENTICAL to
the unpatched builder: same objects in the same order, so campaign
reproducibility and seeds are untouched.

USAGE (on wks720, inside the repo, conda env openmc-env)
--------------------------------------------------------
    python apply_zoned_core.py            # patch + backup + syntax compile
    python apply_zoned_core.py --check    # additionally BUILD one zoned
                                          # model in memory (imports openmc,
                                          # no transport run) to prove the
                                          # geometry assembles

Flags
  --check   after patching, import reactor_model and construct a zoned
            32-assembly model with a 0.85 / balanced / 1.075 ring map to
            verify the code path end to end (seconds, no particles run)

A backup reactor_model.py.zoned.bak is written next to the original. The
script REFUSES to run twice (idempotency guard on the design_map anchor).
"""
import argparse
import py_compile
import shutil
import sys
from pathlib import Path

RM = Path(__file__).with_name("reactor_model.py")

# --------------------------------------------------------------------------- #
# The four anchored edits. Every anchor must occur EXACTLY once.              #
# --------------------------------------------------------------------------- #
A_SIG = """                    core_map=None, refl_thick=None, r_fuel=None,
                    enforce_vessel=True,"""
R_SIG = """                    core_map=None, refl_thick=None, r_fuel=None,
                    design_map=None,
                    enforce_vessel=True,"""

A_BUILD = """    asm_u, fuel_cells, _ = build_assembly_universe(design, mats, geo, pitch)"""
R_BUILD = """    asm_u, fuel_cells, _ = build_assembly_universe(design, mats, geo, pitch)

    # ZONED LOADING (apply_zoned_core.py): optional per-position design
    # overrides, {(row, col): design_dict}. Each distinct override builds its
    # own materials and assembly universe through the SAME builders as the
    # base assembly, so pin layout, gadolinia pattern and the gadolinia-rod
    # enrichment derate stay single-sourced. Variants are cached on their
    # numeric content, and the lattice pitch is ALWAYS the base design's
    # pitch (zoning must never change geometry).
    variant_mats = []
    _variant_cache = {}

    def _variant_u(d):
        key = tuple(sorted((k, round(float(v), 9)) for k, v in d.items()
                           if isinstance(v, (int, float))))
        if key not in _variant_cache:
            mv = build_materials(d, op)
            uv, _fc, _ = build_assembly_universe(d, mv, geo, pitch)
            variant_mats.extend(mv.values())
            _variant_cache[key] = uv
        return _variant_cache[key]"""

A_LOOP = """    for i in range(ny):
        for j in range(nx):
            universes[i, j] = asm_u if core_map[i, j] == 1 else refl_u"""
R_LOOP = """    for i in range(ny):
        for j in range(nx):
            if core_map[i, j] != 1:
                universes[i, j] = refl_u
            elif design_map is not None and (i, j) in design_map:
                universes[i, j] = _variant_u(design_map[(i, j)])
            else:
                universes[i, j] = asm_u"""

A_MATS = """    materials = openmc.Materials([m for m in mats.values()] + [refl_mat])"""
R_MATS = """    materials = openmc.Materials([m for m in mats.values()]
                                 + variant_mats + [refl_mat])"""

PATCHES = [("signature", A_SIG, R_SIG),
           ("variant builder", A_BUILD, R_BUILD),
           ("lattice loop", A_LOOP, R_LOOP),
           ("materials list", A_MATS, R_MATS)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--check", action="store_true",
                    help="after patching, build one zoned model in memory "
                         "(imports openmc, runs no particles)")
    args = ap.parse_args()

    src = RM.read_text()
    if "design_map=None" in src:
        sys.exit("reactor_model.py already contains design_map: patch was "
                 "applied before. Nothing to do.")

    for name, a, _ in PATCHES:
        n = src.count(a)
        if n != 1:
            sys.exit(f"ANCHOR ERROR [{name}]: found {n} occurrences "
                     f"(need exactly 1). reactor_model.py drifted from the "
                     f"campaign5 state this patch was written against. "
                     f"Do NOT force it: inspect and re-anchor.")

    bak = RM.with_suffix(".py.zoned.bak")
    shutil.copy(RM, bak)
    for name, a, r in PATCHES:
        src = src.replace(a, r)
    RM.write_text(src)
    py_compile.compile(str(RM), doraise=True)
    print(f"patched  : {RM}")
    print(f"backup   : {bak}")
    print(f"edits    : {', '.join(p[0] for p in PATCHES)}")
    print("default path (design_map=None) is unchanged, campaign "
          "reproducibility preserved.")

    if args.check:
        import importlib
        import reactor_model as rm
        importlib.reload(rm)
        import zoning as zn
        base = dict(enrich_inner=12.0, enrich_outer=13.0, gd_wt=6.0,
                    pitch=1.20, refl_thick=12.0, gd_pins=16)
        rmap = zn.ring_map()
        m_c, m_m, m_p = zn.balanced_multipliers(0.85, 1.075,
                                                zn.ring_counts(rmap))
        dmap = zn.design_map_for(rmap, zn.zone_designs(base, m_c, m_m, m_p))
        model, _ = rm.make_core_model(base, design_map=dmap,
                                      particles=100, batches=10, inactive=5)
        n_mat = len(model.materials)
        print(f"CHECK ok : zoned model built, {n_mat} materials "
              f"(base set plus three ring variants plus the reflector), "
              f"multipliers C/M/P = {m_c:.4f}/{m_m:.4f}/{m_p:.4f}")


if __name__ == "__main__":
    main()
