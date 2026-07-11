"""
sweep_ktarget.py  --  2-D EDITION (pitch x refl_thick), corrected geometry
==========================================================================
Tabulate the end-of-cycle (EOC) reactivity target as a function of BOTH the
lattice pitch and the reflector thickness,

        K_TARGET(pitch, refl_thick) = k_inf(pitch) / k_eff_core(pitch, refl),

using the CORRECTED make_core_model (envelope-radius fuel cylinder: no fuel
is clipped, the reflector never occupies assembly space).

WHY THE OLD 1-D TABLE (ktarget_vs_refl.json) MUST BE THROWN AWAY
----------------------------------------------------------------
1. It was measured on the OLD core geometry, whose equivalent-area bounding
   cylinder cut ~5.2 % of the fuel and back-filled it with reflector -- every
   k_eff in it is biased.
2. It was 1-D at the nominal pitch 1.26 cm. The new vessel-fit constraint
   g_geom couples pitch and refl_thick (a thick reflector is only buildable
   at SMALL pitch), so designs now systematically visit off-nominal pitches
   exactly where a nominal-pitch table is most wrong. The 2-D table closes
   that documented gap.
3. Fuel isotopics changed (explicit U-234 correlation valid to 19.75 wt%).

ROUTE A vs ROUTE B (unchanged rule)
-----------------------------------
ROUTE B (this script): depletion runs at infinite medium (bc="reflective");
ALL leakage enters through K_TARGET. *** DO NOT COMBINE *** with an explicit
reflector in the depletion model (reflector=True), or reflector leakage is
counted twice.

Assumption (standard): the leakage factor k_inf/k_eff is geometry-driven and
weakly composition/burnup-dependent, so it is evaluated ONCE at BOL with a
fixed reference composition and treated as constant over the cycle.

VESSEL NOTE: grid nodes with refl_thick beyond the vessel budget at that
pitch are built with enforce_vessel=False. They are HYPOTHETICAL
interpolation support (a rectangular table needs values on both sides of the
g_geom line); only vessel-feasible designs ever query the table, and bilinear
interpolation near the line then uses physically sensible neighbours.

Typical cost on a c7a.8xlarge: 3 pitches x 7 thicknesses = 21 core runs
+ 3 assembly runs ~ 2.5-4 h. Run ONCE per finalized geometry, commit the
JSON, and pass it to run_optimization.py with --ktarget-table.

Usage:
    python sweep_ktarget.py                          # defaults below
    python sweep_ktarget.py --fast                   # half fidelity, quick look
    python sweep_ktarget.py --pitches 1.15,1.29,1.43 --refl 2,5,8,11,14,17,19.5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import openmc

import core_geometry as cgm
import reactor_model as rm

# --- fixed reference composition (leakage factor is weakly sensitive to it) --
REF_DESIGN = {"enrich_inner": 4.55, "enrich_outer": 4.05, "gd_wt": 0.0}

OP = rm.Operating()
GEO = rm.Geometry17x17()


def assembly_kinf(pitch: float, tr: dict):
    """Infinite-medium k_inf from the reflective single assembly at THIS
    pitch (k_inf moves with moderation, so it is re-run per pitch row)."""
    d = dict(REF_DESIGN, pitch=pitch, refl_thick=10.0)   # refl unused here
    asm, _fc, _lat = rm.make_assembly_model(
        d, OP, GEO, bc="reflective",
        particles=tr["asm_particles"], batches=tr["asm_batches"],
        inactive=tr["asm_inactive"])
    sp = asm.run(cwd=f"run_asm_p{pitch:g}", output=False)
    with openmc.StatePoint(sp) as s:
        return float(s.keff.nominal_value), float(s.keff.std_dev)


def core_keff(pitch: float, refl_thick: float, tr: dict):
    """Small-core k_eff (vacuum BC) on the CORRECTED geometry. Nodes beyond
    the vessel budget are hypothetical interpolation support, hence
    enforce_vessel=False here (and ONLY here)."""
    d = dict(REF_DESIGN, pitch=pitch, refl_thick=refl_thick)
    core, _fc = rm.make_core_model(
        d, OP, GEO, refl_thick=refl_thick, enforce_vessel=False,
        particles=tr["core_particles"], batches=tr["core_batches"],
        inactive=tr["core_inactive"])
    sp = core.run(cwd=f"run_core_p{pitch:g}_t{refl_thick:g}", output=False)
    with openmc.StatePoint(sp) as s:
        return float(s.keff.nominal_value), float(s.keff.std_dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitches", default="1.15,1.29,1.43",
                    help="comma-separated pitch grid [cm] spanning the "
                         "optimizer bounds (default 1.15,1.29,1.43)")
    ap.add_argument("--refl", default="2,5,8,11,14,17,19.5",
                    help="comma-separated refl_thick grid [cm] spanning the "
                         "optimizer bounds (default 2,...,19.5)")
    ap.add_argument("--out", default="ktarget_table.json",
                    help="output JSON path (default ktarget_table.json)")
    ap.add_argument("--asm-particles", type=int, default=10000)
    ap.add_argument("--asm-batches", type=int, default=120)
    ap.add_argument("--asm-inactive", type=int, default=30)
    ap.add_argument("--core-particles", type=int, default=20000)
    ap.add_argument("--core-batches", type=int, default=150)
    ap.add_argument("--core-inactive", type=int, default=40)
    ap.add_argument("--fast", action="store_true",
                    help="halve particles/batches for a quick provisional "
                         "table (regenerate at full fidelity before the "
                         "production optimization)")
    args = ap.parse_args()

    pitches = [float(s) for s in args.pitches.split(",")]
    refls = [float(s) for s in args.refl.split(",")]
    tr = dict(asm_particles=args.asm_particles, asm_batches=args.asm_batches,
              asm_inactive=args.asm_inactive,
              core_particles=args.core_particles,
              core_batches=args.core_batches,
              core_inactive=args.core_inactive)
    if args.fast:
        for k in ("asm_particles", "asm_batches",
                  "core_particles", "core_batches"):
            tr[k] = max(tr[k] // 2, 10)

    n_runs = len(pitches) * (1 + len(refls))
    print(f"2-D K_TARGET sweep: {len(pitches)} pitches x {len(refls)} "
          f"thicknesses = {len(pitches)*len(refls)} core runs "
          f"+ {len(pitches)} assembly runs ({n_runs} total)")

    kinf_row, Z, SD = [], [], []
    for p in pitches:
        k_inf, k_inf_sd = assembly_kinf(p, tr)
        t_max = cgm.max_refl_thick(p)
        print("=" * 72)
        print(f"pitch {p:.3f} cm | k_inf = {k_inf:.4f} +/- {k_inf_sd:.4f} | "
              f"vessel budget t_max = {t_max:.2f} cm")
        print(f"{'refl[cm]':>9} {'k_eff':>9} {'sd':>7} {'K_TARGET':>9} "
              f"{'in-vessel?':>10}")
        kinf_row.append(k_inf)
        zrow, sdrow = [], []
        for t in refls:
            k_eff, k_eff_sd = core_keff(p, t, tr)
            kt = k_inf / k_eff
            zrow.append(kt)
            sdrow.append(k_eff_sd)
            print(f"{t:>9.1f} {k_eff:>9.4f} {k_eff_sd:>7.4f} {kt:>9.4f} "
                  f"{'yes' if t <= t_max + 1e-9 else 'no (support)':>10}")
        # sanity: k_eff must rise (K_TARGET fall) with thickness, then
        # saturate. A non-monotone blip usually means statistics too loose.
        if any(b > a + 3e-3 for a, b in zip(zrow, zrow[1:])):
            print("WARNING: K_TARGET not monotonically decreasing with "
                  "refl_thick at this pitch -> tighten statistics "
                  "(--core-particles/--core-batches) before trusting it.")
        Z.append(zrow)
        SD.append(sdrow)

    table = {
        "schema": 2,
        "pitch_cm": pitches,
        "refl_thick_cm": refls,
        "k_target": Z,                     # [i_pitch][j_refl]
        "k_inf_assembly": kinf_row,        # per pitch
        "k_eff_sd": SD,                    # per node, for error budgeting
        "design": dict(REF_DESIGN),
        "geometry": "v2-envelope (no fuel clipping; corners+annulus reflector)",
        "note": "Route B: pair with make_assembly_model(bc='reflective') in "
                "openmc_evaluator._cycle_length; bilinear interpolation on "
                "(pitch, refl_thick), clamped at the grid edges.",
    }
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
    print("=" * 72)
    print(f"wrote {args.out}")

    # convenience: show how the evaluator will interpolate it
    for probe in ((1.20, 7.5), (1.26, 12.0), (1.35, 5.0)):
        val = cgm.bilinear_clamped(probe[0], probe[1],
                                   np.array(pitches), np.array(refls),
                                   np.array(Z))
        print(f"  K_TARGET(pitch={probe[0]:.2f}, refl={probe[1]:>4}) "
              f"~ {val:.4f} (bilinear)")


if __name__ == "__main__":
    main()
