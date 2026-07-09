"""
sweep_ktarget.py
================
Tabulate the end-of-cycle (EOC) reactivity target as a FUNCTION of the reflector
thickness, K_TARGET(refl_thick), using the now-parameterized make_core_model.

WHY THIS EXISTS -- two mutually exclusive ways to let refl_thick affect cycle length
------------------------------------------------------------------------------------
ROUTE A (screening lever, already wired into openmc_evaluator._cycle_length):
    depletion uses make_assembly_model(reflector=True). The depleted k already
    contains radial + reflector leakage, so K_TARGET is a SINGLE frozen number
    carrying only the AXIAL leakage a 2-D (two-dimensional) model cannot see
    (~1.02-1.03 for ~1.2 m active height). refl_thick enters through the sleeve.

ROUTE B (magnitude-faithful, what THIS script sets up):
    depletion uses make_assembly_model(bc="reflective"), i.e. the depleted k is
    the infinite-medium k_inf (NO leakage). ALL leakage is then applied through
    K_TARGET(refl_thick) = k_inf / k_eff_core(refl_thick), tabulated below by
    sweeping the real small-core model. Thicker reflector -> lower leakage factor
    -> lower target -> later crossing -> longer effective full power days (EFPD).

*** DO NOT COMBINE THEM. *** Using reflector=True AND a refl_thick-dependent
K_TARGET double-counts the reflector's leakage effect. Pick one:
  - Route A: keep openmc_evaluator as delivered (reflector=True, frozen ~1.03).
  - Route B: set _cycle_length back to bc="reflective" AND load this table so
            k_target becomes k_target(design['refl_thick']).

Assumption (standard): the leakage factor k_inf/k_eff is geometry-driven and
weakly burnup-dependent, so we evaluate it at beginning of life (BOL), exactly as
measure_leakage_target.py does, and treat it as constant over the cycle.

Run this ONCE for your finalized reference geometry, commit ktarget_vs_refl.json,
and (for Route B) load it in the evaluator. Re-run if the core map, pitch band,
enrichment split, or reflector material changes.
"""

import json
import math

import numpy as np
import openmc

import reactor_model as rm

# --- finalized reference design (match measure_leakage_target.py) ------------ #
# refl_thick is overwritten inside the loop, so its value here is a placeholder.
design = {
    "enrich_inner": 4.55, "enrich_outer": 4.05, "gd_wt": 0.0,
    "pitch": 1.26, "refl_thick": 15.0,
}
op = rm.Operating()
geo = rm.Geometry17x17()

# reflector thicknesses [cm] to sample; span the optimizer bounds (2 - 25 cm)
REFL_GRID = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 25.0]

# transport settings (one-off calibration -> keep tight; bump if k stats noisy)
ASM_PARTICLES, ASM_BATCHES, ASM_INACTIVE = 10000, 120, 30
CORE_PARTICLES, CORE_BATCHES, CORE_INACTIVE = 20000, 150, 40

OUT_JSON = "ktarget_vs_refl.json"


def assembly_kinf() -> float:
    """Infinite-medium k_inf from the reflective single assembly.
    Independent of refl_thick, so compute it ONCE."""
    asm, _fc, _lat = rm.make_assembly_model(
        design, op, geo, bc="reflective",
        particles=ASM_PARTICLES, batches=ASM_BATCHES, inactive=ASM_INACTIVE)
    sp = asm.run(cwd="run_assembly_ref", output=False)
    with openmc.StatePoint(sp) as s:
        return float(s.keff.nominal_value), float(s.keff.std_dev)


def core_keff(refl_thick: float) -> float:
    """Small-core k_eff (vacuum BC) with an explicit reflector annulus of the
    given thickness -- this is the number that moves with refl_thick."""
    d = dict(design, refl_thick=refl_thick)
    core, _fc = rm.make_core_model(
        d, op, geo, refl_thick=refl_thick,
        particles=CORE_PARTICLES, batches=CORE_BATCHES, inactive=CORE_INACTIVE)
    sp = core.run(cwd=f"run_core_refl_{refl_thick:g}", output=False)
    with openmc.StatePoint(sp) as s:
        return float(s.keff.nominal_value), float(s.keff.std_dev)


def main():
    k_inf, k_inf_sd = assembly_kinf()
    print("=" * 68)
    print(f"assembly k_inf (reflective, refl-independent) = "
          f"{k_inf:.4f} +/- {k_inf_sd:.4f}")
    print("=" * 68)
    print(f"{'refl_thick[cm]':>14} {'k_eff_core':>12} {'leak=kinf/keff':>16} "
          f"{'K_TARGET':>10}")

    rows = []
    for t in REFL_GRID:
        k_eff, k_eff_sd = core_keff(t)
        leak = k_inf / k_eff
        k_target = 1.0 * leak          # critical core => target = leakage factor
        rows.append({"refl_thick": t, "k_eff": k_eff, "k_eff_sd": k_eff_sd,
                     "leak": leak, "k_target": k_target})
        print(f"{t:>14.1f} {k_eff:>12.4f} {leak:>16.4f} {k_target:>10.4f}")

    print("-" * 68)
    # monotonicity check: k_eff should rise (leak/target fall) with thickness,
    # then saturate. A non-monotone blip usually means statistics too loose.
    keffs = [r["k_eff"] for r in rows]
    if any(b < a - 3e-3 for a, b in zip(keffs, keffs[1:])):
        print("WARNING: k_eff not monotonically increasing -> tighten statistics "
              "(raise CORE_BATCHES/particles) before trusting the fit.")

    table = {
        "k_inf_assembly": k_inf,
        "design": design,
        "refl_thick_cm": [r["refl_thick"] for r in rows],
        "k_target": [r["k_target"] for r in rows],
        "note": "Route B: pair with make_assembly_model(bc='reflective') in "
                "_cycle_length; interpolate k_target at design['refl_thick'].",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(table, f, indent=2)
    print(f"wrote {OUT_JSON}")

    # convenience: show how the evaluator would interpolate it
    xs = np.array(table["refl_thick_cm"])
    ys = np.array(table["k_target"])
    for probe in (7.5, 12.5, 18.0):
        print(f"  K_TARGET({probe:>4} cm) ~ {float(np.interp(probe, xs, ys)):.4f} "
              f"(linear interp)")


if __name__ == "__main__":
    main()
