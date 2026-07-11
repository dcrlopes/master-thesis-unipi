"""
core_geometry.py
================
Pure-math geometry bookkeeping shared by reactor_model.py (the OpenMC builder),
openmc_evaluator.py (the truth evaluator) and reactor_optimization.py (the
optimizer). It has NO OpenMC dependency on purpose, so the analytic demo and
the acquisition loop can use the SAME functions without importing openmc.

WHY THIS FILE EXISTS (the geometry bug it fixes)
------------------------------------------------
The old make_core_model bounded the fuel lattice with an EQUIVALENT-AREA
cylinder r_fuel = sqrt(N_asm/pi) * assembly_pitch (~68.4 cm at pitch 1.26).
But the outermost corner of the 6x6-minus-corners fuel footprint sits at
sqrt(3^2 + 2^2) = sqrt(13) lattice units from the center (~77.2 cm at pitch
1.26). Everything between those two radii -- about 5.2 % of the fuel area,
an 8-10 cm deep bite into every corner-adjacent assembly -- was silently
REPLACED by reflector material. The reflector was occupying part of the
assemblies, exactly the overlap this module now forbids.

The corrected rule, used everywhere below:

    R_env(pitch)              = circumscribed radius of the intact 32-assembly
                                footprint  = sqrt(13) * 17 * pitch  (6x6-corners)
    reflector inner radius    = R_env      (no fuel is ever cut)
    reflector outer radius    = R_env + refl_thick
    vessel-fit constraint     g_geom = R_env + refl_thick - (R_vessel - clearance) <= 0

So `refl_thick` is the MINIMUM (corner-direction) reflector thickness; along
the flat faces the reflector is thicker by (sqrt(13)-3)*17*pitch ~ 0.61*A,
which is physical (real cores have thicker reflector at the flats).

All lengths in cm.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Frozen plant envelope (CTMSP vessel drawing: vessel ID ~ 1.8 m).
# R_VESSEL_INNER is the hard radial budget everything must fit inside.
# VESSEL_CLEARANCE_CM is reserved radial space between the reflector outer
# surface and the vessel inner surface (core barrel wall + downcomer gap).
# It is 0.0 by default = "reflector may touch the vessel"; setting a real
# barrel/downcomer allowance is a DESIGN-BASIS decision (physics owner), and
# tightens g_geom uniformly.
# ---------------------------------------------------------------------------
R_VESSEL_INNER = 90.0        # cm  (vessel inner radius, ID ~ 1.8 m)
VESSEL_CLEARANCE_CM = 0.0    # cm  (barrel + downcomer allowance; edit knowingly)

# The reference 32-assembly core layout: 6x6 grid with the 4 corners removed.
# ONE source of truth -- reactor_model.make_core_model imports THIS map.
CORE_MAP_32 = np.array([
    [0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0],
])


def core_envelope_radius(pitch: float, core_map=None, lattice: int = 17) -> float:
    """Radius [cm] of the smallest z-cylinder that contains EVERY fuel assembly
    of the core map intact (no clipping): the max distance from the core center
    to any corner of any fuel cell.

    For the default 6x6-minus-corners map this equals sqrt(13) * lattice * pitch
    (farthest fuel corner at (3A, 2A), A = assembly pitch), but it is computed
    generically from the map so a future layout change cannot silently break it.
    """
    cmap = CORE_MAP_32 if core_map is None else np.asarray(core_map)
    A = lattice * pitch                       # assembly pitch [cm]
    ny, nx = cmap.shape
    r2max = 0.0
    for i in range(ny):
        for j in range(nx):
            if cmap[i, j] != 1:
                continue
            x0 = -nx * A / 2.0 + j * A        # cell lower-left corner
            y0 = -ny * A / 2.0 + i * A
            for cx in (x0, x0 + A):
                for cy in (y0, y0 + A):
                    r2max = max(r2max, cx * cx + cy * cy)
    return math.sqrt(r2max)


def geometry_margin(pitch: float, refl_thick: float, *,
                    core_map=None, lattice: int = 17,
                    r_vessel: float = R_VESSEL_INNER,
                    clearance: float = VESSEL_CLEARANCE_CM) -> float:
    """The vessel-fit constraint value g_geom (pymoo convention: g <= 0 feasible):

        g_geom = R_env(pitch) + refl_thick - (r_vessel - clearance)

    g_geom <= 0  <=>  the intact fuel footprint PLUS the requested minimum
    reflector thickness fits radially inside the vessel. Positive g_geom means
    the design is unbuildable: either the reflector would have to bite into the
    outer assemblies (the old bug) or it would poke through the vessel.
    """
    return (core_envelope_radius(pitch, core_map, lattice)
            + float(refl_thick) - (float(r_vessel) - float(clearance)))


def max_refl_thick(pitch: float, **kw) -> float:
    """Largest feasible refl_thick [cm] at a given pitch (g_geom = 0 locus)."""
    return -geometry_margin(pitch, 0.0, **kw)


# ---------------------------------------------------------------------------
# End-of-cycle crossing: burnup where k(bu) LAST crosses DOWN through k_target.
# Replaces the old  np.interp(-k_target, -kvals, bu)  which silently returns
# garbage when k is non-monotonic (every Gd design: k rises through the
# gadolinium burnout hump before falling).
# ---------------------------------------------------------------------------
def eoc_crossing_burnup(bu, k, k_target: float):
    """Return the burnup at the LAST downward crossing of k through k_target,
    by linear interpolation inside the bracketing step; None if k never ends
    below the target after having been above it (never-critical designs and
    censored histories both return None -- the caller decides which it is)."""
    bu = np.asarray(bu, dtype=float)
    k = np.asarray(k, dtype=float)
    idx = None
    for i in range(len(k) - 1):
        if k[i] > k_target >= k[i + 1]:
            idx = i
    if idx is None:
        return None
    f = (k[idx] - k_target) / (k[idx] - k[idx + 1])
    return float(bu[idx] + f * (bu[idx + 1] - bu[idx]))


# ---------------------------------------------------------------------------
# Bilinear interpolation on a rectilinear (pitch x refl_thick) k_target table,
# CLAMPED at the grid edges (same convention np.interp uses in 1D). Used by
# openmc_evaluator._k_target_for when sweep_ktarget.py wrote a 2-D table.
# ---------------------------------------------------------------------------
def bilinear_clamped(x: float, y: float, xs, ys, Z) -> float:
    """Z has shape (len(xs), len(ys)); xs, ys strictly increasing."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    Z = np.asarray(Z, dtype=float)
    x = float(np.clip(x, xs[0], xs[-1]))
    y = float(np.clip(y, ys[0], ys[-1]))
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
    tx = 0.0 if xs[i + 1] == xs[i] else (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = 0.0 if ys[j + 1] == ys[j] else (y - ys[j]) / (ys[j + 1] - ys[j])
    return float((1 - tx) * (1 - ty) * Z[i, j] + tx * (1 - ty) * Z[i + 1, j]
                 + (1 - tx) * ty * Z[i, j + 1] + tx * ty * Z[i + 1, j + 1])


if __name__ == "__main__":
    # quick self-check
    for p in (1.15, 1.26, 1.30, 1.43):
        print(f"pitch {p:5.3f}: R_env = {core_envelope_radius(p):6.2f} cm, "
              f"max refl_thick = {max_refl_thick(p):5.2f} cm")
    assert abs(core_envelope_radius(1.0) - math.sqrt(13) * 17) < 1e-9
    # crossing logic: Gd hump (rise then fall), target crossed on the way down
    bu = [0, 1, 2, 3, 4]
    k = [1.04, 1.08, 1.10, 1.06, 1.02]
    assert abs(eoc_crossing_burnup(bu, k, 1.05) - 3.25) < 1e-12
    assert eoc_crossing_burnup(bu, [1.2, 1.15, 1.10, 1.08, 1.06], 1.05) is None
    print("self-checks OK")
