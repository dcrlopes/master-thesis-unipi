"""
zoning.py
=========
Shared logic for the ZONED-LOADING study that runs AFTER Campaign 5 Block 2:

  Stage T (transfer test) : rescore_zoned_core.py
  Stage 2 (refinement)    : refine_zoning.py
  Stage 3 (confirmation)  : confirm_zoned_champion.py

WHY THIS FILE EXISTS (one source of truth, the FUEL_PAD_CM lesson)
------------------------------------------------------------------
Campaign 4 lost a day to a 0.02 cm pad duplicated in two places. Everything
the three zoning scripts share lives HERE once: the ring assignment on the
32-assembly map, the enrichment-balance solve, the construction of the
per-position design map consumed by reactor_model.make_core_model
(design_map argument added by apply_zoned_core.py), the core Beginning of
Life (BOL) transport solve with the peaking extraction copied VERBATIM from
OpenMCEvaluator._bol_core_peaking, and the k(burnup) history reader.

THE ZONING MODEL
----------------
Three concentric rings on the 6x6-minus-corners map (32 assemblies):

  ring C (centre)    :  4 assemblies, Euclidean distance from map centre < 1.0
  ring M (middle)    : 12 assemblies, 1.0 <= distance < 2.3
  ring P (periphery) : 16 assemblies, distance >= 2.3

Both intra-assembly enrichments (enrich_inner, enrich_outer) of a ring are
scaled by ONE multiplier m_z. The middle multiplier is solved from the
fissile balance so the assembly-count-weighted mean multiplier is exactly 1:

    n_C m_C + n_M m_M + n_P m_P = n_C + n_M + n_P

  m_z : enrichment multiplier of ring z (dimensionless)
  n_z : number of assemblies in ring z (4, 12, 16)

Scaling the zone enrichment automatically scales the gadolinia-rod
enrichment through the existing derate in build_materials (e_gd is derived
from the zone enrichment there, floored at 0.2 wt%), so no material logic
is duplicated. Zoning NEVER changes pitch, reflector, gadolinia weight or
gadolinia pin count: it is an enrichment redistribution at fixed
core-average enrichment.

All lengths in cm. All enrichments in wt% U-235.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np

import core_geometry as cg

# Ring thresholds in lattice units of Euclidean distance from the map centre.
RING_THRESHOLDS = (1.0, 2.3)
RING_NAMES = ("C", "M", "P")

# The evaluator's LEU (Low Enriched Uranium) screening cap and the physical
# LEU boundary. g_enr in the campaigns uses 19.75; true LEU means < 20.
LEU_CAP_SCREEN = 19.75
LEU_CAP_PHYSICAL = 20.0


# --------------------------------------------------------------------------- #
# Ring geometry (pure math, no OpenMC import)                                 #
# --------------------------------------------------------------------------- #
def ring_map(core_map=None) -> np.ndarray:
    """Integer map of the core layout: -1 outside the fuel footprint,
    0 = ring C, 1 = ring M, 2 = ring P. Computed from the Euclidean distance
    of each fuel position to the map centre, so it generalises to any
    rectangular core_map, not only CORE_MAP_32."""
    cmap = cg.CORE_MAP_32 if core_map is None else np.asarray(core_map)
    ny, nx = cmap.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    out = np.full(cmap.shape, -1, dtype=int)
    for i in range(ny):
        for j in range(nx):
            if cmap[i, j] != 1:
                continue
            d = math.hypot(i - cy, j - cx)
            if d < RING_THRESHOLDS[0]:
                out[i, j] = 0
            elif d < RING_THRESHOLDS[1]:
                out[i, j] = 1
            else:
                out[i, j] = 2
    return out


def ring_counts(rmap: np.ndarray) -> tuple[int, int, int]:
    return tuple(int((rmap == z).sum()) for z in range(3))


def balanced_multipliers(m_center: float, m_periphery: float,
                         counts=(4, 12, 16)) -> tuple[float, float, float]:
    """Solve the middle-ring multiplier from the fissile balance
    n_C m_C + n_M m_M + n_P m_P = n_C + n_M + n_P, which preserves the
    core-average enrichment multiplier at exactly 1."""
    n_c, n_m, n_p = counts
    for name, m in (("m_C", m_center), ("m_P", m_periphery)):
        if not (0.4 < m < 1.6):
            raise ValueError(
                f"{name} = {m} is outside the sanity window (0.4, 1.6). "
                f"A typo? Zoning multipliers this study explores stay well "
                f"inside 0.8 to 1.15.")
    m_m = (n_c + n_m + n_p - n_c * m_center - n_p * m_periphery) / n_m
    if not (0.4 < m_m < 1.6):
        raise ValueError(
            f"balanced middle multiplier m_M = {m_m:.4f} is outside the "
            f"sanity window (0.4, 1.6) for m_C = {m_center}, "
            f"m_P = {m_periphery}. Choose a milder map.")
    return float(m_center), float(m_m), float(m_periphery)


def zone_designs(base: dict, m_c: float, m_m: float, m_p: float) -> dict:
    """Per-ring design dicts. Only the two enrichment variables are scaled.
    Every other key (gd_wt, gd_pins, pitch, refl_thick, ...) is copied
    unchanged so the ring assemblies are built by the SAME builders with the
    SAME pin layout as the base assembly."""
    out = {}
    for name, m in zip(RING_NAMES, (m_c, m_m, m_p)):
        d = dict(base)
        d["enrich_inner"] = float(base["enrich_inner"]) * m
        d["enrich_outer"] = float(base["enrich_outer"]) * m
        d["zone"] = name           # bookkeeping only, ignored by the builders
        d["zone_mult"] = float(m)
        out[name] = d
    return out


def design_map_for(rmap: np.ndarray, zdesigns: dict) -> dict:
    """{(row, col): design} consumed by make_core_model(design_map=...).
    Ring M positions are omitted on purpose when m_M == 1 would equal the
    base design only if the caller passes the base as ring M. To keep the
    map explicit and auditable, EVERY fuel position gets an entry."""
    out = {}
    ny, nx = rmap.shape
    for i in range(ny):
        for j in range(nx):
            z = rmap[i, j]
            if z >= 0:
                out[(i, j)] = zdesigns[RING_NAMES[z]]
    return out


def max_zoned_enrichment(base: dict, m_p: float) -> float:
    """Highest enrichment anywhere in the zoned core. The periphery carries
    the largest multiplier in every map this study uses."""
    return max(float(base["enrich_inner"]), float(base["enrich_outer"])) * m_p


# --------------------------------------------------------------------------- #
# Archive access (checkpoint written by ActiveLearningMOO.save_checkpoint)    #
# --------------------------------------------------------------------------- #
def load_archive(checkpoint_path):
    """Return (design_variable_names, constraint_names, all_raw list)."""
    ck = json.loads(Path(checkpoint_path).read_text())
    return ck["design_variables"], ck.get("constraint_names", []), \
        ck["all_raw"], ck.get("meta", {})


def design_of(raw: dict, dv_names) -> dict:
    return {n: float(raw[n]) for n in dv_names}


def is_feasible(raw: dict, constraint_names) -> bool:
    return all(float(raw.get(c, 0.0)) <= 1e-9 for c in constraint_names)


# --------------------------------------------------------------------------- #
# Core BOL solve with the campaign peaking extraction                         #
# --------------------------------------------------------------------------- #
def core_bol_solve(base_design: dict, design_map, op, geo, *,
                   particles: int, batches: int, inactive: int,
                   seed: int, case: Path) -> dict:
    """One 2D core Beginning of Life (BOL) eigenvalue solve.

    Peaking extraction (mask zero-fission bins, then max over mean) and the
    Shannon-entropy convergence sentinel are VERBATIM from
    OpenMCEvaluator._bol_core_peaking. Keep the three blocks in sync by
    editing THERE first. Additionally returns the fission-power share of
    each ring, needed by the Stage 3 linear-reactivity combination."""
    import openmc                      # lazy: keeps the pure math importable
    import reactor_model as rm

    case = Path(case)
    case.mkdir(parents=True, exist_ok=True)
    m = rm.make_core_model(base_design, op, geo, design_map=design_map,
                           particles=particles, batches=batches,
                           inactive=inactive)
    model = m[0] if isinstance(m, tuple) else m
    model.settings.seed = int(seed)

    NL = geo.lattice
    pitch = base_design.get("pitch", 1.26)
    rmap = ring_map()
    ny, nx = rmap.shape
    half = nx * NL * pitch / 2.0

    mesh = openmc.RegularMesh()
    mesh.dimension = (nx * NL, ny * NL)
    mesh.lower_left = (-half, -half)
    mesh.upper_right = (half, half)
    t = openmc.Tally(name="core_pin_fission")
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ["fission"]
    model.tallies = openmc.Tallies([t])

    t0 = time.time()
    sp_path = model.run(cwd=str(case), output=False)
    wall = time.time() - t0
    with openmc.StatePoint(sp_path) as sp:
        keff = float(sp.keff.nominal_value)
        keff_sd = float(sp.keff.std_dev)
        v = sp.get_tally(name="core_pin_fission").get_values(
            scores=["fission"]).reshape(ny * NL, nx * NL)
        H = np.asarray(getattr(sp, "entropy", []), dtype=float)

    f = np.ma.masked_equal(v, 0.0)
    fdh = float((f / f.mean()).max())

    conv = None
    if H.size:
        tail = H[inactive + (len(H) - inactive) // 2:]
        mu, sd = float(tail.mean()), float(tail.std(ddof=1))
        Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
        Hs[0], Hs[-1] = H[0], H[-1]
        bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
        conv = int(bad[-1]) + 2 if len(bad) else 1

    shares = ring_power_shares(v, rmap, NL)
    return dict(keff=keff, keff_sd=keff_sd, fdh_core=fdh,
                entropy_conv_batch=conv, wall_s=wall,
                ring_shares=shares, seed=int(seed))


def ring_power_shares(pin_map: np.ndarray, rmap: np.ndarray,
                      NL: int) -> list[float]:
    """Fission-power share of each ring from the core pin mesh. Mesh bins are
    grouped into assemblies by integer division of the bin index by NL."""
    ny, nx = rmap.shape
    tot = float(pin_map.sum())
    shares = [0.0, 0.0, 0.0]
    for i in range(ny):
        for j in range(nx):
            z = rmap[i, j]
            if z < 0:
                continue
            block = pin_map[i * NL:(i + 1) * NL, j * NL:(j + 1) * NL]
            shares[z] += float(block.sum())
    return [s / tot for s in shares]


# --------------------------------------------------------------------------- #
# k(burnup) history of an assembly depletion case                             #
# --------------------------------------------------------------------------- #
def read_k_history(case_dir, spec_power_w_per_g: float):
    """(burnup [MWd/kgHM], k_inf) of an OpenMCEvaluator._cycle_length case.

    The evaluator chains chunks with prev_results, and OpenMC 0.15.x then
    stores the CUMULATIVE history in every later chunk, so the LAST chunk's
    depletion_results.h5 holds the whole trajectory. A monotonic-time check
    guards the assumption. Burnup follows the evaluator's own conversion
    bu [MWd/kg] = t [d] * spec_power [W/g] / 1000."""
    import openmc.deplete
    chunks = sorted(glob.glob(str(Path(case_dir) / "dep_*" /
                                  "depletion_results.h5")))
    if not chunks:
        raise FileNotFoundError(f"no dep_*/depletion_results.h5 under "
                                f"{case_dir}")
    res = openmc.deplete.Results(chunks[-1])
    try:
        t_d, karr = res.get_keff(time_units="d")
    except TypeError:
        t_s, karr = res.get_keff()
        t_d = np.asarray(t_s) / 86400.0
    t_d = np.asarray(t_d, dtype=float)
    if np.any(np.diff(t_d) < -1e-9):
        raise RuntimeError(
            f"non-monotonic time axis in {chunks[-1]}: chunk files are "
            f"chunk-local on this OpenMC version. Stitch manually or rerun "
            f"with the cumulative semantics the evaluator assumes.")
    k = np.asarray(karr, dtype=float)[:, 0]
    bu = t_d * spec_power_w_per_g / 1000.0
    return bu, k


def late_slope_pcm_per_mwdkg(bu: np.ndarray, k: np.ndarray,
                             frac: float = 0.3) -> float:
    """Least-squares slope of k over the LAST `frac` of the burnup axis, in
    pcm per MWd/kgHM. Post-gadolinia the trajectory is close to linear, so
    this is the local reactivity price of one extra MWd/kg."""
    n = max(3, int(len(bu) * frac))
    x, y = bu[-n:], k[-n:]
    A = np.vstack([x, np.ones_like(x)]).T
    slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    return slope * 1e5


# --------------------------------------------------------------------------- #
# Rank statistics (self-contained, ordinal ranks, no scipy dependency)        #
# --------------------------------------------------------------------------- #
def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a, b) -> float:
    ra, rb = _ranks(list(a)), _ranks(list(b))
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) *
                    sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def pearson(a, b) -> float:
    a, b = list(map(float, a)), list(map(float, b))
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) *
                    sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def set_threads(n):
    if n:
        os.environ["OMP_NUM_THREADS"] = str(int(n))
