#!/usr/bin/env python3
r"""
dep_common.py -- shared pieces of the three depletion checks
(c9_dep_replicas.py, c9_dep_core2d.py, c9_dep_core3d.py).

Everything OpenMC-specific is imported lazily inside the functions, so the
arithmetic and the selftest run on any machine.

  fuel_counts(fill)              instances of each fuel material inside a
                                 cell fill (universe or lattice), counted with
                                 the lattice multiplicity
  mark_depletable(model, ...)    volumes and the depletable flag on every fuel
                                 material, from those counts
  unmark_unused(model, rows)     switches the flag OFF on every other material,
                                 including the unplaced base assembly
  run_adaptive(...)              the adaptive chunked depletion of
                                 OpenMCEvaluator._cycle_length, ported so it
                                 accepts ANY model (core, 3D), with an absolute
                                 or a relative end-of-cycle target.
                                 test_dep_common.py proves it identical to
                                 the evaluator's own routine
  k_sd_from_case(case)           Monte Carlo s.d. of every k in the chunk files
  bracket_sigma_efpd(...)        Equations sigma-beoc and sigma-efpd of the
                                 thesis, propagated from the two states that
                                 bracket the crossing
  hump_and_cmax(...)             the Campaign 9 objective arithmetic
                                 (boron_objective.py) applied to a new k
                                 history and the ARCHIVED boron points
  solve_times(case)              wall time of every transport solve, from
                                 the statepoint /runtime/ group
  project_cost(...)              projection of a full cycle from an --estimate run

USAGE
    python dep_common.py --selftest
"""
from __future__ import annotations

import glob
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

import core_geometry as cg


# ---------------------------------------------------------------- fuel ----
def _is_lattice(f):
    return f is not None and hasattr(f, "universes")


def _is_universe(f):
    return f is not None and hasattr(f, "cells") and not hasattr(f, "universes")


def _is_material(f):
    return f is not None and hasattr(f, "nuclides") and not hasattr(f, "cells")


def is_fuel(mat) -> bool:
    """A material is fuel when it carries uranium-235."""
    names = [n[0] if isinstance(n, tuple) else getattr(n, "name", n)
             for n in mat.nuclides]
    return any(str(n).startswith("U235") for n in names)


def fuel_counts(fill, mult: int = 1) -> Counter:
    """{material object: number of pin instances} below `fill`, weighting
    every universe by how many lattice positions place it. The lattice
    `outer` universe is not counted (it is infinite)."""
    out = Counter()
    if _is_lattice(fill):
        occ = Counter(np.asarray(fill.universes, dtype=object).ravel().tolist())
        for u, n in occ.items():
            out.update(fuel_counts(u, mult * n))
    elif _is_universe(fill):
        for c in fill.cells.values():
            out.update(fuel_counts(c.fill, mult))
    elif _is_material(fill):
        if is_fuel(fill):
            out[fill] += mult
    return out


def mark_depletable(model, segments, geo) -> list[dict]:
    """`segments` is a list of (fill, height_cm). Every fuel material found
    under any fill gets volume = sum over segments of instances x pi r_f^2 x
    height, and depletable = True. A material shared by several segments
    (the plain and the grid part of one axial layer) sums its volumes.
    Returns one row per material with its instance count per unit height
    and its volume."""
    area = math.pi * geo.fuel_or ** 2
    vol = Counter()
    inst = {}
    for fill, h in segments:
        for m, n in fuel_counts(fill).items():
            vol[m] += n * area * float(h)
            inst.setdefault(m, n)
    rows = []
    for m, v in vol.items():
        m.volume = float(v)
        m.depletable = True
        rows.append(dict(id=int(m.id), name=str(m.name), pins=int(inst[m]),
                         volume_cm3=float(v)))
    if not rows:
        raise RuntimeError("no fuel material found under the given fills")
    return rows


def unmark_unused(model, rows) -> list[dict]:
    """Switch OFF the depletable flag of every material of model.materials
    that is not one of `rows` (the output of mark_depletable).

    This is not housekeeping. reactor_model.make_core_model always builds a
    base assembly at the core-average enrichment, and when a zoning map
    overrides all 32 positions that assembly is never placed, yet its
    materials stay in model.materials and arrive flagged depletable. The
    operator would then refuse the model ("Volume not specified for
    depletable material") or, worse, count heavy metal that is not in the
    core. Returns one row per material switched off.
    """
    keep = {int(r["id"]) for r in rows}
    off = []
    for m in model.materials:
        if getattr(m, "depletable", False) and int(m.id) not in keep:
            m.depletable = False
            m.volume = None
            off.append(dict(id=int(m.id), name=str(m.name)))
    return off


# ---------------------------------------------------- adaptive depletion ----
def run_adaptive(model, k_target, spec_power: float, *, bol_steps,
                 dep_step: float, chunk_steps: int, max_burnup: float,
                 case, k_target_ratio=None, verbose=True) -> dict:
    """The adaptive chunked depletion of OpenMCEvaluator._cycle_length,
    for a model whose depletable materials are already set. `spec_power` in
    W/gHM converts burnup steps to days and normalises the power, exactly
    as in the evaluator.

    END OF CYCLE. With `k_target` a number, the cycle ends when k falls to
    it (absolute criterion). With `k_target_ratio` instead, the target is
    set AFTER the first solve to ratio x k(BOL), so the model ends its
    cycle when it has lost the same reactivity the campaign's assembly
    loses, ratio = k_target / k_inf(BOL) of the archive. The relative form
    is the one that compares with the assembly proxy, because the campaign's
    k_target is not the criticality point of the core: for C9-47 the 2D core
    at 1000 ppm starts at 1.0273, already below the axial leakage factor
    1.0289, so an absolute target ends the cycle at burnup zero.

    Returns dict(efpd, bu_eoc, censored, k_hist, bu_hist, n_solves, wall_s,
    chunks, k_target)."""
    if (k_target is None) == (k_target_ratio is None):
        raise ValueError("pass k_target or k_target_ratio, not both")
    import openmc.deplete

    case = Path(case)
    case.mkdir(parents=True, exist_ok=True)
    bu_hist = [0.0]
    k_hist: list[float] = []
    state = {"prev": None, "power_w": None, "chunk": 0}
    cwd = Path.cwd()
    t0 = time.time()

    def run_chunk(steps, announce=True):
        op_dep = openmc.deplete.CoupledOperator(
            model, prev_results=state["prev"], diff_burnable_mats=False)
        if state["power_w"] is None:
            state["power_w"] = spec_power * op_dep.heavy_metal
        days = [s * 1000.0 / spec_power for s in steps]
        integrator = openmc.deplete.PredictorIntegrator(
            op_dep, days, power=state["power_w"], timestep_units="d")
        cdir = case / f"dep_{state['chunk']:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(cdir)
            try:
                integrator.integrate(write_rates=True)
            except TypeError:
                integrator.integrate()
            results = openmc.deplete.Results("depletion_results.h5")
            last_rates = getattr(results[-1], "rates", None)
            if last_rates is None or getattr(last_rates, "size", 0) == 0 \
                    or float(np.abs(np.asarray(last_rates)).sum()) == 0.0:
                raise RuntimeError("depletion results carry no reaction rates "
                                   "(the write_rates trap)")
        finally:
            os.chdir(cwd)
        state["chunk"] += 1
        state["prev"] = results
        _t, karr = results.get_keff()
        kvals = [float(v) for v in karr[:, 0]]
        if not k_hist:
            if len(kvals) != len(steps) + 1:
                raise RuntimeError(f"first chunk returned {len(kvals)} k values "
                                   f"for {len(steps)} steps")
            k_hist.extend(kvals)
        else:
            k_hist.extend(kvals[-len(steps):])
        for s in steps:
            bu_hist.append(bu_hist[-1] + s)
        if len(k_hist) != len(bu_hist):
            raise RuntimeError("burnup/k bookkeeping out of sync")
        if verbose and announce:
            print(f"      chunk {state['chunk'] - 1}: B {bu_hist[-1]:6.2f} MWd/kgHM "
                  f"k {k_hist[-1]:.5f} (target {k_target:.5f}) "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    run_chunk([float(s) for s in bol_steps], announce=False)
    if k_target is None:                     # relative criterion, see above
        k_target = float(k_target_ratio) * k_hist[0]
        if verbose:
            print(f"      k(BOL) {k_hist[0]:.5f} x ratio {k_target_ratio:.5f} "
                  f"-> end of cycle at k = {k_target:.5f}", flush=True)
    if verbose:
        for i, (b, kv) in enumerate(zip(bu_hist, k_hist)):
            print(f"      state {i}: B {b:6.2f} MWd/kgHM  k {kv:.5f}"
                  + ("  (target)" if i == 0 else ""), flush=True)
    censored = False
    while True:
        k_op = k_hist[1:] if len(k_hist) > 1 else k_hist
        past_peak = int(np.argmax(k_op)) < len(k_op) - 1
        if past_peak and k_hist[-1] <= k_target:
            break
        remaining = max_burnup - bu_hist[-1]
        if remaining <= 1e-9:
            censored = k_hist[-1] > k_target
            break
        steps = []
        for _ in range(max(1, int(chunk_steps))):
            s = min(dep_step, remaining - sum(steps))
            if s <= 1e-9:
                break
            steps.append(s)
        run_chunk(steps)

    bu_eoc = cg.eoc_crossing_burnup(bu_hist, k_hist, k_target)
    if bu_eoc is None:
        if censored:
            bu_eoc = bu_hist[-1]
            efpd = bu_eoc * 1000.0 / spec_power
        else:
            bu_eoc, efpd = 0.0, 0.0
    else:
        efpd = bu_eoc * 1000.0 / spec_power
    return dict(efpd=float(efpd), bu_eoc=float(bu_eoc), censored=bool(censored),
                k_hist=[float(v) for v in k_hist], bu_hist=[float(v) for v in bu_hist],
                n_solves=len(k_hist), wall_s=time.time() - t0, chunks=state["chunk"],
                k_target=float(k_target))


# ----------------------------------------------------------- statistics ----
def k_sd_from_case(case) -> dict:
    """{k value: Monte Carlo s.d.} over every real entry of every chunk
    results file under `case`. Entries the chunk did not compute are
    zero-filled by this OpenMC version and are skipped."""
    import openmc.deplete
    out = {}
    for ch in sorted(glob.glob(str(Path(case) / "dep_*" / "depletion_results.h5"))):
        _t, karr = openmc.deplete.Results(ch).get_keff()
        karr = np.asarray(karr, float)
        for kv, sd in karr:
            if kv > 0.0:
                out[float(kv)] = float(sd)
    return out


def bracket_indices(bu, k, k_target):
    """Index i of the LAST downward crossing, k[i] > k_target >= k[i+1]."""
    idx = None
    for i in range(len(k) - 1):
        if k[i] > k_target >= k[i + 1]:
            idx = i
    return idx


def bracket_sigma_efpd(bu, k, ksd, k_target, spec_power):
    """Equations sigma-beoc and sigma-efpd: statistical error of the cycle
    length from the two states that bracket the crossing. `ksd` is a list
    aligned with k (None entries allowed). Returns (sigma_efpd or None,
    index of the bracket, slope in pcm per MWd/kgHM or None)."""
    i = bracket_indices(bu, k, k_target)
    if i is None or ksd[i] is None or ksd[i + 1] is None:
        return None, i, None
    dB = bu[i + 1] - bu[i]
    dk = k[i] - k[i + 1]
    a = dB * (k_target - k[i + 1]) / dk ** 2
    b = dB * (k[i] - k_target) / dk ** 2
    sig_b = math.sqrt((a * ksd[i]) ** 2 + (b * ksd[i + 1]) ** 2)
    return 1000.0 * sig_b / spec_power, i, 1.0e5 * dk / dB


def hump_and_cmax(k_hist, k_core_bol, boron_points, noise_pcm, clip=None):
    """Campaign 9 objective from a NEW assembly (or core) k history and the
    ARCHIVED boron points of the same design. When `k_core_bol` is the
    history's own first entry (a core depletion), l_ax = 1 and the hump is
    the core hump measured directly. Returns the merged dict of
    boron_objective.hump_from_history and boron_objective.boron_objective."""
    import boron_objective as bo
    hump = bo.hump_from_history(list(k_hist), float(k_core_bol),
                                noise_pcm=float(noise_pcm))
    points = {float(p): float(v) for p, v in boron_points.items()}
    obj = bo.boron_objective(points, hump,
                             clip=(bo.PPM_CLIP if clip is None else clip))
    out = {}
    out.update(hump)
    out.update(obj)
    out["c_max"] = float(obj["c_max_ppm"])
    return out


def _sp_sorted(cdir):
    """openmc_simulation_n*.h5 of one chunk in NUMERIC order (n10 after n9)."""
    files = list(Path(cdir).glob("openmc_simulation_n*.h5"))
    return sorted(files, key=lambda f: int(re.search(r"_n(\d+)\.h5$", f.name).group(1)))


def solve_times(case) -> list[dict]:
    """One row per transport solve of the depletion under `case`, in solve
    order: chunk, file, wall_s (initialisation + simulation from the
    statepoint /runtime/ group, via campaign_timing.read_statepoint) and
    end (when the file was written). The chain solve of the operator is not
    inside these numbers: it is the gap between consecutive rows."""
    from campaign_timing import read_statepoint
    rows = []
    for cdir in sorted(Path(case).glob("dep_*")):
        for p in _sp_sorted(cdir):
            r = read_statepoint(p)
            rows.append(dict(chunk=cdir.name, file=p.name, wall_s=float(r["wall_s"]),
                             transport_s=float(r["transport_s"]), end=str(r["end"]),
                             runtime_ok=bool(r["runtime_ok"])))
    return rows


def statepoints_in_order(case) -> list[Path]:
    """Every per-solve statepoint under `case`, chunk by chunk, in the order
    the k history was built (one file per new state)."""
    out = []
    for cdir in sorted(Path(case).glob("dep_*")):
        out.extend(_sp_sorted(cdir))
    return out


def project_cost(times, n_solves_total, wall_total_s) -> dict:
    """From an --estimate run (fresh solve + depleted solves), project the
    wall time of a full cycle with `n_solves_total` transport solves.
    Overhead per solve is what the operator spent outside the statepoints."""
    if not times:
        return dict(ok=False)
    fresh = times[0]["wall_s"]
    dep = [t["wall_s"] for t in times[1:]] or [fresh]
    overhead = max(0.0, wall_total_s - sum(t["wall_s"] for t in times)) / len(times)
    per_dep = float(np.mean(dep)) + overhead
    total = fresh + overhead + (n_solves_total - 1) * per_dep
    return dict(ok=True, fresh_s=fresh, depleted_s=float(np.mean(dep)), overhead_s=overhead,
                per_solve_s=per_dep, n_solves=int(n_solves_total), projected_s=total,
                projected_h=total / 3600.0)


# ------------------------------------------------------------- selftest ----
def selftest():
    # bracket sigma: symmetric bracket, equal sd, slope 500 pcm per MWd/kg
    bu = [0.0, 4.0, 8.0]; k = [1.10, 1.09, 1.07]; sd = [1e-3, 2e-3, 2e-3]
    s, i, slope = bracket_sigma_efpd(bu, k, sd, 1.08, 10.0)
    assert i == 1 and abs(slope - 500.0) < 1e-9, (i, slope)
    # at the midpoint a = b = dB/(2 dk): sig_B = dB/(2dk) * sqrt(2) * sd
    exp_b = 4.0 / (2 * 0.02) * math.sqrt(2) * 2e-3
    assert abs(s - 1000.0 * exp_b / 10.0) < 1e-9, s
    assert bracket_sigma_efpd(bu, k, [None] * 3, 1.08, 10.0)[0] is None
    assert bracket_indices([0, 1, 2, 3], [1.1, 1.0, 1.05, 0.99], 1.02) == 2, "LAST crossing"

    # hump and c_max against the archived Campaign 9 arithmetic of C9-47
    pts = {"1000": 2655.231035559427, "2000": -3886.8565238274164}
    k_hist = [1.1157872707434433, 1.0950, 1.1150, 1.115885616565964, 1.10, 1.09, 1.085, 1.08]
    r = hump_and_cmax(k_hist, 1.0272765662069563, pts, 400.0)
    assert abs(r["c_max"] - 1405.869) < 0.01, r["c_max"]           # archived 1405.87
    assert r["hump_core_pcm"] == 0.0 and abs(r["l_ax"] - 1.08616) < 1e-4
    # a real hump moves c_max up by hump / worth
    r2 = hump_and_cmax([1.10, 1.09, 1.12, 1.10, 1.08], 1.0272765662069563, pts, 400.0)
    assert r2["hump_core_pcm"] > 400 and r2["c_max"] > r["c_max"]

    # fuel counting on a duck-typed geometry: 2 pins of A and 1 of B per
    # assembly, assembly placed 3 times, plus a non-fuel material
    class M:
        def __init__(self, name, nuc, i):
            self.name, self.nuclides, self.id = name, nuc, i
    class C:
        def __init__(self, fill): self.fill = fill
    class U:
        def __init__(self, cells): self.cells = {i: c for i, c in enumerate(cells)}
    class L:
        def __init__(self, univ): self.universes = univ
    A, B, W = M("A", [("U235", 1.0), ("O16", 2.0)], 1), M("B", [("U235", 1.0)], 2), M("W", [("H1", 2.0)], 3)
    pin = lambda m: U([C(m), C(W)])
    asm = U([C(L([[pin(A), pin(A)], [pin(B), pin(W)]]))])
    core = L([[asm, asm], [asm, W and U([C(W)])]])
    cnt = fuel_counts(core)
    assert cnt[A] == 6 and cnt[B] == 3 and W not in cnt, cnt

    class Geo:
        fuel_or = 0.4096
    rows = mark_depletable(None, [(core, 120.0)], Geo())
    va = 6 * math.pi * 0.4096 ** 2 * 120.0
    assert A.depletable and abs(A.volume - va) < 1e-9 and len(rows) == 2
    rows = mark_depletable(None, [(core, 120.0), (core, 60.0)], Geo())
    assert abs(A.volume - 1.5 * va) < 1e-9, "shared material sums its segment volumes"

    # a fuel material that is in model.materials but in no lattice position
    # (the unplaced base assembly) must lose its depletable flag
    stray = M("stray", [("U235", 1.0)], 99); stray.depletable = True; stray.volume = 1.0
    class Mod:
        materials = [A, B, W, stray]
    off = unmark_unused(Mod(), rows)
    assert [r["name"] for r in off] == ["stray"], off
    assert stray.depletable is False and stray.volume is None
    assert A.depletable and B.depletable, "placed materials keep their flag"
    assert unmark_unused(Mod(), rows) == [], "idempotent"
    pc = project_cost([dict(wall_s=10.0), dict(wall_s=50.0), dict(wall_s=54.0)], 8, 130.0)
    assert abs(pc["overhead_s"] - 16.0 / 3) < 1e-9 and abs(pc["projected_s"] - (10 + 16 / 3 + 7 * (52 + 16 / 3))) < 1e-9
    assert not project_cost([], 8, 0.0)["ok"]
    print("selftest OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest() if "--selftest" in sys.argv else
             print("usage: python dep_common.py --selftest"))
