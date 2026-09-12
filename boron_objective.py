#!/usr/bin/env python3
r"""
boron_objective.py
==================
The arithmetic of the Campaign 9 objective, kept free of OpenMC so it can
be tested on any machine and reused by the post-analysis.

PHYSICS
-------
Every archived core eigenvalue is at the reference concentration c_ref
(1000 ppm). Campaign 9 measures the unrodded core at one or two further
concentrations, chosen adaptively, and represents the core reactivity as
a piecewise-linear function of the concentration through the measured
points, extrapolated with the nearest segment beyond them. The critical
concentration is the root of that function.

    c_BOL      root of rho(c) = 0                 beginning of life
    c_max      root of rho(c) = -H_core           operating maximum
    c_max_op   root of rho(c) = -H_core_op        operating maximum with
                                                  the xenon credit taken

where H_core is the gadolinium hump of the depleting assembly carried to
the core through the axial factor, in pcm of reactivity:

    H_asm      = rho(k_peak) - rho(k_hist[0])     xenon-free reference
    H_asm_op   = rho(k_peak) - rho(k_hist[0])     same quantity, no floor
    H_core     = max(L_ax * H_asm, 0) with a noise floor
    H_core_op  = L_ax * H_asm_op                  may be negative

    k_hist[0]  xenon-free beginning of life (the state of the core solve)
    k_peak     maximum of k_hist[1:], the operating trajectory
    L_ax       k_hist[0] / k_core_BOL, the leakage factor

The core solve that gives rho(c) is xenon-free, so H_asm is measured
against the xenon-free point for consistency. For a design with no hump
H_asm is negative (the xenon step), the floor sets H_core to zero, and
c_max equals c_BOL: this is the conservative reading and the campaign
objective. c_max_op drops the floor, so the xenon worth reduces the
requirement, and is recorded for re-scoring.

The differential worth w_B is the local slope of the measured curve at
the root, in pcm per ppm, so it is a local derivative where it is used
and not a secant carried from the reference concentration.

ADAPTIVE MEASUREMENT (one or two extra core solves)
---------------------------------------------------
    rho(c_ref) > 0   measure c_ref + step (2000 ppm)
    rho(c_ref) <= 0  measure 0 ppm instead (the low branch)
    any root not bracketed and top not yet measured
                     measure the top (3000 ppm), once
No further solves. A root beyond the top is extrapolated and flagged.

USAGE
    python boron_objective.py --selftest
"""
from __future__ import annotations

import math
import sys

PPM_REF = 1000.0
PPM_STEP = 2000.0
PPM_TOP = 3000.0
PPM_LOW = 0.0
HUMP_NOISE_PCM = 400.0
PPM_FLOOR = 0.0          # a concentration cannot be negative
PPM_CLIP = 6000.0        # BORON-CLIP, twice the top measured point: beyond
                         # the measured support the linear extrapolation is
                         # fiction, exactly as EFPD-CLIP treats cycle length
                         # beyond the depletion ceiling. Ranking is preserved
                         # because every clipped design is far above the
                         # measured moderator-coefficient ceiling anyway.


# ------------------------------------------------------------- reactivity --
def rho_pcm(k: float) -> float:
    """Reactivity in pcm from a multiplication factor."""
    return 1.0e5 * (k - 1.0) / k


def margin_pcm(k: float) -> float:
    """Subcriticality in pcm of reactivity, positive when k < 1."""
    return 1.0e5 * (1.0 - k) / k


def screen_pcm_from_dk(margin_dk: float) -> float:
    """The campaign control screen is stated in dk (k <= 1 - margin_dk).
    Its value in reactivity units, so the operating-maximum screen agrees
    with the beginning-of-life one when the hump is zero."""
    return margin_pcm(1.0 - margin_dk)


# ------------------------------------------------------------------- hump --
def hump_from_history(k_hist, k_core_bol: float, noise_pcm: float = HUMP_NOISE_PCM):
    """Hump of the depleting assembly and its core-level image.

    Returns a dict with
      k_peak, bu_index      the operating maximum and its index in k_hist
      hump_asm_pcm          rho(k_peak) - rho(k_hist[0]), may be negative
      hump_xe_pcm           rho(k_peak) - rho(k_hist[1]), against the first
                            operating point, None when the history is short
      l_ax                  k_hist[0] / k_core_bol
      hump_core_pcm         max(l_ax * hump_asm_pcm, 0), zero below noise
      hump_core_op_pcm      l_ax * hump_asm_pcm, no floor
    """
    if not k_hist:
        raise ValueError("empty k history")
    k0 = float(k_hist[0])
    k_op = [float(v) for v in k_hist[1:]] or [k0]
    i = max(range(len(k_op)), key=lambda j: k_op[j])
    k_peak = k_op[i]
    h_asm = rho_pcm(k_peak) - rho_pcm(k0)
    h_xe = (rho_pcm(k_peak) - rho_pcm(float(k_hist[1]))) if len(k_hist) > 1 else None
    l_ax = k0 / float(k_core_bol)
    h_core_op = l_ax * h_asm
    h_core = h_core_op if h_core_op > noise_pcm else 0.0
    return dict(k_peak=k_peak, bu_index=i + 1, hump_asm_pcm=h_asm,
                hump_xe_pcm=h_xe, l_ax=l_ax, hump_core_pcm=h_core,
                hump_core_op_pcm=h_core_op)


# ------------------------------------------------- piecewise reactivity(c) --
def _sorted_points(points):
    pts = sorted((float(c), float(r)) for c, r in points.items())
    if len(pts) < 2:
        raise ValueError("need at least two concentrations")
    return pts


def root_ppm(points: dict, target_pcm: float) -> tuple[float, float, str]:
    """Concentration at which the piecewise-linear rho(c) equals target.

    points  {ppm: rho_pcm}, at least two entries.
    Returns (c, w_local, status) with w_local the local worth in pcm/ppm
    (positive) of the segment used, and status one of
      "interp"      target bracketed by measured points
      "extrap_hi"   target below the last measured rho, extrapolated up
      "extrap_lo"   target above the first measured rho, extrapolated down
    """
    pts = _sorted_points(points)
    # rho decreases with c, so walk the segments
    for (c1, r1), (c2, r2) in zip(pts[:-1], pts[1:]):
        if r2 == r1:
            continue
        if (r1 >= target_pcm >= r2) or (r2 >= target_pcm >= r1):
            w = (r1 - r2) / (c2 - c1)
            return c1 + (r1 - target_pcm) / w, w, "interp"
    if target_pcm < pts[-1][1]:                      # more boron than measured
        (c1, r1), (c2, r2) = pts[-2], pts[-1]
        w = (r1 - r2) / (c2 - c1)
        return c2 + (r2 - target_pcm) / w, w, "extrap_hi"
    (c1, r1), (c2, r2) = pts[0], pts[1]              # less boron than measured
    w = (r1 - r2) / (c2 - c1)
    return c1 - (target_pcm - r1) / w, w, "extrap_lo"


def bracketed(points: dict, target_pcm: float) -> bool:
    pts = _sorted_points(points)
    lo = min(r for _, r in pts)
    hi = max(r for _, r in pts)
    return lo <= target_pcm <= hi


def next_concentration(points: dict, targets, top=PPM_TOP, step=PPM_STEP,
                       low=PPM_LOW, ref=PPM_REF):
    """Which concentration to measure next, or None when finished.

    points   {ppm: rho_pcm} measured so far, must contain ref
    targets  the rho values whose roots the objective needs
    """
    if ref not in points:
        raise ValueError("the reference concentration must be measured first")
    if len(points) == 1:
        return low if points[ref] <= 0.0 else step
    if all(bracketed(points, t) for t in targets):
        return None
    if max(points) < top and points[ref] > 0.0:
        return top
    return None


# ------------------------------------------------------------- objective --
def clamp_ppm(c: float, status: str, floor: float = PPM_FLOOR,
              clip: float | None = PPM_CLIP) -> tuple[float, str]:
    """Keep the objective inside the range where it means something.

    Below the floor: the core cannot be made critical at any concentration,
    so the extrapolated root is negative. Such a design is a dud and is
    already infeasible on g_kmin, but the objective is MINIMISED, so an
    unclamped negative value would be the best value in the archive and
    would steer the surrogate into a region where nothing operates.

    Above the clip: the root is extrapolated far beyond the measured
    points. Self-shielding makes the true value higher still, so the
    ranking survives, but the magnitude is fiction and it stretches the
    surrogate's length scale. Same treatment as EFPD-CLIP.
    """
    if c < floor:
        return floor, "below_floor"
    if clip is not None and c > clip:
        return clip, "above_clip"
    return c, status


def boron_objective(points: dict, hump: dict, clip: float | None = PPM_CLIP) -> dict:
    """c_BOL, c_max and c_max_op with their local worths and status flags."""
    out = {}
    c, w, s = root_ppm(points, 0.0)
    c, s = clamp_ppm(c, s, clip=clip)
    out.update(c_bol_ppm=c, w_b_bol_pcm_per_ppm=w, c_bol_status=s)
    c, w, s = root_ppm(points, -hump["hump_core_pcm"])
    c, s = clamp_ppm(c, s, clip=clip)
    out.update(c_max_ppm=c, w_b_max_pcm_per_ppm=w, c_max_status=s)
    c, w, s = root_ppm(points, -hump["hump_core_op_pcm"])
    c, s = clamp_ppm(c, s, clip=clip)
    out.update(c_max_op_ppm=c, c_max_op_status=s)
    out["boron_points"] = {f"{k:g}": v for k, v in sorted(points.items())}
    return out


def ctrl_margin_at_peak(k_allre: float, hump: dict, margin_dk: float) -> dict:
    """Four-bank screen moved from beginning of life to the operating
    maximum. g <= 0 is feasible, in k-units like the campaign g_ctrl."""
    m_bol = margin_pcm(k_allre)
    m_peak = m_bol - hump["hump_core_pcm"]
    screen = screen_pcm_from_dk(margin_dk)
    return dict(ctrl_margin_bol_pcm=m_bol, ctrl_margin_peak_pcm=m_peak,
                ctrl_screen_pcm=screen,
                g_ctrl_peak=(screen - m_peak) * 1.0e-5)


# ---------------------------------------------------------------- selftest --
def selftest():
    # reactivity identities
    assert abs(margin_pcm(0.99) - 1010.1) < 0.05
    assert abs(screen_pcm_from_dk(0.01) - 1010.1) < 0.05
    assert abs(rho_pcm(1.10) - 9090.9) < 0.1

    # hump: design-47-like history, xenon step only, no hump
    h = hump_from_history([1.1879, 1.1700, 1.1650, 1.1500], 1.0981)
    assert h["hump_asm_pcm"] < 0 and h["hump_core_pcm"] == 0.0
    assert h["hump_core_op_pcm"] < 0 and abs(h["l_ax"] - 1.0818) < 1e-3
    # hump: design-31-like, rises above the xenon-free point
    h31 = hump_from_history([1.1236, 1.1100, 1.1500, 1.1846, 1.1700], 1.0339)
    assert h31["k_peak"] == 1.1846 and h31["bu_index"] == 3
    assert h31["hump_core_pcm"] > 4000
    # noise floor
    hn = hump_from_history([1.100, 1.090, 1.1003], 1.02, noise_pcm=400.0)
    assert hn["hump_core_pcm"] == 0.0 and hn["hump_core_op_pcm"] > 0.0

    # roots on a measured curve (design 47 report, 570 K points)
    pts = {1000: 9111.0, 2000: 2330.0, 3000: -4184.0}
    c, w, s = root_ppm(pts, 0.0)
    assert s == "interp" and 2300 < c < 2400 and abs(w - 6.514) < 0.01
    c_hi, w_hi, s_hi = root_ppm(pts, -6000.0)
    assert s_hi == "extrap_hi" and c_hi > 3000
    c_lo, w_lo, s_lo = root_ppm(pts, 12000.0)
    assert s_lo == "extrap_lo" and c_lo < 1000

    # adaptive selection, high branch
    p = {1000.0: 9111.0}
    assert next_concentration(p, [0.0]) == 2000.0
    p[2000.0] = 2330.0
    assert next_concentration(p, [0.0]) == 3000.0          # root not bracketed
    p[3000.0] = -4184.0
    assert next_concentration(p, [0.0, -1500.0]) is None
    # adaptive selection, bracketed after one solve
    q = {1000.0: 5000.0, 2000.0: -1500.0}
    assert next_concentration(q, [0.0, -1000.0]) is None
    # adaptive selection, a hump pushes the c_max root past 2000
    assert next_concentration(q, [0.0, -2000.0]) == 3000.0
    # low branch
    r = {1000.0: -800.0}
    assert next_concentration(r, [0.0]) == 0.0
    r[0.0] = 5400.0
    assert next_concentration(r, [0.0]) is None
    # low branch never asks for the top even when unbracketed
    r2 = {1000.0: -800.0, 0.0: -200.0}
    assert next_concentration(r2, [0.0]) is None

    # clamp: a dud reports the floor, not a negative best-in-archive value
    cf, sf = clamp_ppm(-1044.0, "extrap_lo")
    assert cf == 0.0 and sf == "below_floor"
    cc, sc = clamp_ppm(12196.0, "extrap_hi")
    assert cc == 6000.0 and sc == "above_clip"
    assert clamp_ppm(2348.0, "interp") == (2348.0, "interp")
    assert clamp_ppm(12196.0, "extrap_hi", clip=None) == (12196.0, "extrap_hi")
    dud = boron_objective({1000.0: -800.0, 0.0: -200.0}, h)
    assert dud["c_bol_ppm"] == 0.0 and dud["c_bol_status"] == "below_floor"
    far = boron_objective({1000.0: 40000.0, 2000.0: 33000.0, 3000.0: 26500.0}, h)
    assert far["c_bol_ppm"] == 6000.0 and far["c_bol_status"] == "above_clip"

    # objective assembly
    obj = boron_objective(pts, h)
    assert abs(obj["c_max_ppm"] - obj["c_bol_ppm"]) < 1e-9      # no hump
    assert obj["c_max_op_ppm"] < obj["c_bol_ppm"]                # xenon credit
    obj31 = boron_objective(pts, h31)
    assert obj31["c_max_ppm"] > obj31["c_bol_ppm"] + 500

    # control screen at the operating maximum
    g = ctrl_margin_at_peak(0.9248, h, 0.01)
    assert g["g_ctrl_peak"] < 0 and abs(g["ctrl_margin_peak_pcm"] - g["ctrl_margin_bol_pcm"]) < 1e-9
    g31 = ctrl_margin_at_peak(0.9880, h31, 0.01)
    assert g31["g_ctrl_peak"] > 0                                  # fails at the hump
    print("boron_objective selftest OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
