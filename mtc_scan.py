#!/usr/bin/env python3
"""
mtc_scan.py -- moderator temperature coefficient versus soluble boron.

THE QUESTION
------------
Campaign 8 evaluated every design at a fixed 1000 ppm reference, but no
design is critical there. Design 47 needs about 2348 ppm, and the long-cycle
front members need 4000 to 4700 ppm. Whether those concentrations are
operable is decided by the moderator temperature coefficient, because
dissolved boron is expelled with the moderator when it expands and therefore
contributes POSITIVE reactivity on a temperature rise. Above some
concentration that positive term overcomes the negative moderation term and
the reactor stops being self-stabilising.

This script finds that crossing concentration for this lattice.

    MTC(c) = [ rho(T_hi, c) - rho(T_lo, c) ] / (T_hi - T_lo)      pcm/K

evaluated at several boron concentrations, then linearly interpolated to
MTC = 0. The answer is the boron ceiling to draw across the reformulated
front.

WHY NOT tier1_coefficients.py
-----------------------------
That script already computes a coefficient, and it has three limitations
this one removes.

  1. Its water densities are hardcoded and flagged VERIFY. Checked against
     IAPWS-IF97 at 15.5 MPa they read 0.767 / 0.712 / 0.641 g/cm3 at
     550 / 580 / 610 K against true values 0.76970 / 0.71187 / 0.62823.
     The 610 K value is 2.0 per cent high, which shrinks the modelled
     density swing by 11 per cent and biases the coefficient towards zero.
     Here every density comes from IAPWS-IF97 at run time.

  2. It solves the assembly with reflective boundaries, so leakage feedback
     is absent and the result is the lattice component only. Moderator
     density also changes leakage, so the core-level coefficient differs.
     This script defaults to the two-dimensional core.

  3. It measures the worth of the 1000 ppm already present. It does not
     sweep boron, so it cannot find a crossing.

WHY reactor_model.py IS NOT EDITED
----------------------------------
make_water hardcodes set_density("g/cm3", 0.72) and ignores temperature, so
changing mod_T alone changes only the thermal scattering kernel and the
moderator Doppler, never the density, which is the dominant term. Rather
than edit shared campaign code and risk perturbing archived results, this
script wraps make_water at module level, the same idiom tier1_coefficients
already uses and which hardware3d picks up because it calls rm.make_water
by attribute. Nothing on disk changes.

Note also that 0.72 is itself 1.1 per cent off the IAPWS value of 0.71187
at 580 K and 15.5 MPa, a small systematic bias carried by every archived
Campaign 8 result. Worth one sentence in the limitations.

BORON CONVENTION
----------------
make_water adds natural boron as an atom ratio to H2O, ppm*1e-6*(18.015/10.81),
so a fixed ppm is a fixed MASS FRACTION. When the water expands, water and
boron number densities fall together and the mass fraction is preserved.
That is the correct convention for this perturbation, and it is what makes
the boron term positive.

PRESSURE
--------
Pressure matters far less than temperature. Across 12 to 17 MPa the density
at 580 K varies by 1.6 per cent, while a 20 K step changes it by 6 to 7 per
cent. What pressure DOES decide is saturation. At 12.8 MPa the saturation
temperature is 602.8 K, so a perturbation reaching 600 K leaves only 2.8 K
of subcooling and is not defensible. The default pair 570 and 590 K is
subcooled at both 12.8 and 15.5 MPa and is used for that reason.

The script refuses to run if any requested state is within --min-subcool of
saturation.

USAGE (conda env openmc-env, from the repository root)
    python mtc_scan.py --selftest
    python mtc_scan.py --dry-run --pressure 15.5
    setsid nohup python -u mtc_scan.py --idx 47 --pressure 15.5 --threads 32 \
        --out mtc_47_155 > mtc_47_155.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

DEF_BORON = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
DEF_TLO, DEF_THI = 570.0, 590.0


# ------------------------------------------------------------ pure helpers --
def rho_pcm(k):
    return 1e5 * (k - 1.0) / k


def coefficient(k_lo, k_hi, t_lo, t_hi):
    """Reactivity coefficient in pcm per kelvin between two states."""
    return (rho_pcm(k_hi) - rho_pcm(k_lo)) / (t_hi - t_lo)


def coefficient_sd(k_lo, s_lo, k_hi, s_hi, t_lo, t_hi):
    """Propagated uncertainty. d(rho)/dk = 1e5/k**2."""
    a = 1e5 / k_hi ** 2 * s_hi
    b = 1e5 / k_lo ** 2 * s_lo
    return math.sqrt(a * a + b * b) / (t_hi - t_lo)


def crossing(points):
    """Linear interpolation to MTC = 0 over (boron, mtc) pairs, sorted.

    Returns the concentration in ppm, or None when the sign never changes.
    """
    p = sorted(points)
    for (c1, m1), (c2, m2) in zip(p, p[1:]):
        if m1 <= 0.0 < m2 or m2 <= 0.0 < m1:
            if m2 == m1:
                return c1
            return c1 + (0.0 - m1) * (c2 - c1) / (m2 - m1)
    return None


def selftest():
    # a negative coefficient when reactivity falls with temperature
    assert coefficient(1.0100, 1.0000, 570.0, 590.0) < 0
    m = coefficient(1.0100, 1.0000, 570.0, 590.0)
    assert abs(m - (rho_pcm(1.0) - rho_pcm(1.01)) / 20.0) < 1e-9

    # exact crossing, and a bracket that does not contain one
    assert abs(crossing([(1000, -5.0), (2000, 5.0)]) - 1500.0) < 1e-9
    assert abs(crossing([(1000, -3.0), (2000, -1.0), (3000, 1.0)]) - 2500.0) < 1e-9
    assert crossing([(1000, -5.0), (2000, -1.0)]) is None
    assert crossing([(1000, 1.0), (2000, 5.0)]) is None

    # unsorted input must still work
    assert abs(crossing([(2000, 5.0), (1000, -5.0)]) - 1500.0) < 1e-9

    # uncertainty propagation is symmetric and scales as expected
    s1 = coefficient_sd(1.0, 1e-4, 1.0, 1e-4, 570.0, 590.0)
    s2 = coefficient_sd(1.0, 2e-4, 1.0, 2e-4, 570.0, 590.0)
    assert abs(s2 / s1 - 2.0) < 1e-9

    try:
        from iapws import IAPWS97
    except ImportError:
        print("selftest OK (iapws not installed here, density check skipped)")
        return 0
    # IAPWS-IF97 region 1 check value from R7-97 table 5
    assert abs(1.0 / IAPWS97(T=300, P=3.0).rho / 0.100215168e-2 - 1) < 1e-6
    print("selftest OK")
    return 0


# ---------------------------------------------------------------- densities --
def density_table(temps, pressure, min_subcool):
    """IAPWS-IF97 liquid density in g/cm3, with a saturation guard."""
    from iapws import IAPWS97
    tsat = IAPWS97(P=pressure, x=0).T
    out = {}
    for t in temps:
        if tsat - t < min_subcool:
            sys.exit(f"FAIL: T = {t:.1f} K is only {tsat - t:.1f} K below "
                     f"saturation ({tsat:.1f} K) at {pressure} MPa. Lower the "
                     f"temperatures or raise the pressure.")
        out[t] = IAPWS97(T=t, P=pressure).rho / 1000.0
    return out, tsat


# ---------------------------------------------------------------- preflight --
def preflight(args):
    print("=" * 74)
    print(" PREFLIGHT")
    print("=" * 74)
    print(f"  host        : {socket.gethostname()}")
    print(f"  cwd         : {os.getcwd()}")
    env = os.environ.get("CONDA_DEFAULT_ENV", "NONE")
    print(f"  conda env   : {env}")
    if env != "openmc-env":
        sys.exit("FAIL: wrong conda environment. Run: conda activate openmc-env")
    try:
        import openmc
    except Exception as exc:
        sys.exit(f"FAIL: openmc not importable ({exc}). You are probably in (base).")
    print(f"  openmc      : {openmc.__version__}")
    if openmc.__version__ != "0.15.3":
        sys.exit("FAIL: openmc is not 0.15.3, this is not the campaign environment")
    try:
        from iapws import IAPWS97          # noqa: F401
        print("  iapws       : available")
    except ImportError:
        sys.exit("FAIL: the iapws package is missing. Run: pip install iapws")
    br = subprocess.run(["git", "branch", "--show-current"],
                        capture_output=True, text=True).stdout.strip()
    print(f"  branch      : {br}")
    if br != "main":
        sys.exit("FAIL: wrong branch. Run: git checkout main")
    xs = os.environ.get("OPENMC_CROSS_SECTIONS", "")
    if not xs or not Path(xs).is_file():
        sys.exit("FAIL: OPENMC_CROSS_SECTIONS is not set or the file is missing")
    print(f"  XS          : {xs}")
    print(f"  threads     : {args.threads or 'openmc default'} of {os.cpu_count()}")
    busy = subprocess.run(
        ["pgrep", "-af", r"python.*(run_optimization|confirm3d\.py|boron_worth|"
                         r"validate_ktarget|sweep_ktarget|sweep_refl)"],
        capture_output=True, text=True).stdout.strip()
    busy = "\n".join(l for l in busy.splitlines() if "mtc_scan" not in l)
    if busy:
        print("  RUNNING JOBS:\n" + busy)
        sys.exit("FAIL: a simulation is already running. Wait or stop it first.")
    print("  no simulation currently running")
    print("preflight OK\n")


# --------------------------------------------------------------------- main --
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--idx", type=int, default=47, help="archive index, default 47")
    ap.add_argument("--pressure", type=float, default=15.5,
                    help="primary pressure in MPa. 15.5 standard PWR, "
                         "12.8 NuScale-like")
    ap.add_argument("--t-lo", type=float, default=DEF_TLO)
    ap.add_argument("--t-hi", type=float, default=DEF_THI)
    ap.add_argument("--min-subcool", type=float, default=10.0,
                    help="refuse any state closer than this to saturation, K")
    ap.add_argument("--boron", default=",".join(f"{c:g}" for c in DEF_BORON),
                    help="comma-separated boron concentrations in ppm")
    ap.add_argument("--level", choices=("core3d", "core2d", "assembly"), default="core2d",
                    help="core2d carries radial leakage feedback, assembly does "
                         "not and gives the lattice component only")
    ap.add_argument("--doppler", action="store_true",
                    help="also scan fuel temperature at 600, 900 and 1200 K")
    ap.add_argument("--particles", type=int, default=60000)
    ap.add_argument("--batches", type=int, default=170)
    ap.add_argument("--inactive", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="mtc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    if a.selftest:
        return selftest()

    boron = [float(s) for s in a.boron.split(",")]
    temps = sorted({a.t_lo, a.t_hi} | ({600.0, 900.0, 1200.0} and set()))
    rho, tsat = density_table([a.t_lo, a.t_hi], a.pressure, a.min_subcool)

    n = a.seeds * len(boron) * 2 + (a.seeds * 3 if a.doppler else 0)
    print("=" * 74)
    print("Moderator temperature coefficient versus soluble boron")
    print("=" * 74)
    print(f"design            : {a.idx}")
    print(f"level             : {a.level}")
    print(f"pressure          : {a.pressure} MPa, saturation {tsat:.1f} K")
    for t in (a.t_lo, a.t_hi):
        print(f"  T = {t:.1f} K -> rho = {rho[t]:.5f} g/cm3 "
              f"(subcooling {tsat - t:.1f} K)")
    print(f"density change    : {100 * (rho[a.t_hi] / rho[a.t_lo] - 1):+.2f} per cent "
          f"over {a.t_hi - a.t_lo:.0f} K")
    print(f"boron points      : {boron} ppm")
    print(f"seeds             : {a.seeds}")
    print(f"{n} solves in total")
    print("=" * 74)
    if a.dry_run:
        return 0

    preflight(a)
    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)

    import openmc
    import reactor_model as rm

    # ---- wrap make_water so density follows temperature -------------------
    _orig = rm.make_water
    _rho = {"v": None}

    def _patched(boron_ppm, T):
        w = _orig(boron_ppm, T)
        if _rho["v"] is not None:
            w.set_density("g/cm3", _rho["v"])
        return w

    rm.make_water = _patched

    ck = json.loads(Path(a.checkpoint).read_text())
    r = ck["all_raw"][a.idx]
    design = {"enrich_inner": r["enrich_inner"], "enrich_outer": r["enrich_outer"],
              "gd_wt": r["gd_wt"], "pitch": r["pitch"],
              "refl_thick": r["refl_thick"], "gd_pins": r["gd_pins"]}
    print(f"design {a.idx}: enrich {r['enrich']:.3f} wt%, gd_wt {r['gd_wt']:.3f} wt%, "
          f"pins {r['gd_pins_used']:.0f}, refl {r['refl_thick']:.3f} cm\n")

    geo = rm.Geometry17x17()
    tr = dict(particles=a.particles, batches=a.batches, inactive=a.inactive)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    store = outdir / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}

    def solve(ppm, mod_t, dens, fuel_t, seed):
        key = (f"{a.level}_i{a.idx}_b{ppm:g}_T{mod_t:g}_f{fuel_t:g}_s{seed}"
               f"_{a.particles}x{a.batches}x{a.inactive}")
        if key in done:
            return done[key]
        op = rm.Operating(boron_ppm=ppm, mod_T=mod_t, fuel_T=fuel_t)
        _rho["v"] = dens
        if a.level == "core3d":
            import hardware3d as hw
            import zoning as zn
            m, _info = hw.build_model_3d_hw(
                design, op, geo,
                design_map=zn.evaluator_design_map(design),
                rodded_map=None, seed=seed, **tr)
        elif a.level == "assembly":
            m, _fc, _lat = rm.make_assembly_model(design, op, geo,
                                                  bc="reflective", **tr)
        else:
            m, _fc = rm.make_core_model(design, op, geo,
                                        refl_thick=design["refl_thick"],
                                        enforce_vessel=False, **tr)
        m.settings.seed = seed
        m.settings.temperature = {"method": "interpolation",
                                  "range": (294.0, 1500.0), "default": 900.0}
        t0 = time.time()
        sp = m.run(cwd=str(outdir / key), output=False, threads=a.threads)
        with openmc.StatePoint(sp) as s:
            done[key] = dict(k=float(s.keff.nominal_value),
                             sd=float(s.keff.std_dev), wall_s=time.time() - t0)
        _rho["v"] = None
        store.write_text(json.dumps(done, indent=1))
        return done[key]

    def avg(vals):
        import statistics
        k = statistics.fmean(v["k"] for v in vals)
        sd = math.sqrt(sum(v["sd"] ** 2 for v in vals)) / len(vals)
        return k, sd

    seeds = list(range(1, a.seeds + 1))
    fuel_ref = rm.Operating().fuel_T

    L = []
    P = L.append
    P("=" * 74)
    P(f"MODERATOR TEMPERATURE COEFFICIENT, design {a.idx}, {a.level}")
    P("=" * 74)
    P(f"  {a.t_lo:.0f} K -> {a.t_hi:.0f} K at {a.pressure} MPa, "
      f"rho {rho[a.t_lo]:.5f} -> {rho[a.t_hi]:.5f} g/cm3")
    P("")
    P(f"  {'ppm':>7} {'k(lo)':>9} {'k(hi)':>9} {'MTC':>9} {'sd':>7}  verdict")
    pts = []
    for ppm in boron:
        k_lo, s_lo = avg([solve(ppm, a.t_lo, rho[a.t_lo], fuel_ref, s) for s in seeds])
        k_hi, s_hi = avg([solve(ppm, a.t_hi, rho[a.t_hi], fuel_ref, s) for s in seeds])
        m = coefficient(k_lo, k_hi, a.t_lo, a.t_hi)
        sd = coefficient_sd(k_lo, s_lo, k_hi, s_hi, a.t_lo, a.t_hi)
        pts.append((ppm, m))
        tag = ("negative, acceptable" if m < -2 * sd else
               "POSITIVE, not acceptable" if m > 2 * sd else
               "not resolved from zero")
        P(f"  {ppm:>7.0f} {k_lo:>9.5f} {k_hi:>9.5f} {m:>+9.2f} {sd:>7.2f}  {tag}")
    P("")
    P("  MTC in pcm per K. A negative coefficient is the self-stabilising")
    P("  direction. The crossing is the boron ceiling for this lattice.")
    P("")
    c = crossing(pts)
    if c is None:
        allneg = all(m < 0 for _, m in pts)
        P(f"  NO CROSSING inside {min(boron):.0f} to {max(boron):.0f} ppm.")
        P("  The coefficient is negative throughout." if allneg else
          "  The coefficient is positive throughout.")
        P("  Extend the sweep with --boron before drawing any conclusion.")
    else:
        P(f"  CROSSING CONCENTRATION: {c:.0f} ppm")
        P("")
        P(f"  design 47 needs 2348 ppm  -> {'BELOW' if 2348 < c else 'ABOVE'} the ceiling")
        P(f"  design 13 needs 1701 ppm  -> {'BELOW' if 1701 < c else 'ABOVE'} the ceiling")
        P(f"  design  1 needs 4674 ppm  -> {'BELOW' if 4674 < c else 'ABOVE'} the ceiling")
        P("")
        P("  Linear interpolation between the two bracketing points. Add points")
        P("  near the crossing if the bracket is wide.")

    if a.doppler:
        P("")
        P("=" * 74)
        P("FUEL TEMPERATURE (DOPPLER) COEFFICIENT")
        P("=" * 74)
        ppm0 = boron[0]
        ks = {}
        for ft in (600.0, 900.0, 1200.0):
            ks[ft] = avg([solve(ppm0, a.t_lo, rho[a.t_lo], ft, s) for s in seeds])
        P(f"  at {ppm0:.0f} ppm and {a.t_lo:.0f} K moderator")
        for lo, hi in ((600.0, 900.0), (900.0, 1200.0)):
            m = coefficient(ks[lo][0], ks[hi][0], lo, hi)
            sd = coefficient_sd(ks[lo][0], ks[lo][1], ks[hi][0], ks[hi][1], lo, hi)
            P(f"  {lo:.0f} -> {hi:.0f} K : {m:+.2f} +/- {sd:.2f} pcm/K")
        P("")
        P("  Must be negative everywhere. A null Doppler means the library")
        P("  carries a single temperature and the interpolation is not working.")

    txt = "\n".join(L)
    print(txt)
    (outdir / "report.txt").write_text(txt + "\n", encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(dict(
        idx=a.idx, level=a.level, pressure_mpa=a.pressure, t_sat_k=tsat,
        t_lo=a.t_lo, t_hi=a.t_hi, rho_lo=rho[a.t_lo], rho_hi=rho[a.t_hi],
        mtc_pcm_per_k={f"{c0:g}": m for c0, m in pts},
        crossing_ppm=c, seeds=a.seeds, transport=tr,
        density_source="IAPWS-IF97 R7-97(2012) region 1, python package iapws",
    ), indent=1), encoding="utf-8")
    print(f"\nwrote {outdir}/report.txt and {outdir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
