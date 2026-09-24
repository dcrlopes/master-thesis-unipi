#!/usr/bin/env python3
r"""
c9_corrected_postanalysis.py -- what the Campaign 9 search would have produced
if the cycle-length constraint had been the one the resolved axial burnup
imposes. Post-analysis only: no transport, no new campaign, no evaluation.

THE QUESTION
    Campaign 9 held the mission floor against the assembly-level cycle length.
    Resolving the axial burnup costs 15 to 25 per cent of it, so both front
    members miss the floor (Section 5.9). Raising the floor to the value the
    proxy must reach for the depleted core to hold five years changes which
    designs are feasible. This script re-scores the archive under that floor
    and then runs the campaign's own surrogate and NSGA-II search on the
    corrected archive, to see where the optimiser would have gone.

THE CORRECTION
    measured eight-layer cycle length where it exists (seven designs), else
    archive x (0.9025 - 0.0201 gd_wt), the regression of the repository
    (c9_axial_rescore.py: 71 d rms leave-one-out against 169 for a
    hump-aligned table and 276 for the burnup-indexed rho_A table). The
    regression is fitted over 2.6 to 6.9 wt% gadolinia and extrapolates
    outside it, which is flagged per design.

WHAT IT IS NOT
    The NSGA-II front it reports is a SURROGATE PREDICTION. No design in it
    has been evaluated by the transport code, and the thesis documents twice
    what a proxy does to a ranking near the optimum. It is read here as a
    direction, not as a result.

USAGE (repository root, no OpenMC needed)
    python c9_corrected_postanalysis.py
    python c9_corrected_postanalysis.py --floor 1826 --out figs_c9_corrected
"""
import argparse
import json
from pathlib import Path

import numpy as np

import reactor_optimization as ro
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

A, B = 0.9025, -0.0201                     # ratio = A + B * gd_wt
GD_LOW, GD_HIGH = 2.6, 6.9
MEASURED_3D = {47: 1573.3, 35: 1641.4, 27: 1956.1, 1: 2271.2,
               24: 1915.3, 16: 1908.7, 11: 1839.0}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="out_c9/optimization_checkpoint.json")
ap.add_argument("--floor", type=float, default=1826.0)
ap.add_argument("--f-max", type=float, default=1.65)
ap.add_argument("--n-doe", type=int, default=24)
ap.add_argument("--pop", type=int, default=300)
ap.add_argument("--gen", type=int, default=400)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="figs_c9_corrected")
a = ap.parse_args()

ck = json.loads(Path(a.ckpt).read_text())
raw = ck["all_raw"]
spec = ro.campaign9_problem(efpd_req=a.floor, f_max=a.f_max)
names = spec.design_space.names
cons = spec.constraint_names


def corrected(i, r):
    gd = float(r["gd_wt"])
    if i in MEASURED_3D:
        return MEASURED_3D[i], "measured 3D", False
    return float(r["cycle_length"]) * (A + B * gd), "projected", not (GD_LOW <= gd <= GD_HIGH)


def front(pts):
    """Non-dominated in (peaking, c_max), both minimised."""
    return [p for p in pts if not any(
        q is not p and q[1] <= p[1] and q[2] <= p[2] and (q[1] < p[1] or q[2] < p[2]) for q in pts)]


rows, X, F, G_arch, G_corr = [], [], [], [], []
for i, r in enumerate(raw):
    t_corr, src, extra = corrected(i, r)
    g = [float(r.get(c, 0.0)) / spec.g_scale(c) for c in cons]
    gc = list(g)
    gc[cons.index("g_efpd")] = (a.floor - t_corr) / spec.g_scale("g_efpd")
    X.append([float(r[n]) for n in names])
    F.append([float(r["peaking"]), float(r["c_max"])])
    G_arch.append(g); G_corr.append(gc)
    rows.append(dict(i=i, doe=i < a.n_doe, gd=float(r["gd_wt"]), enr=float(r["enrich"]),
                     refl=float(r["refl_thick"]), pins=float(r["gd_pins"]),
                     t=float(r["cycle_length"]), t_corr=t_corr, src=src, extrapolated=extra,
                     F=float(r["peaking"]), c=float(r["c_max"]),
                     feas_arch=max(g) <= 1e-9, feas_corr=max(gc) <= 1e-9))
X, F = np.array(X), np.array(F)
G_arch, G_corr = np.array(G_arch), np.array(G_corr)

fa = [(r["i"], r["F"], r["c"]) for r in rows if r["feas_arch"]]
fc = [(r["i"], r["F"], r["c"]) for r in rows if r["feas_corr"]]
front_a, front_c = front(fa), front(fc)
print(f"archive: {len(fa)} feasible, front {sorted(i for i, _, _ in front_a)}")
print(f"corrected floor {a.floor:.0f} d: {len(fc)} feasible, front {sorted(i for i, _, _ in front_c)}")
print(f"  of the corrected front, measured in 3D: "
      f"{sorted(i for i, _, _ in front_c if i in MEASURED_3D)}")
print(f"  designs whose corrected value extrapolates the regression: "
      f"{sorted(r['i'] for r in rows if r['extrapolated'] and r['feas_corr'])}")

results = {}
for tag, idx in (("all 60", list(range(len(rows)))), ("DOE only", list(range(a.n_doe)))):
    obj = ro.GPSurrogate().fit(X[idx], F[idx])
    con = ro.GPSurrogate().fit(X[idx], G_corr[idx])
    prob = ro._SurrogateProblem(spec, obj, con)
    res = minimize(prob, NSGA2(pop_size=a.pop, sampling=LHS()),
                   ("n_gen", a.gen), seed=a.seed, verbose=False)
    if res.X is None:
        print(f"\n{tag}: the surrogate search returned no feasible design")
        results[tag] = None
        continue
    Xp = np.atleast_2d(res.X); Fp = np.atleast_2d(res.F)
    print(f"\n{tag}: surrogate front of {len(Xp)} predicted designs (never evaluated)")
    for nm, col in zip(names, Xp.T):
        print(f"   {nm:11s} {col.min():7.3f} to {col.max():7.3f}   "
              f"(archive feasible: {min(r[nm[:4]] if False else 0 for r in rows):.0f})"
              if False else f"   {nm:11s} {col.min():7.3f} to {col.max():7.3f}")
    print(f"   predicted peaking {Fp[:, 0].min():.3f} to {Fp[:, 0].max():.3f}, "
          f"c_max {Fp[:, 1].min():.0f} to {Fp[:, 1].max():.0f} ppm")
    d = np.sqrt(((Xp[:, None, :] - X[None, :, :]) / (spec.design_space.xu - spec.design_space.xl)) ** 2).mean(2)
    print(f"   mean normalised distance to the nearest archive design: {d.min(1).mean():.3f}")
    results[tag] = dict(n=len(Xp), X=Xp.tolist(), F=Fp.tolist(),
                        var_ranges={nm: [float(c.min()), float(c.max())] for nm, c in zip(names, Xp.T)},
                        nearest_archive_distance=float(d.min(1).mean()))

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "c9_corrected_postanalysis.json").write_text(json.dumps(dict(
    floor=a.floor, correction=dict(ratio=[A, B], band=[GD_LOW, GD_HIGH], measured=MEASURED_3D),
    designs=rows, front_archive=[i for i, _, _ in front_a], front_corrected=[i for i, _, _ in front_c],
    surrogate_fronts=results), indent=1))
print(f"\nwrote {out}/c9_corrected_postanalysis.json")
