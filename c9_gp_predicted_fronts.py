#!/usr/bin/env python3
r"""
c9_gp_predicted_fronts.py -- the Pareto front the surrogate predicts, block by
block, against the front the transport code measured. Post-analysis only:
no transport, no new evaluation.

WHAT IT DOES
    The checkpoint stores no fitted Gaussian process, and it does not need
    to: the GP is a deterministic function of the archive. For each infill
    block the campaign's own surrogate (reactor_optimization.GPSurrogate, one
    Matern 5/2 process with ARD and a white-noise term per output) is refitted
    on the evaluations that preceded the block, exactly as the loop did when
    it chose that block, and the campaign's own search (NSGA-II, 300 x 400)
    is run on it. The non-dominated set of the predictions is the front the
    loop believed in at that moment. The final refit, on the whole archive,
    gives the front the campaign ends believing in.

WHAT IT IS NOT
    No design on a predicted front has been evaluated by the transport code.
    The thesis documents twice what a proxy does to a ranking near the
    optimum. The figure is read as the surrogate's belief, not as a result.

USAGE (repository root, pymoo and scikit-learn, no OpenMC)
    python c9_gp_predicted_fronts.py
    python c9_gp_predicted_fronts.py --ckpt out_c9f/optimization_checkpoint.json --n-doe 60 --floor 2270 --out figs_c9f_gp
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import reactor_optimization as ro
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="out_c9/optimization_checkpoint.json")
ap.add_argument("--n-doe", type=int, default=24, help="evaluations before the first infill block")
ap.add_argument("--n-infill", type=int, default=6)
ap.add_argument("--floor", type=float, default=1826.0)
ap.add_argument("--f-max", type=float, default=1.65)
ap.add_argument("--ceil", type=float, default=2763.0, help="MTC boron limit, drawn as a line")
ap.add_argument("--pop", type=int, default=300)
ap.add_argument("--gen", type=int, default=400)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--label", default="C9")
ap.add_argument("--out", default="figs_c9_gp")
a = ap.parse_args()

ck = json.loads(Path(a.ckpt).read_text())
raw = ck["all_raw"]
spec = ro.campaign9_problem(efpd_req=a.floor, f_max=a.f_max)
names, cons = spec.design_space.names, spec.constraint_names

X = np.array([[float(r[n]) for n in names] for r in raw])
F = np.array([[float(r["peaking"]), float(r["c_max"])] for r in raw])
G = np.array([[float(r.get(c, 0.0)) / spec.g_scale(c) for c in cons] for r in raw])
feas = (G.max(1) <= 1e-9)


def nd(pts):
    """Indices of the non-dominated rows of pts (both columns minimised)."""
    keep = []
    for i, p in enumerate(pts):
        if not any(j != i and q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])
                   for j, q in enumerate(pts)):
            keep.append(i)
    return sorted(keep, key=lambda i: pts[i][0])


def predicted_front(n):
    obj = ro.GPSurrogate().fit(X[:n], F[:n])
    con = ro.GPSurrogate().fit(X[:n], G[:n])
    res = minimize(ro._SurrogateProblem(spec, obj, con), NSGA2(pop_size=a.pop, sampling=LHS()),
                   ("n_gen", a.gen), seed=a.seed, verbose=False)
    if res.F is None:
        return np.empty((0, 2)), np.empty((0, len(names)))
    Fp, Xp = np.atleast_2d(res.F), np.atleast_2d(res.X)
    o = np.argsort(Fp[:, 0])
    return Fp[o], Xp[o]


cuts = list(range(a.n_doe, len(raw) + 1, a.n_infill))
if cuts[-1] != len(raw):
    cuts.append(len(raw))
fronts = {}
for n in cuts:
    Fp, Xp = predicted_front(n)
    fronts[n] = (Fp, Xp)
    print(f"GP on the first {n:3d} evaluations: predicted front of {len(Fp):3d} designs, "
          f"peaking {Fp[:,0].min():.3f} to {Fp[:,0].max():.3f}, c_max {Fp[:,1].min():.0f} to {Fp[:,1].max():.0f} ppm"
          if len(Fp) else f"GP on the first {n:3d} evaluations: no feasible prediction")

# how far each predicted front moved from the previous one (mean nearest distance, normalised)
lo = np.array([F[:, 0].min(), F[:, 1].min()]); hi = np.array([F[:, 0].max(), F[:, 1].max()])
moves = []
for k in range(1, len(cuts)):
    A, B = fronts[cuts[k - 1]][0], fronts[cuts[k]][0]
    if len(A) and len(B):
        An, Bn = (A - lo) / (hi - lo), (B - lo) / (hi - lo)
        d = np.sqrt(((Bn[:, None, :] - An[None, :, :]) ** 2).sum(2)).min(1).mean()
        moves.append((cuts[k], float(d)))
        print(f"  belief moved by {d:.3f} (normalised) between {cuts[k-1]} and {cuts[k]} evaluations")

ev_front = nd(F[feas]) if feas.any() else []
ev_idx = np.where(feas)[0][ev_front] if len(ev_front) else []
seed_feas = np.where(feas[:a.n_doe])[0]
seed_front = seed_feas[nd(F[seed_feas])] if len(seed_feas) else []
print(f"\nevaluated front ({a.label}): {[f'{a.label}-{i}' for i in ev_idx]}")
print(f"front of the design of experiments alone: {[f'{a.label}-{i}' for i in seed_front]}")

# ---------------------------------------------------------------- figure --
# House style: crosses for infeasible, circles for feasible, Okabe-Ito colours,
# and only three predicted fronts (first, middle, last) so the panel stays readable.
C_FIRST, C_MID, C_LAST, C_GREY, C_EVAL = "#999999", "#D55E00", "#0072B2", "#999999", "black"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "pdf.fonttype": 42,
})
shown = [cuts[0], cuts[len(cuts) // 2], cuts[-1]]
styles = {shown[0]: (C_FIRST, "--", 1.0), shown[1]: (C_MID, "-.", 1.1), shown[2]: (C_LAST, "-", 1.5)}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
ax.scatter(F[~feas, 0], F[~feas, 1], s=20, marker="x", color=C_GREY, alpha=0.6, label="Evaluated, infeasible")
ax.scatter(F[feas, 0], F[feas, 1], s=26, marker="o", facecolors="none", edgecolors="#333333",
           linewidths=1.0, label="Evaluated, feasible")
if len(ev_idx):
    ax.plot(F[ev_idx, 0], F[ev_idx, 1], "o-", color=C_EVAL, ms=5, lw=1.3, label="Evaluated front")
for n in shown:
    Fp = fronts[n][0]
    if not len(Fp):
        continue
    c, ls, lw = styles[n]
    ax.plot(Fp[:, 0], Fp[:, 1], ls, color=c, lw=lw, label=f"Predicted after {n} evaluations")
ax.axhline(a.ceil, color="#B23A48", ls=":", lw=1, label=f"MTC boron limit, {a.ceil:.0f} ppm")
ax.set_xlabel("Hot channel factor $F_{\Delta H}$")
ax.set_ylabel("Critical boron concentration $c_\mathrm{max}$ [ppm]")
ax.set_title(f"{a.label}: fronts predicted by the surrogate against the evaluated front")
ax.legend(fontsize=7, loc="upper right")
# The boron axis is clipped: a few infeasible designs reach several thousand ppm
# and would squash the region where the fronts lie.
y_top = max(a.ceil * 1.2, float(np.percentile(F[feas, 1], 95)) * 1.1)
n_out = int((F[:, 1] > y_top).sum())
ax.set_ylim(max(0.0, float(F[:, 1].min()) - 100), y_top)
if n_out:
    ax.text(0.02, 0.02, f"{n_out} evaluated designs above {y_top:.0f} ppm are outside the axis",
            transform=ax.transAxes, fontsize=6.5, color="#555555")

if moves:
    bx.plot([m[0] for m in moves], [m[1] for m in moves], "o-", color=C_LAST, ms=5)
    bx.set_xlabel("Evaluations in the archive")
    bx.set_ylabel("Movement of the predicted front, normalised")
    bx.set_title("Convergence of the surrogate belief")
    bx.set_ylim(bottom=0)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "gp_predicted_fronts.pdf"); fig.savefig(out / "gp_predicted_fronts.png")
(out / "gp_predicted_fronts.json").write_text(json.dumps({
    "ckpt": a.ckpt, "floor": a.floor, "cuts": cuts,
    "predicted": {str(n): {"F": fronts[n][0].tolist(), "X": fronts[n][1].tolist()} for n in cuts},
    "moves": moves, "evaluated_front": [int(i) for i in ev_idx], "seed_front": [int(i) for i in seed_front],
    "caveat": "no predicted design has been evaluated by the transport code"}, indent=1))
print(f"\nwrote {out}/gp_predicted_fronts.pdf, .png, .json")
