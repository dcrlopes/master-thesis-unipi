#!/usr/bin/env python3
"""
make_c9_infill_step_figure.py -- one iteration of the active-learning loop of
Campaign 9 in the objective plane, for the presentation.

The Gaussian processes are refitted on the evaluations that preceded infill
batch k and NSGA-II (300 x 400, the campaign seed) is executed on them, as
c9_acquisition_replay.py does. The figure shows
  - the designs evaluated before the batch, at their OpenMC values
  - the 300 candidates of the final NSGA-II population, at their predicted values
  - the Pareto front predicted by the surrogate (non-dominated candidates)
  - the six designs of the batch, at their predicted values

The six designs are those the campaign evaluated. The replay is not bitwise
identical to the campaign (library versions), and its own six selections are
within 0.002 to 0.05 of them in the unit design box (c9_replay_fidelity.py).

No transport. About one minute.
Usage: python make_c9_infill_step_figure.py <prefix> [batch, 1 to 6, default 3] [measured] [full]
With measured, the OpenMC result of each of the six designs is added.
With full, the axes contain every evaluated design instead of the region of the front.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c9_acquisition_replay import _NoEvaluator, blocks_of, build
from reactor_optimization import ActiveLearningMOO

PREFIX = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MEASURED = "measured" in sys.argv[3:]
FULL = "full" in sys.argv[3:]                  # axes that contain every evaluated design
F_MAX = 1.65

ck = json.load(open("out_c9/optimization_checkpoint.json"))
raw, names, CN = ck["all_raw"], ck["design_variables"], ck["constraint_names"]
s, e = blocks_of(ck)[K - 1]
spec, cfg = build(ck, "asrun")
opt = ActiveLearningMOO(spec, _NoEvaluator(), cfg)
opt._seed_from_raw(raw[:s])
_, info = opt.propose(it=K - 1, verbose=True)
cand = info["cand"]
assert len(cand) == cfg.nsga_pop, len(cand)
P = np.atleast_2d(info["obj_sur"].predict(cand)[0])
batch = np.array([[float(r[n]) for n in names] for r in raw[s:e]])
B = np.atleast_2d(info["obj_sur"].predict(batch)[0])
# non-dominated candidates in the predicted objectives (both minimised)
nd = np.array([not np.any(np.all(P <= p, axis=1) & np.any(P < p, axis=1)) for p in P])
front = P[nd][np.argsort(P[nd][:, 0])]
feas = np.array([all(r[g] is not None and r[g] <= 0 for g in CN) for r in raw])
T = np.array([[r["peaking"], r["c_max"]] for r in raw])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
old = np.arange(len(raw)) < s
ax.scatter(*T[old & ~feas].T, s=26, facecolors="none", edgecolors="0.55", linewidths=0.8,
           label=f"Evaluated designs, infeasible ({int((old & ~feas).sum())})", zorder=2)
ax.scatter(*T[old & feas].T, s=30, color="0.45", edgecolors="k", linewidths=0.4,
           label=f"Evaluated designs, feasible ({int((old & feas).sum())})", zorder=3)
ax.scatter(*P.T, s=12, color="gold", edgecolors="#B8860B", linewidths=0.3,
           label=f"NSGA-II candidates, predicted ({len(P)})", zorder=6)
ax.plot(*front.T, color="#0072B2", lw=1.6, label="Pareto front predicted by the surrogate", zorder=5)
ax.scatter(*B.T, s=95, marker="*", color="crimson", edgecolors="k", linewidths=0.5,
           label=f"Designs of infill batch {K} (6)", zorder=7)
# the name of each design of the batch: in free space with a leader on the zoomed axes of batch 2,
# beside the star otherwise
LABEL = {2: {30: (1.540, 1660), 31: (1.488, 2450), 32: (1.497, 2200), 33: (1.543, 1120),
             34: (1.520, 1380), 35: (1.510, 1870)}}
ARROW = dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=0, shrinkB=3)
for j, i in enumerate(range(s, e)):
    if not FULL and K in LABEL:
        ax.annotate(f"C9-{i}", B[j], xytext=LABEL[K][i], textcoords="data", fontsize=7, color="crimson",
                    ha="center", va="center", arrowprops=ARROW, zorder=8)
    else:
        off = (5, 5) if not FULL or j % 2 == 0 else (-5, -10)       # alternate sides on the full axes
        ax.annotate(f"C9-{i}", B[j], xytext=off, textcoords="offset points", fontsize=7, color="crimson",
                    ha="left" if off[0] > 0 else "right", va="bottom" if off[1] > 0 else "top", zorder=8)
if MEASURED:
    M = T[s:e]
    fb = feas[s:e]
    for b, m in zip(B, M):
        ax.plot([b[0], m[0]], [b[1], m[1]], color="crimson", lw=0.6, ls=":", zorder=6)
    ax.scatter(*M[fb].T, s=40, marker="D", color="crimson", edgecolors="k", linewidths=0.4,
               label=f"OpenMC result, feasible ({int(fb.sum())})", zorder=6)
    ax.scatter(*M[~fb].T, s=40, marker="D", facecolors="white", edgecolors="crimson", linewidths=1.1,
               label=f"OpenMC result, infeasible ({int((~fb).sum())})", zorder=6)
ax.axvline(F_MAX, color="tab:orange", ls="--", lw=0.8)
ax.text(F_MAX - 0.002, 0.85, "Limit, 1.65", transform=ax.get_xaxis_transform(), fontsize=7,
        color="tab:orange", ha="right", va="center", rotation=90)
ax.set_xlabel(r"Radial peaking factor $F_{\Delta H}$ [-]")
ax.set_ylabel(r"Critical boron concentration $c_\mathrm{max}$ [ppm]")
ax.set_title(f"Campaign 9, infill batch {K}: surrogates fitted on {s} evaluations", fontsize=9)
XL, YL = (1.47, 1.67), (800, 3300)           # the region of the predicted front
if FULL:
    XL, YL = (1.47, T[old, 0].max() + 0.02), (-150, 6300)
else:
    inside = (T[old, 0] > XL[0]) & (T[old, 0] < XL[1]) & (T[old, 1] > YL[0]) & (T[old, 1] < YL[1])
    ax.text(0.01, 0.015, f"{int((~inside).sum())} of the {s} evaluated designs are outside the axes",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="left", va="bottom")
ax.set_xlim(*XL)
ax.set_ylim(*YL)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.tight_layout()
OUT = f"{PREFIX}_batch{K}" + ("_measured" if MEASURED else "") + ("_full" if FULL else "")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
print("written", OUT + ".pdf")
print("candidates", len(P), "| non-dominated", int(nd.sum()), "| margin-feasible", int(np.sum(info["acq"]["eligible"])))
# table: design variables, objectives and constraints, predicted against OpenMC
GM, GS = (np.atleast_2d(v) for v in info["con_sur"].predict(batch))
gp_cols = [c for c in spec.constraint_names if c not in spec.exact_constraints]
rows = ["| Design | " + " | ".join(names) + " | gd_pins used | F pred | F OpenMC | c_max pred | c_max OpenMC | "
        + " | ".join(f"{c} pred | {c} OpenMC" for c in gp_cols) + " | cycle length OpenMC [d] |",
        "|" + "---|" * (len(names) + 6 + 2 * len(gp_cols))]
for j, i in enumerate(range(s, e)):
    r = raw[i]
    g = " | ".join(f"{GM[j, spec.constraint_names.index(c)] * spec.g_scale(c):.3f} | {float(r[c]):.3f}" for c in gp_cols)
    rows.append(f"| C9-{i} | " + " | ".join(f"{float(r[n]):.3f}" for n in names) + f" | {r['gd_pins_used']} | "
                f"{B[j, 0]:.3f} | {r['peaking']:.3f} | {B[j, 1]:.0f} | {r['c_max']:.0f} | {g} | {r['cycle_length']:.1f} |")
TAB = f"figs_c9_acq/c9_infill_step_batch{K}_table.md"
open(TAB, "w", encoding="utf-8").write(chr(10).join(rows) + chr(10))
print(chr(10).join(rows))
print("written", TAB)
