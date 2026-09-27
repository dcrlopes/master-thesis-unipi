#!/usr/bin/env python3
"""
make_c9_gp_fronts_figure.py -- Figure of the fronts that the Campaign 9
surrogate predicts after each refit, against the evaluated front, with the
transport test of the last predicted front. Drawing only: it reads the
predicted fronts stored by c9_gp_predicted_fronts.py and the transport test
stored by c9_eval_predicted_front.py; no refit, no transport.

Reads  figs_c9_gp/gp_predicted_fronts.json
       c9_post/predicted_front_eval.json
       out_c9/optimization_checkpoint.json
Usage  python make_c9_gp_fronts_figure.py OUT.pdf [OUT.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

G = json.load(open("figs_c9_gp/gp_predicted_fronts.json"))
E = json.load(open("c9_post/predicted_front_eval.json"))
D = json.load(open("out_c9/optimization_checkpoint.json"))
A = D["all_raw"]
CN = D["constraint_names"]

F = np.array([[a["peaking"], a["c_max"]] for a in A])
feas = np.array([all(a[g] <= 0 for g in CN) for a in A])
front = G["evaluated_front"]
cuts = G["cuts"]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(8.2, 5.2))

# ---------------- (a) fronts, zoomed on the region of the fronts
cmap = plt.get_cmap("viridis")
cols = {n: cmap(0.05 + 0.85 * k / (len(cuts) - 1)) for k, n in enumerate(cuts)}
ax.scatter(F[~feas, 0], F[~feas, 1], s=22, marker="o", facecolors="none", edgecolors="#9a9a9a",
           linewidths=0.8, zorder=2)
ax.scatter(F[feas, 0], F[feas, 1], s=22, marker="o", color="#6b6b6b", zorder=3)
for n in cuts:
    Fp = np.array(G["predicted"][str(n)]["F"])
    o = np.argsort(Fp[:, 0])
    ax.plot(Fp[o, 0], Fp[o, 1], "-", color=cols[n], lw=1.6 if n == cuts[-1] else 1.0, zorder=4)
fo = sorted(front, key=lambda i: F[i, 0])
ax.plot(F[fo, 0], F[fo, 1], "o-", color="black", ms=5, lw=1.4, zorder=6)
LAB = {35: (-14, 10, "right"), 40: (-16, 2, "right"), 34: (4, -14, "left"), 44: (0, 12, "center"), 47: (6, -12, "left")}
for i in fo:
    dx, dy, ha = LAB.get(i, (6, 4, "left"))
    ax.annotate(f"C9-{i}", (F[i, 0], F[i, 1]), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                ha=ha, arrowprops=dict(arrowstyle="-", color="0.45", lw=0.5, shrinkA=0, shrinkB=2), zorder=8)
ax.set_xlim(1.478, 1.600)
ax.set_ylim(1000, 3450)
ax.set_xlabel(r"Core $F_{\Delta H}$ [-]")
ax.set_ylabel(r"Critical boron concentration $c_\mathrm{max}$ [ppm]")
handles = [Line2D([], [], color=cols[n], lw=1.6 if n == cuts[-1] else 1.0,
                  label=f"Predicted after {n} evaluations") for n in cuts]
handles += [Line2D([], [], marker="o", ls="-", color="black", ms=5, label="Evaluated front"),
            Line2D([], [], marker="o", ls="", color="#6b6b6b", ms=5, label="Evaluated, feasible"),
            Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#9a9a9a", ms=5,
                   label="Evaluated, infeasible")]
fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1])
