#!/usr/bin/env python3
"""
make_c9_pareto_blocks_figure.py -- objective plane of the Campaign 9 archive
in the style of the Campaign 8 figure (c8_pareto_figure.py): feasible
designs as filled markers coloured by block, infeasible designs as open markers,
the Pareto front joined by a step line, the two-bank designs ringed, the
MTC boron limits of C9-47 and the peaking bound as reference lines.

Reads out_c9/optimization_checkpoint.json and c9_post/c9_front.json.
Drawing only, no transport.
Usage: python make_c9_pareto_blocks_figure.py <out.pdf> [out.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = json.load(open("out_c9/optimization_checkpoint.json"))
M = json.load(open("c9_post/c9_front.json"))
A = D["all_raw"]
CN = D["constraint_names"]
feas = [i for i, a in enumerate(A) if all(a[g] <= 0 for g in CN)]
front = M["front"]
two = set(M["two_bank"])
assert set(feas) == set(M["feasible"]), "feasible set differs from the manifest"

EDGES = [0, 24, 30, 36, 42, 48, 54, 60]
NAMES = ["DOE (block 1)"] + [f"Infill {k}" for k in range(1, 7)]
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999933"]
MK = ["o", "s", "D", "^", "v", "P", "X"]
LIMIT_128, LIMIT_155, F_MAX, TRUNC = 2897, 3266, 1.65, 6000


def block(i):
    return next(b for b in range(7) if EDGES[b] <= i < EDGES[b + 1])


F = np.array([a["peaking"] for a in A])
C = np.array([a["c_max"] for a in A])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8.0, 5.2))
ax.axhspan(TRUNC - 150, TRUNC + 150, color="0.9", zorder=0)
ax.text(1.415, TRUNC + 200, f"Truncation, {TRUNC} ppm", fontsize=7.5, color="0.35", va="bottom")
for i in range(len(A)):
    b = block(i)
    if i in feas:
        ax.scatter(F[i], C[i], marker=MK[b], s=44, color=COL[b], edgecolors="k", linewidths=0.4, zorder=4)
    else:
        ax.scatter(F[i], C[i], marker=MK[b], s=36, facecolors="none", edgecolors=COL[b], linewidths=0.9, alpha=0.9, zorder=3)
for i in two:
    ax.scatter(F[i], C[i], marker="o", s=150, facecolors="none", edgecolors="#6A3D9A", linewidths=1.3, zorder=5)
fx = np.array([F[i] for i in front]); fy = np.array([C[i] for i in front])
o = np.argsort(fx)
ax.step(fx[o], fy[o], where="post", color="crimson", lw=1.4, zorder=2)
LAB = {47: (0, -11, "center", "top"), 44: (0, 10, "center", "bottom"), 34: (-10, -6, "right", "top"),
       40: (10, 5, "left", "bottom"), 35: (-10, 6, "right", "bottom")}
for i in front:
    dx, dy, ha, va = LAB[i]
    ax.annotate(f"C9-{i}", (F[i], C[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.5)
ax.axhline(LIMIT_128, color="tab:red", ls="--", lw=0.8)
ax.axhline(LIMIT_155, color="tab:red", ls=":", lw=0.8)
ax.text(1.735, LIMIT_128 + 60, "MTC boron limit of C9-47, 12.8 MPa", fontsize=7, color="tab:red", ha="right")
ax.text(1.735, LIMIT_155 + 60, "15.5 MPa", fontsize=7, color="tab:red", ha="right")
ax.axvline(F_MAX, color="tab:orange", ls="--", lw=0.8)
ax.text(F_MAX + 0.004, 5200, r"$F_{\Delta H}$ limit, 1.65", rotation=90, fontsize=7.5, color="tab:orange", va="top")
ax.set_xlim(1.41, 1.74)
ax.set_ylim(0, 6400)
ax.set_xlabel(r"Core $F_{\Delta H}$ [-]")
ax.set_ylabel(r"Critical boron at the operating maximum $c_\mathrm{max}$ [ppm]")
ax.grid(alpha=0.3)
nf = [sum(1 for i in feas if block(i) == b) for b in range(7)]
handles = [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="0.4", markersize=6, label="Infeasible (38), open marker of its block")]
handles += [Line2D([], [], marker=MK[b], ls="", color=COL[b], markeredgecolor="k", markersize=6,
                   label=f"{NAMES[b]}, feasible ({nf[b]})") for b in range(7)]
handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#6A3D9A",
                   markersize=10, label=f"Two-bank controllable ({len(two)})"),
            Line2D([], [], color="crimson", lw=1.4, label=f"Pareto front ({len(front)} designs)")]
ax.legend(handles=handles, loc="upper right", fontsize=7.2, frameon=True, framealpha=0.95)
fig.tight_layout()
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1], "feasible per block", nf, "two-bank", sorted(two))
