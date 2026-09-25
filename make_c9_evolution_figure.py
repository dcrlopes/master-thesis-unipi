#!/usr/bin/env python3
"""
make_c9_evolution_figure.py -- the two objectives of Campaign 9 along the
evaluation order: (a) the radial peaking factor and (b) the critical boron
concentration at the operating maximum of every evaluated design, with the
design of experiments and the six infill blocks distinguished, the feasible
designs filled, and the five front members labelled.

Reads out_c9/optimization_checkpoint.json and c9_post/c9_front.json.
Drawing only, no transport.
Usage: python make_c9_evolution_figure.py <out.pdf> [out.png]
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
feas = {i for i, a in enumerate(A) if all(a[g] <= 0 for g in CN)}
front = M["front"]
EDGES = [0, 24, 30, 36, 42, 48, 54, 60]
NAMES = ["DOE"] + [f"Infill {k}" for k in range(1, 7)]
SHORT = ["DOE"] + [str(k) for k in range(1, 7)]
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999933"]
MK = ["o", "s", "D", "^", "v", "P", "X"]
F_MAX, TRUNC, LIMIT = 1.65, 6000, 2897


def block(i):
    return next(b for b in range(7) if EDGES[b] <= i < EDGES[b + 1])


n = np.arange(1, len(A) + 1)
F = np.array([a["peaking"] for a in A])
C = np.array([a["c_max"] for a in A])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.0, 3.3))
for axis, y in ((ax, F), (bx, C)):
    for b in range(1, 7):
        axis.axvline(EDGES[b] + 0.5, color="0.75", lw=0.7, ls=":")
    for i in range(len(A)):
        b = block(i)
        if i in feas:
            axis.scatter(n[i], y[i], marker=MK[b], s=40, color=COL[b], edgecolors="k", linewidths=0.4, zorder=4)
        else:
            axis.scatter(n[i], y[i], marker=MK[b], s=34, facecolors="none", edgecolors=COL[b], linewidths=0.9, alpha=0.9, zorder=3)
    for i in front:
        axis.scatter(n[i], y[i], marker="o", s=150, facecolors="none", edgecolors="crimson", linewidths=1.2, zorder=5)
    axis.grid(alpha=0.3)
for b in range(7):
    ax.text((EDGES[b] + EDGES[b + 1]) / 2 + 0.5, 1.755, SHORT[b], ha="center", va="top", fontsize=7, color="0.35")
ax.axhline(F_MAX, color="tab:orange", ls="--", lw=0.8)
ax.text(60.4, F_MAX + 0.004, "Limit, 1.65", fontsize=6.5, color="tab:orange", ha="right", va="bottom")
ax.set_ylim(1.40, 1.76)
ax.set_ylabel(r"Core $F_{\Delta H}$ [-]")
ax.set_title("(a) Radial peaking factor", fontsize=9)
bx.axhspan(TRUNC - 150, TRUNC + 150, color="0.9", zorder=0)
bx.text(1, TRUNC + 200, f"Truncation, {TRUNC} ppm", fontsize=6.5, color="0.35", va="bottom")
bx.axhline(LIMIT, color="tab:red", ls="--", lw=0.8)
bx.text(60.4, LIMIT + 80, "MTC limit of C9-47", fontsize=6.5, color="tab:red", ha="right", va="bottom")
bx.set_ylim(0, 6500)
bx.set_xlim(0, 61)
ax.set_xlim(0, 61)
ax.set_xlabel("Evaluation")
bx.set_xlabel("Evaluation")
bx.set_ylabel(r"$c_\mathrm{max}$ [ppm]")
bx.set_title("(b) Critical boron at the operating maximum", fontsize=9)
OFA = {34: (-8, -10, "right", "top"), 35: (0, -12, "center", "top"), 40: (8, -10, "left", "top"),
       44: (0, 12, "center", "bottom"), 47: (0, -12, "center", "top")}
OFB = {34: (0, -12, "center", "top"), 35: (8, 8, "left", "bottom"), 40: (0, -12, "center", "top"),
       44: (0, -12, "center", "top"), 47: (8, 8, "left", "bottom")}
for i in front:
    dx, dy, ha, va = OFA[i]
    ax.annotate(f"C9-{i}", (n[i], F[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=6.5)
    dx, dy, ha, va = OFB[i]
    bx.annotate(f"C9-{i}", (n[i], C[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=6.5)
handles = [Line2D([], [], marker=MK[b], ls="", color=COL[b], markeredgecolor="k", markersize=6, label=f"{NAMES[b]}, feasible")
           for b in range(7)]
handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="0.4", markersize=6, label="Infeasible, open marker of its block"),
            Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="crimson", markersize=10, label="Front member")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1])
