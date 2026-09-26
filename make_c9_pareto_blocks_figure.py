#!/usr/bin/env python3
"""
make_c9_pareto_blocks_figure.py -- Campaign 9 archive in two panels, in the
style of the Campaign 3 figure (make_c3_space_figure.py).

  (a) the objective plane: feasible designs as filled markers coloured by
      block, infeasible designs as open markers of their block, the Pareto
      front joined by a step line, the two-bank designs ringed, the MTC
      boron limits of C9-47 and the peaking bound as reference lines.

  (b) the design vectors in parallel coordinates, each variable normalised
      to its bounds, one colour per block (the same colours as panel a), the
      front members drawn as thick lines. The pitch is fixed at 1.26 cm in
      Campaign 9, so the four live variables are e, w_Gd, n_Gd and t_refl.

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
NAMES = ["Design of experiments"] + [f"Infill {k}" for k in range(1, 7)]
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999933"]
MK = ["o", "s", "D", "^", "v", "P", "X"]
LIMIT_128, LIMIT_155, F_MAX, TRUNC = 2897, 3266, 1.65, 6000
# live design variables and the bounds of the Campaign 9 specification
VARS = [("enrich", r"$e$", 2.0, 13.913),
        ("gd_wt", r"$w_\mathrm{Gd}$", 0.0, 8.0),
        ("gd_pins_used", r"$n_\mathrm{Gd}$", 12.0, 40.0),
        ("refl_thick", r"$t_\mathrm{refl}$", 2.0, 5.66)]


def block(i):
    return next(b for b in range(7) if EDGES[b] <= i < EDGES[b + 1])


F = np.array([a["peaking"] for a in A])
C = np.array([a["c_max"] for a in A])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]})

# ---------------- (a) objective plane
ax.axhspan(TRUNC - 150, TRUNC + 150, color="0.9", zorder=0)
ax.text(1.415, TRUNC + 200, f"Truncation, {TRUNC} ppm", fontsize=7.5, color="0.35", va="bottom")
for i in range(len(A)):
    b = block(i)
    if i in feas:
        ax.scatter(F[i], C[i], marker=MK[b], s=44, color=COL[b], edgecolors="k", linewidths=0.4, zorder=4)
    else:
        ax.scatter(F[i], C[i], marker=MK[b], s=36, facecolors="none", edgecolors=COL[b],
                   linewidths=0.9, alpha=0.9, zorder=3)
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
ax.set_title("(a) Objective plane", fontsize=9)

# ---------------- (b) parallel coordinates, one colour per block
x = np.arange(len(VARS))
norm = np.empty((len(A), len(VARS)))
for j, (key, _, lo, hi) in enumerate(VARS):
    v = np.array([a[key] for a in A], float)
    norm[:, j] = (v - lo) / (hi - lo)
for i in range(len(A)):
    b = block(i)
    bx.plot(x, norm[i], color=COL[b], lw=0.9, alpha=0.8 if i in feas else 0.4,
            ls="-" if i in feas else ":", zorder=2)
for i in front:
    bx.plot(x, norm[i], color="crimson", lw=2.6, alpha=0.9, zorder=4)
    bx.plot(x, norm[i], color=COL[block(i)], lw=1.0, zorder=5)
for xi in x:
    bx.axvline(xi, color="#c9d2d9", lw=0.8, zorder=0)
bx.set_xticks(x)
bx.set_xticklabels([lab for _, lab, _, _ in VARS])
bx.set_xlim(-0.15, len(VARS) - 0.85)
bx.set_ylim(-0.05, 1.05)
bx.set_yticks([0.0, 0.5, 1.0])
bx.set_yticklabels(["Lower\nbound", "Mid", "Upper\nbound"])
bx.set_title("(b) Design vectors by block", fontsize=9)

# ---------------- shared legend
nf = [sum(1 for i in feas if block(i) == b) for b in range(7)]
handles = [Line2D([], [], marker=MK[b], ls="-", color=COL[b], markeredgecolor="k", markersize=6, lw=1.2,
                  label=f"{NAMES[b]}, feasible ({nf[b]})") for b in range(7)]
handles += [Line2D([], [], marker="o", ls=":", color="0.4", markerfacecolor="none", markersize=6,
                   label="Infeasible (38), open marker, dotted line"),
            Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#6A3D9A",
                   markersize=10, label=f"Two-bank controllable ({len(two)})"),
            Line2D([], [], color="crimson", lw=1.8, label=f"Pareto front ({len(front)} designs)")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, -0.13), columnspacing=1.4)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1], "feasible per block", nf, "two-bank", sorted(two))
