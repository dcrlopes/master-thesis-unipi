#!/usr/bin/env python3
"""
make_c9a_evolution_figure.py -- the two objectives along the evaluation order
of the continuation of Campaign 9, with the peaking factor of the
three-dimensional core. The 60 evaluations of Campaign 9 are the starting
data, drawn as one block, followed by the six infill batches of the
continuation (C9-60 to C9-95).

  (a) radial peaking factor of the three-dimensional core, all rods out
  (b) critical boron concentration at the operating maximum

Filled markers: feasible under the constraints stored in the continuation
checkpoint (adjusted cycle-length constraint). Circled: the final front.

Reads out_c9a/optimization_checkpoint.json, confirm3d_c9_all, confirm3d_c9_front,
confirm3d_c9a, confirm3d_c9a_rest /summary.json and c9_post/d*_p128/mtc_ceiling_table.json.
Drawing only, no transport.
Usage: python make_c9a_evolution_figure.py <prefix>     # writes <prefix>.pdf and <prefix>.png
       python make_c9a_evolution_figure.py <prefix> cycle
With cycle, a third panel (c) gives the cycle length of the assembly depletion of
each evaluation and the cycle-length constraint that applied to it.
       python make_c9a_evolution_figure.py <prefix> a2d
With a2d, panel (a) alone, with the two-dimensional core peaking factor of the
campaign evaluator (the "peaking" of the checkpoint) in place of the
three-dimensional one: <prefix>_a_2d.pdf and .png.
       python make_c9a_evolution_figure.py <prefix> 2d
With 2d, the two panels as in the default mode, panel (a) with the
two-dimensional core peaking factor: <prefix>_2d.pdf and .png.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = json.load(open("out_c9a/optimization_checkpoint.json"))
A, CN = D["all_raw"], D["constraint_names"]
assert len(A) == 96
F3 = {}
for f in ("confirm3d_c9_all", "confirm3d_c9_front", "confirm3d_c9a", "confirm3d_c9a_rest"):
    F3.update({int(k): v["ARO_3Dhw"]["F"] for k, v in json.load(open(f + "/summary.json")).items()})
assert len(F3) == 96
feas = {i for i, a in enumerate(A) if all(a[g] is not None and a[g] <= 0 for g in CN)}
FRONT = [27, 69, 70]
MTC = [json.load(open(f"c9_post/d{d}_p128/mtc_ceiling_table.json"))[0]["ceiling"] for d in FRONT]
EDGES = [0, 60, 66, 72, 78, 84, 90, 96]
# the 60 evaluations of Campaign 9 contain the design of experiments and infill
# batches 1 to 6, so the batches of the continuation are 7 to 12
NAMES = ["Campaign 9"] + [f"Infill {k}" for k in range(7, 13)]
SHORT = ["Campaign 9, 60 evaluations"] + [str(k) for k in range(7, 13)]
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999933"]
MK = ["o", "s", "D", "^", "v", "P", "X"]
F_MAX, TRUNC = 1.65, 6000

n = np.arange(1, 97)
A2D = len(sys.argv) > 2 and sys.argv[2] == "a2d"
TWO_D = A2D or (len(sys.argv) > 2 and sys.argv[2] == "2d")
F = np.array([a["peaking"] for a in A]) if TWO_D else np.array([F3[i] for i in range(96)])
C = np.array([a["c_max"] for a in A], dtype=float)
block = lambda i: next(b for b in range(7) if EDGES[b] <= i < EDGES[b + 1])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
CYCLE = len(sys.argv) > 2 and sys.argv[2] == "cycle"
if len(sys.argv) > 2 and sys.argv[2] not in ("cycle", "a2d", "2d"):
    sys.exit(f"unknown second argument {sys.argv[2]!r}: use cycle, a2d or 2d")
E_LO, E_HI = 1400.0, 3200.0                 # axis of panel (c); values outside are drawn on its edge
E_RAW = np.array([a["cycle_length"] for a in A], dtype=float)
E = np.clip(E_RAW, E_LO, E_HI)
if A2D:
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    bx = cx = None
    PANELS = ((ax, F, 1.838),)
elif CYCLE:
    fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(9.6, 10.3), sharex=True)
    PANELS = ((ax, F, 1.838), (bx, C, 6460), (cx, E, E_HI - 20))
else:
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.6, 7.0), sharex=True)
    PANELS = ((ax, F, 1.838), (bx, C, 6460))
for axis, y, top in PANELS:
    for b in range(1, 7):
        axis.axvline(EDGES[b] + 0.5, color="0.45", lw=0.8, ls=":")
    for i in range(96):
        b = block(i)
        if i in feas:
            axis.scatter(n[i], y[i], marker=MK[b], s=40, color=COL[b], edgecolors="k", linewidths=0.4, zorder=4)
        else:
            axis.scatter(n[i], y[i], marker=MK[b], s=34, facecolors="none", edgecolors=COL[b], linewidths=0.9,
                         alpha=0.9, zorder=3)
    for i in FRONT:
        axis.scatter(n[i], y[i], marker="o", s=150, facecolors="none", edgecolors="crimson", linewidths=1.2, zorder=5)
    for b in range(7):
        axis.text((EDGES[b] + EDGES[b + 1]) / 2 + 0.5, top, SHORT[b], ha="center", va="top", fontsize=7, color="0.35")
    axis.grid(alpha=0.3, axis="y")
ax.axhline(F_MAX, color="tab:orange", ls="--", lw=0.8)
ax.text(96.4, F_MAX + 0.004, "Limit, 1.65", fontsize=6.5, color="tab:orange", ha="right", va="bottom")
ax.set_ylim(1.42, 1.84)
ax.set_ylabel(r"$F_{\Delta H}$, " + ("two" if TWO_D else "three") + "-dimensional core [-]")
ax.set_title(("Radial peaking factor of the two-dimensional core" if A2D
              else "(a) Radial peaking factor of the " + ("two" if TWO_D else "three") + "-dimensional core"), fontsize=9)
if A2D:                                     # panel (a) alone: own x axis, labels, legend, output
    ax.set_xlim(0, 97)
    ax.set_xlabel("Evaluation")
    ARROW = dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=0, shrinkB=2)
    LA = {27: (31.0, 1.70), 69: (66.0, 1.445), 70: (75.0, 1.445)}
    for i in FRONT:
        ax.annotate(f"C9-{i}", (n[i], F[i]), xytext=LA[i], textcoords="data", ha="center", va="center",
                    fontsize=6.5, arrowprops=ARROW, zorder=6)
    handles = [Line2D([], [], marker=MK[b], ls="", color=COL[b], markeredgecolor="k", markersize=6,
                      label=f"{NAMES[b]}, feasible") for b in range(7)]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="0.4", markersize=6,
                       label="Infeasible, open marker of its block"),
                Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="crimson", markersize=10,
                       label="Member of the final front")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT = sys.argv[1] + "_a_2d"
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=170, bbox_inches="tight")
    print("written", OUT + ".pdf")
    print("feasible:", len(feas), "of 96;", sum(1 for i in feas if i >= 60), "in the continuation")
    print("outside the axes:", [(i, round(F[i], 3)) for i in range(96) if not 1.42 <= F[i] <= 1.84])
    sys.exit(0)
bx.axhspan(TRUNC - 150, TRUNC + 150, color="0.9", zorder=0)
bx.text(1, TRUNC - 200, f"Truncation, {TRUNC} ppm", fontsize=6.5, color="0.35", va="top")
bx.axhspan(min(MTC), max(MTC), color="tab:red", alpha=0.12, zorder=0)
bx.text(1, max(MTC) + 60, f"MTC boron limits of the final front, {min(MTC):.0f} to {max(MTC):.0f} ppm",
        fontsize=6.5, color="tab:red", va="bottom")
bx.set_ylim(0, 6500)
bx.set_xlim(0, 97)
(cx if CYCLE else bx).set_xlabel("Evaluation")
if CYCLE:
    cx.hlines(1826, 0, 60.5, colors="tab:green", linestyles="--", lw=1.0)
    cx.text(1, 1826 - 25, "Minimum of Campaign 9, 1826 d", fontsize=6.5, color="tab:green", va="top")
    cx.hlines(2270, 60.5, 78.5, colors="tab:purple", linestyles="--", lw=1.0)
    cx.text(59.5, 2270, "Adjusted minimum, 2270 d", fontsize=6.5, color="tab:purple", va="center", ha="right")
    req = [A[i]["axial_req_efpd"] for i in range(78, 96)]
    cx.hlines(req, n[78:] - 0.42, n[78:] + 0.42, colors="#008080", lw=1.6)
    cx.text(96.6, max(req) + 60, "Adjusted minimum of each design", fontsize=6.5, color="#008080",
            va="bottom", ha="right")
    out = int((E_RAW > E_HI).sum()), int((E_RAW < E_LO).sum())
    cx.text(1, E_LO + 30, f"On the axis edges: {out[0]} designs above {E_HI:.0f} d, {out[1]} below {E_LO:.0f} d",
            fontsize=6.5, color="0.35", va="bottom")
    cx.set_ylim(E_LO, E_HI)
    cx.set_ylabel("Assembly cycle length [d]")
    cx.set_title("(c) Cycle length of the assembly depletion and its constraint", fontsize=9)
    LC = {27: (33.0, 2750), 69: (66.0, 2700), 70: (75.0, 1900)}
bx.set_ylabel(r"$c_\mathrm{max}$ [ppm]")
bx.set_title("(b) Critical boron at the operating maximum", fontsize=9)
ARROW = dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=0, shrinkB=2)
LA = {27: (31.0, 1.70), 69: (66.0, 1.445), 70: (75.0, 1.445)}
LB = {27: (33.0, 700), 69: (66.0, 3900), 70: (75.0, 900)}
for i in FRONT:
    ax.annotate(f"C9-{i}", (n[i], F[i]), xytext=LA[i], textcoords="data", ha="center", va="center",
                fontsize=6.5, arrowprops=ARROW, zorder=6)
    bx.annotate(f"C9-{i}", (n[i], C[i]), xytext=LB[i], textcoords="data", ha="center", va="center",
                fontsize=6.5, arrowprops=ARROW, zorder=6)
    if CYCLE:
        cx.annotate(f"C9-{i}", (n[i], E[i]), xytext=LC[i], textcoords="data", ha="center", va="center",
                    fontsize=6.5, arrowprops=ARROW, zorder=6)
handles = [Line2D([], [], marker=MK[b], ls="", color=COL[b], markeredgecolor="k", markersize=6,
                  label=f"{NAMES[b]}, feasible") for b in range(7)]
handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="0.4", markersize=6,
                   label="Infeasible, open marker of its block"),
            Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="crimson", markersize=10,
                   label="Member of the final front")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02 if CYCLE else -0.03))
fig.tight_layout(rect=(0, 0.03 if CYCLE else 0.04, 1, 1))
OUT = sys.argv[1] + ("_cycle" if CYCLE else "_2d" if TWO_D else "")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=170, bbox_inches="tight")
print("written", OUT + ".pdf")
print("feasible:", len(feas), "of 96;", sum(1 for i in feas if i >= 60), "in the continuation")
print("outside the axes:", [(i, round(F[i], 3)) for i in range(96) if not 1.42 <= F[i] <= 1.84])
