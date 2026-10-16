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
       python make_c9_evolution_figure.py <prefix> split
With split, each objective is a figure of its own, wider than a panel:
<prefix>_fdh.pdf and <prefix>_cmax.pdf (and the .png of each).
       python make_c9_evolution_figure.py <prefix> stack
With stack, the two wide panels are one above the other in one figure with
one legend: <prefix>_stacked.pdf and .png.
       python make_c9_evolution_figure.py <prefix> cycle
With cycle, the stacked figure has a third panel (c) with the cycle length of
the assembly depletion and its constraint: <prefix>_stacked_cycle.pdf and .png.
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
MODE = sys.argv[2] if len(sys.argv) > 2 else ""
if MODE and MODE not in ("split", "stack", "cycle") and not MODE.endswith(".png"):
    sys.exit(f"unknown second argument {MODE!r}: use split, stack, cycle or a .png path")
SPLIT = MODE == "split"
CYCLE = MODE == "cycle"
STACK = MODE in ("stack", "cycle")
WIDE = SPLIT or STACK                       # output named from a prefix, wide panels
E_LO, E_HI = 1400.0, 3200.0                 # axis of panel (c); values outside are drawn on its edge
E_RAW = np.array([a["cycle_length"] for a in A], dtype=float)
E = np.clip(E_RAW, E_LO, E_HI)
cx = None
if CYCLE:
    fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(9.6, 10.3), sharex=True)
    FIGS = [(fig, "_stacked_cycle")]
elif STACK:
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.6, 7.0), sharex=True)
    FIGS = [(fig, "_stacked")]
elif SPLIT:
    fig_a, ax = plt.subplots(figsize=(9.6, 3.9))
    fig_b, bx = plt.subplots(figsize=(9.6, 3.9))
    FIGS = [(fig_a, "_fdh"), (fig_b, "_cmax")]
else:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.0, 3.3))
    FIGS = [(fig, "")]
for axis, y in ((ax, F), (bx, C)) + (((cx, E),) if CYCLE else ()):
    for b in range(1, 7):
        axis.axvline(EDGES[b] + 0.5, color="0.45", lw=0.8, ls=":")
    for i in range(len(A)):
        b = block(i)
        if i in feas:
            axis.scatter(n[i], y[i], marker=MK[b], s=40, color=COL[b], edgecolors="k", linewidths=0.4, zorder=4)
        else:
            axis.scatter(n[i], y[i], marker=MK[b], s=34, facecolors="none", edgecolors=COL[b], linewidths=0.9, alpha=0.9, zorder=3)
    for i in front:
        axis.scatter(n[i], y[i], marker="o", s=150, facecolors="none", edgecolors="crimson", linewidths=1.2, zorder=5)
    axis.grid(alpha=0.3, axis="y")
for b in range(7):
    ax.text((EDGES[b] + EDGES[b + 1]) / 2 + 0.5, 1.755, SHORT[b], ha="center", va="top", fontsize=7, color="0.35")
ax.axhline(F_MAX, color="tab:orange", ls="--", lw=0.8)
ax.text(60.4, F_MAX + 0.004, "Limit, 1.65", fontsize=6.5, color="tab:orange", ha="right", va="bottom")
ax.set_ylim(1.40, 1.76)
ax.set_ylabel(r"Core $F_{\Delta H}$ [-]")
ax.set_title("Radial peaking factor" if SPLIT else "(a) Radial peaking factor", fontsize=9)
bx.axhspan(TRUNC - 150, TRUNC + 150, color="0.9", zorder=0)
bx.text(1, TRUNC + 200, f"Truncation, {TRUNC} ppm", fontsize=6.5, color="0.35", va="bottom")
bx.axhline(LIMIT, color="tab:red", ls="--", lw=0.8)
bx.annotate("MTC boron limit of C9-47", (47, LIMIT), xytext=(40, 3900), textcoords="data",
            ha="center", va="center", fontsize=6.5, color="tab:red",
            arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.6, shrinkA=0, shrinkB=0), zorder=6)
bx.set_ylim(0, 6500)
if WIDE:                                    # the block labels of panel (a), for the figure on its own
    for b in range(7):
        bx.text((EDGES[b] + EDGES[b + 1]) / 2 + 0.5, 6460, SHORT[b], ha="center", va="top", fontsize=7, color="0.35")
bx.set_xlim(0, 61)
ax.set_xlim(0, 61)
if not STACK:
    ax.set_xlabel("Evaluation")
(cx if CYCLE else bx).set_xlabel("Evaluation")
bx.set_ylabel(r"$c_\mathrm{max}$ [ppm]")
if CYCLE:
    for b in range(7):
        cx.text((EDGES[b] + EDGES[b + 1]) / 2 + 0.5, E_HI - 20, SHORT[b], ha="center", va="top", fontsize=7, color="0.35")
    cx.axhline(1826, color="tab:green", ls="--", lw=1.0)
    cx.text(1, 1826 - 25, "Minimum cycle length, 1826 d", fontsize=6.5, color="tab:green", va="top")
    cx.text(1, E_LO + 30, f"On the axis edges: {int((E_RAW > E_HI).sum())} designs above {E_HI:.0f} d, "
            f"{int((E_RAW < E_LO).sum())} below {E_LO:.0f} d", fontsize=6.5, color="0.35", va="bottom")
    cx.set_ylim(E_LO, E_HI)
    cx.set_ylabel("Assembly cycle length [d]")
    cx.set_title("(c) Cycle length of the assembly depletion and its constraint", fontsize=9)
bx.set_title(("" if SPLIT else "(b) ") + "Critical boron at the operating maximum", fontsize=9)
# labels in free space, ordered as the points, with a thin leader to each point
ARROW = dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=0, shrinkB=2)
LA = {34: (30.0, 1.452), 35: (37.0, 1.442), 40: (44.0, 1.452),
      44: (42.0, 1.592), 47: (51.0, 1.592)}
LB = {34: (30.0, 650), 35: (36.0, 2550), 40: (41.0, 650),
      44: (47.0, 650), 47: (53.0, 800)}
LC = {34: (31.0, 2250), 35: (37.0, 2450), 40: (42.0, 2250), 44: (46.0, 2450), 47: (52.0, 2300)}
for i in front:
    ax.annotate(f"C9-{i}", (n[i], F[i]), xytext=LA[i], textcoords="data",
                ha="center", va="center", fontsize=6.5, arrowprops=ARROW, zorder=6)
    bx.annotate(f"C9-{i}", (n[i], C[i]), xytext=LB[i], textcoords="data",
                ha="center", va="center", fontsize=6.5, arrowprops=ARROW, zorder=6)
    if CYCLE:
        cx.annotate(f"C9-{i}", (n[i], E[i]), xytext=LC[i], textcoords="data",
                    ha="center", va="center", fontsize=6.5, arrowprops=ARROW, zorder=6)
handles = [Line2D([], [], marker=MK[b], ls="", color=COL[b], markeredgecolor="k", markersize=6, label=f"{NAMES[b]}, feasible")
           for b in range(7)]
handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="0.4", markersize=6, label="Infeasible, open marker of its block"),
            Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="crimson", markersize=10, label="Front member")]
for f, suffix in FIGS:
    f.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5 if WIDE else 6.8, frameon=False,
             bbox_to_anchor=(0.5, -0.02 if CYCLE else -0.03 if STACK else -0.08))
    f.tight_layout(rect=(0, 0.03 if CYCLE else 0.04, 1, 1))
    if WIDE:
        f.savefig(sys.argv[1] + suffix + ".pdf", bbox_inches="tight")
        f.savefig(sys.argv[1] + suffix + ".png", dpi=170, bbox_inches="tight")
        print("written", sys.argv[1] + suffix + ".pdf")
    else:
        f.savefig(sys.argv[1], bbox_inches="tight")
        if len(sys.argv) > 2:
            f.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
        print("written", sys.argv[1])
