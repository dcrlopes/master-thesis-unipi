#!/usr/bin/env python3
"""make_bg_nsga2_figure.py -- Figure 3.13 of the thesis: one elitist
generation of NSGA-II, redrawn after Figure 2 and the main-loop description
of Deb et al. (2002), IEEE Trans. Evol. Comput. 6(2), pp. 185-186.

The population size and the generation count are those of the surrogate
search from Campaign 6, 300 x 400 (Section 4.11.4).

usage (plot only):
    python make_bg_nsga2_figure.py OUT.pdf
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = sys.argv[1]
N_LABEL = "$N = 300$"          # population size shown in the boxes
GENERATIONS = 400              # generation count in the loop label

NAVY = "#16324F"
STEEL = "#4A6D9C"
MIST = "#7D9CC0"
PALE = "#AEC3D8"
INK = "#1B2A3A"
GREY = "#5A6672"
WHITE = "#FFFFFF"
FS_HDR, FS_BOX, FS_ANN = 9.0, 8.5, 7.6

fig, ax = plt.subplots(figsize=(9.8, 4.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, color, text, tcolor, fs=FS_BOX, hatch=None, ec=WHITE):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=ec,
                           linewidth=1.2, hatch=hatch))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tcolor, linespacing=1.35)


def arrow(p0, p1, ls="-", lw=1.4, color=GREY):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, linestyle=ls, color=color,
                                 shrinkA=2, shrinkB=2))


# column A: P_t and Q_t
box(3, 34, 15, 14, NAVY, f"Parents $P_t$\n({N_LABEL})", WHITE)
box(3, 12, 15, 14, STEEL, f"Offspring $Q_t$\n({N_LABEL})", WHITE)
arrow((18.4, 41), (32.4, 32.5))
arrow((18.4, 19), (32.4, 28.5))
ax.text(25.5, 38.8, "1. Pool", fontsize=FS_ANN, color=GREY, ha="center", va="center")

# column B: R_t sorted into fronts
BX, BW = 33, 15
box(BX, 39, BW, 13, NAVY, "$F_1$", WHITE)
box(BX, 29, BW, 10, STEEL, "$F_2$", WHITE)
box(BX, 18, BW, 11, MIST, "$F_3$", INK)
box(BX, 9, BW, 9, PALE, "$F_4$", INK)
ax.text(BX + BW / 2, 57.6, r"2. $R_t = P_t \cup Q_t$  ($2N$)", fontsize=FS_HDR,
        color=INK, ha="center", va="center")
ax.text(BX + BW / 2, 54.4, "After non-dominated sorting", fontsize=FS_HDR,
        color=INK, ha="center", va="center")
arrow((48.4, 41), (61.6, 41))
ax.text(55.0, 36.2, "Crowding-distance\nsorting on $F_3$", fontsize=FS_ANN,
        color=GREY, ha="center", va="center", linespacing=1.3)

# column C: next population
CX, CW = 62, 15
box(CX, 39, CW, 13, NAVY, "$F_1$: all", WHITE)
box(CX, 29, CW, 10, STEEL, "$F_2$: all", WHITE)
box(CX, 23, CW, 6, MIST, "$F_3$: largest\ncrowding distance", INK, fs=7.0)
ax.text(CX + CW / 2, 57.6, r"3. Next population $P_{t+1}$", fontsize=FS_HDR,
        color=INK, ha="center", va="center")
ax.text(CX + CW / 2, 54.4, f"({N_LABEL})", fontsize=FS_HDR, color=INK,
        ha="center", va="center")

# rejected designs
arrow((48.4, 17.5), (85.6, 14.2))
box(86, 9.2, 12, 9.6, WHITE, "", INK, hatch="///", ec=GREY)
ax.text(92, 5.6, "Rest of $F_3$ and $F_4$:\nrejected", fontsize=FS_ANN,
        color=GREY, ha="center", va="center", linespacing=1.3)

# generation loop
ax.plot([CX + CW / 2, CX + CW / 2], [22.4, 3.2], color=GREY, lw=1.2, ls=(0, (4, 3)))
ax.plot([10.5, CX + CW / 2], [3.2, 3.2], color=GREY, lw=1.2, ls=(0, (4, 3)))
arrow((10.5, 3.2), (10.5, 11.4), ls=(0, (4, 3)), lw=1.2)
ax.text(40.0, 1.1, "4. Selection, crossover and mutation on $P_{t+1}$ create "
        f"$Q_{{t+1}}$, repeated for {GENERATIONS} generations",
        fontsize=FS_ANN, color=GREY, ha="center", va="center")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("written", OUT)
