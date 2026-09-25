#!/usr/bin/env python3
"""make_restart_defect_figure.py -- the restarted-depletion defect on design C5-7,
for Section 5.3.2.

Horizontal, two bars on one burnup axis:

  top     the burnup label the evaluator recorded, nine blocks, with the first
          step of every restarted block hatched: the label advanced by 4.0
          MWd/kgHM while the fuel did not burn (OpenMC 0.15.3 restarted without
          reaction rates, Section 4.4.5);
  bottom  the physical burnup, the same blocks with the eight lost steps
          removed, Equation (4.x) of the methodology.

The schedule is the Campaign 5 one: five beginning-of-life steps of 0.5, 1, 2,
4 and 6 MWd/kgHM in the first block, then blocks of two steps of 4.0, the last
clipped at the 75.0 MWd/kgHM horizon. That gives 22 depletion solves, which is
the count the archive stores for C5-7, and 13.5 + 7 x 8 + 5.5 = 75.0 recorded
against 13.5 + 7 x 4 + 1.5 = 43.0 physical. The cycle length scales with the
burnup at constant specific power, 7513 to 4307 EFPD.

usage (plot only, runs anywhere):
    python make_restart_defect_figure.py OUT.pdf [OUT.png]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

OUT = sys.argv[1]

BOL = [0.5, 1.0, 2.0, 4.0, 6.0]          # first block, no restart
STEP, CHUNK, HORIZON = 4.0, 2, 75.0
EFPD_REC, B_REC = 7512.505, 75.0          # archive values of C5-7
K = EFPD_REC / B_REC                      # EFPD per MWd/kgHM at constant specific power

# ---- the recorded blocks, then the physical ones ---------------------------
blocks_rec = []                           # (start, end, lost_end) in recorded label
b = sum(BOL)
blocks_rec.append((0.0, b, None))
while b < HORIZON:
    end = min(b + CHUNK * STEP, HORIZON)
    blocks_rec.append((b, end, b + STEP))  # the first step of the block is the lost one
    b = end
n_restart = len(blocks_rec) - 1
lost = n_restart * STEP

blocks_phys, shift = [], 0.0
for s, e, l in blocks_rec:
    if l is None:
        blocks_phys.append((s, e))
    else:
        shift += STEP
        blocks_phys.append((l - shift, e - shift))
B_PHYS = blocks_phys[-1][1]

C_BURN, C_LOST, C_EDGE = "#4C72B0", "#D55E00", "0.25"
fig, ax = plt.subplots(figsize=(10.8, 3.9))
Y_REC, Y_PHY, H = 1.0, 0.0, 0.42

def bar(y, s, e, colour, hatch=None):
    ax.add_patch(Rectangle((s, y - H / 2), e - s, H, facecolor=colour, edgecolor=C_EDGE,
                           lw=0.8, hatch=hatch, zorder=3))

for i, (s, e, l) in enumerate(blocks_rec):
    if l is None:
        bar(Y_REC, s, e, C_BURN)
    else:
        bar(Y_REC, s, l, "white", hatch="////")
        bar(Y_REC, l, e, C_BURN)
    ax.text((s + e) / 2, Y_REC + H / 2 + 0.06, str(i + 1), ha="center", va="bottom",
            fontsize=8.5, color=C_EDGE)
for i, (s, e) in enumerate(blocks_phys):
    bar(Y_PHY, s, e, C_BURN)
    ax.text((s + e) / 2, Y_PHY - H / 2 - 0.06, str(i + 1), ha="center", va="top",
            fontsize=8.5, color=C_EDGE)

# the shift of every block end, recorded to physical
for (s, e, l), (ps, pe) in zip(blocks_rec[1:], blocks_phys[1:]):
    ax.plot([e, pe], [Y_REC - H / 2, Y_PHY + H / 2], color="0.6", lw=0.7, ls=":", zorder=2)

# the two end points, written beyond the bars
ax.annotate(f"{B_REC:.1f} MWd/kgHM\n{EFPD_REC:.0f} EFPD", xy=(B_REC, Y_REC),
            xytext=(B_REC + 1.2, Y_REC), va="center", ha="left", fontsize=9, color=C_EDGE)
ax.annotate(f"{B_PHYS:.1f} MWd/kgHM\n{B_PHYS * K:.0f} EFPD", xy=(B_PHYS, Y_PHY),
            xytext=(B_PHYS + 1.2, Y_PHY), va="center", ha="left", fontsize=9, color=C_EDGE)
ax.text(HORIZON + 1.2, (Y_REC + Y_PHY) / 2,
        f"{n_restart} restarts $\\times$ {STEP:.1f} = {lost:.1f} MWd/kgHM\nlabelled but not burned",
        va="center", ha="left", fontsize=9, color=C_LOST)

ax.set_yticks([Y_REC, Y_PHY])
ax.set_yticklabels(["Recorded label", "Physical burnup"], fontsize=9.5)
ax.set_ylim(-0.7, 1.7)
ax.set_xlim(0, 98)
ax.set_xlabel("Burnup [MWd/kgHM]")
ax.grid(axis="x", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
top = ax.secondary_xaxis("top", functions=(lambda x: x * K, lambda d: d / K))
top.set_xlabel("Cycle length at constant specific power [EFPD]", fontsize=9.5)
for sp in ("left", "right"):
    ax.spines[sp].set_visible(False)

ax.legend(handles=[Patch(facecolor=C_BURN, edgecolor=C_EDGE, label="Step in which the fuel burned"),
                   Patch(facecolor="white", edgecolor=C_EDGE, hatch="////",
                         label="First step after a restart, no reaction rates")],
          fontsize=8.5, loc="lower right", bbox_to_anchor=(0.995, 0.01), framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight")

print(f"blocks {len(blocks_rec)}, restarts {n_restart}, solves {1 + len(BOL) + CHUNK * n_restart}")
print(f"recorded {B_REC:.1f} MWd/kgHM = {EFPD_REC:.0f} EFPD; physical {B_PHYS:.1f} = {B_PHYS * K:.0f} EFPD")
print("wrote", OUT)
