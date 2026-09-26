#!/usr/bin/env python3
"""
make_c9_hump_replicas_figure.py -- stability of the boron objective under
the depletion noise, from the seed replicas of the Campaign 9 front
(c9_dep_replicas.py). Two panels, side by side:

  (a) the core hump L x hump of every replica, the archived evaluation and
      the high-fidelity run, per design, against the 400 pcm threshold below
      which the boron objective sets the hump to zero; filled markers are
      humps that enter c_max, open markers humps set to zero;
  (b) c_max of every replica minus the archived c_max, per design.

Reads c9_dep_replicas/runs.json and out_c9/optimization_checkpoint.json.
Drawing only, no transport.
Usage: python make_c9_hump_replicas_figure.py <out.pdf> [out.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = json.load(open("c9_dep_replicas/runs.json"))
A = json.load(open("out_c9/optimization_checkpoint.json"))["all_raw"]
DES = [34, 47, 35, 40, 44]          # ordered by archived hump
THR = 400.0
C_REP, C_ARC, C_HI = "#0072B2", "#D55E00", "#009E73"

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 3.9))
rng = np.random.default_rng(5)

for x, d in enumerate(DES):
    reps = [v for v in R.values() if v["idx"] == d and v["tier"] == "campaign"]
    hi = [v for v in R.values() if v["idx"] == d and v["tier"] == "production"]
    j = rng.uniform(-0.13, 0.13, len(reps))
    for jj, v in zip(j, reps):
        h = v["hump_core_op_pcm"]
        ax.scatter(x + jj, h, s=30, marker="o", color=C_REP if h >= THR else "white",
                   edgecolors=C_REP, linewidths=1.0, zorder=3)
        bx.scatter(x + jj, v["c_max"] - A[d]["c_max"], s=30, marker="o", color=C_REP,
                   edgecolors="k", linewidths=0.3, zorder=3)
    for v in hi:
        h = v["hump_core_op_pcm"]
        ax.scatter(x + 0.25, h, s=34, marker="s", color=C_HI if h >= THR else "white",
                   edgecolors=C_HI, linewidths=1.0, zorder=3)
        bx.scatter(x + 0.25, v["c_max"] - A[d]["c_max"], s=34, marker="s", color=C_HI,
                   edgecolors="k", linewidths=0.3, zorder=3)
    h = A[d]["hump_core_op_pcm"]
    ax.scatter(x - 0.25, h, s=44, marker="D", color=C_ARC if h >= THR else "white",
               edgecolors=C_ARC, linewidths=1.1, zorder=4)
    bx.scatter(x - 0.25, 0.0, s=44, marker="D", color=C_ARC, edgecolors="k", linewidths=0.3, zorder=4)

ax.axhline(THR, color="k", ls="--", lw=0.9)
ax.text(len(DES) - 0.55, THR + 40, "Threshold, 400 pcm", ha="right", va="bottom", fontsize=7.5)
ax.axhspan(-1000, THR, color="0.93", zorder=0)
ax.text(len(DES) - 0.55, -950, "Hump set to zero", fontsize=7.5, color="0.35", va="bottom", ha="right")
ax.set_xticks(range(len(DES)))
ax.set_xticklabels([f"C9-{d}" for d in DES])
ax.set_xlim(-0.6, len(DES) - 0.4)
ax.set_ylim(-1000, 1600)
ax.set_ylabel("Core hump (pcm)")
ax.set_title("(a) Hump of each depletion against the threshold", fontsize=9, loc="left")

bx.axhline(0, color="k", lw=0.6)
bx.set_xticks(range(len(DES)))
bx.set_xticklabels([f"C9-{d}" for d in DES])
bx.set_xlim(-0.6, len(DES) - 0.4)
bx.set_ylim(-60, 100)
bx.set_ylabel(r"$c_\mathrm{max}$ minus archived value (ppm)")
bx.set_title(r"(b) Critical boron $c_\mathrm{max}$ of each depletion", fontsize=9, loc="left")

for a in (ax, bx):
    a.grid(axis="y", alpha=0.3)
handles = [Line2D([], [], marker="D", ls="", color=C_ARC, markeredgecolor=C_ARC, markersize=6, label="Archived evaluation"),
           Line2D([], [], marker="o", ls="", color=C_REP, markeredgecolor=C_REP, markersize=6, label="Seed replica, low fidelity"),
           Line2D([], [], marker="s", ls="", color=C_HI, markeredgecolor=C_HI, markersize=6, label="High fidelity"),
           Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="0.4", markersize=6,
                  label="Open marker in (a): hump set to zero")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1])
