#!/usr/bin/env python3
"""make_c9_axial_layers_figure.py -- Figure 5.18, the layer convergence of C9-47.

Two panels, both read from the layered depletion runs, no transport:

  (a) cycle length of the 3D core against the number of axial burnup layers,
      with the assembly mean of five replicas, the zoned 2D core and the
      mission floor as reference lines.

  (b) burnup along the height at the last computed state of the twelve-layer
      run, against the uniform assumption at the same core-average burnup.

usage (repository root):
    python make_c9_axial_layers_figure.py OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]

LAYER_RUNS = {1: "c9_dep_core3d_L1", 4: "c9_dep_core3d_L4",
              8: "c9_dep_core3d", 12: "c9_dep_core3d_L12"}
ASM_MEAN, ASM_SE = 1962.0, 13.1   # five replicas of C9-47, Table 5.31: +136 d at t = +10.4
ZONED_2D, FLOOR = 1910.0, 1826.0

runs = {L: json.load(open(f"{d}/runs.json"))["d47"] for L, d in LAYER_RUNS.items()}
L = sorted(runs)
efpd = [runs[k]["efpd"] for k in L]
sig = [runs[k]["sigma_efpd"] for k in L]

r12 = runs[12]
edges = np.asarray(r12["info"]["layer_edges"], dtype=float)
edges -= edges[0]                                   # height above the bottom of the fuel
mid = 0.5 * (edges[:-1] + edges[1:])
bu = np.asarray(r12["layer_burnup"][-1], dtype=float)

C_3D, C_ASM, C_2D, C_FLOOR = "#0072B2", "#444444", "#009E73", "#CC79A7"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9))

# ---- (a) cycle length against the number of layers ------------------------
ax.axhspan(ASM_MEAN - ASM_SE, ASM_MEAN + ASM_SE, color="0.85", zorder=0)
ax.axhline(ASM_MEAN, color=C_ASM, lw=1.1, label="Assembly, mean of 5 replicas")
ax.axhline(ZONED_2D, color=C_2D, lw=1.2, ls="-.", label="Zoned 2D core")
ax.axhline(FLOOR, color=C_FLOOR, lw=1.1, ls="--", label="Mission floor")
ax.errorbar(L, efpd, yerr=sig, fmt="o-", color=C_3D, ms=5, lw=1.4, capsize=3,
            label="3D core, C9-47", zorder=3)
ax.set_xlabel("Axial burnup layers")
ax.set_ylabel("Cycle length [d]")
ax.set_xticks(L)
ax.set_xlim(0.4, 12.6)
ax.set_ylim(1500, 2000)
ax.legend(fontsize=8, loc="center right")
ax.set_title("(a)", loc="left", fontweight="bold")
ax.grid(alpha=0.25, lw=0.6)

# ---- (b) burnup along the height at the last state ------------------------
bx.barh(mid, bu, height=9.0, color=C_3D, label="12 layers", zorder=2)
bx.axvline(bu.mean(), color="#D55E00", ls="--", lw=1.3, zorder=3,
           label="Uniform, 1 layer")
bx.set_xlabel("Burnup at the last state [MWd/kgHM]")
bx.set_ylabel("Height above the bottom of the fuel [cm]")
bx.set_ylim(0, edges[-1])
bx.set_xlim(0, 1.06 * bu.max())
bx.legend(fontsize=8, loc="upper right")
bx.set_title("(b)", loc="left", fontweight="bold")
bx.grid(alpha=0.25, lw=0.6, axis="x")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3 or len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight")

print("layers", L, "cycle", [round(e) for e in efpd], "sigma", [round(s, 1) for s in sig])
print(f"last state: mean {bu.mean():.2f}, peak {bu.max():.2f} ({bu.max() / bu.mean():.2f} x), "
      f"ends {bu[0] / bu.mean():.2f} and {bu[-1] / bu.mean():.2f} x")
print("wrote", OUT)
