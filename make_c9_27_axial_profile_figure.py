#!/usr/bin/env python3
"""
make_c9_27_axial_profile_figure.py -- the axial power profile of C9-27 during
the eight-layer depletion, in the form of panel (c) of the thesis figure
c9_axial_peaking (c9_axial_correction.py, which draws C9-47 with 12 layers).
Drawing only, no transport.

Each curve is the power of each of the 8 layers of 15 cm divided by the
core average, at one core-average burnup, from the layer_share of the
stored depletion steps. The five seeds of C9-27 are averaged, and the
error bars give the standard deviation of the seeds.

Reads c9_dep_core3d_d27_L8 and c9_dep_core3d_d27_seed2..5 /runs.json.
Usage: python make_c9_27_axial_profile_figure.py <prefix>     # <prefix>.pdf and .png
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRS = ["c9_dep_core3d_d27_L8"] + [f"c9_dep_core3d_d27_seed{k}" for k in (2, 3, 4, 5)]
runs = [json.load(open(d + "/runs.json"))["d27"] for d in DIRS]
bu = runs[0]["bu_hist"]
assert all(r["bu_hist"] == bu for r in runs) and all(r["layers"] == 8 for r in runs)
n = 8
z = (np.arange(n) + 0.5) * 120.0 / n                     # mid-height of each layer, cm
# layer power / average: shape (seed, step, layer)
S = np.array([[np.array(s["layer_share"]) * n for s in r["states"]] for r in runs])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(4.2, 4.6))
shades = ["#999999", "#56B4E9", "#0072B2", "#000000"]
for j, Bt in enumerate((0.0, 7.5, 13.5, 21.5)):
    i = bu.index(Bt)
    m, sd = S[:, i].mean(axis=0), S[:, i].std(axis=0, ddof=1)
    ax.errorbar(m, z, xerr=sd, fmt="-o", ms=3.5, lw=1.4, capsize=2, color=shades[j],
                label=f"B = {Bt:g} MWd/kgHM")
    print(f"B = {Bt:4.1f}: layer power / average", np.round(m, 3).tolist(), "max", round(m.max(), 3))
ax.axvline(1.0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("Layer power / core average")
ax.set_ylabel("Height from the bottom of the active fuel [cm]")
ax.set_ylim(0, 120)
ax.set_xlim(0.2, 1.6)
ax.set_yticks(np.arange(0, 121, 15))
ax.set_title("C9-27, 8 layers, 5 seeds: axial power profile", fontsize=9, loc="left")
ax.legend(loc="center left", frameon=False, fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[1] + ".pdf", bbox_inches="tight")
fig.savefig(sys.argv[1] + ".png", dpi=200, bbox_inches="tight")
print("written", sys.argv[1] + ".pdf")
