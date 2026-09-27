#!/usr/bin/env python3
"""
make_c9_hump_figure.py -- Single-panel figure of the mid-cycle hump of the
Campaign 9 feasible designs against the gadolinia weight fraction, with the
400 pcm threshold and the enrichment-matched sample of sec:res-c9-gd
filled. Drawing only: it reads the campaign checkpoint.

Reads  out_c9/optimization_checkpoint.json
Usage  python make_c9_hump_figure.py OUT.pdf [OUT.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND_C, BAND_H = 4.31, 0.35   # enrichment-matched sample of sec:res-c9-gd, wt%
C_FEAS, C_GREY = "#0072B2", "#444444"

D = json.load(open("out_c9/optimization_checkpoint.json"))
CN = D["constraint_names"]
F = [r for r in D["all_raw"] if all(r[c] is not None and r[c] <= 0 for c in CN)]
inb = [r for r in F if abs(r["enrich"] - BAND_C) <= BAND_H]
outb = [r for r in F if abs(r["enrich"] - BAND_C) > BAND_H]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.axhline(400, color=C_GREY, ls=":", lw=1.0)
ax.text(0.15, 560, "Threshold 400 pcm", fontsize=7.5, color=C_GREY, ha="left")
ax.scatter([r["gd_wt"] for r in outb], [r["hump_core_pcm"] for r in outb],
           s=30, facecolors="none", edgecolors=C_FEAS, linewidths=1.0,
           label="Other feasible designs")
ax.scatter([r["gd_wt"] for r in inb], [r["hump_core_pcm"] for r in inb],
           s=34, color=C_FEAS, edgecolors="k", linewidths=0.4,
           label=f"Enrichment-matched sample, $e = {BAND_C} \\pm {BAND_H}$ wt%")
ax.legend(frameon=False, loc="upper right", fontsize=7.5)
ax.set_xlabel(r"Gd$_2$O$_3$ weight fraction (wt%)")
ax.set_ylabel(r"Core hump $\Delta\rho_\mathrm{hump}$ (pcm)")
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
for out in sys.argv[1:]:
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
print(len(F), "feasible,", len(inb), "in the sample")
