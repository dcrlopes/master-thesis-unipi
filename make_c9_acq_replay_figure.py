#!/usr/bin/env python3
"""make_c9_acq_replay_figure.py -- where each acquisition rule sends the batch,
from the offline replay of the six Campaign 9 blocks (c9_acquisition_replay.py).

One panel per rule: the 36 picks in the gadolinia against enrichment plane,
coloured by block, the corner below 1 wt% gadolinia shaded (the region the
reported front excludes on the 2763 ppm MTC boron limit), and the 36 designs
Campaign 9 actually evaluated in its infill blocks as grey crosses.

USAGE (repository root, after the replay)
    python make_c9_acq_replay_figure.py --replay figs_c9_acq/c9_acquisition_replay.json --out figs_c9_acq/c9_acq_replay
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL = {"asrun": "As run: margin, kappa 1.5", "pof": "Probability of feasibility",
         "boron": "Margin, boron limit constrained", "pof_boron": "Probability of feasibility, boron limit constrained",
         "kappa_efpd": "Margin, kappa 0.5 on the cycle length", "gate_no_efpd": "Margin, gate without the cycle length"}
BLOCK_COLOR = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.5, "figure.dpi": 150,
                     "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "pdf.fonttype": 42})

ap = argparse.ArgumentParser()
ap.add_argument("--replay", default="figs_c9_acq/c9_acquisition_replay.json")
ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
ap.add_argument("--out", default="figs_c9_acq/c9_acq_replay")
a = ap.parse_args()

rep = json.loads(Path(a.replay).read_text())
raw = json.loads(Path(a.checkpoint).read_text())["all_raw"]
actual = np.array([[float(r["gd_wt"]), float(r["enrich"])] for r in raw[24:60]])
rules = [r for r in LABEL if r in rep["rules"]]
n = len(rules)
ncol = 3 if n > 4 else 2
nrow = int(np.ceil(n / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow), sharex=True, sharey=True)
axs = np.atleast_1d(axs).ravel()
for ax, rule in zip(axs, rules):
    R = rep["rules"][rule]
    ax.axvspan(0, 1.0, color="#D55E00", alpha=0.08, lw=0)
    ax.scatter(actual[:, 0], actual[:, 1], marker="x", s=18, c="#888888", lw=0.8, label="Evaluated in Campaign 9", zorder=2)
    for b in R["blocks"]:
        P = np.array([[p["gd_wt"], p["enrich"]] for p in b["picks"]])
        ax.scatter(P[:, 0], P[:, 1], s=26, c=BLOCK_COLOR[b["block"] - 1], edgecolors="black", linewidths=0.4,
                   label=f"Block {b['block']}", zorder=3)
    s = R["summary"]
    ax.set_title(f"{LABEL[rule]}\n{s['low_gd']} of 36 below 1 wt%, spread {s['mean_sep']:.3f}", fontsize=8.5, loc="left")
    ax.set_xlim(-0.2, 8.2)
for ax in axs[n:]:
    ax.axis("off")
for i, ax in enumerate(axs[:n]):
    if i % ncol == 0:
        ax.set_ylabel("Enrichment [wt%]")
    if i >= n - ncol:
        ax.set_xlabel(r"Gadolinia content [wt% Gd$_2$O$_3$]")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(f"{a.out}.{ext}", bbox_inches="tight")
    print(f"written -> {a.out}.{ext}")
