#!/usr/bin/env python3
"""make_c5_normalisation_figure.py -- the normalisation of the constraint
violation on the Campaign 5 archive, for Section 5.3.2.

Read from the Campaign 5 checkpoint, no transport. With no feasible design,
NSGA-II orders the population by the sum of the constraint violations. The
raw sum adds each violation in its own unit; from Campaign 6 each violation is
divided by its limit first (Section 4.4.5, Table 4.x, the Campaign 6 scales:
k_min 1.02, k_max 1.35, enrichment 19.75 wt%, peaking 2.0, vessel 90 cm).

  (a) the share of the archive's total violation held by each constraint under
      the two weightings;
  (b) the rank of every design under the raw sum against its rank under the
      normalised sum, with the Spearman coefficient.

On this archive the vessel-fit and enrichment constraints are never violated,
so the constraints whose units differ most from the others contribute nothing
and the two orderings nearly coincide. The figure records that.

usage (plot only, runs anywhere):
    python make_c5_normalisation_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SRC, OUT = sys.argv[1], sys.argv[2]
raw = json.load(open(SRC))["all_raw"]

NAMES = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom"]
LABEL = {"g_kmin": r"$g_{k\min}$", "g_kmax": r"$g_{k\max}$", "g_enr": r"$g_\mathrm{enr}$",
         "g_peak": r"$g_\mathrm{peak}$", "g_geom": r"$g_\mathrm{geom}$"}
SCALE = {"g_kmin": 1.02, "g_kmax": 1.35, "g_enr": 19.75, "g_peak": 2.0, "g_geom": 90.0}
COL = {"g_kmin": "#56B4E9", "g_kmax": "#0072B2", "g_enr": "#CC79A7", "g_peak": "#D55E00",
       "g_geom": "#009E73"}

G = np.array([[max(r[n], 0.0) for n in NAMES] for r in raw])       # positive part only
S = np.array([SCALE[n] for n in NAMES])
raw_sum, norm_sum = G.sum(1), (G / S).sum(1)
share_raw = G.sum(0) / G.sum()
share_norm = (G / S).sum(0) / (G / S).sum()
rank_raw, rank_norm = stats.rankdata(raw_sum), stats.rankdata(norm_sum)
rho = stats.spearmanr(raw_sum, norm_sum)[0]
shift = int(np.abs(rank_raw - rank_norm).max())

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": (1.0, 1.15)})

# ---- (a) share of the archive violation by constraint ---------------------
for y, share, lab in ((1.0, share_raw, "Raw sum, each violation\nin its own unit"),
                      (0.0, share_norm, "Normalised sum, each\nviolation over its limit")):
    left = 0.0
    for n, s in zip(NAMES, share):
        if s > 0:
            ax.barh(y, s, left=left, height=0.5, color=COL[n], edgecolor="white", lw=0.6)
            if s > 0.08:
                ax.text(left + s / 2, y, f"{100 * s:.0f} %", ha="center", va="center",
                        fontsize=9, color="white")
            left += s
ax.set_yticks([1.0, 0.0])
ax.set_yticklabels(["Raw sum, each violation\nin its own unit",
                    "Normalised sum, each\nviolation over its limit"], fontsize=9)
ax.set_xlim(0, 1)
ax.set_xlabel("Share of the archive's total violation")
ax.set_title("(a) Which constraint the sum is made of", fontsize=10.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[n]) for n in NAMES]
ax.legend(handles, [LABEL[n] for n in NAMES], fontsize=8.5, loc="lower center",
          bbox_to_anchor=(0.5, -0.42), ncol=5, frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---- (b) rank against rank -------------------------------------------------
bx.plot([1, 60], [1, 60], color="0.6", lw=0.9, ls="--", zorder=1)
bx.scatter(rank_raw, rank_norm, s=26, color="#4C72B0", edgecolors="white", lw=0.5, zorder=3)
bx.set_xlabel("Rank under the raw sum")
bx.set_ylabel("Rank under the normalised sum")
bx.set_title("(b) Ordering of the 60 designs", fontsize=10.5)
bx.set_xlim(0, 61)
bx.set_ylim(0, 61)
bx.set_aspect("equal")
bx.grid(alpha=0.22, lw=0.6)
bx.text(0.04, 0.96, f"Rank 1 = least violation\nSpearman $\\rho_S = {rho:.3f}$\nlargest shift {shift} places",
        transform=bx.transAxes, ha="left", va="top", fontsize=9, color="0.15")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")

print("share raw:", {n: round(float(s), 3) for n, s in zip(NAMES, share_raw)})
print("share normalised:", {n: round(float(s), 3) for n, s in zip(NAMES, share_norm)})
print(f"Spearman {rho:.3f}, largest rank shift {shift}, violated constraints:",
      [n for n, c in zip(NAMES, (G > 0).sum(0)) if c > 0])
print("wrote", OUT)
