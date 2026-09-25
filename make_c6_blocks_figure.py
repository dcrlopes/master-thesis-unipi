#!/usr/bin/env python3
"""make_c6_blocks_figure.py -- Campaign 6 acquisition figure for Section 5.4.2.

Two panels, both read from the Campaign 6 checkpoint, no transport:

  (a) hypervolume against the number of evaluations, with the design of
      experiments and the four infill blocks separated. Each block is labelled
      by the revision of the acquisition function in force during it. The
      hypervolume is flat until the third revision is in place.

  (b) what each block returned: the fraction of its evaluations that are
      feasible, and the minimum normalised separation of its picks from the
      designs already in the archive. The separation of the first infill block
      shows the batch collapse; the feasible fraction of the second shows the
      criterion ranking without regard to the constraints.

usage (plot only, runs anywhere):
    python make_c6_blocks_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(SRC))
hv = np.asarray(d["hv_history"], dtype=float)
raw, CN = d["all_raw"], d["constraint_names"]

# block boundaries as the repository snapshots record them, 42 -> 60 -> 66 -> 84 -> 114
BLOCKS = [("Design of experiments", "DoE", 0, 42, 0),
          ("Bare uncertainty ranking", "Uncertainty", 42, 60, 3),
          ("Batch diversity", "Diversity", 60, 66, 4),
          ("Feasibility margin", "Margin", 66, 84, 7),
          ("Convergence", "Convergence", 84, 114, 12)]

VARS = ["enrich_inner", "enrich_outer", "gd_wt", "pitch", "refl_thick", "gd_pins"]
span = [max(r[v] for r in raw) - min(r[v] for r in raw) or 1.0 for v in VARS]


def nearest(a, pool):
    """Minimum normalised mean-per-variable distance of a to the designs in pool."""
    best = math.inf
    for b in pool:
        s = sum(((a[v] - b[v]) / sp) ** 2 for v, sp in zip(VARS, span))
        best = min(best, math.sqrt(s) / math.sqrt(len(VARS)))
    return best


feas, seps = [], []
for _, _, a, b, _ in BLOCKS:
    blk = raw[a:b]
    feas.append(sum(1 for r in blk if all(r[c] <= 0 for c in CN)) / len(blk))
    seps.append(min(nearest(r, raw[:a + k]) for k, r in enumerate(blk)) if a else np.nan)

C_HV, C_FE, C_SEP, C_SHADE = "#0072B2", "#009E73", "#D55E00", "#EAEAEA"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.3))

# ---- (a) hypervolume against the evaluation count -------------------------
n = np.array([42] + [42 + 6 * k for k in range(1, len(hv))])
ax.step(n, hv, where="post", color=C_HV, lw=1.8, zorder=3)
ax.plot(n, hv, "o", ms=4.2, color=C_HV, zorder=4)
for k, (_, short, a, b, _) in enumerate(BLOCKS[1:], 1):
    if k % 2:
        ax.axvspan(a, b, color=C_SHADE, zorder=0)
    ax.text((a + b) / 2, 1680 if k == 2 else 1655, f"{short}\n{a + 1} to {b}",
            ha="center", va="center", fontsize=8.0, linespacing=1.25)
    ax.plot([a, b], [1636] * 2, color="0.45", lw=0.9, solid_capstyle="butt")
    for e in (a, b):
        ax.plot([e, e], [1630, 1636], color="0.45", lw=0.9)
ax.axvline(42, color="0.5", lw=0.8, ls=":")
ax.annotate(f"+{100 * (hv[5] - hv[4]) / hv[4]:.1f} per cent in the\nfirst margin iteration,\n"
            f"+{100 * (hv[7] - hv[4]) / hv[4]:.1f} per cent over\nthe margin block",
            xy=(72, hv[5]), xytext=(86, 1330), fontsize=8.5, color=C_HV,
            arrowprops=dict(arrowstyle="->", color=C_HV, lw=1.0))
ax.set_xlabel("Transport evaluations")
ax.set_ylabel("Hypervolume")
ax.set_xlim(36, 118)
ax.set_ylim(1250, 1720)
ax.set_title("(a) Hypervolume by acquisition revision", fontsize=10)
ax.grid(alpha=0.25, lw=0.6)

# ---- (b) feasible fraction and minimum separation per block ---------------
x = np.arange(len(BLOCKS))
bx.bar(x, feas, width=0.56, color=C_FE, alpha=0.85, zorder=2,
       label="Feasible fraction of the block")
for xi, f in zip(x, feas):
    bx.text(xi, f + 0.02, f"{f:.2f}", ha="center", fontsize=8.4, color=C_FE)
bx.set_ylim(0, 1.15)
bx.set_ylabel("Feasible fraction", color=C_FE)
bx.tick_params(axis="y", colors=C_FE)
bx.set_xticks(x)
bx.set_xticklabels([b[0].replace("\n", " ") for b in BLOCKS], fontsize=8.0, rotation=18,
                   ha="right")

cx = bx.twinx()
cx.semilogy(x, seps, "D--", ms=6, color=C_SEP, lw=1.3, zorder=3,
            label="Minimum separation of the picks")
cx.axhline(0.14, color=C_SEP, lw=0.9, ls=":")
cx.text(-0.45, 0.155, r"$\delta_{\min}=0.14$", color=C_SEP,
        fontsize=8.2, ha="left")
cx.set_ylabel("Minimum normalised separation", color=C_SEP)
cx.tick_params(axis="y", colors=C_SEP)
cx.set_ylim(5e-4, 1.0)
bx.set_title("(b) What each block returned", fontsize=10)
bx.grid(alpha=0.25, lw=0.6, axis="y")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
print("wrote", OUT, "feasible", [round(f, 3) for f in feas],
      "separation", [None if np.isnan(s) else round(s, 4) for s in seps])
