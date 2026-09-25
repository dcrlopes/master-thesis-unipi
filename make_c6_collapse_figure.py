#!/usr/bin/env python3
"""make_c6_collapse_figure.py -- batch collapse and the minimum separation in
Campaign 6, for Section 5.4.1.

For every infill design, the separation d(a, b) of Equation 4.40 from the
nearest design evaluated before it (the archive and the earlier picks of the
same batch), each variable divided by its range in the Campaign 6 design space.
Read from the checkpoint, no transport.

  (a) the separation against the evaluation index, by block, with the
      threshold in force in each iteration drawn as a step: delta_min = 0.14
      from evaluation 61, halved as many times as the batch needed
      (Section 4.12.3). The relaxation depth is not recorded in the archive, so
      the step shows the shallowest level that admits every design of the
      batch, the lowest level the selection can have used.
  (b) the same separations sorted within each block, smallest first, with the
      three levels 0.14, 0.07 and 0.035.

usage (plot only, runs anywhere):
    python make_c6_collapse_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
B6 = {"enrich_inner": (2.0, 17.174), "enrich_outer": (2.0, 17.174), "gd_wt": (0.0, 8.0),
      "pitch": (1.15, 1.43), "refl_thick": (2.0, 19.5), "gd_pins": (12, 40)}
DMIN, N_DOE, N_INFILL, FIRST_DIV = 0.14, 42, 6, 3      # the threshold enters at iteration 4

ck = json.load(open(SRC))
raw, dv = ck["all_raw"], ck["design_variables"]
lo = np.array([B6[v][0] for v in dv])
hi = np.array([B6[v][1] for v in dv])
Z = (np.array([[r[v] for v in dv] for r in raw], float) - lo) / (hi - lo)
n = len(dv)
d = np.full(len(Z), np.nan)
for i in range(1, len(Z)):
    d[i] = min(np.linalg.norm(Z[i] - Z[j]) / np.sqrt(n) for j in range(i))

# the shallowest relaxation that admits each batch, iterations with the threshold in force
n_it = (len(raw) - N_DOE) // N_INFILL
level = []
for it in range(n_it):
    a, b = N_DOE + N_INFILL * it, N_DOE + N_INFILL * (it + 1)
    m = np.nanmin(d[a:b])
    if it < FIRST_DIV:
        level.append(np.nan)
    else:
        k = 0
        while DMIN / 2 ** k > m:
            k += 1
        level.append(DMIN / 2 ** k)

BLOCKS = [("Bare uncertainty ranking", 42, 60, "#E69F00", "o"),
          ("Batch diversity", 60, 66, "#0072B2", "s"),
          ("Feasibility margin", 66, 84, "#009E73", "D"),
          ("Convergence", 84, 114, "#6A3D9A", "^")]
C_THR = "#B22222"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True,
                             gridspec_kw={"width_ratios": (1.5, 1.0)})

# ---- (a) against the evaluation index ------------------------------------------
for k, (name, a, b, col, mk) in enumerate(BLOCKS):
    if k % 2 == 0:
        ax.axvspan(a + 0.5, b + 0.5, color="0.93", zorder=0)
    ax.scatter(np.arange(a + 1, b + 1), d[a:b], s=30, color=col, marker=mk,
               edgecolors="white", lw=0.5, zorder=3, label=name)
xs, ys = [], []
for it, lv in enumerate(level):
    if np.isnan(lv):
        continue
    a, b = N_DOE + N_INFILL * it, N_DOE + N_INFILL * (it + 1)
    xs += [a + 0.5, b + 0.5]
    ys += [lv, lv]
ax.step(xs, ys, where="post", color=C_THR, lw=1.6, zorder=2,
        label="Threshold in force, shallowest level that admits the batch")
ax.axhline(DMIN, color=C_THR, lw=1.1, ls="-", zorder=1)
for lv, lab in ((DMIN, r"$\delta_\mathrm{min}=0.14$"), (DMIN / 2, "0.07"),
                (DMIN / 4, "0.035"), (DMIN / 8, "0.0175")):
    ax.text(115.6, lv, lab, fontsize=8.2, color=C_THR, va="center", ha="left")
ax.set_xlim(42, 115)
ax.set_xlabel("Transport evaluation")
ax.set_ylabel(r"Separation $d$ from the nearest earlier design [-]", fontsize=9.5)
ax.set_title("(a) By evaluation, with the threshold in force", fontsize=10)

# ---- (b) sorted within each block --------------------------------------------------
for name, a, b, col, mk in BLOCKS:
    v = np.sort(d[a:b])
    frac = (np.arange(len(v)) + 0.5) / len(v)
    bx.plot(frac, v, marker=mk, ms=4.5, color=col, lw=1.2, zorder=3)
for lv, ls in ((DMIN, "-"), (DMIN / 2, "--"), (DMIN / 4, "-.")):
    bx.axhline(lv, color=C_THR, lw=1.1, ls=ls, zorder=2)
bx.set_xlim(0, 1)
bx.set_xlabel("Fraction of the block, smallest separation first")
bx.set_title("(b) Sorted within each block", fontsize=10)

for axis in (ax, bx):
    axis.set_yscale("log")
    axis.set_ylim(4e-4, 0.6)
    axis.grid(alpha=0.22, lw=0.6, which="major")

handles, labels = ax.get_legend_handles_labels()
handles += [plt.Line2D([], [], color=C_THR, lw=1.1, ls="-"),
            plt.Line2D([], [], color=C_THR, lw=1.1, ls="--"),
            plt.Line2D([], [], color=C_THR, lw=1.1, ls="-.")]
labels += [r"$\delta_\mathrm{min} = 0.14$", "First relaxation, 0.07", "Second relaxation, 0.035"]
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
for it, lv in enumerate(level):
    a, b = N_DOE + N_INFILL * it + 1, N_DOE + N_INFILL * (it + 1)
    print(f"iteration {it + 1:2d}, evaluations {a}-{b}: min d {np.nanmin(d[a - 1:b]):.4f}, "
          f"threshold in force {'none' if np.isnan(lv) else f'{lv:.4f}'}")
print("wrote", OUT)
