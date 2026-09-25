#!/usr/bin/env python3
"""make_c3_asm_core_figure.py -- Campaign 3 proxy figure for Section 5.2.3.

Every feasible Campaign 3 design rescored at core level. No transport: the
figure reads core_rescore/core_rescore.csv, which holds the archived assembly
value, the measured core value and its standard error, the two ranks, and the
cycle length of each design.

The figure carries the three statements of the subsection: the discrepancy is a
factor between the two guide lines, the peaking constraint is violated by every
design once it is evaluated on the core, and the ordering near the optimum does
not survive the change of model.

usage (plot only, runs anywhere):
    python make_c3_asm_core_figure.py CORE_RESCORE.csv OUT.pdf [OUT.png]
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(SRC)))
for r in rows:
    for k in ("idx", "asm_rank", "core_rank"):
        r[k] = int(float(r[k]))
    for k in ("fdh_asm", "fdh_core", "fdh_core_sem", "EFPD", "ratio"):
        r[k] = float(r[k])

asm = np.array([r["fdh_asm"] for r in rows])
core = np.array([r["fdh_core"] for r in rows])
sem = np.array([r["fdh_core_sem"] for r in rows])
efpd = np.array([r["EFPD"] for r in rows])

n = len(rows)
d2 = sum((r["asm_rank"] - r["core_rank"]) ** 2 for r in rows)
rho = 1 - 6 * d2 / (n * (n * n - 1))
lo, hi = min(r["ratio"] for r in rows), max(r["ratio"] for r in rows)

fig, ax = plt.subplots(figsize=(8.6, 5.2))

# the envelope of the discrepancy
xs = np.array([asm.min() - 0.04, asm.max() + 0.04])
for k, ls in ((hi, "--"), (lo, ":")):
    ax.plot(xs, k * xs, ls, color="0.55", lw=1.0, zorder=1)
YTOP = core.max() + 0.12
for k, dy in ((hi, 6), (lo, -14)):
    xl = min(xs[1] - 0.02, 0.96 * YTOP / k)      # keep the label inside the axes
    ax.annotate(f"Ratio {k:.2f}", xy=(xl, k * xl), xytext=(0, dy),
                textcoords="offset points", ha="right", fontsize=8.4, color="0.4")

# the peaking constraint at core level
ax.axhline(2.0, color="#b03a3a", lw=1.3, zorder=2)
ax.annotate(r"Peaking constraint at core level, $F_{\Delta H} \leq 2.0$."
            "\nEvery one of the 36 designs is above it.",
            xy=(asm.max() + 0.03, 2.0), xytext=(-2, -8), textcoords="offset points",
            ha="right", va="top", fontsize=8.6, color="#b03a3a")

ax.errorbar(asm, core, yerr=sem, fmt="none", ecolor="0.6", lw=0.8, zorder=3)
sc = ax.scatter(asm, core, c=efpd, cmap="viridis", s=52, zorder=4,
                edgecolor="0.25", linewidth=0.4)

# the three designs whose rank moves most between the two models, labelled in the
# empty band to the right of the dense cluster so that no point is covered
MARK = [("C3-51", 1.335, 2.34), ("C3-49", 1.335, 2.24), ("C3-47", 1.335, 2.14)]
for name, tx, ty in MARK:
    idx = int(name.split("-")[1])
    r = next(q for q in rows if q["idx"] == idx)
    ax.annotate(f"{name}: assembly #{r['asm_rank']} to core #{r['core_rank']}",
                xy=(r["fdh_asm"], r["fdh_core"]), xytext=(tx, ty),
                textcoords="data", fontsize=8.4, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9,
                                shrinkA=2, shrinkB=4,
                                connectionstyle="arc3,rad=0.12"),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.7))

ax.text(0.025, 0.975, rf"Spearman $\rho = {rho:+.3f}$", transform=ax.transAxes,
        ha="left", va="top", fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#1f3b57", lw=1.0))

ax.set_xlabel(r"Assembly $F_{\Delta H}$, the Campaign 3 objective")
ax.set_ylabel(r"Core $F_{\Delta H}$, measured with 3 to 8 seeds")
ax.set_xlim(xs[0], xs[1])
ax.set_ylim(1.80, YTOP)
ax.grid(alpha=0.25, lw=0.6)

cb = fig.colorbar(sc, ax=ax, pad=0.015)
cb.set_label("Cycle length [EFPD]")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
print(f"wrote {OUT} | n={n} rho={rho:+.3f} ratio {lo:.3f} to {hi:.3f} "
      f"| above 2.0: {int((core > 2.0).sum())}/{n}")
