#!/usr/bin/env python3
"""make_c4_trajectory_figure.py -- Campaign 4 constraint plane for Section 5.3.1.

Two panels, both read from the Campaign 4 checkpoint, no transport:

  (a) all 54 designs on the plane of the two active constraints, the reactivity
      limit at 1.35 and the peaking bound at 2.0. The feasible quadrant is
      shaded and empty.

  (b) the same plane over the window the three infill iterations occupy, which
      is the dashed box of panel (a). The two designs that come closest to the
      feasible quadrant are marked: one misses the reactivity limit by 30 pcm
      while feasible on peaking, the other misses the peaking bound by 0.010
      while feasible on reactivity, together with C4-50, the only other
      iteration-3 design that satisfies the reactivity limit.

usage (plot only, runs anywhere):
    python make_c4_trajectory_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC, OUT = sys.argv[1], sys.argv[2]
raw = json.load(open(SRC))["all_raw"]

K_LIM, F_LIM = 1.35, 2.0
PHASES = [("Design of experiments (36)", 0, 36, "#8C8C8C", "o", 26),
          ("Iteration 1 (6)", 36, 42, "#E69F00", "s", 38),
          ("Iteration 2 (6)", 42, 48, "#009E73", "D", 38),
          ("Iteration 3 (6)", 48, 54, "#6A3D9A", "^", 44)]

k = np.array([r["k_bol"] for r in raw])
F = np.array([r["peaking"] for r in raw])

C_K, C_F, C_FEAS = "#1B7837", "#B22222", "#1B7837"
ZOOM = (1.335, 1.412, 1.895, 2.165)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 5.0))

for panel, (x0, x1, y0, y1) in ((ax, (1.10, 1.56, 1.80, 4.30)), (bx, ZOOM)):
    panel.add_patch(Rectangle((x0, y0), K_LIM - x0, F_LIM - y0, facecolor=C_FEAS,
                              alpha=0.10, edgecolor="none", zorder=0))
    panel.axvline(K_LIM, color=C_K, lw=1.4, zorder=2)
    panel.axhline(F_LIM, color=C_F, lw=1.4, zorder=2)
    for name, a, b, colour, mk, size in PHASES:
        panel.scatter(k[a:b], F[a:b], marker=mk, s=size, color=colour,
                      edgecolors="white", lw=0.5, zorder=3, label=name)
    panel.set_xlim(x0, x1)
    panel.set_ylim(y0, y1)
    panel.set_xlabel(r"Core $k_\mathrm{BOL}$ [-]")
    panel.grid(alpha=0.22, lw=0.6)

# ---- (a) the whole campaign ----------------------------------------------
ax.set_ylabel(r"Core $F_{\Delta H}$ [-]")
ax.set_title("(a) All 54 designs on the constraint plane", fontsize=10.5)
ax.legend(fontsize=8.4, loc="upper right", framealpha=0.92)
ax.text(1.115, 1.87, "Feasible region (FR)", color=C_FEAS, fontsize=10.5, weight="bold")
ax.text(K_LIM - 0.006, 4.22, r"$k_{\max} = 1.35$", color=C_K, fontsize=9,
        rotation=90, va="top", ha="right")
ax.text(1.555, F_LIM + 0.05, r"$F_{\Delta H} = 2.0$", color=C_F, fontsize=9, ha="right")
ax.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2],
                       facecolor="none", edgecolor="0.25", ls="--", lw=1.0, zorder=4))
ax.annotate("Window of panel (b)", xy=(ZOOM[1], ZOOM[3]), xytext=(1.398, 3.00),
            fontsize=9, color="0.25", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0))

# ---- (b) the walk along the boundary --------------------------------------
bx.set_ylabel(r"Core $F_{\Delta H}$ [-]")
bx.set_title("(b) The walk along the boundary, enlarged", fontsize=10.5)
bx.text(1.3385, 1.945, "FR", color=C_FEAS, fontsize=13, weight="bold")
bx.text(K_LIM + 0.0010, 2.160, r"$k_{\max}$", color=C_K, fontsize=9,
        rotation=90, va="top", ha="left")
bx.text(1.410, F_LIM + 0.005, r"$F_{\Delta H} = 2.0$", color=C_F, fontsize=9, ha="right")

i_peak = 53   # feasible on peaking, misses the reactivity limit by 30 pcm
i_reac = 49   # feasible on reactivity, misses the peaking bound by 0.010
i_reac2 = 50  # the other iteration-3 design that satisfies the reactivity limit
i_doe = int(np.argmin(F[:36]))

for idx, off in ((i_reac, (-46, 13)), (i_reac2, (-50, 36)), (i_peak, (-46, -16)),
                 (i_doe, (-44, 15))):
    bx.annotate(f"C4-{idx}", xy=(k[idx], F[idx]), xytext=off,
                textcoords="offset points", fontsize=9.5, color="0.15",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="0.15", lw=1.0,
                                shrinkA=2, shrinkB=3))

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")

print(f"closest on peaking: idx {i_peak}, k {k[i_peak]:.4f}, F {F[i_peak]:.3f}")
print(f"closest on reactivity: idx {i_reac}, k {k[i_reac]:.4f}, F {F[i_reac]:.3f}")
print(f"best DOE peaking: idx {i_doe}, k {k[i_doe]:.4f}, F {F[i_doe]:.4f}")
print(f"designs in the feasible quadrant: {int(((k <= K_LIM) & (F <= F_LIM)).sum())}")
print("wrote", OUT)
