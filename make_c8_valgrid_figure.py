#!/usr/bin/env python3
"""make_c8_valgrid_figure.py -- Figure for Section 5.5.6.3, the Campaign 8
surrogate-assisted search against exhaustive enumeration on the slice
(enrichment x gadolinia weight fraction, reflector 4.29 cm and 12 poisoned pins
held at C8-47). Reads the two archives through valgrid_compare.load, so
feasibility and the fronts are those of the comparison. No transport.

  (a) objective plane: the 35 grid points (feasible, infeasible), the
      enumerated front, the 20 search evaluations and the recovered front;
  (b) design plane, enrichment against gadolinia weight fraction: the same
      designs, with the members of both fronts marked.

usage:
    python make_c8_valgrid_figure.py OUT.pdf [OUT.png]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from valgrid_compare import load, front_of

OUT = sys.argv[1]
g = load("out_valgrid_grid/optimization_checkpoint.json")
o = load("out_valgrid_opt/optimization_checkpoint.json")
fg, fo = front_of(g), front_of(o)
assert len(g["X"]) == 35 and len(o["X"]) == 20, (len(g["X"]), len(o["X"]))
ie, ig = g["names"].index("enrich"), g["names"].index("gd_wt")

C_GRID, C_SRCH, C_REC = "0.45", "#E69F00", "#C8102E"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.2))


def ordered(F, idx):
    pts = F[idx]
    k = np.argsort(-pts[:, 0])          # by cycle length, ascending (F[:,0] = -EFPD)
    return -pts[k, 0], pts[k, 1]


# ---- (a) objective plane ----------------------------------------------------------
E_g, P_g = -g["F"][:, 0], g["F"][:, 1]
E_o, P_o = -o["F"][:, 0], o["F"][:, 1]
ax.scatter(E_g[~g["feas"]], P_g[~g["feas"]], s=30, facecolors="none", edgecolors=C_GRID, lw=1.0,
           zorder=2, label="Grid point, infeasible")
ax.scatter(E_g[g["feas"]], P_g[g["feas"]], s=30, color=C_GRID, zorder=2, label="Grid point, feasible")
x, y = ordered(g["F"], fg)
ax.plot(x, y, color="0.25", lw=1.4, zorder=3, label=f"Enumerated front ({len(fg)})")
ax.scatter(E_o, P_o, s=40, marker="^", color=C_SRCH, edgecolors="0.2", lw=0.4, zorder=4,
           label=f"Search evaluation ({len(E_o)})")
x, y = ordered(o["F"], fo)
ax.plot(x, y, color=C_REC, lw=1.6, zorder=5, label=f"Recovered front ({len(fo)})")
ax.scatter(x, y, s=60, marker="^", color=C_REC, edgecolors="0.2", lw=0.5, zorder=6)
ax.set_xlabel("Cycle length [EFPD]")
ax.set_ylabel(r"Core $F_{\Delta H}$ [-]")
ax.set_title("(a) Objective plane", fontsize=10)
ax.grid(alpha=0.22, lw=0.6)
ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95)

# ---- (b) design plane ----------------------------------------------------------------
X_g, X_o = g["X"], o["X"]
bx.scatter(X_g[~g["feas"], ie], X_g[~g["feas"], ig], s=46, facecolors="none", edgecolors=C_GRID,
           lw=1.0, zorder=2, label="Grid point, infeasible")
bx.scatter(X_g[g["feas"], ie], X_g[g["feas"], ig], s=46, color=C_GRID, zorder=2,
           label="Grid point, feasible")
bx.scatter(X_g[fg, ie], X_g[fg, ig], s=150, facecolors="none", edgecolors="0.1", lw=1.4, zorder=3,
           label="Enumerated-front member")
bx.scatter(X_o[:, ie], X_o[:, ig], s=40, marker="^", color=C_SRCH, edgecolors="0.2", lw=0.4, zorder=4,
           label="Search evaluation")
bx.scatter(X_o[fo, ie], X_o[fo, ig], s=64, marker="^", color=C_REC, edgecolors="0.2", lw=0.5, zorder=5,
           label="Recovered-front member")
bx.set_xlim(2.7, 7.3)
bx.set_ylim(-0.6, 8.7)
bx.set_xlabel(r"Enrichment $e$ [wt\%]".replace("\\%", "%"))
bx.set_ylabel(r"Gadolinia weight fraction $w_\mathrm{Gd}$ [wt%]")
bx.set_title("(b) Design plane", fontsize=10)
bx.grid(alpha=0.22, lw=0.6)
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight")
print(f"grid {len(X_g)} ({int(g['feas'].sum())} feasible), front {len(fg)} | "
      f"search {len(X_o)} ({int(o['feas'].sum())} feasible), front {len(fo)}")
