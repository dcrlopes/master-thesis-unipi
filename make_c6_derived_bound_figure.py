#!/usr/bin/env python3
"""make_c6_derived_bound_figure.py -- the upper reactivity bound derived from
the bank-worth screen of eight Campaign 6 designs, for Section 5.4.1.4.
No transport: reads the screen written by rod_bank_worth.py --screen.

  (a) the four regulating banks: rodded against unrodded core eigenvalue, the
      limit of 0.99, and the curve of the smallest measured reactivity worth
      assumed for every design, which meets the limit at the bound of 1.126;
  (b) the same with all 32 control rod assemblies inserted, retrospective
      bound of 1.368.

The bound is the one derive_kmax.py writes:
    k_max = 1 / (1 - (rho_min - margin)),  rho = 1/k_rod - 1/k0.

usage:
    python make_c6_derived_bound_figure.py banks_screen_v3/banks_B4C.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(SRC))
MARGIN = d["margin_pcm"] * 1e-5
rows = [s for s in d["states"] if s.get("mode") == "screen"]
idx = [s["idx"] for s in rows]
k0 = np.array([s["k0"] for s in rows])

C_PT, C_LIM, C_CURVE, C_BOUND = "#0072B2", "#B22222", "#D55E00", "0.35"
fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
panels = [("ALLRE", "(a) Four regulating banks", r"Core $k_\mathrm{eff}$, regulating banks inserted [-]",
           (0.66, 1.13), {59: (8, -28), 110: (-34, 6), 107: (-34, 10), 26: (-34, 10), 7: (-46, 4),
                          101: (4, -24), 85: (34, -22), 86: (34, -4)}),
          ("SCRAM", "(b) All rods in", r"Core $k_\mathrm{eff}$, all rods in [-]",
           (0.66, 1.13), {59: (26, -8), 110: (-30, 10), 107: (26, -12), 26: (-30, 10), 7: (-30, 10),
                          101: (26, -12), 85: (26, -14), 86: (26, -8)})]

for p, (grp, title, ylab, ylim, off) in zip(axes, panels):
    k = np.array([s["states"][grp]["k"] for s in rows])
    rho = 1.0 / k - 1.0 / k0
    rho_min = rho.min()
    k_max = 1.0 / (1.0 - (rho_min - MARGIN))
    assert abs(k_max - {"ALLRE": 1.126, "SCRAM": 1.368}[grp]) < 5e-4, (grp, k_max)
    kk = np.linspace(1.0, 1.42, 200)
    p.plot(kk, 1.0 / (1.0 / kk + rho_min), color=C_CURVE, lw=1.3, ls="--", zorder=2,
           label=f"Smallest measured worth, {rho_min * 1e5:.0f} pcm, assumed for every design")
    p.axhline(1.0 - MARGIN, color=C_LIM, lw=1.2, label="Controllability limit, 0.99")
    p.axvline(k_max, color=C_BOUND, lw=1.0, ls=":", label="Derived upper reactivity bound")
    p.scatter(k0, k, s=44, color=C_PT, edgecolors="white", lw=0.5, zorder=3,
              label="Campaign 6 designs, measured")
    for i, x, y in zip(idx, k0, k):
        p.annotate(f"C6-{i}", (x, y), xytext=off[i], textcoords="offset points", fontsize=8.5,
                   ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color="0.35", lw=0.6, shrinkA=0, shrinkB=4))
    p.text(k_max, ylim[1] - 0.012 * (ylim[1] - ylim[0]), f" {k_max:.3f}", fontsize=8.5,
           ha="left", va="top", color=C_BOUND)
    p.set_xlim(1.03, 1.40)
    p.set_ylim(*ylim)
    p.set_xlabel(r"Core $k_\mathrm{eff}$ at beginning of life, all rods out [-]")
    p.set_ylabel(ylab)
    p.set_title(title, fontsize=10)
    p.grid(alpha=0.22, lw=0.6)
    print(f"{grp}: rho_min = {rho_min * 1e5:.0f} pcm (C6-{idx[int(rho.argmin())]}), k_max = {k_max:.4f}")

h, l = axes[0].get_legend_handles_labels()
h = [h[3], h[1], h[2], h[0]]
l = [l[3], l[1], l[2], "Smallest measured worth assumed for every design"]
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.6, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.10, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
