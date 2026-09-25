#!/usr/bin/env python3
"""make_c6_margin_figure.py -- the batch-diversity block of Campaign 6 against
the upper reactivity bound, for Section 5.4.1.

The checkpoint does not store the constraint surrogate's predictions, so they
are recomputed: the campaign's own GPSurrogate (reactor_optimization.py,
random_state=0, deterministic) is refitted on the 60 designs archived before
the block, on the normalised constraint g_kmax / 1.35, and asked for the six
designs the block selected. No transport.

  (a) predicted core k_eff, mean and one standard deviation, against the
      measured value, for the six designs, with the bound of 1.35;
  (b) inner enrichment against gadolinia fraction for the 60 archived designs
      and the six picks, with the enrichment bound of 17.174 wt%, and archive
      design C6-25, the one measured design in the corner the picks went to.

usage (plot only, runs anywhere):
    python make_c6_margin_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from reactor_optimization import GPSurrogate

SRC, OUT = sys.argv[1], sys.argv[2]
ck = json.load(open(SRC))
raw, dv = ck["all_raw"], ck["design_variables"]
K_MAX, E_BOUND = 1.35, 17.174
A, B = 60, 66                                     # the batch-diversity block, 0-based

X = np.array([[r[v] for v in dv] for r in raw], float)
g = np.array([r["g_kmax"] for r in raw]) / K_MAX
sur = GPSurrogate().fit(X[:A], g[:A, None])
m, s = sur.predict(X[A:B])
k_pred, k_sd = K_MAX + K_MAX * m[:, 0], K_MAX * s[:, 0]
k_meas = np.array([raw[i]["keff_core_bol"] for i in range(A, B)])
names = [f"C6-{i}" for i in range(A, B)]

C_PRED, C_MEAS, C_BOUND, C_ARCH = "#0072B2", "#D55E00", "#B22222", "0.72"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 5.2))

# ---- (a) predicted against measured ------------------------------------------
x = np.arange(B - A)
ax.axhspan(K_MAX, 1.46, color=C_BOUND, alpha=0.07, lw=0)
ax.axhline(K_MAX, color=C_BOUND, lw=1.3, label="Upper reactivity bound, 1.35")
ax.errorbar(x - 0.08, k_pred, yerr=k_sd, fmt="o", ms=7, mfc="white", mec=C_PRED, color=C_PRED,
            capsize=4, lw=1.2, label="Surrogate prediction, mean and one standard deviation")
ax.plot(x + 0.08, k_meas, "s", ms=7, color=C_MEAS, label="Measured core eigenvalue")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylim(1.11, 1.46)
ax.set_ylabel(r"Core $k_\mathrm{eff}$ at beginning of life [-]")
ax.set_title("(a) Upper reactivity bound: predicted and measured", fontsize=10)
ax.grid(alpha=0.22, lw=0.6)
ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)

# ---- (b) where the picks went -------------------------------------------------
e_in = X[:, dv.index("enrich_inner")]
gd = X[:, dv.index("gd_wt")]
bx.axvline(E_BOUND, color="0.35", lw=1.0, ls="--", label="Enrichment bound, 17.174 wt%")
bx.scatter(e_in[:A], gd[:A], s=22, color=C_ARCH, edgecolors="white", lw=0.4,
           label="Archive before the block")
bx.scatter(e_in[25], gd[25], s=70, marker="D", color=C_ARCH, edgecolors="0.2", lw=1.0,
           label=f"C6-25, measured $k_\\mathrm{{eff}}$ = {raw[25]['keff_core_bol']:.3f}")
feas = k_meas <= K_MAX
bx.scatter(e_in[A:B][~feas], gd[A:B][~feas], s=60, marker="s", color=C_MEAS,
           edgecolors="white", lw=0.5, zorder=3, label="Picks above the bound")
bx.scatter(e_in[A:B][feas], gd[A:B][feas], s=60, marker="s", facecolors="white",
           edgecolors=C_MEAS, lw=1.3, zorder=3, label="Pick below the bound")
bx.set_xlabel(r"Inner enrichment $e_\mathrm{in}$ [wt%]")
bx.set_ylabel(r"Gadolinia fraction $w_\mathrm{Gd}$ [wt%]")
bx.set_xlim(1.0, 18.8)
bx.set_ylim(-0.4, 8.6)
bx.set_title("(b) Position of the picks in the design space", fontsize=10)
bx.grid(alpha=0.22, lw=0.6)
bx.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
for n_, p, sd, me in zip(names, k_pred, k_sd, k_meas):
    print(f"{n_}: predicted {p:.4f} +/- {sd:.4f}, measured {me:.4f}")
print("wrote", OUT)
