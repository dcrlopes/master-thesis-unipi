#!/usr/bin/env python3
"""
make_c9_fz_gate_figure.py -- the axial peaking factor F_z as a candidate
objective, from the test of c10_fz_gate.py. Three panels, side by side:

  (a) reproducibility: F_z of the first seed against the second, for the
      22 feasible designs solved twice, with the identity line and the
      band of two seed standard deviations around it;
  (b) spread against noise: F_z of all 60 designs, sorted, as a deviation
      from the archive mean with the one-sigma seed error of one solve and
      the band of two seed standard deviations; the spans of F_z and of the
      three-dimensional F_dH, (max - min) / mean as in the thesis, printed;
  (c) ranking: rank of each design by F_dH x F_z against its rank by F_dH
      alone, with the identity line.

Reads c10_gate/fz_gate.json. Drawing only, no transport.
Usage: python make_c9_fz_gate_figure.py <out.pdf> [out.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata, spearmanr

D = json.load(open("c10_gate/fz_gate.json"))
T = D["table"]
SD = D["sd_seed"]
feas = np.array([r["feasible"] for r in T])
fz = np.array([r["fz"][0] for r in T])
fdh = np.array([r["fdh3d"] for r in T])
two = [r for r in T if len(r["fz"]) >= 2]
s1 = np.array([r["fz"][0] for r in two])
s2 = np.array([r["fz"][1] for r in two])

C_FEAS, C_INF, C_BAND = "#0072B2", "#999999", "#56B4E9"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(12.0, 3.9))

# (a) seed 1 against seed 2
lo = min(s1.min(), s2.min()) - 0.003
hi = max(s1.max(), s2.max()) + 0.003
x = np.linspace(lo, hi, 50)
ax.fill_between(x, x - 2 * SD, x + 2 * SD, color=C_BAND, alpha=0.3, lw=0)
ax.plot(x, x, color="k", lw=0.8, ls="--")
ax.scatter(s1, s2, s=26, color=C_FEAS, edgecolors="k", linewidths=0.4, zorder=3)
r = np.corrcoef(s1, s2)[0, 1]
ax.text(0.04, 0.96, "22 feasible designs\nCorrelation between seeds %.2f" % r,
        transform=ax.transAxes, va="top", fontsize=7.5)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel(r"$F_z$, first seed")
ax.set_ylabel(r"$F_z$, second seed")
ax.set_title("(a) Reproducibility between seeds", fontsize=9, loc="left")

# (b) F_z of every design, sorted, with its seed error and the noise band
order = np.argsort(fz)
dz = 100 * (fz / fz.mean() - 1)
ez = 100 * SD / fz.mean()
xs = np.arange(1, len(T) + 1)
bx.axhspan(-2 * ez, 2 * ez, color=C_BAND, alpha=0.3, lw=0)
for k, i in enumerate(order):
    col = C_FEAS if feas[i] else C_INF
    bx.errorbar(xs[k], dz[i], yerr=ez, fmt="o", ms=3.5, color=col,
                mfc=col if feas[i] else "white", mec=col, elinewidth=0.7, capsize=0, zorder=3)
bx.axhline(0, color="k", lw=0.6)
span_z = 100 * (fz.max() - fz.min()) / fz.mean()
span_f = 100 * (fdh.max() - fdh.min()) / fdh.mean()
bx.text(0.04, 0.96, "Span of $F_z$: %.2f %%\nSpan of 3D $F_{\\Delta H}$: %.1f %%" % (span_z, span_f),
        transform=bx.transAxes, va="top", fontsize=7.5)
bx.set_xlim(0, len(T) + 1)
bx.set_ylim(-1.6, 1.6)
bx.set_xlabel(r"Design, sorted by $F_z$")
bx.set_ylabel(r"$F_z$, deviation from the archive mean (%)")
bx.set_title("(b) Spread between designs and seed noise", fontsize=9, loc="left")
bx.legend(handles=[Line2D([], [], marker="o", ls="", color=C_FEAS, markersize=5, label="Feasible"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=C_INF,
                          markersize=5, label="Infeasible"),
                   plt.Rectangle((0, 0), 1, 1, color=C_BAND, alpha=0.3, label=r"$\pm 2$ seed s.d.")],
          loc="lower right", fontsize=7, frameon=False)

# (c) ranking by the product against ranking by F_dH alone
rk_f = rankdata(fdh)
rk_q = rankdata(fdh * fz)
cx.plot([1, len(T)], [1, len(T)], color="k", lw=0.8, ls="--")
cx.scatter(rk_f[feas], rk_q[feas], s=18, color=C_FEAS, edgecolors="k", linewidths=0.3, zorder=3)
cx.scatter(rk_f[~feas], rk_q[~feas], s=16, facecolors="none", edgecolors=C_INF, linewidths=0.8, zorder=3)
rho = spearmanr(fdh, fdh * fz)[0]
shift = int(np.max(np.abs(rk_f - rk_q)))
cx.text(0.04, 0.96, "Spearman %.3f\nLargest rank shift %d" % (rho, shift),
        transform=cx.transAxes, va="top", fontsize=7.5)
cx.set_xlim(0, len(T) + 1)
cx.set_ylim(0, len(T) + 1)
cx.set_aspect("equal")
cx.set_xlabel(r"Rank by 3D $F_{\Delta H}$")
cx.set_ylabel(r"Rank by $F_{\Delta H}\,F_z$")
cx.set_title("(c) Ranking of the 60 designs", fontsize=9, loc="left")

for a in (ax, bx, cx):
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1], "seed corr %.3f, spearman %.4f, shift %d, spans %.2f / %.1f %%"
      % (r, rho, shift, span_z, span_f))
