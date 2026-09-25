#!/usr/bin/env python3
"""make_c4_vertex_figure.py -- Campaign 4: the design variables by phase, and
the two correlations with the core peaking factor, for Section 5.3.1.

Read from the Campaign 4 checkpoint, no transport.

  Top row, one panel per design variable against the evaluation index, with
  the bounds of the design space as dashed lines. The reflector is shown as
  its margin to the vessel bound of Equation (4.4), t_max(p) - t_refl, so the
  bound is the zero line whatever the pitch. The active bounds on the
  gadolinia fraction, the pitch and the reflector from iteration 2, and the
  walk of the enrichment, are read directly.

  Bottom row, the core peaking factor against the volume-weighted enrichment
  and against the gadolinia fraction, every design shown, with the Pearson
  coefficient over the 36 designs of the design of experiments written in the
  panel. The inner zone holds 9 of the 32 assemblies, hence the weight 9/32.

usage (plot only, runs anywhere):
    python make_c4_vertex_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
raw = json.load(open(SRC))["all_raw"]

PHASES = [("Design of experiments (36)", 0, 36, "#8C8C8C", "o", 22),
          ("Iteration 1 (6)", 36, 42, "#E69F00", "s", 30),
          ("Iteration 2 (6)", 42, 48, "#009E73", "D", 30),
          ("Iteration 3 (6)", 48, 54, "#6A3D9A", "^", 36)]
W_IN = 9 / 32                                   # inner-zone share of the 32 assemblies
R_VESSEL, DR_DP = 90.0, np.sqrt(13) * 17        # Equation (4.4): R_env(p) = sqrt(13) * 17 p

g = lambda key: np.array([r[key] for r in raw], dtype=float)
e_in, e_out, gd, p, t = g("enrich_inner"), g("enrich_outer"), g("gd_wt"), g("pitch"), g("refl_thick")
F = g("peaking")
e_vw = W_IN * e_in + (1 - W_IN) * e_out
margin = (R_VESSEL - DR_DP * p) - t             # reflector margin to the vessel bound
idx = np.arange(len(raw))

VARS = [(r"$e_\mathrm{in}$ [wt%]", e_in, (2.0, 19.75)),
        (r"$e_\mathrm{out}$ [wt%]", e_out, (2.0, 19.75)),
        (r"$w_\mathrm{Gd}$ [wt%]", gd, (0.0, 8.0)),
        (r"$p$ [cm]", p, (1.15, 1.43)),
        (r"Reflector margin [cm]", margin, (0.0, None))]

fig = plt.figure(figsize=(11.6, 7.4))
gs = fig.add_gridspec(2, 10, height_ratios=(1.0, 1.25), hspace=0.42, wspace=1.6)
top = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(5)]
bot = [fig.add_subplot(gs[1, 0:5]), fig.add_subplot(gs[1, 5:10])]

# ---- top row: each variable against the evaluation index -------------------
for ax, (lab, y, (lo, hi)) in zip(top, VARS):
    for name, a, b, colour, mk, size in PHASES:
        ax.scatter(idx[a:b], y[a:b], marker=mk, s=size, color=colour,
                   edgecolors="white", lw=0.4, zorder=3)
    for v in (lo, hi):
        if v is not None:
            ax.axhline(v, color="0.35", lw=0.9, ls="--", zorder=1)
    for x in (36, 42, 48):
        ax.axvline(x - 0.5, color="0.75", lw=0.7, zorder=0)
    ax.set_xlim(-1.5, 54.5)
    ax.set_xlabel("Evaluation")
    ax.set_title(lab, fontsize=9.5)
    ax.grid(alpha=0.2, lw=0.5)
    ax.tick_params(labelsize=8.5)
top[0].set_ylabel("Value, bounds dashed", fontsize=9)

# ---- bottom row: the two correlations ---------------------------------------
def rdoe(x):
    return np.corrcoef(x[:36], F[:36])[0, 1]

for ax, x, lab in ((bot[0], e_vw, r"Volume-weighted enrichment $e_\mathrm{vw}$ [wt%]"),
                   (bot[1], gd, r"Gadolinia fraction $w_\mathrm{Gd}$ [wt%]")):
    for name, a, b, colour, mk, size in PHASES:
        ax.scatter(x[a:b], F[a:b], marker=mk, s=size + 10, color=colour,
                   edgecolors="white", lw=0.5, zorder=3, label=name)
    ax.axhline(2.0, color="#B22222", lw=1.2, zorder=2)
    ax.set_xlabel(lab)
    ax.set_ylabel(r"Core $F_{\Delta H}$ [-]")
    ax.grid(alpha=0.22, lw=0.6)
    ax.margins(x=0.06)
    ax.text(0.97, 0.95, f"$r = {rdoe(x):+.2f}$ over the design of experiments",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="0.15")
bot[0].text(0.03, 0.03, r"$F_{\Delta H} = 2.0$", transform=bot[0].transAxes,
            color="#B22222", fontsize=9, ha="left", va="bottom")
bot[0].legend(fontsize=8.2, loc="center right", bbox_to_anchor=(0.97, 0.62), framealpha=0.92)

fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")

print(f"e_vw: iteration 2 {e_vw[42]:.2f}, iteration 3 {e_vw[48]:.2f} wt%")
print(f"r over the DOE: e_vw {rdoe(e_vw):+.3f}, gd {rdoe(gd):+.3f}; "
      f"over all 54: e_vw {np.corrcoef(e_vw, F)[0, 1]:+.3f}, gd {np.corrcoef(gd, F)[0, 1]:+.3f}")
for name, a, b, *_ in PHASES:
    print(f"  {name:<28} gd {gd[a:b].min():.2f}-{gd[a:b].max():.2f}  p {p[a:b].min():.3f}-{p[a:b].max():.3f}"
          f"  margin {margin[a:b].min():.2f}-{margin[a:b].max():.2f} cm")
print("wrote", OUT)
