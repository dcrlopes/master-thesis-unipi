#!/usr/bin/env python3
"""
plot_feasibility_envelope.py -- the reactivity-feasibility envelope of the
Campaign 3 design space, in the plane of volume-weighted enrichment vs
gadolinium loading.

WHAT THE FIGURE SHOWS
---------------------
The binding constraint of the campaign is g_kmax = k_BOL - 1.35 <= 0
(beginning-of-life assembly k_infinity must not exceed 1.35). Fitting the
36-point DOE (Design of Experiments) gives, with R^2 = 0.945,

    k_BOL = a + b * ln(e_vw) + c * gd

where gd is the gadolinium loading [wt%] and e_vw is the VOLUME-WEIGHTED
enrichment. The weighting matters: build_assembly_universe() defines the
inner zone as a centred 9x9 block of the 17x17 lattice, so only 81 of 289
pin positions (28%) carry the inner enrichment and 208 (72%) the outer.
Assembly k_infinity is a volume-weighted quantity, hence

    e_vw = 0.72 * e_outer + 0.28 * e_inner

Setting k_BOL = 1.35 turns the fit into a boundary curve: for each
gadolinium loading, the maximum enrichment that remains licensable. The
curve is the burnable-absorber trade-off made quantitative -- gadolinium
buys enrichment headroom, but only about 2.8 percentage points across its
full range, because enrichment outweighs it roughly fivefold.

PANELS
  LEFT  -- the envelope: feasible region shaded, boundary curve, the 36 DOE
           designs coloured by measured k_BOL, iso-k contours from the fit.
  RIGHT -- the same designs in the (e_outer, e_inner) plane, showing why the
           volume weighting is the structural insight: high INNER enrichment
           is affordable, high OUTER enrichment is not.

The figure reads the checkpoint directly, refits from the data, and prints
the fit quality -- nothing is hard-coded, so it stays correct as the
campaign grows.

USAGE
  python3 plot_feasibility_envelope.py \
      --checkpoint out_c3_atf75/optimization_checkpoint.json \
      --out figs/feasibility_envelope.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend: no display needed over SSH
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--k-max", type=float, default=1.35,
                help="the g_kmax limit on beginning-of-life k_infinity")
ap.add_argument("--k-min", type=float, default=1.02,
                help="the g_kmin lower bound on k_infinity")
ap.add_argument("--w-outer", type=float, default=None,
                help="volume weight of the outer zone; default 208/289 from "
                     "the 9x9-inner / 17x17-lattice geometry")
ap.add_argument("--out", default="feasibility_envelope.png")
ap.add_argument("--dpi", type=int, default=220)
args = ap.parse_args()

# --- geometry-derived volume weights --------------------------------------- #
N, HALF = 17, 4                       # 17x17 lattice, inner = centred 9x9
n_in = (2 * HALF + 1) ** 2            # 81
w_out = args.w_outer if args.w_outer is not None else (N * N - n_in) / (N * N)
w_in = 1.0 - w_out

ck = json.loads(Path(args.checkpoint).read_text())
raw = [r for r in ck["all_raw"] if "k_bol" in r]
eo = np.array([float(r["enrich_outer"]) for r in raw])
ei = np.array([float(r["enrich_inner"]) for r in raw])
gd = np.array([float(r["gd_wt"]) for r in raw])
k = np.array([float(r["k_bol"]) for r in raw])
evw = w_out * eo + w_in * ei

# --- fit  k = a + b ln(e_vw) + c gd ---------------------------------------- #
A = np.c_[np.ones(len(k)), np.log(evw), gd]
coef, *_ = np.linalg.lstsq(A, k, rcond=None)
a, b, c = coef
pred = A @ coef
r2 = 1.0 - ((k - pred) ** 2).sum() / ((k - k.mean()) ** 2).sum()
rms = float(np.sqrt(((k - pred) ** 2).mean()))


def e_boundary(gd_val, klim):
    """Max volume-weighted enrichment still satisfying k_BOL <= klim."""
    return np.exp((klim - a - c * gd_val) / b)


print(f"volume weights: outer {w_out:.3f} ({N*N-n_in} pins), "
      f"inner {w_in:.3f} ({n_in} pins)")
print(f"fit: k_BOL = {a:.4f} + {b:.4f} ln(e_vw) {c:+.5f} gd"
      f"   R2={r2:.4f}  rms={rms:.4f}  (n={len(k)})")
print(f"gadolinium authority over its full range 0->{gd.max():.1f} wt%: "
      f"{abs(c) * gd.max():.4f} in k_BOL")
for g in (0.0, 4.0, gd.max()):
    print(f"  gd={g:4.1f} wt%  ->  feasible up to e_vw = {e_boundary(g, args.k_max):5.2f} wt%")

# =========================================================================== #
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444"})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.5),
                               gridspec_kw={"width_ratios": [1.28, 1]})
FEAS, INF = "#2E6F4E", "#B23A48"

# --------------------------- LEFT: the envelope ---------------------------- #
gd_grid = np.linspace(0, max(8.0, gd.max() * 1.02), 300)
e_lo, e_hi = 1.5, max(evw.max() * 1.06, 18.0)
bnd = e_boundary(gd_grid, args.k_max)

axL.fill_betweenx(gd_grid, e_lo, np.clip(bnd, e_lo, e_hi),
                  color=FEAS, alpha=0.10, zorder=1)
axL.fill_betweenx(gd_grid, np.clip(bnd, e_lo, e_hi), e_hi,
                  color=INF, alpha=0.09, zorder=1)

# iso-k contours from the fit: the shape of the reactivity surface
EE, GG = np.meshgrid(np.linspace(e_lo, e_hi, 320), gd_grid)
KK = a + b * np.log(EE) + c * GG
cs = axL.contour(EE, GG, KK, levels=[1.10, 1.20, 1.25, 1.30, 1.40, 1.45],
                 colors="#8a8a8a", linewidths=0.6, alpha=0.75, zorder=2)
axL.clabel(cs, fmt=r"$k_\infty$=%.2f", fontsize=6.8, inline_spacing=2)

axL.plot(bnd, gd_grid, color="#111111", lw=2.4, zorder=5,
         label=rf"$k_\infty = {args.k_max:g}$ boundary (fit)")

ok = k <= args.k_max
sc = axL.scatter(evw[ok], gd[ok], c=k[ok], cmap="viridis", vmin=k.min(),
                 vmax=k.max(), s=96, marker="o", edgecolors="white",
                 linewidths=0.9, zorder=6)
axL.scatter(evw[~ok], gd[~ok], c=k[~ok], cmap="viridis", vmin=k.min(),
            vmax=k.max(), s=112, marker="X", edgecolors="white",
            linewidths=0.9, zorder=6)

# the headroom gadolinium actually buys, drawn horizontally at max loading
g1 = gd.max()
e_at_0, e_at_max = e_boundary(0.0, args.k_max), e_boundary(g1, args.k_max)
axL.annotate("", xy=(e_at_max, g1 * 1.005), xytext=(e_at_0, g1 * 1.005),
             arrowprops=dict(arrowstyle="<|-|>", color="#0E2A47", lw=1.8,
                             shrinkA=0, shrinkB=0), zorder=9)
axL.plot([e_at_0, e_at_0], [0, g1], color="#0E2A47", lw=0.9, ls=":", zorder=4)
axL.text(0.5 * (e_at_0 + e_at_max), g1 * 0.84,
         f"gadolinium buys only\n{e_at_max - e_at_0:.1f} wt% of enrichment\n"
         f"headroom across 0-{g1:.0f} wt%",
         fontsize=8.2, color="#0E2A47", ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#0E2A47",
                   lw=0.7, alpha=0.95), zorder=10)

axL.text(e_lo + 0.4, gd_grid.max() * 0.95, "FEASIBLE", fontsize=11,
         color=FEAS, weight="bold", alpha=0.85, va="top")
axL.text(e_hi - 0.4, gd_grid.max() * 0.95, "OVER-REACTIVE", fontsize=11,
         color=INF, weight="bold", alpha=0.85, ha="right", va="top")

axL.set_xlim(e_lo, e_hi)
axL.set_ylim(0, gd_grid.max())
axL.set_xlabel(r"volume-weighted enrichment  $e_{\rm vw}="
               rf"{w_out:.2f}\,e_{{\rm out}}+{w_in:.2f}\,e_{{\rm in}}$  [wt\%]"
               if matplotlib.rcParams["text.usetex"] else
               f"volume-weighted enrichment  "
               f"{w_out:.2f}·e_outer + {w_in:.2f}·e_inner  [wt%]")
axL.set_ylabel("gadolinium loading  [wt%]")
axL.set_title(f"Reactivity-feasibility envelope\n"
              f"fit R$^2$={r2:.3f}, {len(k)} DOE designs", fontsize=11)
axL.grid(alpha=0.18, lw=0.5)
cb = fig.colorbar(sc, ax=axL, pad=0.015)
cb.set_label(r"measured $k_\infty$ at BOL", fontsize=9)
cb.ax.axhline(args.k_max, color="black", lw=1.6)
axL.legend(handles=[
    Line2D([], [], color="#111111", lw=2.4,
           label=rf"$k_\infty={args.k_max:g}$ boundary"),
    Line2D([], [], marker="o", ls="", mfc="#5b8ac0", mec="white",
           ms=8, label="feasible design"),
    Line2D([], [], marker="X", ls="", mfc="#B23A48", mec="white",
           ms=9, label="over-reactive design")],
    fontsize=8, loc="lower left", framealpha=0.94)

# ------------- RIGHT: why the volume weighting is the insight -------------- #
lim = max(eo.max(), ei.max()) * 1.05
# iso-e_vw lines are straight in this plane
for ev, style in ((e_boundary(0.0, args.k_max), ":"),
                  (e_boundary(gd.max(), args.k_max), "--")):
    x = np.linspace(0, lim, 50)
    y = (ev - w_out * x) / w_in
    m = (y >= 0) & (y <= lim)
    axR.plot(x[m], y[m], style, color="#111111", lw=1.7,
             label=rf"$k_\infty={args.k_max:g}$ at "
                   f"gd={'0' if style == ':' else f'{gd.max():.0f}'} wt%")

axR.scatter(eo[ok], ei[ok], c=k[ok], cmap="viridis", vmin=k.min(), vmax=k.max(),
            s=96, marker="o", edgecolors="white", linewidths=0.9, zorder=5)
axR.scatter(eo[~ok], ei[~ok], c=k[~ok], cmap="viridis", vmin=k.min(),
            vmax=k.max(), s=112, marker="X", edgecolors="white",
            linewidths=0.9, zorder=5)
axR.plot([0, lim], [0, lim], color="#bbbbbb", lw=0.8, zorder=1)
axR.text(lim * 0.78, lim * 0.81, "uniform\nenrichment", fontsize=7.2,
         color="#999999", rotation=45, ha="center", va="center")

axR.annotate(f"inner zone: {n_in} of {N*N} pins ({100*w_in:.0f}%)\n"
             f"hot INNER is affordable",
             xy=(2.6, lim * 0.88), xytext=(lim * 0.055, lim * 0.985),
             fontsize=8.2, color="#0E2A47", ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color="#0E2A47", lw=1.2,
                             connectionstyle="arc3,rad=-0.25"),
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#0E2A47",
                       lw=0.7, alpha=0.95), zorder=8)
axR.annotate(f"outer zone: {N*N-n_in} of {N*N} pins ({100*w_out:.0f}%)\n"
             f"hot OUTER is not",
             xy=(lim * 0.90, 2.4), xytext=(lim * 0.30, lim * 0.075),
             fontsize=8.2, color="#0E2A47", ha="left", va="bottom",
             arrowprops=dict(arrowstyle="->", color="#0E2A47", lw=1.2,
                             connectionstyle="arc3,rad=0.2"),
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#0E2A47",
                       lw=0.7, alpha=0.95), zorder=8)

axR.set_xlim(0, lim)
axR.set_ylim(0, lim)
axR.set_xlabel("outer-zone enrichment  [wt%]")
axR.set_ylabel("inner-zone enrichment  [wt%]")
axR.set_title("The volume weighting, seen directly\n"
              f"outer:inner pin ratio = {(N*N-n_in)/n_in:.2f}:1", fontsize=11)
axR.grid(alpha=0.18, lw=0.5)
axR.legend(fontsize=7.6, loc="center left", framealpha=0.94)
axR.set_aspect("equal", adjustable="box")

fig.suptitle(r"Why half the design space is inaccessible: "
             r"$k_\infty(\mathrm{BOL}) \leq $"
             f"{args.k_max:g} is set by enrichment, "
             f"gadolinium only shifts it",
             fontsize=12.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.945))

dest = Path(args.out)
dest.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dest, dpi=args.dpi)
fig.savefig(dest.with_suffix(".pdf"))
print(f"\nwritten: {dest}  and  {dest.with_suffix('.pdf')}")
