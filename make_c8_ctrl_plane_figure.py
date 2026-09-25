#!/usr/bin/env python3
"""make_c8_ctrl_plane_figure.py -- Campaign 8 control plane for Section 5.5.2.

Two panels, both read from the Campaign 8 checkpoint, no transport:

  (a) the rodded eigenvalue with the four regulating banks inserted against the
      unrodded eigenvalue at beginning of life. This state carries the
      controllability constraint, so the horizontal line at 0.99 is the
      boundary that rejects 35 of the 60 designs.

  (b) the same plane for the two-bank state, which the campaign records but
      does not constrain.

Both panels share the axis limits, so the vertical offset between the two
rodded states, and the distance of each fitted line from the 0.99 boundary,
can be read across the pair.

usage (plot only, runs anywhere):
    python make_c8_ctrl_plane_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(SRC))
raw, CN = d["all_raw"], d["constraint_names"]

K_CTRL, K_MIN, K_MAX = 0.99, 1.02, 1.166
BLOCKS = [("DOE", 0, 36, "#0072B2", "o"),
          ("Infill 1", 36, 42, "#E69F00", "s"),
          ("Infill 2", 42, 48, "#009E73", "D"),
          ("Infill 3", 48, 54, "#CC79A7", "^"),
          ("Infill 4", 54, 60, "#D55E00", "v")]

kc = np.array([r["keff_core_bol"] for r in raw])
feas = np.array([all(r[c] <= 0 for c in CN) for r in raw])

XLIM, YLIM = (0.82, 1.35), (0.70, 1.26)
C_BOUND, C_CTRL = "#B22222", "#4B0082"

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), sharex=True, sharey=True)

PANELS = [("k_allre", "(a) Four banks, RE1 to RE4, sixteen assemblies: the constrained state",
           1.206),
          ("k_re12", "(b) Two banks, RE1 and RE2, eight assemblies: recorded only",
           1.200)]

for ax, (field, title, x_note) in zip(axes, PANELS):
    y = np.array([r[field] for r in raw])

    ax.axhspan(K_CTRL, YLIM[1], color=C_CTRL, alpha=0.055, zorder=0)
    ax.axvspan(K_MAX, XLIM[1], color=C_BOUND, alpha=0.055, zorder=0)

    # least-squares line and the unrodded eigenvalue at which it reaches 0.99
    p = np.polyfit(kc, y, 1)
    cross = (K_CTRL - p[1]) / p[0]
    xx = np.linspace(*XLIM, 50)
    ax.plot(xx, np.polyval(p, xx), "-", color="0.45", lw=0.9, zorder=1)

    for name, a, b, colour, mk in BLOCKS:
        sel = np.zeros(len(raw), bool)
        sel[a:b] = True
        for ok, face, lab in ((False, "none", "infeasible"), (True, colour, "feasible")):
            m = sel & (feas == ok)
            if m.any():
                ax.scatter(kc[m], y[m], marker=mk, s=34, lw=1.1, facecolors=face,
                           edgecolors=colour, zorder=3,
                           label=f"{name}, {lab} ({m.sum()})")

    ax.axhline(K_CTRL, color=C_CTRL, lw=1.4, zorder=2)
    ax.axvline(K_MAX, color=C_BOUND, lw=1.3, zorder=2)
    ax.axvline(K_MIN, color=C_BOUND, lw=1.0, ls=":", zorder=2)
    ax.text(0.835, K_CTRL + 0.012, "Subcritical by 1000 pcm", color=C_CTRL, fontsize=9)
    ax.text(K_MIN - 0.008, 0.715, r"$k_{\min} = 1.02$", color=C_BOUND, fontsize=8.5,
            rotation=90, va="bottom", ha="right")
    ax.text(K_MAX - 0.008, 0.715, r"$k_{\max} = 1.166$", color=C_BOUND, fontsize=8.5,
            rotation=90, va="bottom", ha="right")

    ax.plot([cross], [K_CTRL], marker="o", ms=7, mfc="white", mec="0.2", mew=1.3, zorder=5)
    ax.annotate(f"Line reaches 0.99 at\n$k_\\mathrm{{eff}} = {cross:.3f}$",
                xy=(cross, K_CTRL), xytext=(x_note, 0.80),
                fontsize=9, color="0.2", ha="left", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color="0.2", lw=1.0))

    ax.set_xlabel(r"Core $k_\mathrm{eff}$ at beginning of life, all rods out [-]")
    ax.set_title(title, fontsize=10.0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(alpha=0.25, lw=0.6)

axes[0].set_ylabel(r"Core $k_\mathrm{eff}$ with the banks inserted [-]")
axes[0].legend(fontsize=7.6, loc="upper left", ncol=2, framealpha=0.92,
               borderpad=0.5, handletextpad=0.5, columnspacing=1.0)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")

for field, title, _ in PANELS:
    y = np.array([r[field] for r in raw])
    p = np.polyfit(kc, y, 1)
    print(f"{title}: slope {p[0]:.3f}, reaches 0.99 at {(K_CTRL - p[1]) / p[0]:.3f}, "
          f"residual {1e5 * np.std(y - np.polyval(p, kc)):.0f} pcm, "
          f"{(y <= K_CTRL).sum()} of {len(y)} subcritical")
print("wrote", OUT)
