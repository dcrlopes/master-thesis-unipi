#!/usr/bin/env python3
"""make_fid_ladder_figure.py -- the assembly fidelity ladder, for Section 5.2.

Two panels, no transport:

  (a) the seed-to-seed standard deviation of the assembly peaking factor against
      the number of active histories, on logarithmic axes, with the power law
      fitted to the three converged settings. The 4000-particle setting falls
      below that law, because its estimator is biased rather than merely noisy.

  (b) the bias of each setting against the 256000-particle reference, with the
      number of paired designs beside each point.

The name of each setting is carried by a second axis along the top of both
panels, so that no text is written inside the axes over a marker or over the
fitted line. In (b) the two settings at the lower right are labelled through
leader arrows, their text placed in the empty region above the tail.

usage (repository root):
    python make_fid_ladder_figure.py OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
d = json.load(open("figs_bg/bg_mc_noise_summary.json"))

pts = [d["point_4k"]] + d["points"]
N = np.array([p["N"] for p in pts], dtype=float)
sd = np.array([p["sd"] for p in pts])
se = np.array([p["se"] for p in pts])
SETTING = ["4000 $\\times$ 60", "16 000 $\\times$ 120",
           "64 000 $\\times$ 120", "256 000 $\\times$ 120"]

# bias against the 256k reference and wall time, Table 5.2 of the manuscript
bias = np.array([0.055, 0.0089, 0.0047, 0.0])
nb = np.array([1, 2, 5, 7])                      # designs with a reference solve

C_OK, C_LOW, C_FIT, C_TXT = "#0072B2", "#D55E00", "0.35", "0.25"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.4))


def setting_axis(a):
    """Name every setting on a second axis above the panel, outside the frame."""
    t = a.secondary_xaxis("top")
    t.set_xticks(N)
    t.set_xticklabels(SETTING, fontsize=8.2)
    t.tick_params(axis="x", length=3, pad=2)
    t.minorticks_off()
    return t


# ---- (a) noise against the number of active histories --------------------
slope, c = d["slope_free"], d["fit_logc"]
SHORT = d["shortfall_4k_vs_fit"]
xx = np.logspace(np.log10(N.min() * 0.7), np.log10(N.max() * 1.4), 50)
ax.plot(xx, np.exp(c) * xx ** slope, "-", color=C_FIT, lw=1.1, zorder=1,
        label=f"Fit to the converged settings, slope {slope:.2f}")
ax.errorbar(N[1:], sd[1:], yerr=se[1:], fmt="o", ms=7, color=C_OK, capsize=3,
            zorder=3, label="Converged settings")
ax.errorbar(N[:1], sd[:1], yerr=se[:1], fmt="s", ms=8, color=C_LOW, capsize=3,
            zorder=3,
            label=f"4000 $\\times$ 60, {100 * SHORT:.0f} per cent below the law")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.0013, 0.055)
ax.set_xlabel("Active histories")
ax.set_ylabel(r"Seed-to-seed standard deviation of $F_{\Delta H}$")
ax.set_title("(a) Noise against the number of histories", fontsize=10.5, pad=26)
ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
for x, y, e in zip(N, sd, se):
    ax.annotate(f"{y:.4f}", (x, y - e), textcoords="offset points",
                xytext=(0, -9), ha="center", va="top", fontsize=8.5,
                color=C_TXT)
ax.grid(alpha=0.25, lw=0.6, which="both")
setting_axis(ax)

# ---- (b) bias against the reference setting ------------------------------
bx.axhline(0, color="0.5", lw=0.9, ls=":")
bx.plot(N, bias, "o-", color=C_OK, ms=7, lw=1.4, zorder=3)
bx.plot(N[:1], bias[:1], "s", color=C_LOW, ms=8, zorder=4)
bx.set_xscale("log")
bx.set_xlim(1.05e5, 3.6e7)
bx.set_ylim(-0.006, 0.070)

# the two settings clear of the crowded tail, written in open space
bx.annotate("1 design", (N[0], bias[0]), textcoords="offset points",
            xytext=(13, -1), va="center", ha="left", fontsize=8.2, color=C_TXT)
bx.annotate("2 designs", (N[1], bias[1]), textcoords="offset points",
            xytext=(9, 13), ha="left", fontsize=8.2, color=C_TXT)

# the two settings at the lower right, reached by leader arrows
for x, y, txt, tx, ty in [
        (N[2], bias[2], "5 designs", 3.0e6, 0.029),
        (N[3], bias[3], "7 designs", 1.05e7, 0.0115)]:
    bx.annotate(txt, xy=(x, y), xytext=(tx, ty), fontsize=8.2, ha="left",
                va="bottom", linespacing=1.3, color=C_TXT,
                arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0,
                                shrinkA=3, shrinkB=5))

bx.set_xlabel("Active histories")
bx.set_ylabel(r"Bias of $F_{\Delta H}$ against the reference setting")
bx.set_title("(b) Bias against the reference setting", fontsize=10.5, pad=26)
bx.grid(alpha=0.25, lw=0.6)
setting_axis(bx)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight")

print("fitted slope %.3f, 4k shortfall %.1f per cent" % (slope, 100 * d["shortfall_4k_vs_fit"]))
for s, x, y, b in zip(SETTING, N, sd, bias):
    print(f"  {s:<22} N={x:>10.0f}  sd={y:.4f}  bias={b:+.4f}")
print("wrote", OUT)
