#!/usr/bin/env python3
"""
make_c8_hump_figure.py -- reactivity history of the eleven Campaign 8
post-analysis designs relative to beginning of life, with the most
reactive operating point marked, for Section 5.5.6.6 (Table tab:c8-swing).

Reads swing_c8/swing.json (assembly depletion histories bu, k and the hump
of Equation eq:c7-hump). Panel (a) the five designs with a positive hump,
panel (b) the six whose beginning of life is the most reactive state.

Drawing only, no transport.
Usage: python make_c8_hump_figure.py <out.pdf> [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("swing_c8/swing.json"))
POS = ["29", "31", "21", "44", "59"]
NEG = ["53", "47", "42", "23", "13", "1"]


def rho(k):
    return 1e5 * (k - 1.0) / k


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
COL = plt.get_cmap("tab10").colors
MK = ["o", "s", "^", "D", "v", "P"]
for ax, ids, title in [(axes[0], POS, "(a) Positive hump"),
                       (axes[1], NEG, "(b) Beginning of life most reactive")]:
    for n, d in enumerate(sorted(ids, key=lambda i: S[i]["enrich"])):
        r = S[d]
        r0 = rho(r["k"][0])
        x = r["bu"]
        y = [rho(k) - r0 for k in r["k"]]
        c = COL[n]
        ax.plot(x, y, "-", marker=MK[n], ms=3.5, lw=1.2, color=c,
                label=f"C8-{d}, {r['enrich']:.2f} wt%, hump {r['hump_pcm']:+.0f} pcm")
        if r["hump_pcm"] > 0:
            j = max(range(1, len(y)), key=lambda i: y[i])
            ax.plot(x[j], y[j], marker="*", ms=11, color=c, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=5)
    ax.axhline(0.0, color="0.3", lw=0.8, ls="--")
    ax.set_xlim(0, 32)
    ax.set_xlabel("Assembly burnup [MWd/kgHM]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)
axes[0].set_ylabel(r"$\rho(B) - \rho_\mathrm{BOL}$ [pcm]")
fig.tight_layout()
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1])
