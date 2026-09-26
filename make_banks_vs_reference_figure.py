#!/usr/bin/env python3
"""
make_banks_vs_reference_figure.py -- control rod banks of the 32-assembly core
of this work beside those of the NuScale-like reference specification
(Fridman, Bilodid and Valtavirta 2023, Fig. 1), for Section 5.5.6.4.

Panel (a) uses the bank positions of make_cr_banks_figure.py. Panel (b) is
redrawn from Fig. 1 of the reference specification: 37 assemblies, RE1 around
the centre, RE2 at the ends of the axes, SH3 and SH4 on the diagonals.

Drawing only, no transport.
Usage: python make_banks_vs_reference_figure.py <out.pdf> <out.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2]

C = {"RE1": "#B23A48", "RE2": "#E08A3C", "RE3": "#3f6d8c", "RE4": "#5f9a7f",
     "SH": "#8f9aa3", "SH3": "#8f9aa3", "SH4": "#b9c2c9", "": "#f2f4f6"}

# (a) this work, 6 x 6 minus corners
WORK = {}
for i in range(6):
    for j in range(6):
        if (i, j) not in {(0, 0), (0, 5), (5, 0), (5, 5)}:
            WORK[(i, j)] = "SH" if i in (0, 5) or j in (0, 5) else ""
for name, cells in [("RE1", {(2, 2), (2, 3), (3, 2), (3, 3)}),
                    ("RE2", {(1, 1), (1, 4), (4, 1), (4, 4)}),
                    ("RE3", {(1, 2), (2, 4), (4, 3), (3, 1)}),
                    ("RE4", {(1, 3), (3, 4), (4, 2), (2, 1)})]:
    for c in cells:
        WORK[c] = name

# (b) reference specification, rows of 3, 5, 7, 7, 7, 5, 3 assemblies
REF = {}
for i, (a, b) in enumerate([(2, 4), (1, 5), (0, 6), (0, 6), (0, 6), (1, 5), (2, 4)]):
    for j in range(a, b + 1):
        REF[(i, j)] = ""
for name, cells in [("RE1", {(2, 3), (3, 2), (3, 4), (4, 3)}),
                    ("RE2", {(0, 3), (3, 0), (3, 6), (6, 3)}),
                    ("SH3", {(1, 4), (2, 1), (4, 5), (5, 2)}),
                    ("SH4", {(1, 2), (2, 5), (4, 1), (5, 4)})]:
    for c in cells:
        REF[c] = name


def draw(ax, cells, n, title):
    for (i, j), lab in cells.items():
        ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=C[lab],
                               edgecolor="white", lw=1.6))
        if lab:
            ax.text(j + 0.5, n - 1 - i + 0.5, lab, ha="center", va="center",
                    fontsize=9, weight="bold",
                    color="white" if lab.startswith("RE") else "#2b3238")
    ax.set_xlim(-0.1, n + 0.1)
    ax.set_ylim(-0.1, n + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10)


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 5.0))
draw(a1, WORK, 6, "(a) This work, 32 assemblies")
draw(a2, REF, 7, "(b) NuScale-like reference, 37 assemblies")

handles = [Patch(facecolor=C[k], edgecolor="white", label=l) for k, l in [
    ("RE1", "Regulating bank RE1"), ("RE2", "Regulating bank RE2"),
    ("RE3", "Regulating bank RE3"), ("RE4", "Regulating bank RE4"),
    ("SH", "Shutdown bank SH, SH3"), ("SH4", "Shutdown bank SH4"),
    ("", "No control rod assembly")]]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print("written", OUT_PDF)
