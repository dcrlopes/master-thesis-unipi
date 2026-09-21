#!/usr/bin/env python3
"""
make_cr_banks_figure.py -- the control-rod bank layout of the core, for the
controllability screen of the Methodology chapter.

The regulating banks RE1 to RE4 are the sixteen central assemblies listed in
zoning.RE_BANK_POSITIONS, with RE1 and RE2 the subset of zoning.RE12_POSITIONS
read by the two-bank diagnostic. The remaining sixteen assemblies of the core
map carry the shutdown banks, withdrawn in the screen and reserved for scram.

Drawing only, no transport.
Usage: python make_cr_banks_figure.py <out.pdf> <out.png>
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2]

CORE_MAP = np.array([[0, 1, 1, 1, 1, 0]] + [[1] * 6] * 4 + [[0, 1, 1, 1, 1, 0]])

RE1 = {(2, 2), (2, 3), (3, 2), (3, 3)}
RE2 = {(1, 1), (1, 4), (4, 1), (4, 4)}
RE3 = {(1, 2), (2, 4), (4, 3), (3, 1)}
RE4 = {(1, 3), (3, 4), (4, 2), (2, 1)}
BANKS = [("RE1", RE1, "#B23A48"), ("RE2", RE2, "#E08A3C"),
         ("RE3", RE3, "#3f6d8c"), ("RE4", RE4, "#5f9a7f")]
SH_COLOUR, EMPTY = "#b9c2c9", "#f2f4f6"

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(5.4, 5.0))

for i in range(6):
    for j in range(6):
        if not CORE_MAP[i, j]:
            continue
        colour, label = SH_COLOUR, "SH"
        for name, cells, c in BANKS:
            if (i, j) in cells:
                colour, label = c, name
                break
        ax.add_patch(Rectangle((j, 5 - i), 1, 1, facecolor=colour,
                               edgecolor="white", lw=1.6))
        ax.text(j + 0.5, 5 - i + 0.5, label, ha="center", va="center",
                fontsize=9.5, color="white" if label != "SH" else "#3f4a52",
                weight="bold")

ax.set_xlim(-0.1, 6.1)
ax.set_ylim(-0.1, 6.1)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Control rod banks of the 32-assembly core", fontsize=10)

handles = [Patch(facecolor=c, edgecolor="white", label=f"Regulating bank {n}")
           for n, _, c in BANKS]
handles.append(Patch(facecolor=SH_COLOUR, edgecolor="white",
                     label="Shutdown banks, withdrawn in the screen"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          ncol=2, fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print("written", OUT_PDF)
