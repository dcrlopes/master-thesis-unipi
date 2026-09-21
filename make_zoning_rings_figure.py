"""Radial enrichment zoning of the 32-assembly core: rings C, M and P by the
Euclidean distance of each assembly centre from the core centre, in lattice units.

usage: python make_zoning_rings_figure.py OUT.pdf OUT.png
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2]

CORE_MAP = np.array([[0, 1, 1, 1, 1, 0]] + [[1] * 6] * 4 + [[0, 1, 1, 1, 1, 0]])
RINGS = [("C", "Centre", 0.720, "#3f6d8c"),
         ("M", "Middle", 0.893, "#E08A3C"),
         ("P", "Periphery", 1.150, "#B23A48")]


def ring(dist):
    return "C" if dist < 1.0 else ("M" if dist < 2.3 else "P")


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(5.6, 5.6))
colour = {r: c for r, _, _, c in RINGS}
count = {r: 0 for r, _, _, _ in RINGS}
for i in range(6):
    for j in range(6):
        if not CORE_MAP[i, j]:
            continue
        x, y = j - 2.5, 2.5 - i                  # assembly centre, lattice units
        dist = float(np.hypot(x, y))
        r = ring(dist)
        count[r] += 1
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=colour[r],
                               edgecolor="white", lw=1.6))
        ax.text(x, y + 0.13, r, ha="center", va="center", color="white",
                fontsize=11, weight="bold")
        ax.text(x, y - 0.2, f"d = {dist:.2f}", ha="center", va="center",
                color="white", fontsize=7.5)
assert count == {"C": 4, "M": 12, "P": 16}

for rad in (1.0, 2.3):
    ax.add_patch(Circle((0, 0), rad, fill=False, ls="--", lw=1.2, color="black"))
ax.plot(0, 0, "k+", ms=9)
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.2, 3.2)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Radial enrichment zones of the 32-assembly core", fontsize=10)
handles = [Patch(facecolor=c, label=f"Ring {r}, {name}: {count[r]} assemblies, multiplier {m:.3f}")
           for r, name, m, c in RINGS]
handles.append(plt.Line2D([], [], color="black", ls="--",
                          label="Ring boundaries at d = 1.0 and d = 2.3 (lattice units)"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
          fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print(count)
