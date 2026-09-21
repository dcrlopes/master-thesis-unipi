#!/usr/bin/env python3
"""
make_gd_ladder_figure.py -- the six poisoned-pin patterns of the ladder, for
the design-space section of the Methodology chapter.

The patterns and the guide-tube map are those of reactor_model.py. They are
rebuilt here from the same octant orbits so that the figure can be drawn on a
machine without OpenMC; when reactor_model imports, the rebuilt patterns are
checked against it and the script stops on any difference.

Each panel shows one rung of the ladder, with the pins inherited from the rung
below in one colour and the pins added at that rung in another, which is what
makes the nesting visible.

Drawing only, no transport.
Usage: python make_gd_ladder_figure.py <out.pdf> <out.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

N = 17


def orbit(i, j):
    """The octant orbit of (i, j) in a 17 x 17 lattice, as in reactor_model."""
    return sorted({(i, j), (j, i), (16 - i, j), (i, 16 - j),
                   (16 - i, 16 - j), (16 - j, i), (j, 16 - i),
                   (16 - j, 16 - i)})


PATTERNS = {12: sorted(set(orbit(2, 2) + orbit(6, 6) + orbit(3, 8)))}
PATTERNS[16] = sorted(set(PATTERNS[12] + orbit(4, 4)))
PATTERNS[20] = sorted(set(PATTERNS[16] + orbit(6, 8)))
PATTERNS[24] = sorted(set(PATTERNS[20] + orbit(7, 7)))
PATTERNS[32] = sorted(set(PATTERNS[24] + orbit(3, 5)))
PATTERNS[40] = sorted(set(PATTERNS[32] + orbit(4, 6)))
LADDER = sorted(PATTERNS)

GUIDE_TUBES = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13), (5, 2), (5, 5), (5, 8),
               (5, 11), (5, 14), (8, 2), (8, 5), (8, 8), (8, 11), (8, 14), (11, 2),
               (11, 5), (11, 8), (11, 11), (11, 14), (13, 3), (13, 13), (14, 5),
               (14, 8), (14, 11)]

try:                                    # check against the model when possible
    import reactor_model as rm
    assert {n: sorted(map(tuple, p)) for n, p in rm.GD_PATTERNS.items()} == \
           {n: sorted(map(tuple, p)) for n, p in PATTERNS.items()}
    assert sorted(map(tuple, rm.GUIDE_TUBE_POSITIONS)) == sorted(GUIDE_TUBES)
    print("patterns checked against reactor_model")
except ImportError:
    print("reactor_model not importable here, patterns rebuilt locally")

GT = set(GUIDE_TUBES)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 3, figsize=(8.6, 6.0))

for k, (n, ax) in enumerate(zip(LADDER, axes.ravel())):
    here = set(PATTERNS[n])
    below = set(PATTERNS[LADDER[k - 1]]) if k else set()
    added = here - below
    ax.add_patch(Rectangle((-0.5, -0.5), N, N, facecolor="#f2f4f6",
                           edgecolor="#33506b", lw=0.8))
    for i in range(N):
        for j in range(N):
            if (i, j) in GT:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#bdbdbd", edgecolor="none"))
                continue
            if (i, j) in added:
                fc, ec = "#B23A48", "#7a1f2b"
            elif (i, j) in below:
                fc, ec = "#e8a0a8", "#B23A48"
            else:
                fc, ec = "#ffffff", "#c9d2d9"
            ax.add_patch(Circle((j, i), 0.40, facecolor=fc, edgecolor=ec, lw=0.5))
    ax.set_xlim(-0.6, N - 0.4)
    ax.set_ylim(N - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    title = f"$n_\\mathrm{{Gd}}$ = {n}"
    if k:
        title += f",  +{len(added)} pins"
    ax.set_title(title, fontsize=9)

handles = [Circle((0, 0), 1, facecolor="#e8a0a8", edgecolor="#B23A48"),
           Circle((0, 0), 1, facecolor="#B23A48", edgecolor="#7a1f2b"),
           Rectangle((0, 0), 1, 1, facecolor="#bdbdbd", edgecolor="none"),
           Circle((0, 0), 1, facecolor="#ffffff", edgecolor="#c9d2d9")]
labels = ["Inherited from the rung below", "Added at this rung",
          "Guide tube", "Uranium dioxide pin"]
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.2,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT_PDF := sys.argv[1], bbox_inches="tight")
fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", OUT_PDF, "| ladder", LADDER)
