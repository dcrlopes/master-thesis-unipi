#!/usr/bin/env python3
"""make_summary_model_figure.py -- Figure 1 of the thesis summary.

Variant of make_model_geometry_figure.py for the Campaign 8 and 9 model:
  (a) the 17x17 assembly with one enrichment e and the guide-tube positions;
  (b) the core plan view at the fixed pitch of 1.26 cm and the reflector
      thickness of the final front, 5.66 cm.
No design-variable table, no vessel-fit expression.

usage (plot only, no transport):
    python make_summary_model_figure.py OUT.pdf
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

OUT = sys.argv[1]

GUIDE_TUBE_POSITIONS = [
    (2, 5), (2, 8), (2, 11),
    (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13),
    (14, 5), (14, 8), (14, 11),
]
N = 17
GT = set(GUIDE_TUBE_POSITIONS)


def _orbit(i, j):
    """Octant orbit of lattice position (i, j), as in reactor_model.py."""
    return {(i, j), (j, i), (16 - i, j), (i, 16 - j), (16 - i, 16 - j),
            (16 - j, i), (j, 16 - i), (16 - j, 16 - i)}


# the 12-pin gadolinia pattern of reactor_model.GD_PATTERNS[12]
GD12 = _orbit(2, 2) | _orbit(6, 6) | _orbit(3, 8)
assert len(GD12) == 12 and not (GD12 & GT)

# colours of th_model_3d.pdf (Figure 4.3), for one visual identity
C_FUEL = "#d29097"
C_GD = "#7a4a26"
C_GT = "#ffffff"
C_EDGE = "#9a6575"
C_TXT = "#3b2a2e"
C_REFL = "#a0b0ba"
C_VESSEL = "#6f7d86"
# radial enrichment rings (Section 4.4.5): distance d from the core centre in
# assembly pitches, ring C below 1.0, ring M from 1.0 to 2.3, ring P above
RING_COLOUR = {"C": "#efcfd2", "M": "#d29097", "P": "#a3505c"}
RING_LABEL = {"C": "Ring C, centre, multiplier 0.720",
              "M": "Ring M, middle, multiplier 0.893",
              "P": "Ring P, periphery, multiplier 1.150"}

PITCH, TREFL = 1.26, 5.66
R_VESSEL = 90.0

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(12.6, 5.6))
axa = fig.add_axes([0.03, 0.10, 0.40, 0.82])
axb = fig.add_axes([0.50, 0.05, 0.48, 0.87])

# ------------------------------------------------------------------ (a) assembly
for i in range(N):
    for j in range(N):
        y = N - 1 - i
        if (i, j) in GT:
            axa.add_patch(Circle((j, y), 0.40, facecolor=C_GT,
                                 edgecolor=C_EDGE, lw=0.9, zorder=3))
        elif (i, j) in GD12:
            axa.add_patch(Circle((j, y), 0.40, facecolor=C_GD,
                                 edgecolor=C_EDGE, lw=0.6, zorder=3))
        else:
            axa.add_patch(Circle((j, y), 0.40, facecolor=C_FUEL,
                                 edgecolor=C_EDGE, lw=0.4, zorder=2))

axa.annotate("", xy=(0.4, N + 0.45), xytext=(1.4, N + 0.45),
             arrowprops=dict(arrowstyle="<->", color="#404b54", lw=0.9))
axa.text(0.9, N + 0.75, f"Pitch $p$ = {PITCH} cm", ha="left", va="bottom",
         fontsize=8.5, color="#404b54")
axa.set_xlim(-1.1, N + 0.6)
axa.set_ylim(-2.6, N + 1.6)
axa.set_aspect("equal")
axa.axis("off")
axa.set_title("(a) The $17\\times17$ assembly", fontsize=10, loc="left", x=-0.02)
handles = [
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_FUEL, mec=C_EDGE, mew=0.5,
           label="Fuel rod, enrichment $e$"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_GD, mec=C_EDGE, mew=0.6,
           label="Gadolinia-bearing rod, 12-rod pattern"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_GT, mec=C_EDGE, mew=0.9,
           label="Guide tube"),
]
axa.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           ncol=3, frameon=False, fontsize=8.5, handletextpad=0.4,
           columnspacing=1.6)

# ------------------------------------------------------------------ (b) core plan
a_w = N * PITCH
r_env = np.sqrt(13.0) * a_w
r_refl = r_env + TREFL

axb.add_patch(Circle((0, 0), R_VESSEL, facecolor="#c9d0d5",
                     edgecolor=C_VESSEL, lw=1.4, zorder=1))
axb.add_patch(Circle((0, 0), r_refl, facecolor=C_REFL,
                     edgecolor="none", zorder=2))
axb.add_patch(Circle((0, 0), r_env, facecolor="#f7f1f2", edgecolor="none",
                     zorder=3))
count = {"C": 0, "M": 0, "P": 0}
for i in range(6):
    for j in range(6):
        if (i, j) in {(0, 0), (0, 5), (5, 0), (5, 5)}:
            continue
        d = np.hypot(j - 2.5, i - 2.5)
        ring = "C" if d < 1.0 else ("M" if d < 2.3 else "P")
        count[ring] += 1
        axb.add_patch(Rectangle(((j - 3) * a_w, (i - 3) * a_w), a_w, a_w,
                                facecolor=RING_COLOUR[ring], edgecolor=C_EDGE,
                                lw=0.8, zorder=4))
        axb.text((j - 2.5) * a_w, (i - 2.5) * a_w, ring, ha="center",
                 va="center", fontsize=9, weight="bold", zorder=5,
                 color="white" if ring == "P" else C_TXT)
assert count == {"C": 4, "M": 12, "P": 16}, count
ring_handles = [Rectangle((0, 0), 1, 1, facecolor=RING_COLOUR[r],
                          edgecolor=C_EDGE, lw=0.8,
                          label=f"{RING_LABEL[r]}, {count[r]} assemblies")
                for r in ("C", "M", "P")]
axb.legend(handles=ring_handles, loc="upper left", bbox_to_anchor=(0.66, 0.99),
           frameon=False, fontsize=8.5, handlelength=1.4, labelspacing=0.6)
axb.annotate("Steel reflector", xy=(-(r_env + TREFL / 2) * 0.707,
                                     (r_env + TREFL / 2) * 0.707),
             xytext=(-R_VESSEL * 1.40, R_VESSEL * 0.95), fontsize=8.5,
             ha="left", va="center", color=C_TXT,
             arrowprops=dict(arrowstyle="->", color=C_TXT, lw=0.9))
axb.annotate("", xy=(r_env, -6), xytext=(r_refl, -6),
             arrowprops=dict(arrowstyle="<->", color="#7a3b2e", lw=1.1))
axb.text(R_VESSEL + 4, -6, "Reflector thickness $t_\\mathrm{refl}$,\n"
         "design variable, 2.0 to 5.66 cm", fontsize=8.5, ha="left",
         va="center", color="#7a3b2e")
axb.annotate("Vessel, $R$ = 90 cm", xy=(-R_VESSEL * 0.62, -R_VESSEL * 0.80),
             xytext=(-R_VESSEL * 1.40, -R_VESSEL * 1.10), fontsize=8.5,
             ha="left", va="center", color=C_TXT,
             arrowprops=dict(arrowstyle="->", color=C_TXT, lw=0.9))
axb.set_xlim(-R_VESSEL * 1.45, R_VESSEL * 2.35)
axb.set_ylim(-R_VESSEL * 1.22, R_VESSEL * 1.18)
axb.set_aspect("equal")
axb.axis("off")
axb.set_title(f"(b) The core plan view, drawn at $t_\\mathrm{{refl}}$ = {TREFL} cm",
              fontsize=10, loc="left", x=-0.02)

fig.savefig(OUT, bbox_inches="tight")
print("written", OUT)
