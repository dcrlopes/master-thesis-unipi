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
C_GT = "#c9eef7"
C_GT_EDGE = "#5fb3c8"
C_COOL = "#8fd9ec"
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
axb = fig.add_axes([0.50, 0.03, 0.48, 0.91])

# ------------------------------------------------------------------ (a) assembly
axa.add_patch(Rectangle((-0.5, -0.5), N, N, facecolor=C_COOL, edgecolor="none",
                        zorder=1))
for i in range(N):
    for j in range(N):
        y = N - 1 - i
        if (i, j) in GT:
            axa.add_patch(Circle((j, y), 0.40, facecolor=C_GT,
                                 edgecolor=C_GT_EDGE, lw=0.9, zorder=3))
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
axa.set_ylim(-4.4, N + 1.6)
axa.set_aspect("equal")
axa.axis("off")
# both panel titles are set on the figure, at the same height
fig.text(0.03, 0.965, "(a) The $17\\times17$ assembly", fontsize=10,
         ha="left", va="bottom")
# legend: a coolant square with the rod drawn inside it, two columns
LEGEND = [("UO$_2$ fuel rod", C_FUEL, C_EDGE),
          ("UO$_2$-Gd$_2$O$_3$ fuel rod", C_GD, C_EDGE),
          ("Guide tube (GT)", C_GT, C_GT_EDGE),
          ("Coolant", None, None)]
for k, (lab, fc, ec) in enumerate(LEGEND):
    x0 = -0.5 + (k % 2) * 8.6
    y0 = -2.3 - (k // 2) * 1.5
    axa.add_patch(Rectangle((x0, y0), 1.0, 1.0, facecolor=C_COOL, edgecolor="none",
                            zorder=2, clip_on=False))
    if fc is not None:
        axa.add_patch(Circle((x0 + 0.5, y0 + 0.5), 0.40, facecolor=fc, edgecolor=ec,
                             lw=0.7, zorder=3, clip_on=False))
    axa.text(x0 + 1.5, y0 + 0.5, lab, ha="left", va="center", fontsize=8.5,
             color=C_TXT)

# ------------------------------------------------------------------ (b) core plan
# Radial stack of the three-dimensional model (Figure 2b), in cm. The heavy
# reflector fills everything between the assemblies and the barrel: the model
# places reflector material in the removed corners, in the gaps at the flat
# faces and in the annulus of minimum thickness t_refl.
DELTA, T_BARREL, T_WALL = 0.02, 5.08, 10.0
a_w = N * PITCH
r_env = np.sqrt(13.0) * a_w
r_refl = r_env + DELTA + TREFL
r_barrel = r_refl + T_BARREL
assert r_barrel <= R_VESSEL
C_BARREL = "#6f7d86"
C_WALL = "#3f4a53"
C_HREFL = "#c3ccd2"

axb.add_patch(Circle((0, 0), R_VESSEL + T_WALL, facecolor=C_WALL,
                     edgecolor="none", zorder=1))
axb.add_patch(Circle((0, 0), R_VESSEL, facecolor=C_COOL, edgecolor="none",
                     zorder=2))
axb.add_patch(Circle((0, 0), r_barrel, facecolor=C_BARREL, edgecolor="none",
                     zorder=3))
axb.add_patch(Circle((0, 0), r_refl, facecolor=C_HREFL, edgecolor="none",
                     zorder=4))
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
                                lw=0.8, zorder=5))
        axb.text((j - 2.5) * a_w, (i - 2.5) * a_w, ring, ha="center",
                 va="center", fontsize=9, weight="bold", zorder=6,
                 color="white" if ring == "P" else C_TXT)
assert count == {"C": 4, "M": 12, "P": 16}, count
ring_handles = [Rectangle((0, 0), 1, 1, facecolor=RING_COLOUR[r],
                          edgecolor=C_EDGE, lw=0.8,
                          label=f"{RING_LABEL[r]}, {count[r]} assemblies")
                for r in ("C", "M", "P")]
# ring legend under the drawing, as the legend of panel (a)
axb.legend(handles=ring_handles, loc="upper left",
           bbox_to_anchor=(-(R_VESSEL + 10.0) - 4.0, -(R_VESSEL + 10.0) - 6.0),
           bbox_transform=axb.transData, borderaxespad=0.0, borderpad=0.0,
           frameon=False, fontsize=8.5, handlelength=1.4, labelspacing=0.6)

arrow = dict(arrowstyle="->", color=C_TXT, lw=0.9)
# heavy reflector: the removed corner of the array, upper left
axb.annotate("Heavy reflector", xy=(-2.45 * a_w, 2.45 * a_w),
             xytext=(-R_VESSEL * 1.22, R_VESSEL * 1.22), fontsize=8.5,
             ha="left", va="center", color=C_TXT, arrowprops=arrow, zorder=8)
# core barrel, lower right
ang = np.radians(-52.0)
rb = 0.5 * (r_refl + r_barrel)
axb.annotate(f"Core barrel, {T_BARREL} cm", xy=(rb * np.cos(ang), rb * np.sin(ang)),
             xytext=(R_VESSEL * 1.25, -R_VESSEL * 0.72), fontsize=8.5,
             ha="left", va="center", color=C_TXT, arrowprops=arrow, zorder=8)
# downcomer, right
ang = np.radians(-22.0)
rd = 0.5 * (r_barrel + R_VESSEL)
axb.annotate(f"Downcomer, {R_VESSEL - r_barrel:.1f} cm",
             xy=(rd * np.cos(ang), rd * np.sin(ang)),
             xytext=(R_VESSEL * 1.25, -R_VESSEL * 0.30), fontsize=8.5,
             ha="left", va="center", color=C_TXT, arrowprops=arrow, zorder=8)
# vessel wall, lower left
ang = np.radians(-62.0)
rw = R_VESSEL + 0.5 * T_WALL
axb.annotate(f"Vessel wall, {T_WALL:.0f} cm,\ninner radius {R_VESSEL:.0f} cm",
             xy=(rw * np.cos(ang), rw * np.sin(ang)),
             xytext=(R_VESSEL * 1.25, -R_VESSEL * 1.12), fontsize=8.5,
             ha="left", va="center", color=C_TXT, arrowprops=arrow, zorder=8)
# minimum reflector thickness, along the direction of the farthest lattice corner
th = np.arctan2(2.0, 3.0)
ux, uy = np.cos(th), np.sin(th)
axb.plot([(r_env + DELTA) * ux, r_refl * ux], [(r_env + DELTA) * uy, r_refl * uy],
         color="#7a3b2e", lw=2.2, solid_capstyle="butt", zorder=9)
axb.annotate("Minimum reflector\nthickness $t_\\mathrm{refl}$,\n"
             "design variable,\n2.0 to 5.66 cm",
             xy=((r_env + 0.5 * TREFL) * ux, (r_env + 0.5 * TREFL) * uy),
             xytext=(R_VESSEL * 1.25, R_VESSEL * 0.30), fontsize=8.5, ha="left",
             va="center", color="#7a3b2e", zorder=9,
             arrowprops=dict(arrowstyle="-", color="#7a3b2e", lw=0.7))
axb.set_xlim(-R_VESSEL * 1.25, R_VESSEL * 2.05)
axb.set_ylim(-R_VESSEL * 1.75, R_VESSEL * 1.30)
axb.set_aspect("equal")
axb.axis("off")
fig.text(0.50, 0.965, f"(b) The core plan view, drawn at $t_\\mathrm{{refl}}$ = {TREFL} cm",
         fontsize=10, ha="left", va="bottom")

fig.savefig(OUT, bbox_inches="tight")
print("written", OUT)
