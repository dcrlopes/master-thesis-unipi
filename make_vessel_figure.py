#!/usr/bin/env python3
"""make_vessel_figure.py -- the adopted vessel and core layout, Figure 4.2.

  (a) elevation: the vessel envelope, the nozzle penetration, the lower core plate
      and the fuel assembly, with the elevations that fix the axial layout.
  (b) plan view at core mid-height: the assembly array, the reflector, the barrel
      and the vessel, with the radii that fix the radial zones.

The radial stack is computed from the pitch, the reflector thickness, the radial
tolerance and the barrel thickness of Chapter 4, so the printed radii cannot
drift from the model. The reflector is drawn at 5.66 cm, the upper bound of the
design space of Campaigns 8 and 9, at which the downcomer has its minimum
width of 2.0 cm. Both vessel heads have the constant wall thickness of 100 mm
(the lower head is a hemisphere, the closure head a half ellipse). Colours
follow th_model_3d.pdf (Figure 4.3).

usage (plot only, no transport):
    python make_vessel_figure.py OUT.pdf
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

OUT = sys.argv[1]

# ---- geometry, all in mm, elevations from the outer bottom of the lower head
H_TOT = 4700.0
R_IN, T_WALL = 900.0, 100.0
R_OUT = R_IN + T_WALL
PLATE_LO, PLATE_HI = 1090.0, 1165.0
NOZ_LO, NOZ_HI = 2614.0, 3095.0
NOZ_MID = 0.5 * (NOZ_LO + NOZ_HI)
ASM_LO, ASM_HI = 1165.0, 2800.6
FUEL_LO, FUEL_HI = 1278.6, 2478.6

PITCH_CM, T_REFL_CM, DELTA_CM, T_BARREL_CM = 1.26, 5.66, 0.02, 5.08
A_PITCH = 17.0 * PITCH_CM * 10.0             # assembly pitch, 214.2
R_ENV = np.sqrt(13.0) * A_PITCH              # circumscribed radius, 772.3
R_REFL = R_ENV + 10.0 * (DELTA_CM + T_REFL_CM)   # reflector outer, 829.1
R_BARREL = R_REFL + 10.0 * T_BARREL_CM       # barrel outer, 879.9
FUEL_FLAT = 3 * A_PITCH                      # 642.6
assert R_BARREL <= R_IN, "the radial stack does not fit in the vessel"

# colours of th_model_3d.pdf (Figure 4.3)
C_STEEL = "#c9d0d5"
C_STEEL_EDGE = "#6f7d86"
C_WATER = "#dfeaf2"
C_FUEL = "#d29097"
C_FUEL_EDGE = "#9a6575"
C_REFL = "#a0b0ba"
C_EDGE = "#4a5b68"
C_DIM = "#404b54"

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(11.2, 4.9))
axa = fig.add_axes([0.035, 0.02, 0.40, 0.90])
axb = fig.add_axes([0.495, 0.02, 0.48, 0.90])


def dim_v(ax, x, y0, y1, label, side="left", pad=55):
    """A vertical dimension with its ticks and label."""
    ax.annotate("", xy=(x, y0), xytext=(x, y1),
                arrowprops=dict(arrowstyle="<->", color=C_DIM, lw=0.8))
    for y in (y0, y1):
        ax.plot([x - 40, x + 40], [y, y], color=C_DIM, lw=0.5, ls=":")
    ax.text(x + (-pad if side == "left" else pad), 0.5 * (y0 + y1), label,
            rotation=90, ha="center", va="center", fontsize=8, color=C_DIM)


def dim_h(ax, y, x0, x1, label, pad=70):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=C_DIM, lw=0.8))
    ax.text(0.5 * (x0 + x1), y - pad, label, ha="center", va="top",
            fontsize=8, color=C_DIM)


# ================================================================== (a) elevation
H_DOME = 750.0                       # rise of the closure dome, inner
body_lo = R_OUT                      # tangent line of the lower head, inner and outer
body_hi = H_TOT - H_DOME - T_WALL    # tangent line of the closure head


def vessel_outline(r, rise):
    """Lower hemisphere of radius r centred on the tangent line, cylindrical
    body, elliptical closure dome of rise `rise`. The inner and the outer
    outline share both centres, so the wall thickness is constant."""
    th = np.linspace(np.pi, 2 * np.pi, 120)
    low = np.column_stack([r * np.cos(th), body_lo + r * np.sin(th)])
    th2 = np.linspace(0, np.pi, 120)
    dome = np.column_stack([r * np.cos(th2), body_hi + rise * np.sin(th2)])
    return np.vstack([low, [[r, body_lo], [r, body_hi]], dome,
                      [[-r, body_hi], [-r, body_lo]]])


axa.add_patch(Polygon(vessel_outline(R_OUT, H_DOME + T_WALL), closed=True,
                      facecolor=C_STEEL, edgecolor=C_STEEL_EDGE, lw=0.9, zorder=1))
axa.add_patch(Polygon(vessel_outline(R_IN, H_DOME), closed=True,
                      facecolor=C_WATER, edgecolor=C_STEEL_EDGE, lw=0.7, zorder=2))

# the coolant nozzles
for s_ in (-1, 1):
    axa.add_patch(Rectangle((s_ * R_IN, NOZ_LO), s_ * (R_OUT + 300 - R_IN),
                            NOZ_HI - NOZ_LO, facecolor=C_STEEL,
                            edgecolor=C_STEEL_EDGE, lw=0.8, zorder=4))

# the lower core plate
axa.add_patch(Rectangle((-R_REFL, PLATE_LO), 2 * R_REFL, PLATE_HI - PLATE_LO,
                        facecolor=C_REFL, edgecolor=C_EDGE, lw=0.7, zorder=4))

# the reflector annulus and the assembly stack
for s_ in (-1, 1):
    axa.add_patch(Rectangle((s_ * FUEL_FLAT, ASM_LO), s_ * (R_REFL - FUEL_FLAT),
                            ASM_HI - ASM_LO, facecolor=C_REFL,
                            edgecolor=C_EDGE, lw=0.6, zorder=4))
axa.add_patch(Rectangle((-FUEL_FLAT, ASM_LO), 2 * FUEL_FLAT, ASM_HI - ASM_LO,
                        facecolor="#f3eaec", edgecolor=C_FUEL_EDGE, lw=0.7, zorder=4))
axa.add_patch(Rectangle((-FUEL_FLAT, FUEL_LO), 2 * FUEL_FLAT, FUEL_HI - FUEL_LO,
                        facecolor=C_FUEL, edgecolor=C_FUEL_EDGE, lw=0.8, zorder=5))

# the elevations that fix the axial layout
dim_v(axa, -R_OUT - 980, 0.0, H_TOT, "Overall 4700")
dim_v(axa, -R_OUT - 620, 0.0, NOZ_MID, "Nozzle centreline 2854")
dim_v(axa, -R_OUT - 260, 0.0, PLATE_HI, "Plate top 1165")
dim_v(axa, R_OUT + 300, FUEL_LO, FUEL_HI, "Active fuel 1200", side="right", pad=105)
dim_v(axa, R_OUT + 760, ASM_LO, ASM_HI, "Assembly 1636", side="right", pad=105)
axa.annotate("Nozzle bottom 2614", xy=(R_OUT + 250, NOZ_LO),
             xytext=(R_OUT + 180, NOZ_LO + 980), fontsize=8, color=C_DIM,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_DIM, lw=0.7))

dim_h(axa, -700, -R_OUT, 0, "R 900 inner, R 1000 outer", pad=130)
axa.annotate("Head R 1000", xy=(0, H_TOT - 60), xytext=(R_IN + 200, H_TOT + 260),
             fontsize=8, color=C_DIM, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_DIM, lw=0.7))
axa.annotate("Wall 100", xy=(R_IN + 0.5 * T_WALL, NOZ_HI + 820),
             xytext=(R_OUT + 180, NOZ_HI + 1420), fontsize=8, color=C_DIM,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_DIM, lw=0.7))

axa.set_xlim(-2450, 2350)
axa.set_ylim(-1000, H_TOT + 620)
axa.set_aspect("equal")
axa.axis("off")
axa.set_title("(a) Elevation, mm", fontsize=9.5, loc="left", x=0.12)

# ================================================================== (b) plan view
axb.add_patch(Circle((0, 0), R_OUT, facecolor=C_STEEL, edgecolor=C_STEEL_EDGE,
                     lw=0.9, zorder=1))
axb.add_patch(Circle((0, 0), R_IN, facecolor=C_WATER, edgecolor=C_STEEL_EDGE,
                     lw=0.7, zorder=2))
axb.add_patch(Circle((0, 0), R_BARREL, facecolor=C_STEEL, edgecolor=C_STEEL_EDGE,
                     lw=0.7, zorder=3))
axb.add_patch(Circle((0, 0), R_REFL, facecolor=C_REFL, edgecolor=C_EDGE,
                     lw=0.7, zorder=4))
axb.add_patch(Circle((0, 0), R_ENV, facecolor="#f7f1f2", edgecolor=C_FUEL_EDGE,
                     lw=0.8, ls=(0, (3, 3)), zorder=5))

for i in range(6):
    for j in range(6):
        if (i, j) in {(0, 0), (0, 5), (5, 0), (5, 5)}:
            continue
        axb.add_patch(Rectangle(((j - 3) * A_PITCH, (i - 3) * A_PITCH),
                                A_PITCH, A_PITCH, facecolor=C_FUEL,
                                edgecolor=C_FUEL_EDGE, lw=0.6, zorder=6))

# the radii that fix the radial zones
rows = [(R_ENV, f"Fuel envelope $R_\\mathrm{{env}}$ {R_ENV:.0f}, dashed"),
        (FUEL_FLAT, f"Fuel at flat {FUEL_FLAT:.0f}"),
        (R_REFL, f"Reflector outer {R_REFL:.0f}"),
        (R_BARREL, f"Barrel outer {R_BARREL:.0f}"),
        (R_IN, f"Vessel inner {R_IN:.0f}")]
for k, (r, lab) in enumerate(rows):
    y = -R_OUT - 200 - k * 175
    axb.annotate("", xy=(0, y), xytext=(r, y),
                 arrowprops=dict(arrowstyle="<->", color=C_DIM, lw=0.8))
    axb.text(r + 60, y, lab, fontsize=8, color=C_DIM, ha="left", va="center")

axb.set_xlim(-1300, 1900)
axb.set_ylim(-2130, 1250)
axb.set_aspect("equal")
axb.axis("off")
axb.set_title("(b) Plan view at core mid-height, mm", fontsize=9.5, loc="left",
              x=0.06)

fig.savefig(OUT, bbox_inches="tight")
print(f"R_env {R_ENV:.1f}, fuel at flat {FUEL_FLAT:.1f}, reflector outer {R_REFL:.1f}, "
      f"barrel outer {R_BARREL:.1f}, downcomer {R_IN - R_BARREL:.1f}, "
      f"assembly {ASM_HI - ASM_LO:.1f}, active fuel {FUEL_HI - FUEL_LO:.1f}")
