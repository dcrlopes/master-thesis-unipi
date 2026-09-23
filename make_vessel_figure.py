#!/usr/bin/env python3
"""make_vessel_figure.py -- the adopted vessel and core layout, Section 4.2.

  (a) elevation: the vessel envelope, the nozzle penetration, the lower core plate
      and the fuel assembly, with the elevations that fix the axial layout.
  (b) plan view at core mid-height: the assembly array, the reflector, the barrel
      and the vessel, with the radii that fix the radial zones.

The component sizes of the fuel assembly are deliberately NOT annotated here, they
are tabulated with the axial layout. Only the geometry the two views fix is drawn.

usage (plot only, no transport):
    python make_vessel_figure.py OUT.pdf [OUT.png]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge, Polygon

OUT = sys.argv[1]
PNG = sys.argv[2] if len(sys.argv) > 2 else None

# ---- geometry, all in mm, elevations from the outer bottom of the lower head
H_TOT = 4700.0
R_IN, T_WALL = 900.0, 100.0
R_HEAD = 1000.0
PLATE_LO, PLATE_HI = 1090.0, 1165.0
NOZ_LO, NOZ_HI = 2614.0, 3095.0
NOZ_MID = 0.5 * (NOZ_LO + NOZ_HI)
ASM_LO, ASM_HI = 1165.0, 2800.6
FUEL_LO, FUEL_HI = 1278.6, 2478.6

A_PITCH = 214.2                      # assembly pitch, 17 x 1.26 cm
R_ENV = np.sqrt(13.0) * A_PITCH      # circumscribed radius of the 32-assembly array
R_REFL = 813.0
R_BARREL = 863.0
FUEL_FLAT = 3 * A_PITCH              # 642.6

C_STEEL = "#b9bec2"
C_WATER = "#dfeaf2"
C_FUEL = "#f2cfa0"
C_REFL = "#9aa7b1"
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
R_OUT = R_IN + T_WALL
H_DOME = 750.0                       # rise of the closure dome
body_lo, body_hi = R_IN, H_TOT - H_DOME


def vessel_outline(r, rise):
    """Lower hemisphere, cylindrical body, elliptical closure dome."""
    th = np.linspace(np.pi, 2 * np.pi, 120)
    low = np.column_stack([r * np.cos(th), r + r * np.sin(th)])
    th2 = np.linspace(0, np.pi, 120)
    dome = np.column_stack([r * np.cos(th2), body_hi + rise * np.sin(th2)])
    return np.vstack([low, [[r, body_lo], [r, body_hi]], dome,
                      [[-r, body_hi], [-r, body_lo]]])


axa.add_patch(Polygon(vessel_outline(R_OUT, H_DOME + T_WALL), closed=True,
                      facecolor=C_STEEL, edgecolor=C_EDGE, lw=0.9, zorder=1))
axa.add_patch(Polygon(vessel_outline(R_IN, H_DOME), closed=True,
                      facecolor=C_WATER, edgecolor=C_EDGE, lw=0.7, zorder=2))

# the coolant nozzles
for s_ in (-1, 1):
    axa.add_patch(Rectangle((s_ * R_IN, NOZ_LO), s_ * (R_OUT + 300 - R_IN),
                            NOZ_HI - NOZ_LO, facecolor=C_STEEL,
                            edgecolor=C_EDGE, lw=0.8, zorder=4))

# the lower core plate
axa.add_patch(Rectangle((-R_REFL, PLATE_LO), 2 * R_REFL, PLATE_HI - PLATE_LO,
                        facecolor=C_REFL, edgecolor=C_EDGE, lw=0.7, zorder=4))

# the reflector annulus and the assembly stack
for s_ in (-1, 1):
    axa.add_patch(Rectangle((s_ * FUEL_FLAT, ASM_LO), s_ * (R_REFL - FUEL_FLAT),
                            ASM_HI - ASM_LO, facecolor=C_REFL,
                            edgecolor=C_EDGE, lw=0.6, zorder=4))
axa.add_patch(Rectangle((-FUEL_FLAT, ASM_LO), 2 * FUEL_FLAT, ASM_HI - ASM_LO,
                        facecolor="#e8eef3", edgecolor=C_EDGE, lw=0.7, zorder=4))
axa.add_patch(Rectangle((-FUEL_FLAT, FUEL_LO), 2 * FUEL_FLAT, FUEL_HI - FUEL_LO,
                        facecolor=C_FUEL, edgecolor=C_EDGE, lw=0.8, zorder=5))

# the elevations that fix the axial layout
dim_v(axa, -R_OUT - 980, 0.0, H_TOT, "4700 overall")
dim_v(axa, -R_OUT - 620, 0.0, NOZ_MID, "2854 nozzle centreline")
dim_v(axa, -R_OUT - 260, 0.0, PLATE_HI, "1165 plate top")
dim_v(axa, R_OUT + 300, FUEL_LO, FUEL_HI, "1200 active fuel", side="right", pad=105)
dim_v(axa, R_OUT + 760, ASM_LO, ASM_HI, "1636 assembly", side="right", pad=105)
axa.annotate("2614 nozzle bottom", xy=(R_OUT + 250, NOZ_LO),
             xytext=(R_OUT + 180, NOZ_LO + 980), fontsize=8, color=C_DIM,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_DIM, lw=0.7))

dim_h(axa, body_lo - 700, -R_IN, 0, "R 900", pad=130)
axa.annotate("R 1000 head", xy=(0, H_TOT - 60), xytext=(R_IN + 200, H_TOT + 260),
             fontsize=8, color=C_DIM, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_DIM, lw=0.7))
axa.annotate("wall 100", xy=(R_IN + 0.5 * T_WALL, NOZ_HI + 820),
             xytext=(R_OUT + 180, NOZ_HI + 1420), fontsize=8, color=C_DIM,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_DIM, lw=0.7))

axa.set_xlim(-2450, 2350)
axa.set_ylim(-420, H_TOT + 620)
axa.set_aspect("equal")
axa.axis("off")
axa.set_title("(a) Elevation, mm", fontsize=9.5, loc="left", x=0.12)

# ================================================================== (b) plan view
axb.add_patch(Circle((0, 0), R_IN + T_WALL, facecolor=C_STEEL,
                     edgecolor=C_EDGE, lw=0.9, zorder=1))
axb.add_patch(Circle((0, 0), R_IN, facecolor=C_WATER, edgecolor=C_EDGE,
                     lw=0.7, zorder=2))
axb.add_patch(Circle((0, 0), R_BARREL, facecolor=C_STEEL, edgecolor=C_EDGE,
                     lw=0.7, zorder=3))
axb.add_patch(Circle((0, 0), R_REFL, facecolor=C_REFL, edgecolor=C_EDGE,
                     lw=0.7, zorder=4))
axb.add_patch(Circle((0, 0), R_ENV, facecolor="#eef4f8", edgecolor=C_EDGE,
                     lw=0.8, ls=(0, (3, 3)), zorder=5))

for i in range(6):
    for j in range(6):
        if (i, j) in {(0, 0), (0, 5), (5, 0), (5, 5)}:
            continue
        axb.add_patch(Rectangle(((j - 3) * A_PITCH, (i - 3) * A_PITCH),
                                A_PITCH, A_PITCH, facecolor=C_FUEL,
                                edgecolor="#a8845a", lw=0.6, zorder=6))

axb.plot([0, R_ENV * np.cos(np.pi / 4)], [0, R_ENV * np.sin(np.pi / 4)],
         color="#7a3b2e", lw=0.8, ls=(0, (4, 3)), zorder=7)
axb.text(R_ENV * 0.42, R_ENV * 0.50, f"$R_\\mathrm{{env}}$ {R_ENV:.0f}",
         fontsize=8, color="#7a3b2e", ha="left", va="bottom", zorder=8)

# the radii that fix the radial zones
for k, (r, lab) in enumerate([(FUEL_FLAT, "Fuel at flat 643"),
                              (R_REFL, "Reflector outer 813"),
                              (R_BARREL, "Barrel outer 863"),
                              (R_IN, "Vessel inner 900")]):
    y = -(R_IN + T_WALL) - 200 - k * 175
    axb.annotate("", xy=(0, y), xytext=(r, y),
                 arrowprops=dict(arrowstyle="<->", color=C_DIM, lw=0.8))
    axb.text(r + 60, y, lab, fontsize=8, color=C_DIM, ha="left", va="center")

axb.set_xlim(-1300, 1900)
axb.set_ylim(-1950, 1250)
axb.set_aspect("equal")
axb.axis("off")
axb.set_title("(b) Plan view at core mid-height, mm", fontsize=9.5, loc="left",
              x=0.06)

fig.savefig(OUT, bbox_inches="tight")
if PNG:
    fig.savefig(PNG, dpi=160, bbox_inches="tight")
print(f"R_env {R_ENV:.1f}, fuel at flat {FUEL_FLAT:.1f}, "
      f"assembly {ASM_HI - ASM_LO:.1f}, active fuel {FUEL_HI - FUEL_LO:.1f}")
