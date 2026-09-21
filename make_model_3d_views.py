#!/usr/bin/env python3
"""
make_model_3d_views.py -- three-dimensional views of the fuel assembly and of
the core, for the Reactor model section of the Methodology chapter.

Drawing only, no transport. Every dimension is one this work already uses:
the axial stack and the radial build of hardware3d.py, the envelope and vessel
radii of core_geometry.py, and the lattice of reactor_model.py. The axial
elevations are those of the axial-layout table of the methodology.

Usage: python make_model_3d_views.py <out.pdf> <out.png>
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2]

N, PITCH = 17, 1.26
A = N * PITCH                 # assembly pitch, cm
H_FUEL = 120.0
REFL, BARREL = 5.66, 5.08     # confirmed reflector, benchmark barrel, cm
R_VESSEL_IN, WALL = 90.0, 10.0

STACK = [("Bottom nozzle, 10.16 cm", 10.16, "#8fa2ae"),
         ("Lower end cap, 1.21 cm", 1.205, "#c8ced4"),
         ("Active fuel, 120 cm", H_FUEL, "#c0616b"),
         ("Plenum spring, 13.49 cm", 13.49, "#e3c46b"),
         ("Upper end cap, 1.21 cm", 1.205, "#c8ced4"),
         ("Coolant gap, 8.48 cm", 8.481, "#9fc7e8"),
         ("Top nozzle, 9.02 cm", 9.020, "#8fa2ae")]
GRID_H, GRID_FIRST, GRID_STEP = 4.445, 3.555, 41.41
GRID_COLOUR = "#3f6d8c"

CORE_MAP = np.array([[0, 1, 1, 1, 1, 0]] + [[1] * 6] * 4 + [[0, 1, 1, 1, 1, 0]])


def box(ax, x0, x1, y0, y1, z0, z1, fc, alpha, ec="#33506b", lw=0.4):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], facecolor=fc,
                                         edgecolor=ec, lw=lw, alpha=alpha))


def cylinder(ax, r, z0, z1, colour, alpha, n=72, lw=0.0):
    th = np.linspace(0, 2 * np.pi, n)
    x, y = r * np.cos(th), r * np.sin(th)
    quads = [[(x[i], y[i], z0), (x[i + 1], y[i + 1], z0),
              (x[i + 1], y[i + 1], z1), (x[i], y[i], z1)] for i in range(n - 1)]
    ax.add_collection3d(Poly3DCollection(quads, facecolor=colour, edgecolor="none",
                                         alpha=alpha))


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(10.6, 5.0))

# ---------------- (a) the fuel assembly ------------------------------------
ax = fig.add_subplot(1, 2, 1, projection="3d")
half, z = A / 2.0, 0.0
for name, h, colour in STACK:
    box(ax, -half, half, -half, half, z, z + h, colour, 0.45 if name.startswith("Active") else 0.6)
    if name.startswith("Active"):
        z_fuel0 = z
    z += h
z_top = z
for i in range(4):
    zg = z_fuel0 + GRID_FIRST + i * GRID_STEP
    box(ax, -half, half, -half, half, zg, zg + GRID_H, GRID_COLOUR, 0.85)

ax.set_title("(a) Fuel assembly", fontsize=10)
ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(0, z_top)
ax.set_box_aspect((1, 1, 2.6))
ax.set_zlabel("Elevation above the bottom nozzle [cm]", fontsize=8.5)
ax.view_init(elev=16, azim=-60)
clean(ax)
ax.legend(handles=[Patch(facecolor=c, edgecolor="#33506b", label=n) for n, _, c in STACK]
                  + [Patch(facecolor=GRID_COLOUR, edgecolor="none",
                           label="Spacer grids, 4 of 4.45 cm")],
          loc="upper left", bbox_to_anchor=(-0.28, 0.98), fontsize=7.6, frameon=False)

# ---------------- (b) the core --------------------------------------------
bx = fig.add_subplot(1, 2, 2, projection="3d")
r_env = np.hypot(3 * A, 2 * A)
r_refl, r_barrel = r_env + REFL, r_env + REFL + BARREL

for i in range(6):
    for j in range(6):
        if CORE_MAP[i, j]:
            box(bx, (j - 3) * A, (j - 2) * A, (2 - i) * A, (3 - i) * A,
                0, H_FUEL, "#c0616b", 0.30, ec="#7a1f2b", lw=0.4)
cylinder(bx, r_env, 0, H_FUEL, "#8fa2ae", 0.16)
cylinder(bx, r_refl, 0, H_FUEL, "#5d6770", 0.16)
cylinder(bx, r_barrel, 0, H_FUEL, "#3f4a52", 0.12)
cylinder(bx, R_VESSEL_IN + WALL, 0, H_FUEL, "#2b3238", 0.10)

bx.set_title("(b) Core, 32 assemblies", fontsize=10)
lim = (R_VESSEL_IN + WALL) * 1.02
bx.set_xlim(-lim, lim); bx.set_ylim(-lim, lim); bx.set_zlim(0, H_FUEL)
bx.set_box_aspect((1, 1, 0.62))
bx.set_zlabel("Active height [cm]", fontsize=8.5)
bx.view_init(elev=26, azim=-60)
clean(bx)
bx.legend(handles=[Patch(facecolor="#c0616b", edgecolor="#7a1f2b", label="Fuel assembly"),
                   Patch(facecolor="#8fa2ae", edgecolor="none",
                         label=f"Fuel envelope, r = {r_env:.1f} cm"),
                   Patch(facecolor="#5d6770", edgecolor="none",
                         label=f"Heavy reflector, {REFL:g} cm"),
                   Patch(facecolor="#3f4a52", edgecolor="none",
                         label=f"Core barrel, {BARREL:g} cm"),
                   Patch(facecolor="#2b3238", edgecolor="none",
                         label=f"Vessel wall, {WALL:g} cm")],
          loc="upper right", bbox_to_anchor=(1.20, 0.98), fontsize=7.6, frameon=False)

fig.subplots_adjust(left=0.10, right=0.92, wspace=0.10)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print("written", OUT_PDF, "| r_env %.1f  r_refl %.1f  r_barrel %.1f cm" % (r_env, r_refl, r_barrel))
