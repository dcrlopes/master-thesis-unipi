#!/usr/bin/env python3
"""
make_c9_layered_model_figure.py -- the model of the axially resolved
depletion (c9_dep_core3d.py), drawn from the same constants the code uses.
Drawing only, no transport.

  (a) the C9-47 assembly, radial section: the 17 x 17 lattice of
      reactor_model.build_assembly_universe, with the 20-rod gadolinia
      pattern GD_PATTERNS[20], the guide-tube positions, and the central
      9 x 9 block whose pins carry separate materials ("fuel_in",
      "fuel_gd_in") from the outer pins ("fuel_out", "fuel_gd_out").
  (b) the core, vertical half section along the row of assemblies next to
      the core centre (y = A/2), which crosses one assembly of each ring,
      C, M and P (zoning.ring_map): the active fuel cut into 8 layers of
      15 cm (fuel_cuts), the spacer grids, end caps, nozzles, plenum and
      coolant gaps of hardware3d.elevations, the water above and below,
      the heavy reflector, the barrel, the downcomer and the vessel wall.
      Radii from core_geometry.core_envelope_radius + FUEL_PAD_CM,
      refl_thick of C9-47, and HardwareSpec (barrel 5.08, vessel 90 / 100).

Usage: python make_c9_layered_model_figure.py <out.pdf> [out.png]
"""
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

# ------------------------------------------------------------- constants --
N, PITCH = 17, 1.26
A = N * PITCH                                    # assembly pitch, cm
GT = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13), (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
      (8, 2), (8, 5), (8, 8), (8, 11), (8, 14), (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
      (13, 3), (13, 13), (14, 5), (14, 8), (14, 11)]


def orbit(i, j):
    return {(i, j), (j, i), (16 - i, j), (i, 16 - j), (16 - i, 16 - j), (16 - j, i), (j, 16 - i), (16 - j, 16 - i)}


GD = orbit(2, 2) | orbit(6, 6) | orbit(3, 8) | orbit(4, 4) | orbit(6, 8)   # GD_PATTERNS[20]
assert len(GD) == 20 and not (GD & set(GT))
INNER = range(N // 2 - 4, N // 2 + 5)            # central 9 x 9 block
FUEL_OR, CLAD_OR, GT_OR = 0.4058, 0.4750, 0.6121      # reactor_model.Geometry17x17

E, GDW = 4.4596, 4.4178                           # C9-47
M = {"C": 0.720, "M": (32 - 4 * 0.720 - 16 * 1.15) / 12, "P": 1.15}
REFL, BARREL, R_VIN, WALL = 5.66, 5.08, 90.0, 10.0
CORE = np.array([[0, 1, 1, 1, 1, 0]] + [[1] * 6] * 4 + [[0, 1, 1, 1, 1, 0]])

H = 120.0
Z = {"fuel": (-60.0, 60.0)}
Z["lower_cap"] = (-61.205, -60.0)
Z["bottom_nozzle"] = (-71.365, -61.205)
Z["water_below"] = (-86.365, -71.365)
Z["plenum"] = (60.0, 73.49)
Z["upper_cap"] = (73.49, 74.695)
Z["upper_gap"] = (74.695, 83.176)
Z["top_nozzle"] = (83.176, 92.196)
Z["water_above"] = (92.196, 107.196)
GRIDS = [(-60 + 3.555 + g * 41.4, -60 + 3.555 + g * 41.4 + 4.445, "htm" if g == 0 else "htp") for g in range(4)]
LAYERS = np.linspace(-60, 60, 9)


def envelope():
    ny, nx = CORE.shape
    r = 0.0
    for i in range(ny):
        for j in range(nx):
            if CORE[i, j]:
                for dy in (0, 1):
                    for dx in (0, 1):
                        r = max(r, math.hypot((j + dx - nx / 2) * A, (i + dy - ny / 2) * A))
    return r + 0.02


R_FUEL = envelope()
R_REFL, R_BAR, R_VOUT = R_FUEL + REFL, R_FUEL + REFL + BARREL, R_VIN + WALL
YC = A / 2                                         # chord through the row next to the centre
X = lambda r: math.sqrt(r * r - YC * YC)

C_FUEL, C_GD, C_GT, C_W = "#e8a07f", "#8c2d04", "#9ecae1", "#d8ecf7"
RING_C = {"C": ["#6baed6", "#3182bd"], "M": ["#74c476", "#31a354"], "P": ["#fd8d3c", "#e6550d"]}
C_REFL, C_SS, C_VES = "#bdbdbd", "#737373", "#525252"
C_NOZ, C_CAP, C_PLEN, C_GAP = "#8fa2ae", "#c8ced4", "#e3c46b", "#9fc7e8"
C_HTM, C_HTP = "#3f007d", "#807dba"

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(12.0, 6.6))
ax = fig.add_axes([0.03, 0.22, 0.36, 0.70])
bx = fig.add_axes([0.62, 0.12, 0.52, 0.82])

# --------------------------------------------------------------- (a) ------
ax.add_patch(Rectangle((0, 0), A, A, facecolor=C_W, edgecolor="k", lw=1.0))
for i in range(N):
    for j in range(N):
        x, y = (j + 0.5) * PITCH, (N - i - 0.5) * PITCH
        if (i, j) in GT:
            ax.add_patch(Circle((x, y), GT_OR, facecolor=C_W, edgecolor=C_GT, lw=1.6))
            continue
        ax.add_patch(Circle((x, y), CLAD_OR, facecolor="#cccccc", edgecolor="none"))
        ax.add_patch(Circle((x, y), FUEL_OR, facecolor=C_GD if (i, j) in GD else C_FUEL, edgecolor="none"))
lo = (N // 2 - 4) * PITCH
ax.add_patch(Rectangle((lo, (N - (N // 2 + 5)) * PITCH), 9 * PITCH, 9 * PITCH, fill=False,
                       edgecolor="k", lw=1.2, ls="--"))
ax.set_xlim(-0.3, A + 0.3); ax.set_ylim(-0.3, A + 0.3); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) Assembly of C9-47, radial section", fontsize=10, loc="left")
leg_a = [Patch(facecolor=C_FUEL, label="UO$_2$ fuel rod"),
         Patch(facecolor=C_GD, label="UO$_2$–Gd$_2$O$_3$ rod, 20 rods, %.2f wt%% Gd$_2$O$_3$" % GDW),
         Patch(facecolor=C_W, edgecolor=C_GT, lw=1.6, label="Guide tube, all rods out"),
         Patch(facecolor="none", edgecolor="k", ls="--", label="Central 9 $\\times$ 9 block, own materials")]
ax.legend(handles=leg_a, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7.8, frameon=False)
fig.text(0.21, 0.05, "Enrichment %.2f wt%% scaled by ring: C %.2f, M %.2f, P %.2f wt%%"
         % (E, E * M["C"], E * M["M"], E * M["P"]), ha="center", fontsize=7.8)

# --------------------------------------------------------------- (b) ------
zb, zt = Z["water_below"][0], Z["water_above"][1]
x_lat = 3 * A                                      # C, M, P assemblies along the chord
x_fuel, x_refl, x_bar, x_vin, x_vout = X(R_FUEL), X(R_REFL), X(R_BAR), X(R_VIN), X(R_VOUT)


def box(x0, x1, z0, z1, **kw):
    bx.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, **kw))


# full-height radial regions outside the fuel cylinder
box(x_fuel, x_refl, zb, zt, facecolor=C_REFL, edgecolor="none")
box(x_refl, x_bar, zb, zt, facecolor=C_SS, edgecolor="none")
box(x_bar, x_vin, zb, zt, facecolor=C_W, edgecolor="none")
box(x_vin, x_vout, zb, zt, facecolor=C_VES, edgecolor="none")
# inside the fuel cylinder: water slabs, and the reflector lattice positions beyond the assemblies
box(0, x_fuel, *Z["water_below"], facecolor=C_W, edgecolor="none")
box(0, x_fuel, *Z["water_above"], facecolor=C_W, edgecolor="none")
box(x_lat, x_fuel, Z["bottom_nozzle"][0], Z["top_nozzle"][1], facecolor=C_REFL, edgecolor="none")
# axial structures across the assemblies
for key, col in (("bottom_nozzle", C_NOZ), ("lower_cap", C_CAP), ("plenum", C_PLEN),
                 ("upper_cap", C_CAP), ("upper_gap", C_GAP), ("top_nozzle", C_NOZ)):
    box(0, x_lat, *Z[key], facecolor=col, edgecolor="none")
# fuel layers, coloured by ring, alternating shade by layer
for k, ring in enumerate(("C", "M", "P")):
    for l in range(8):
        box(k * A, (k + 1) * A, LAYERS[l], LAYERS[l + 1], facecolor=RING_C[ring][l % 2], edgecolor="none")
    bx.text((k + 0.5) * A, 62.5, "Ring " + ring, ha="center", va="bottom", fontsize=8, color="k")
for l in range(9):
    bx.plot([0, x_lat], [LAYERS[l]] * 2, color="k", lw=0.6)
for l in range(8):
    zm = 0.5 * (LAYERS[l] + LAYERS[l + 1])
    for g0, g1, _ in GRIDS:                     # keep the number clear of a grid band
        if g0 - 2.5 <= zm <= g1 + 2.5:
            zm = g1 + 3.5 if LAYERS[l + 1] - g1 > 6 else g0 - 3.5
    bx.text(1.2, zm, str(l + 1), ha="left", va="center", fontsize=7.5, color="w", fontweight="bold")
for g0, g1, kind in GRIDS:
    box(0, x_lat, g0, g1, facecolor=C_HTM if kind == "htm" else C_HTP, edgecolor="none", alpha=0.85)
for k in range(1, 3):
    bx.plot([k * A, k * A], [Z["bottom_nozzle"][0], Z["top_nozzle"][1]], color="k", lw=0.6)
bx.plot([x_lat, x_lat], [Z["bottom_nozzle"][0], Z["top_nozzle"][1]], color="k", lw=0.6)
bx.plot([0, x_vout], [zb, zb], color="k", lw=1.0); bx.plot([0, x_vout], [zt, zt], color="k", lw=1.0)
bx.plot([x_vout, x_vout], [zb, zt], color="k", lw=1.0)
bx.set_xlim(0, x_vout + 1)
bx.set_ylim(zb - 2, zt + 2)
bx.set_xlabel("Distance from the core axis along the section (cm)")
bx.set_ylabel("Elevation from the core mid-plane (cm)")
bx.set_title("(b) Core, vertical half section: 8 depletion layers of 15 cm", fontsize=10, loc="left")
leg_b = [Patch(facecolor=RING_C["C"][1], label="Fuel, ring C"), Patch(facecolor=RING_C["M"][1], label="Fuel, ring M"),
         Patch(facecolor=RING_C["P"][1], label="Fuel, ring P"),
         Patch(facecolor=C_HTM, label="Spacer grid, Inconel"), Patch(facecolor=C_HTP, label="Spacer grid, Zircaloy"),
         Patch(facecolor=C_CAP, label="End cap"), Patch(facecolor=C_PLEN, label="Plenum"),
         Patch(facecolor=C_GAP, label="Coolant gap"), Patch(facecolor=C_NOZ, label="Nozzle"),
         Patch(facecolor=C_W, edgecolor="0.6", label="Water, 1000 ppm boron"), Patch(facecolor=C_REFL, label="Heavy reflector"),
         Patch(facecolor=C_SS, label="Barrel"), Patch(facecolor=C_VES, label="Vessel wall")]
bx.legend(handles=leg_b, loc="upper right", bbox_to_anchor=(-0.09, 1.0), fontsize=7.5, frameon=False)
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1], "r_fuel %.2f, chord x: lattice %.2f fuel %.2f refl %.2f barrel %.2f vessel %.2f / %.2f"
      % (R_FUEL, x_lat, x_fuel, x_refl, x_bar, x_vin, x_vout))
