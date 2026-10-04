#!/usr/bin/env python3
"""make_model_geometry_figure.py -- the reference model of Section 4.2, Figure 4.1.

  (a) the 17x17 assembly, showing only what the assembly geometry fixes: the two
      enrichment zones and the guide tubes. The gadolinia rod pattern is NOT
      drawn here, it belongs to the section that treats the burnable absorbers.

  (b) the core plan view at the drawing pitch and reflector thickness.

The lattice map is imported from reactor_model.py when that module is importable,
so the figure cannot drift from the model. It falls back to the same literals.
Colours follow th_model_3d.pdf (Figure 4.3).

usage (plot only, no transport):
    python make_model_geometry_figure.py OUT.pdf
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

OUT = sys.argv[1]

try:
    from reactor_model import GUIDE_TUBE_POSITIONS
except Exception:                                    # plotting outside the env
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
INNER_LO, INNER_HI = N // 2 - 4, N // 2 + 4          # central 9x9 block
GT = set(GUIDE_TUBE_POSITIONS)

# colours of th_model_3d.pdf (Figure 4.3)
C_IN = "#c4717c"        # inner zone, e_in
C_OUT = "#efcfd2"       # outer zone, e_out
C_GT = "#ffffff"        # guide tube
C_EDGE = "#9a6575"
C_ZONE = "#6e2a35"
C_TXT = "#3b2a2e"
C_REFL = "#a0b0ba"
C_VESSEL_EDGE = "#6f7d86"

PITCH_DRAW, TREFL_DRAW = 1.26, 12.0
R_VESSEL = 90.0

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(12.6, 5.6))
axa = fig.add_axes([0.03, 0.10, 0.40, 0.82])
axb = fig.add_axes([0.50, 0.05, 0.48, 0.87])

# ------------------------------------------------------------------ (a) assembly
for i in range(N):
    for j in range(N):
        y = N - 1 - i                                 # row 0 at the top
        if (i, j) in GT:
            axa.add_patch(Circle((j, y), 0.40, facecolor=C_GT,
                                 edgecolor=C_EDGE, lw=0.9, zorder=3))
            continue
        inner = INNER_LO <= i <= INNER_HI and INNER_LO <= j <= INNER_HI
        axa.add_patch(Circle((j, y), 0.40, facecolor=C_IN if inner else C_OUT,
                             edgecolor=C_EDGE, lw=0.4, zorder=2))

# the inner zone boundary
axa.add_patch(Rectangle((INNER_LO - 0.5, N - 1 - INNER_HI - 0.5), 9, 9,
                        fill=False, edgecolor=C_ZONE, lw=1.6, ls=(0, (5, 3)),
                        zorder=5))

# the pitch
axa.annotate("", xy=(0.4, N + 0.45), xytext=(1.4, N + 0.45),
             arrowprops=dict(arrowstyle="<->", color="#404b54", lw=0.9))
axa.text(0.9, N + 0.75, "Pitch $p$", ha="center", va="bottom", fontsize=8.5,
         color="#404b54")

axa.set_xlim(-1.1, N + 0.6)
axa.set_ylim(-2.6, N + 1.6)
axa.set_aspect("equal")
axa.axis("off")
axa.set_title("(a) The $17\\times17$ assembly", fontsize=10, loc="left", x=-0.02)

handles = [
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_IN, mec=C_EDGE, mew=0.5,
           label="Inner zone, $e_\\mathrm{in}$"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_OUT, mec=C_EDGE, mew=0.5,
           label="Outer zone, $e_\\mathrm{out}$"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_GT, mec=C_EDGE, mew=0.9,
           label="Guide tube"),
]
axa.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
           ncol=3, frameon=False, fontsize=8.5, handletextpad=0.4,
           columnspacing=1.4)

# ------------------------------------------------------------------ (b) core plan
a_w = N * PITCH_DRAW                                  # assembly width
r_env = np.sqrt(13.0) * a_w                           # circumscribed radius
r_refl = r_env + TREFL_DRAW

axb.add_patch(Circle((0, 0), R_VESSEL, facecolor="#c9d0d5",
                     edgecolor=C_VESSEL_EDGE, lw=1.4, zorder=1))
axb.add_patch(Circle((0, 0), r_refl, facecolor=C_REFL,
                     edgecolor="none", zorder=2))
axb.add_patch(Circle((0, 0), r_env, facecolor="#f7f1f2", edgecolor=C_ZONE,
                     lw=1.1, ls=(0, (3, 3)), zorder=3))

for i in range(6):
    for j in range(6):
        if (i, j) in {(0, 0), (0, 5), (5, 0), (5, 5)}:
            continue
        axb.add_patch(Rectangle(((j - 3) * a_w, (i - 3) * a_w), a_w, a_w,
                                facecolor=C_OUT, edgecolor=C_EDGE, lw=0.7,
                                zorder=4))

axb.text(0, 0, "32 assemblies\n($6\\times6$ minus the corners)", ha="center",
         va="center", fontsize=9, zorder=6,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.92))

axb.annotate("Steel reflector", xy=(-r_env * 0.76, r_env * 0.76),
             xytext=(-R_VESSEL * 1.35, R_VESSEL * 0.92), fontsize=8.5,
             ha="left", va="center", color=C_TXT,
             arrowprops=dict(arrowstyle="->", color=C_TXT, lw=0.9))
axb.annotate("$R_\\mathrm{env} = \\sqrt{13}\\,\\cdot\\,17p$",
             xy=(r_env * 0.70, r_env * 0.70),
             xytext=(R_VESSEL * 0.42, R_VESSEL * 1.16), fontsize=8.5,
             ha="left", va="center", color=C_TXT,
             arrowprops=dict(arrowstyle="->", color=C_TXT, lw=0.9))
axb.annotate("", xy=(r_env, -6), xytext=(r_refl, -6),
             arrowprops=dict(arrowstyle="<->", color="#7a3b2e", lw=1.1))
axb.text(r_refl + 5, -6, "Reflector thickness $t_\\mathrm{refl}$,\n"
         "design variable, 2.0 to 19.5 cm",
         fontsize=8.5, ha="left", va="center", color="#7a3b2e")
axb.annotate("Vessel, $R$ = 90 cm",
             xy=(-R_VESSEL * 0.62, -R_VESSEL * 0.80),
             xytext=(-R_VESSEL * 1.35, -R_VESSEL * 1.10), fontsize=8.5,
             ha="left", va="center", color=C_TXT,
             arrowprops=dict(arrowstyle="->", color=C_TXT, lw=0.9))

axb.set_xlim(-R_VESSEL * 1.42, R_VESSEL * 1.62)
axb.set_ylim(-R_VESSEL * 1.30, R_VESSEL * 1.30)
axb.set_aspect("equal")
axb.axis("off")
axb.set_title(f"(b) The core plan view, drawn at $p$ = {PITCH_DRAW} cm and "
              f"$t_\\mathrm{{refl}}$ = {TREFL_DRAW:.0f} cm",
              fontsize=10, loc="left", x=-0.02)

fig.savefig(OUT, bbox_inches="tight")
print(f"R_env {r_env:.2f}, reflector outer {r_refl:.2f}, guide tubes {len(GT)}")
