#!/usr/bin/env python3
"""make_model_geometry_figure.py -- the reference model of Section 4.2, two panels.

  (a) the 17x17 assembly, showing only what the assembly geometry fixes: the two
      enrichment zones, the 24 guide tubes and the single central instrument tube.
      The gadolinia rod pattern is NOT drawn here, it belongs to the section that
      treats the burnable absorbers.

  (b) the core plan view at the drawing pitch and reflector thickness, with the
      vessel-fit budget that couples p and t_refl.

The lattice map is imported from reactor_model.py when that module is importable,
so the figure cannot drift from the model. It falls back to the same literals.

usage (plot only, no transport):
    python make_model_geometry_figure.py OUT.pdf [OUT.png]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

OUT = sys.argv[1]
PNG = sys.argv[2] if len(sys.argv) > 2 else None

try:
    from reactor_model import GUIDE_TUBE_POSITIONS, INSTRUMENT_TUBE_POSITION
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
    INSTRUMENT_TUBE_POSITION = (8, 8)

N = 17
INNER_LO, INNER_HI = N // 2 - 4, N // 2 + 4          # central 9x9 block
GT = set(GUIDE_TUBE_POSITIONS) - {INSTRUMENT_TUBE_POSITION}

C_IN = "#6f9fc0"        # inner zone, e_in
C_OUT = "#cfe0ec"       # outer zone, e_out
C_GT = "#ffffff"        # guide tube
C_IT = "#9aa7b1"        # instrument tube
C_EDGE = "#4a5b68"
C_ZONE = "#1f3b52"

PITCH_DRAW, TREFL_DRAW = 1.26, 12.0
R_VESSEL, PAD = 90.0, 0.02

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(12.6, 7.4))
axa = fig.add_axes([0.045, 0.38, 0.40, 0.58])
axb = fig.add_axes([0.525, 0.38, 0.44, 0.58])

# ------------------------------------------------------------------ (a) assembly
n_in = n_out = 0
for i in range(N):
    for j in range(N):
        y = N - 1 - i                                 # row 0 at the top
        if (i, j) == INSTRUMENT_TUBE_POSITION:
            axa.add_patch(Circle((j, y), 0.40, facecolor=C_IT,
                                 edgecolor=C_EDGE, lw=0.6, zorder=3))
            axa.add_patch(Circle((j, y), 0.13, facecolor="white",
                                 edgecolor="none", zorder=4))
            continue
        if (i, j) in GT:
            axa.add_patch(Circle((j, y), 0.40, facecolor=C_GT,
                                 edgecolor=C_EDGE, lw=0.9, zorder=3))
            continue
        inner = INNER_LO <= i <= INNER_HI and INNER_LO <= j <= INNER_HI
        n_in, n_out = (n_in + 1, n_out) if inner else (n_in, n_out + 1)
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
axa.set_ylim(-1.5, N + 1.6)
axa.set_aspect("equal")
axa.axis("off")
axa.set_title("(a) The $17\\times17$ assembly", fontsize=10, loc="left", x=-0.02)

handles = [
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_IN, mec=C_EDGE, mew=0.5,
           label=f"Inner zone, $e_\\mathrm{{in}}$ ({n_in} rods)"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_OUT, mec=C_EDGE, mew=0.5,
           label=f"Outer zone, $e_\\mathrm{{out}}$ ({n_out} rods)"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_GT, mec=C_EDGE, mew=0.9,
           label=f"Guide tube ({len(GT)})"),
    Line2D([], [], marker="o", ls="", ms=9, mfc=C_IT, mec=C_EDGE, mew=0.5,
           label="Instrument tube (1)"),
]
axa.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
           ncol=2, frameon=False, fontsize=8.5, handletextpad=0.4,
           columnspacing=1.4, labelspacing=0.5)

# ------------------------------------------------------------------ (b) core plan
a_w = N * PITCH_DRAW                                  # assembly width
r_env = np.sqrt(13.0) * a_w                           # circumscribed radius
r_refl = r_env + TREFL_DRAW

axb.add_patch(Circle((0, 0), R_VESSEL, facecolor="#b9bec2",
                     edgecolor="#6b7075", lw=1.2, zorder=1))
axb.add_patch(Circle((0, 0), r_refl, facecolor="#9aa7b1",
                     edgecolor="none", zorder=2))
axb.add_patch(Circle((0, 0), r_env, facecolor="#f2f7fa", edgecolor=C_ZONE,
                     lw=1.1, ls=(0, (3, 3)), zorder=3))

for i in range(6):
    for j in range(6):
        if (i, j) in {(0, 0), (0, 5), (5, 0), (5, 5)}:
            continue
        x0 = (j - 3) * a_w
        y0 = (i - 3) * a_w
        axb.add_patch(Rectangle((x0, y0), a_w, a_w, facecolor="#dce9f2",
                                edgecolor="#5b8fb0", lw=0.7, zorder=4))

axb.text(0, 0, "32 assemblies\n($6\\times6$ minus the corners)", ha="center",
         va="center", fontsize=9, zorder=6,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.92))

axb.annotate("Steel reflector", xy=(-r_env * 0.76, r_env * 0.76),
             xytext=(-R_VESSEL * 1.35, R_VESSEL * 0.92), fontsize=8.5,
             ha="left", va="center", color="#243b4a",
             arrowprops=dict(arrowstyle="->", color="#243b4a", lw=0.9))
axb.annotate("$R_\\mathrm{env} = \\sqrt{13}\\,\\cdot\\,17p$",
             xy=(r_env * 0.70, r_env * 0.70),
             xytext=(R_VESSEL * 0.42, R_VESSEL * 1.16), fontsize=8.5,
             ha="left", va="center", color="#243b4a",
             arrowprops=dict(arrowstyle="->", color="#243b4a", lw=0.9))
axb.annotate("", xy=(r_env, -6), xytext=(r_refl, -6),
             arrowprops=dict(arrowstyle="<->", color="#7a3b2e", lw=1.1))
axb.text(r_refl + 5, -6, "$t_\\mathrm{refl}$, design variable\n2.0 to 19.5 cm",
         fontsize=8.5, ha="left", va="center", color="#7a3b2e")
axb.annotate("Vessel, $R = 90$ cm",
             xy=(-R_VESSEL * 0.62, -R_VESSEL * 0.80),
             xytext=(-R_VESSEL * 1.35, -R_VESSEL * 1.10), fontsize=8.5,
             ha="left", va="center", color="#243b4a",
             arrowprops=dict(arrowstyle="->", color="#243b4a", lw=0.9))

axb.set_xlim(-R_VESSEL * 1.42, R_VESSEL * 1.62)
axb.set_ylim(-R_VESSEL * 1.30, R_VESSEL * 1.30)
axb.set_aspect("equal")
axb.axis("off")
axb.set_title(f"(b) The core plan view, drawn at $p = {PITCH_DRAW}$ cm and "
              f"$t_\\mathrm{{refl}} = {TREFL_DRAW:.0f}$ cm",
              fontsize=10, loc="left", x=-0.02)

# ------------------------------------------------------------------ the footnotes
fig.text(0.525, 0.310,
         "$g_\\mathrm{geom} = R_\\mathrm{env}(p) + \\delta_\\mathrm{pad} + "
         "t_\\mathrm{refl} - 90 \\leq 0$,   "
         f"$\\delta_\\mathrm{{pad}} = {PAD}$ cm",
         fontsize=8.5, ha="left", va="top")
fig.text(0.525, 0.265,
         f"Radial budget at this drawing: $R_\\mathrm{{env}} = {r_env:.2f}$ cm, "
         f"reflector outer $= {r_refl:.2f}$ cm, "
         f"slack $= {R_VESSEL - r_refl - PAD:.2f}$ cm",
         fontsize=8, ha="left", va="top", color="#404b54")

fig.text(0.045, 0.215, "The six design variables", fontsize=9.5, weight="bold",
         ha="left", va="top")
fig.add_artist(Line2D([0.045, 0.955], [0.196, 0.196], color="#9aa7b1",
                      lw=0.8, transform=fig.transFigure))
ROWS = [("$e_\\mathrm{in}$", "Inner-zone enrichment", "2.00 to 19.75", "wt% $^{235}$U",
         "$p$", "Lattice pitch", "1.15 to 1.43", "cm"),
        ("$e_\\mathrm{out}$", "Outer-zone enrichment", "2.00 to 19.75", "wt% $^{235}$U",
         "$t_\\mathrm{refl}$", "Reflector thickness", "2.0 to 19.5", "cm"),
        ("$w_\\mathrm{Gd}$", "Gadolinia weight fraction", "0.0 to 8.0", "wt% Gd$_2$O$_3$",
         "$n_\\mathrm{Gd}$", "Gadolinia rod count", "12 to 40", "rods")]
XL = [0.048, 0.085, 0.235, 0.310, 0.525, 0.562, 0.712, 0.788]
for k, row in enumerate(ROWS):
    y = 0.172 - k * 0.034
    for x, cell in zip(XL, row):
        fig.text(x, y, cell, fontsize=8.5, ha="left", va="top")

fig.text(0.045, 0.030,
         "264 fuel rods per assembly, 72 in the inner zone and 192 in the outer "
         "zone. $p$ and $t_\\mathrm{refl}$ are coupled through "
         "$g_\\mathrm{geom}$, so the ranges above bound a box and not a feasible "
         "region.", fontsize=8, ha="left", va="bottom", color="#404b54")

fig.savefig(OUT, bbox_inches="tight")
if PNG:
    fig.savefig(PNG, dpi=160, bbox_inches="tight")
print(f"inner {n_in}, outer {n_out}, guide tubes {len(GT)}, "
      f"instrument tube 1, total {n_in + n_out + len(GT) + 1}")
print(f"R_env {r_env:.2f}, reflector outer {r_refl:.2f}, "
      f"slack {R_VESSEL - r_refl - PAD:.2f}")
