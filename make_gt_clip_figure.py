"""Guide-tube clipping at small pitch: the tube annulus against the lattice cell
at 1.26 cm and at 1.150 cm, with the wall fraction lost outside the cell.

usage: python make_gt_clip_figure.py OUT.pdf OUT.png
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2]
R_IN, R_OUT = 0.5715, 0.6121          # cm, guide tube (reactor_model.Geometry17x17)
PITCHES = (1.26, 1.150)


def segment(r, h):
    """Area of the circle of radius r beyond the chord at distance h."""
    return 0.0 if r <= h else r * r * np.arccos(h / r) - h * np.sqrt(r * r - h * h)


def lost_fraction(pitch):
    """Exact fraction of the annulus area outside the square cell (four
    non-overlapping segments, valid while R_OUT < half-pitch * sqrt(2))."""
    h = pitch / 2
    assert R_OUT < h * np.sqrt(2)
    lost = 4 * (segment(R_OUT, h) - segment(R_IN, h))
    return lost / (np.pi * (R_OUT**2 - R_IN**2))


assert abs(lost_fraction(1.150) - 0.2768) < 1e-4 and lost_fraction(1.26) == 0.0


fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))
for ax, p in zip(axes, PITCHES):
    h = p / 2
    ax.add_patch(Rectangle((-h, -h), p, p, facecolor="#cfe3f5", edgecolor="none"))
    # wall: grey inside the cell, red where the lattice discards it
    x = np.linspace(-R_OUT, R_OUT, 1201)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    wall = (r2 >= R_IN**2) & (r2 <= R_OUT**2)
    outside = (np.abs(X) > h) | (np.abs(Y) > h)
    img = np.full(X.shape + (4,), 0.0)
    img[wall & ~outside] = (0.45, 0.45, 0.45, 1.0)
    img[wall & outside] = (0.84, 0.15, 0.15, 1.0)
    ax.imshow(img, extent=(-R_OUT, R_OUT, -R_OUT, R_OUT), origin="lower", zorder=2)
    ax.add_patch(Circle((0, 0), R_IN, facecolor="#cfe3f5", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((-h, -h), p, p, fill=False, edgecolor="black",
                           linestyle="--", linewidth=1.4, zorder=3))
    f = lost_fraction(p)
    ax.set_title(f"Pitch {p:.3f} cm, half-pitch {h:.3f} cm\n"
                 f"Wall lost outside the cell: {100 * f:.1f} %", fontsize=11)
    lim = 0.70
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")

handles = [Patch(facecolor="#737373", label="Guide-tube wall kept"),
           Patch(facecolor="#d62626", label="Guide-tube wall clipped (lost)"),
           Patch(facecolor="#cfe3f5", label="Moderator"),
           plt.Line2D([], [], color="black", linestyle="--", label="Lattice cell boundary")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print({p: round(lost_fraction(p), 4) for p in PITCHES}, "clip onset", 2 * R_OUT)
