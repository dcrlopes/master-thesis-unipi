#!/usr/bin/env python3
"""
make_c9_readings_figure.py -- the Campaign 9 front under each reading of
tab:c9-readings, over the evaluated designs coloured by origin (Campaign 9,
stage 1 and stage 2 of the continuation). Drawing only.

Panel (a): two-dimensional peaking of the campaign, readings 1 and 3.
Panel (b): three-dimensional peaking at 150 000 particles, readings 2, 4, 5.
Filled markers: feasible under the constraints stored by the evaluator of the
design; open markers: infeasible.

Reads  out_c9a/optimization_checkpoint.json   (C9-0 to C9-95)
       confirm3d_c9_all/summary.json          (3D peaking, 60 designs, 2 seeds)
       confirm3d_c9_front/summary.json        (3D peaking, 5 front designs)
       confirm3d_c9a/summary.json             (3D peaking, C9-69 and C9-70)
       c9_dep_core3d_d1_L8/runs.json          (c_max of C9-1 from its 8-layer hump)
Usage  python make_c9_readings_figure.py OUT.pdf [OUT.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = json.load(open("out_c9a/optimization_checkpoint.json"))
A, CN = B["all_raw"], B["constraint_names"]
F3 = {int(k): v["ARO_3Dhw"]["F"] for k, v in json.load(open("confirm3d_c9_all/summary.json")).items()}
for f in ("confirm3d_c9_front/summary.json", "confirm3d_c9a/summary.json"):
    F3.update({int(k): v["ARO_3Dhw"]["F"] for k, v in json.load(open(f)).items()})
F2 = {i: r["peaking"] for i, r in enumerate(A)}
C = {i: r["c_max"] for i, r in enumerate(A)}
C_AX = dict(C)                                        # c_max on the eight-layer hump
C_AX[1] = json.load(open("c9_dep_core3d_d1_L8/runs.json"))["d1"]["c_max"]
feas = {i: all(r[c] is not None and r[c] <= 0 for c in CN) for i, r in enumerate(A)}
C9 = json.load(open("out_c9/optimization_checkpoint.json"))       # Campaign 9 constraints for C9-0 to C9-59
feas.update({i: all(r[c] is not None and r[c] <= 0 for c in C9["constraint_names"])
             for i, r in enumerate(C9["all_raw"])})

ORIGIN = [("Campaign 9, C9-0 to C9-59", range(0, 60), "#9E9E9E"),
          ("Continuation, stage 1, C9-60 to C9-77", range(60, 78), "#56B4E9"),
          ("Continuation, stage 2, C9-78 to C9-95", range(78, 96), "#E69F00")]

# readings of tab:c9-readings: label, members, peaking map, c_max map, colour, marker
R2D = [("1. Campaign values", [35, 40, 34, 44, 47], F2, C, "#000000", "o"),
       ("3. Axially resolved depletion", [16, 1, 27], F2, C_AX, "#6A3D9A", "s")]
R3D = [("2. Three-dimensional peaking", [34, 40, 47], F3, C, "#D55E00", "^"),
       ("4. Both corrections", [16, 11, 27], F3, C_AX, "#009E73", "D"),
       ("5. Continuation, both corrections", [27, 69, 70], F3, C_AX, "#CC79A7", "v")]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), sharey=True)


def background(ax, fmap):
    for _, idx, col in ORIGIN:
        for i in idx:
            if i not in fmap or C[i] is None:
                continue
            ax.scatter(fmap[i], C[i], s=16, zorder=2, linewidths=0.8,
                       facecolors=col if feas[i] else "none", edgecolors=col, alpha=0.75)


def front(ax, name, members, fmap, cmap, col, mk, offsets):
    pts = sorted((fmap[i], cmap[i], i) for i in members)
    xs, ys = [], []
    for k, (x, y, _) in enumerate(pts):             # Pareto staircase, peaking then boron
        if k:
            xs.append(x); ys.append(ys[-1])
        xs.append(x); ys.append(y)
    ax.plot(xs, ys, color=col, lw=1.4, zorder=3)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=48, marker=mk, color=col,
               edgecolors="white", linewidths=0.6, zorder=4)
    for x, y, i in pts:
        if (round(x, 4), round(y, 1)) in LABELLED:
            continue
        LABELLED.add((round(x, 4), round(y, 1)))
        dx, dy = offsets.get((name[0], i), (5, 4))
        ax.annotate(f"C9-{i}", (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7, color=col, zorder=5)


LABELLED = set()
OFF = {("1", 35): (-30, 4), ("1", 40): (5, -12), ("1", 34): (-34, -2), ("3", 1): (5, 2), ("1", 47): (6, -4), ("2", 47): (5, -2),
       ("2", 34): (-12, 6), ("4", 11): (5, -9), ("5", 27): (5, 4), ("4", 27): (5, -10),
       ("4", 16): (6, -12), ("3", 16): (-8, -14), ("5", 69): (5, 4), ("5", 70): (5, -9), ("2", 40): (-30, -4)}

MTC_REF = 2763.0                                     # MTC boron limit of C8-47, 3D core, 12.8 MPa
for ax in axes:
    ax.axhline(MTC_REF, color="#CC3311", ls="--", lw=1.0, zorder=1)
    ax.text(1.715, MTC_REF + 30, "MTC boron limit of C8-47, 2763 ppm", color="#CC3311",
            fontsize=7.5, ha="right", va="bottom")

# MTC boron limit of the C9-27 lattice, 12.8 MPa (mtc_front_table.py, c9_post/d27_p128)
LIM27 = json.load(open("c9_post/d27_p128/mtc_ceiling_table.json"))[0]
for ax, fmap in ((axes[0], F2), (axes[1], F3)):
    x27, y27, s27 = fmap[27], LIM27["ceiling"], LIM27["sigma"]
    ax.fill_between([x27 - 0.025, x27 + 0.025], y27 - s27, y27 + s27, color="#882255", alpha=0.12, lw=0, zorder=1)
    ax.hlines(y27, x27 - 0.025, x27 + 0.025, colors="#882255", linestyles="-", lw=1.6, zorder=2)
    ax.annotate(f"MTC boron limit of C9-27\n{y27:.0f} $\\pm$ {s27:.0f} ppm", (x27, y27 + s27 + 25),
                fontsize=7.5, color="#882255", va="bottom", ha="center")

ax = axes[0]
background(ax, F2)
for r in R2D:
    front(ax, *r, OFF)
ax.set_title("(a) Two-dimensional peaking of the campaign", fontsize=9, loc="left")
ax.set_xlabel(r"Radial peaking factor $F_{\Delta H}$, 2D")
ax.set_ylabel(r"Critical boron concentration $c_\mathrm{max}$ [ppm]")

ax = axes[1]
background(ax, F3)
for i in (69, 70):                                    # the only continuation designs with a 3D solve
    ax.scatter(F3[i], C[i], s=150, facecolors="none", edgecolors=ORIGIN[1][2], linewidths=1.6, zorder=6)
ax.text(1.715, 1030, "Only C9-69 and C9-70 of the continuation\nhave a three-dimensional solve",
        fontsize=7.5, ha="right", va="bottom", color="0.3")
for r in R3D:
    front(ax, *r, OFF)
ax.set_title("(b) Three-dimensional peaking, 150 000 particles", fontsize=9, loc="left")
ax.set_xlabel(r"Radial peaking factor $F_{\Delta H}$, 3D")

for ax in axes:
    ax.set_xlim(1.42, 1.72)
    ax.set_ylim(1000, 3600)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

h = [Line2D([], [], ls="", marker="o", ms=5, mfc=c, mec=c, label=n) for n, _, c in ORIGIN]
h.append(Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="0.4", label="Open marker: infeasible in its own campaign or stage"))
h += [Line2D([], [], color=r[4], marker=r[5], ms=6, lw=1.4, label=r[0]) for r in (R2D[0], R3D[0], R2D[1], R3D[1], R3D[2])]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7.8, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1))
for out in sys.argv[1:]:
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
n_off = sum(1 for i in range(96) if C[i] is not None and not (1000 <= C[i] <= 3600))
print(n_off, "designs outside the c_max window; 3D peaking for", len(F3), "designs")
