#!/usr/bin/env python3
"""
make_summary_final_front_figure.py -- Figure 7 of the thesis summary. Drawing only.

Panel (a): the objective plane with the three-dimensional peaking factor, as
           panel (b) of make_c9_readings_figure.py, reduced to two fronts: the
           Campaign 9 front (assembly cycle length) and the final front
           (eight-layer cycle length).
Panel (b): the axial peaking factor F_z of the final front against the
           core-average burnup, eight-layer depletion, with the one-layer
           depletion of C9-27 as the uniform-burnup reference.

Reads  out_c9a/optimization_checkpoint.json, out_c9/optimization_checkpoint.json
       confirm3d_c9_all, confirm3d_c9_front, confirm3d_c9a /summary.json
       c9_post/d27_p128/mtc_ceiling_table.json
       c9_dep_core3d_d27_L8, c9_dep_core3d_d27_seed2..5, c9_dep_core3d_d27_L1,
       c9f_dep_core3d, c9a_dep_core3d_seed2..5 /runs.json
Usage (repository root)
       python make_summary_final_front_figure.py ../thesis_vFinal/summary/sum_final_front.pdf
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- data, panel (a)
B = json.load(open("out_c9a/optimization_checkpoint.json"))
A = B["all_raw"]
F3 = {int(k): v["ARO_3Dhw"]["F"] for k, v in json.load(open("confirm3d_c9_all/summary.json")).items()}
for f in ("confirm3d_c9_front/summary.json", "confirm3d_c9a/summary.json", "confirm3d_c9a_rest/summary.json"):
    F3.update({int(k): v["ARO_3Dhw"]["F"] for k, v in json.load(open(f)).items()})
C = {i: r["c_max"] for i, r in enumerate(A)}
C9 = json.load(open("out_c9/optimization_checkpoint.json"))       # Campaign 9 constraints, C9-0 to C9-59
feas = {i: all(r[c] is not None and r[c] <= 0 for c in C9["constraint_names"])
        for i, r in enumerate(C9["all_raw"])}
feas.update({i: all(A[i][c] is not None and A[i][c] <= 0 for c in B["constraint_names"]) for i in range(60, 96)})
LIM27 = json.load(open("c9_post/d27_p128/mtc_ceiling_table.json"))[0]

# label, members, colour, marker. The five members of the Campaign 9 front show
# where the front was before the correction. On this axis two of them are
# dominated, so the Pareto staircase joins only the non-dominated members.
FRONTS = [("Campaign 9 front, assembly cycle length", [35, 40, 34, 44, 47], "#D55E00", "^"),
          ("Final front, eight-layer cycle length", [27, 69, 70], "#009E73", "D")]
OFF = {34: (-50, -26), 40: (-32, -4), 35: (-12, 7), 44: (7, -1), 47: (3, -11),
       27: (6, 5), 69: (4, 6), 70: (6, 4)}
LEADER = {34}                                         # labels placed away from the marker, with a line

# ---------------------------------------------------------------- data, panel (b)


def fz_runs(key, dirs):
    """Burnup grid and F_z of each eight-layer run of one design."""
    runs = [json.load(open(f"{d}/runs.json"))[key] for d in dirs]
    bu = np.asarray(runs[0]["bu_hist"], dtype=float)
    assert all(np.allclose(r["bu_hist"], bu) for r in runs)
    return bu, np.array([[s["fz"] for s in r["states"]] for r in runs]), runs[0]["spec_power"]


SEEDS = [f"_seed{k}" for k in (2, 3, 4, 5)]
L8 = [("C9-27, 8 layers, 5 seeds", "d27", ["c9_dep_core3d_d27_L8"] + ["c9_dep_core3d_d27" + s for s in SEEDS], "#0072B2", "D"),
      ("C9-70, 8 layers, 5 seeds", "d70", ["c9f_dep_core3d"] + ["c9a_dep_core3d" + s for s in SEEDS], "#000000", "s"),
      ("C9-69, 8 layers, 5 seeds", "d69", ["c9f_dep_core3d"] + ["c9a_dep_core3d_d69" + s for s in SEEDS], "#CC79A7", "o")]
bu1, fz1, spec_power = fz_runs("d27", ["c9_dep_core3d_d27_L1"])
BU_MIN = 1826.0 * spec_power / 1000.0                 # burnup of the minimum cycle length, MWd/kgHM

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "pdf.fonttype": 42})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.9))    # printed at 0.86 of the text width

# (a) objective plane, three-dimensional peaking
for i in range(96):
    if i in F3 and C[i] is not None:
        ax.scatter(F3[i], C[i], s=16, zorder=2, linewidths=0.8, alpha=0.75,
                   facecolors="#9E9E9E" if feas[i] else "none", edgecolors="#9E9E9E")
y27 = LIM27["ceiling"]                                # its uncertainty is in Table 3 of the summary
ax.axhline(y27, color="#882255", ls="--", lw=1.1, zorder=1)
ax.text(1.715, y27 + 30, f"MTC boron limit of C9-27, {y27:.0f} ppm", fontsize=7.5, color="#882255",
        va="bottom", ha="right")
shown = [i for i in range(96) if i in F3 and C[i] is not None]
inside = [i for i in shown if 1.42 <= F3[i] <= 1.72 and 1000 <= C[i] <= 4200]
print(f"Evaluated designs with a 3D peaking factor and a c_max: {len(shown)} of 96, "
      f"{len(inside)} inside the axes; continuation designs with a 3D solve: "
      f"{sorted(i for i in F3 if i >= 60)}")
for name, members, col, mk in FRONTS:
    pts = sorted((F3[i], C[i], i) for i in members)
    nd = [p for p in pts if not any(q is not p and q[0] <= p[0] and q[1] <= p[1] for q in pts)]
    print(f"{name}: non-dominated on this axis {[i for _, _, i in nd]} of {members}")
    xs, ys = [], []
    for k, (x, y, _) in enumerate(nd):                # Pareto staircase, peaking then boron
        if k:
            xs.append(x); ys.append(ys[-1])
        xs.append(x); ys.append(y)
    ax.plot(xs, ys, color=col, lw=1.4, zorder=3)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=48, marker=mk, color=col,
               edgecolors="white", linewidths=0.6, zorder=4)
    for x, y, i in pts:
        ax.annotate(f"C9-{i}", (x, y), xytext=OFF[i], textcoords="offset points",
                    fontsize=7, color=col, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6) if i in LEADER else None,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.4))
ax.set_xlim(1.42, 1.72)
ax.set_ylim(1000, 4200)
ax.set_xlabel(r"Radial peaking factor $F_{\Delta H}$, three-dimensional core")
ax.set_ylabel(r"Critical boron concentration $c_\mathrm{max}$ [ppm]")
ax.set_title("(a) Objective plane", fontsize=9, loc="left")
h = [Line2D([], [], ls="", marker="o", ms=5, mfc="#9E9E9E", mec="#9E9E9E", label="Evaluated design, feasible"),
     Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="#9E9E9E", label="Evaluated design, infeasible")]
h += [Line2D([], [], color=c, marker=m, ms=6, lw=1.4, label=n) for n, _, c, m in FRONTS]
ax.legend(handles=h, fontsize=7.5, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

# (b) axial peaking factor against burnup
for name, key, dirs, col, mk in L8:
    bu, fz, _ = fz_runs(key, dirs)
    m = fz.mean(axis=0)
    if len(fz) > 1:
        bx.errorbar(bu, m, yerr=fz.std(axis=0, ddof=1), color=col, marker=mk, ms=4.5, lw=1.4,
                    capsize=2.5, label=name)
    else:
        bx.plot(bu, m, color=col, marker=mk, ms=4.5, lw=1.4, label=name)
    print(f"{name}: F_z {m[0]:.3f} at BOL, {m[-1]:.3f} at {bu[-1]:.1f} MWd/kgHM")
# the one-layer reference has the colour and the marker of C9-27, dashed and open;
# panel (b) uses no colour of panel (a)
bx.plot(bu1, fz1[0], color="#0072B2", ls="--", lw=1.3, marker="D", ms=4.5, mfc="white",
        label="C9-27, 1 layer, 1 seed")
bx.axvline(BU_MIN, color="0.45", ls=":", lw=1.2)
bx.text(BU_MIN - 0.3, 1.497, "Minimum cycle length,\n1826 d", fontsize=7.5, color="0.3",
        ha="right", va="top")
bx.set_xlim(-0.5, 22.5)
bx.set_ylim(1.10, 1.50)
bx.set_xlabel("Core-average burnup [MWd/kgHM]")
bx.set_ylabel(r"Axial peaking factor $F_z$")
bx.set_title("(b) Axial peaking factor during the depletion", fontsize=9, loc="left")
hb, lb = bx.get_legend_handles_labels()
order = sorted(range(len(lb)), key=lambda k: (lb[k][:5], "1 layer" in lb[k]))   # each design together, 8 layers first
bx.legend([hb[k] for k in order], [lb[k] for k in order], fontsize=7.5, loc="lower left",
          frameon=True, framealpha=0.9, edgecolor="none")

for a in (ax, bx):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
fig.tight_layout()
for out in sys.argv[1:]:
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
print(f"1 layer: F_z {fz1[0][0]:.3f} at BOL, {fz1[0][bu1 <= 21.5][-1]:.3f} at 21.5; "
      f"minimum cycle length at {BU_MIN:.2f} MWd/kgHM")
for _, members, _, _ in FRONTS:
    print([(i, round(F3[i], 3), round(C[i])) for i in members])
