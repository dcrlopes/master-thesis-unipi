#!/usr/bin/env python3
"""make_c3_space_figure.py -- Campaign 3 in two panels, for Section 5.3.

  (a) the archive in the plane that the campaign optimises: discharge burnup against
      radial peaking factor. Burnup and cycle length are proportional at constant
      specific power, so this is the objective plane read in the unit the discharge
      screen acts on. Every evaluation is shown, feasible and infeasible designs are
      separated, the feasible non-dominated set is marked and the in-loop burnup
      screen is drawn.

  (b) the design vectors in parallel coordinates, each variable normalised to its own
      box bounds, coloured by phase: the design of experiments and the two infill
      blocks. Built like panel (b) of the Campaign 2 figure and sharing its colours.

usage (plot only, no transport):
    python make_c3_space_figure.py CHECKPOINT.json OUT.pdf [OUT.png] [--screen 75]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
PNG = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
SCREEN = float(sys.argv[sys.argv.index("--screen") + 1]) if "--screen" in sys.argv else 75.0
N_DOE, BATCH = 36, 18

VARS = [("enrich_inner", "$e_\\mathrm{in}$", 2.0, 19.75),
        ("enrich_outer", "$e_\\mathrm{out}$", 2.0, 19.75),
        ("gd_wt", "$w_\\mathrm{Gd}$", 0.0, 8.0),
        ("pitch", "$p$", 1.15, 1.43),
        ("refl_thick", "$t_\\mathrm{refl}$", 2.0, 19.5)]

d = json.load(open(SRC))
recs, names = d["all_raw"], d["constraint_names"]
bu = np.array([r["bu_eoc_mwd_kg"] for r in recs], float)
f = np.array([r["peaking"] for r in recs], float)
ok = np.array([all(r[n] <= 0 for n in names) for r in recs])

# feasible non-dominated set: maximum burnup, minimum peaking
idx = [i for i in np.where(ok)[0]
       if not any(bu[j] >= bu[i] and f[j] <= f[i] and (bu[j] > bu[i] or f[j] < f[i])
                  for j in np.where(ok)[0])]
idx.sort(key=lambda i: bu[i])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

# ---------------- (a) objective plane
ax.scatter(bu[~ok], f[~ok], marker="x", s=32, color="#9aa7b1", label="Infeasible")
ax.scatter(bu[ok], f[ok], marker="o", s=34, facecolor="#3f6d8c", edgecolor="white",
           linewidth=0.6, label="Feasible", zorder=3)
ax.plot(bu[idx], f[idx], "s--", ms=8, mfc="none", mec="#B23A48", color="#B23A48",
        lw=1.1, label="Pareto front", zorder=4)
ax.axvline(SCREEN, color="black", ls=":", lw=1.2)
ymid = 0.5 * (f.min() + f.max())
ax.annotate(f"Burnup screen, {SCREEN:.0f} MWd/kgHM", (SCREEN, ymid),
            textcoords="offset points", xytext=(-8, 0), ha="center", va="center",
            rotation=90, fontsize=8)
ax.set_xlabel("Discharge burnup [MWd/kgHM]")
ax.set_ylabel("Radial peaking factor $F_{\\Delta H}$")
ax.legend(fontsize=8, loc="upper left")
ax.margins(x=0.06, y=0.10)
ax.set_title("(a) Objective plane", fontsize=9)

# ---------------- (b) parallel coordinates by phase, coloured as in the Campaign 2 figure
phase = np.array([0 if i < N_DOE else 1 + (i - N_DOE) // BATCH for i in range(len(recs))])
n_ph = phase.max() + 1
cmap = plt.get_cmap("viridis")
colors = ["#9aa7b1"] + [cmap(0.12 + 0.76 * k / max(1, n_ph - 2)) for k in range(n_ph - 1)]
labels = ["Design of experiments"] + [f"Block {k}" for k in range(1, n_ph)]

x = np.arange(len(VARS))
norm = np.empty((len(recs), len(VARS)))
for j, (key, _, lo, hi) in enumerate(VARS):
    v = np.array([r[key] for r in recs], float)
    norm[:, j] = (v - lo) / (hi - lo)
for p in range(n_ph):
    sel = phase == p
    ax2.plot(x, norm[sel].T, color=colors[p], lw=1.0, alpha=0.75 if p else 0.45)
    ax2.plot([], [], color=colors[p], lw=1.6, label=labels[p])
for xi in x:
    ax2.axvline(xi, color="#c9d2d9", lw=0.8, zorder=0)
ax2.set_xticks(x)
ax2.set_xticklabels([lab for _, lab, _, _ in VARS])
ax2.set_ylim(-0.05, 1.05)
ax2.set_yticks([0.0, 0.5, 1.0])
ax2.set_yticklabels(["lower\nbound", "mid", "upper\nbound"])
ax2.set_title("(b) Design vectors by phase", fontsize=9)
ax2.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12),
           frameon=False, columnspacing=1.2, handlelength=1.6)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if PNG:
    fig.savefig(PNG, dpi=170, bbox_inches="tight")
print(f"designs {len(recs)}, feasible {ok.sum()}, at the screen {(bu >= SCREEN - 0.05).sum()}, "
      f"front {[int(i) for i in idx]}")
print("front burnup", [round(bu[i], 1) for i in idx], "peaking", [round(f[i], 4) for i in idx])
print("phases", n_ph, [int((phase == p).sum()) for p in range(n_ph)])
