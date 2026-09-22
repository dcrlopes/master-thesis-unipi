#!/usr/bin/env python3
"""make_c2_figure.py -- Campaign 2 in two panels, for Section 5.1.2.

  (a) the objective plane of the campaign: radial peaking factor against discharge
      burnup, feasible and infeasible designs separated, the feasible non-dominated
      set marked and the depletion horizon of the campaign drawn.

  (b) the design vectors in parallel coordinates, each variable normalised to its own
      box bounds, coloured by phase: the design of experiments and each infill
      iteration.

usage (plot only, no transport):
    python make_c2_figure.py CHECKPOINT.json OUT.pdf [OUT.png] [--horizon 100]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
PNG = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
HOR = float(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 100.0
N_DOE, BATCH = 54, 6

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

idx = [i for i in np.where(ok)[0]
       if not any(bu[j] >= bu[i] and f[j] <= f[i] and (bu[j] > bu[i] or f[j] < f[i])
                  for j in np.where(ok)[0])]
idx.sort(key=lambda i: bu[i])

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

# ---------------- (a) objective plane
ax1.scatter(bu[~ok], f[~ok], marker="x", s=30, color="#9aa7b1", label="Infeasible")
ax1.scatter(bu[ok], f[ok], marker="o", s=32, facecolor="#3f6d8c", edgecolor="white",
            linewidth=0.6, label="Feasible", zorder=3)
ax1.plot(bu[idx], f[idx], "s--", ms=8, mfc="none", mec="#B23A48", color="#B23A48",
         lw=1.1, label="Pareto front", zorder=4)
ax1.axvline(HOR, color="black", ls=":", lw=1.2)
ax1.annotate(f"Depletion horizon, {HOR:.0f} MWd/kgHM", (HOR, f.max()),
             textcoords="offset points", xytext=(-6, 6), ha="right", va="bottom",
             fontsize=8)
ax1.set_xlabel("Discharge burnup [MWd/kgHM]")
ax1.set_ylabel("Radial peaking factor $F_{\\Delta H}$")
ax1.set_title("(a) Objective plane", fontsize=9)
ax1.legend(fontsize=8, loc="upper left", framealpha=0.95)
ax1.margins(x=0.07, y=0.22)

# ---------------- (b) parallel coordinates by phase
phase = np.array([0 if i < N_DOE else 1 + (i - N_DOE) // BATCH for i in range(len(recs))])
n_ph = phase.max() + 1
cmap = plt.get_cmap("viridis")
colors = ["#9aa7b1"] + [cmap(0.12 + 0.76 * k / max(1, n_ph - 2)) for k in range(n_ph - 1)]
labels = ["Design of experiments"] + [f"Infill {k}" for k in range(1, n_ph)]

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
print(f"designs {len(recs)}, feasible {int(ok.sum())}, at the horizon {int((bu >= HOR - 0.05).sum())}, "
      f"front {[int(i) for i in idx]}, phases {n_ph}")
