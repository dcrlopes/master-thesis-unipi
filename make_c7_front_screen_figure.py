#!/usr/bin/env python3
"""make_c7_front_screen_figure.py -- Campaign 7, Section 5.4.2: the Pareto front
and the controllability screen, from the campaign checkpoint. No transport.

  (a) objective plane, cycle length against core F_dH, with the front as run
      (upper reactivity bound 1.126 and controllability constraint both active)
      and the front under the controllability constraint alone;
  (b) core k_eff at beginning of life against k_eff with the sixteen
      regulating-bank assemblies inserted, with the bound of 1.126 and the
      controllability limit of 0.99.

Categories: feasible as run; rejected only by the upper reactivity bound
(controllable, every other constraint met); rejected by any other constraint.

usage:
    python make_c7_front_screen_figure.py out_c7/optimization_checkpoint.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
ck = json.load(open(SRC))
raw, names = ck["all_raw"], ck["constraint_names"]
K_MAX, K_CTRL = ck["meta"]["limits"]["k_max"], 0.99

E = np.array([r["cycle_length"] for r in raw])
F = np.array([r["peaking"] for r in raw])
kc = np.array([r["keff_core_bol"] for r in raw])
kr = np.array([r["k_allre"] for r in raw])
feas = np.array([all(r[k] <= 0 for k in names) for r in raw])
only_kmax = np.array([r["g_kmax"] > 0 and all(r[k] <= 0 for k in names if k != "g_kmax") for r in raw])
other = ~feas & ~only_kmax


def front(mask):
    idx = np.where(mask)[0]
    nd = [i for i in idx if not any(E[j] >= E[i] and F[j] <= F[i] and (E[j] > E[i] or F[j] < F[i]) for j in idx)]
    return sorted(nd, key=lambda i: F[i])


fr_run, fr_ctrl = front(feas), front(feas | only_kmax)
assert len(fr_run) == 3 and len(fr_ctrl) == 6 and only_kmax.sum() == 14 and feas.sum() == 16

C_FE, C_KM, C_OT, C_LIM = "#0072B2", "#D55E00", "0.72", "#B22222"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 5.4))


def scatter(p, x, y):
    p.scatter(x[other], y[other], s=22, color=C_OT, edgecolors="white", lw=0.4,
              label="Rejected by another constraint")
    p.scatter(x[feas], y[feas], s=38, color=C_FE, edgecolors="white", lw=0.5, zorder=3,
              label="Feasible as run")
    p.scatter(x[only_kmax], y[only_kmax], s=40, marker="s", facecolors="white", edgecolors=C_KM,
              lw=1.3, zorder=3, label="Rejected only by the upper reactivity bound")


# ---- (a) objective plane, cycle length against F_dH as in the other front
# ---- figures of the thesis, with the front region enlarged in an inset ------
def fronts(p):
    p.plot(E[fr_run], F[fr_run], "-", color=C_FE, lw=1.6, zorder=2, label="Pareto front as run")
    p.plot(E[fr_ctrl], F[fr_ctrl], "--", color=C_KM, lw=1.4, zorder=2,
           label="Pareto front under the controllability constraint alone")


scatter(ax, E, F)
fronts(ax)
ax.set_xlabel("Cycle length [EFPD]")
ax.set_ylabel(r"Core $F_{\Delta H}$ [-]")
ax.set_xlim(0, 7000)
ax.set_ylim(1.48, 2.40)
ax.set_title("(a) Objective plane", fontsize=10)
ax.grid(alpha=0.22, lw=0.6)

ix = ax.inset_axes([0.42, 0.62, 0.56, 0.36])
scatter(ix, E, F)
fronts(ix)
ix.set_xlim(3650, 5300)
ix.set_ylim(1.492, 1.615)
ix.tick_params(labelsize=7.5)
ix.grid(alpha=0.22, lw=0.6)
ax.indicate_inset_zoom(ix, edgecolor="0.4", lw=0.8)
off = {59: (-4, 18), 57: (22, -14), 55: (24, -14), 48: (28, -10), 53: (26, -12), 49: (-30, 4)}
for i in fr_ctrl:
    ix.annotate(f"C7-{i}", (E[i], F[i]), xytext=off[i], textcoords="offset points", fontsize=7.5,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.6, shrinkA=0, shrinkB=3))

# ---- (b) controllability screen ------------------------------------------------
bx.axvspan(K_MAX, 1.26, ymax=(K_CTRL - 0.65) / (1.11 - 0.65), color=C_KM, alpha=0.07, lw=0,
           label="Controllable, above the upper reactivity bound")
scatter(bx, kc, kr)
bx.axvline(K_MAX, color="0.35", lw=1.0, ls="--", label=f"Upper reactivity bound, {K_MAX}")
bx.axhline(K_CTRL, color=C_LIM, lw=1.2, label=f"Controllability limit, {K_CTRL}")
offb = {59: (12, -92), 57: (34, -72), 55: (22, 62)}
for i in (59, 57, 55):
    bx.annotate(f"C7-{i}", (kc[i], kr[i]), xytext=offb[i], textcoords="offset points", fontsize=8.5,
                ha="center", arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7, shrinkB=4))
bx.set_xlabel(r"Core $k_\mathrm{eff}$ at beginning of life, all rods out [-]")
bx.set_ylabel(r"Core $k_\mathrm{eff}$, regulating banks inserted [-]")
bx.set_xlim(0.77, 1.26)
bx.set_ylim(0.65, 1.11)
bx.set_title("(b) Controllability screen", fontsize=10)
bx.grid(alpha=0.22, lw=0.6)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = bx.get_legend_handles_labels()
fig.legend(h1 + h2[:1] + h2[4:], l1 + l2[:1] + l2[4:], loc="lower center", ncol=2, fontsize=8.6, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
print("front as run:", [f"C7-{i}" for i in fr_run], " controllability front:", [f"C7-{i}" for i in fr_ctrl])
