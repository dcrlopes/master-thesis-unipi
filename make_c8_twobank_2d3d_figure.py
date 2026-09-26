#!/usr/bin/env python3
"""
make_c8_twobank_2d3d_figure.py -- the two-bank reading of Campaign 8 in two
and in three dimensions, for the merged section on the designs
controllable with two regulating banks.

(a) Objective plane of the archive with the two-bank front in 2D (archive)
    and in 3D at the operating maximum (confirmed designs), five years at
    full power as a vertical line.
(b) Two-bank margin M_8 in 3D against M_8 in 2D for the eleven confirmed
    designs, with the 1010 pcm constraint on both axes, the 1:1 line and
    the linear fit of the 3D-2D offset.

Reads out_c8/optimization_checkpoint.json, confirm3d_c8/summary.json and
swing_c8/swing.json. Drawing only, no transport.
Usage: python make_c8_twobank_2d3d_figure.py <out.pdf> [out.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open("out_c8/optimization_checkpoint.json"))["all_raw"]
S = json.load(open("confirm3d_c8/summary.json"))
W = json.load(open("swing_c8/swing.json"))
CN = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_ctrl", "g_geom"]
MARGIN = -1e5 * (0.99 - 1.0) / 0.99          # 1010 pcm
FIVE = 5 * 365.25


def rho(k):
    return 1e5 * (k - 1.0) / k


def pareto(ids):
    out = []
    for i in ids:
        L, P = A[i]["cycle_length"], A[i]["peaking"]
        if not any(A[j]["cycle_length"] >= L and A[j]["peaking"] <= P
                   and (A[j]["cycle_length"] > L or A[j]["peaking"] < P) for j in ids if j != i):
            out.append(i)
    return sorted(out, key=lambda i: A[i]["cycle_length"])


feas = [i for i, r in enumerate(A) if all(r[g] <= 0 for g in CN)]
infeas = [i for i in range(len(A)) if i not in feas]
two2d = [i for i in feas if -rho(A[i]["k_re12"]) >= MARGIN]
conf = [int(d) for d in S]
two3d_bol = [d for d in conf if S[str(d)]["RE12_margin3D_pcm"] >= MARGIN]
two3d_pk = [d for d in conf if W[str(d)]["RE12_margin_peak_3d_pcm"] >= MARGIN]
front4 = pareto(feas)
front2_2d = pareto(two2d)
front2_3d = pareto(two3d_pk)
print("two-bank 2D", two2d, "3D BOL", two3d_bol, "3D operating maximum", two3d_pk)
print("two-bank front 2D", front2_2d, "3D at the operating maximum", front2_3d)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))

X = lambda ids: [A[i]["cycle_length"] for i in ids]
Y = lambda ids: [A[i]["peaking"] for i in ids]
ax.scatter(X(infeas), Y(infeas), marker="x", s=20, color="0.6", linewidths=0.8, label="Infeasible")
four = [i for i in feas if i not in two2d]
ax.scatter(X(four), Y(four), marker="o", s=28, color="tab:blue", label="Feasible, four regulating banks needed")
ax.scatter(X(two2d), Y(two2d), marker="s", s=34, color="tab:green", label="Two-bank controllable in 2D")
gain = [d for d in two3d_pk if d not in two2d]
ax.scatter(X(gain), Y(gain), marker="s", s=110, facecolors="none", edgecolors="tab:orange", linewidths=1.6,
           label="Two-bank controllable in 3D at the operating maximum")
lose = [d for d in two2d if d in conf and d not in two3d_pk]
ax.scatter(X(lose), Y(lose), marker="x", s=60, color="tab:red", linewidths=1.4,
           label="Loses two-bank control at the operating maximum")
unconf = [d for d in two2d if d not in conf]
ax.scatter(X(unconf), Y(unconf), marker="s", s=110, facecolors="none", edgecolors="0.4", linewidths=1.2,
           linestyle="--", label="Not confirmed in 3D")
ax.plot(X(front4), Y(front4), "-", color="tab:blue", lw=1.1, label="Four-bank front, 2D")
ax.plot(X(front2_2d), Y(front2_2d), "-", color="tab:green", lw=1.3, label="Two-bank front, 2D")
ax.plot(X(front2_3d), Y(front2_3d), "--", color="tab:orange", lw=1.5, label="Two-bank front, 3D at the operating maximum")
ax.axvline(FIVE, color="tab:red", ls=":", lw=0.9)
ax.text(FIVE - 60, 1.795, "Five years at full power", rotation=90, ha="right", va="top", fontsize=7, color="tab:red")
LAB = {53: (-7, 0, "right", "center"), 13: (7, -2, "left", "center"), 31: (7, 0, "left", "center"),
       47: (9, -3, "left", "center"), 42: (-9, 3, "right", "center"), 23: (7, -8, "left", "top"),
       12: (7, 0, "left", "center")}
for d, (dx, dy, ha, va) in LAB.items():
    ax.annotate(f"C8-{d}", (A[d]["cycle_length"], A[d]["peaking"]), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va=va, fontsize=7.5)
ax.set_xlim(-150, 6500)
ax.set_ylim(1.49, 1.80)
ax.set_xlabel("Cycle length [EFPD]")
ax.set_ylabel(r"$F_{\Delta H}$ [-]")
ax.set_title("(a) Objective plane", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7, frameon=False)

m2 = np.array([S[str(d)]["RE12_margin2D_pcm"] for d in conf])
m3 = np.array([S[str(d)]["RE12_margin3D_pcm"] for d in conf])
m3p = np.array([W[str(d)]["RE12_margin_peak_3d_pcm"] for d in conf])
p = np.polyfit(m2, m3 - m2, 1)
xx = np.linspace(-7500, 7000, 2)
bx.plot(xx, xx, "-", color="0.5", lw=0.8, label="No change from 2D to 3D")
bx.plot(xx, xx + np.polyval(p, xx), "--", color="tab:orange", lw=1.2,
        label=f"Linear fit, offset {np.polyval(p, 0):.0f} pcm at $M_8^\\mathrm{{2D}} = 0$")
bx.axhline(MARGIN, color="tab:red", ls=":", lw=0.9)
bx.axvline(MARGIN, color="tab:red", ls=":", lw=0.9)
bx.scatter(m2, m3, marker="o", s=34, color="tab:blue", label="Beginning of life", zorder=4)
bx.scatter(m2, m3p, marker="v", s=34, color="tab:purple", label="Operating maximum", zorder=4)
BL = {53: (-7, -4, "right", "top"), 13: (7, -4, "left", "top"), 31: (-7, 4, "right", "bottom"),
      47: (7, 2, "left", "bottom"), 42: (-7, 2, "right", "bottom"), 23: (7, 0, "left", "center"),
      29: (7, -2, "left", "top"), 21: (7, 4, "left", "bottom"), 44: (7, -4, "left", "top"),
      59: (-7, -3, "right", "top"), 1: (-7, -2, "right", "top")}
for d, x, y in zip(conf, m2, m3):
    dx, dy, ha, va = BL[d]
    bx.annotate(f"C8-{d}", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7)
bx.text(MARGIN + 150, -7200, "Constraint, 1010 pcm", rotation=90, fontsize=7, color="tab:red", va="bottom")
bx.set_xlim(-7500, 7000)
bx.set_ylim(-8000, 9500)
bx.set_xlabel(r"$M_8$ in 2D [pcm]")
bx.set_ylabel(r"$M_8$ in 3D [pcm]")
bx.set_title("(b) Two-bank margin, 2D against 3D", fontsize=10)
bx.grid(alpha=0.3)
bx.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig(sys.argv[1], bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=170, bbox_inches="tight")
print("written", sys.argv[1], "fit dk = %.4f M8_2D + %.0f" % (p[0], p[1]))
