#!/usr/bin/env python3
"""make_c8_front_split_figure.py -- Figure 5.18, the Campaign 8 front and the
two-bank controllable designs in their rod configurations. Read from the
checkpoint, no transport.

  (a) controllability margin: the rodded eigenvalue with the four regulating
      banks inserted (k_allre, the state of the controllability constraint
      g_ctrl = k_RE - 0.99 <= 0) and with RE1 and RE2 only (k_re12, computed and
      stored, not constrained), for the eight front designs and the four
      feasible designs that the first two banks hold below 0.99; a dotted
      segment joins the two states of each design;
  (b) the core peaking factor of the eight front designs with all rods out,
      RE1 and RE2 inserted, and RE1 to RE4 inserted, with the limit of 2.0.

usage:
    python make_c8_front_split_figure.py out_c8/optimization_checkpoint.json OUT.pdf [OUT.png]
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
feas = [i for i, r in enumerate(raw) if all(r[c] <= 0 for c in names)]
E = {i: raw[i]["cycle_length"] for i in feas}
F = {i: raw[i]["peaking"] for i in feas}
front = sorted((i for i in feas if not any(E[j] >= E[i] and F[j] <= F[i] and (E[j] > E[i] or F[j] < F[i])
                                              for j in feas)), key=lambda i: E[i])
two = sorted((i for i in feas if raw[i]["k_re12"] <= 0.99), key=lambda i: E[i])
assert front == [47, 42, 23, 29, 21, 44, 59, 1], front
assert two == [12, 53, 13, 31], two

C_FR, C_TW, C_LIM = "#0B2545", "#6A3D9A", "#B22222"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.6))

# ---- (a) controllability margin -------------------------------------------------
for grp, col, mk in ((front, C_FR, "o"), (two, C_TW, "s")):
    for i in grp:
        ax.plot([E[i], E[i]], [raw[i]["k_allre"], raw[i]["k_re12"]], ls=":", color=col, lw=0.9, zorder=2)
ax.scatter([E[i] for i in front], [raw[i]["k_allre"] for i in front], s=48, marker="o", color=C_FR,
           zorder=3, label="Front, four banks inserted")
ax.scatter([E[i] for i in front], [raw[i]["k_re12"] for i in front], s=48, marker="o", facecolors="white",
           edgecolors=C_FR, lw=1.3, zorder=3, label="Front, first two banks inserted")
ax.scatter([E[i] for i in two], [raw[i]["k_allre"] for i in two], s=52, marker="s", color=C_TW,
           zorder=3, label="Two-bank controllable, four banks inserted")
ax.scatter([E[i] for i in two], [raw[i]["k_re12"] for i in two], s=52, marker="s", facecolors="white",
           edgecolors=C_TW, lw=1.3, zorder=3, label="Two-bank controllable, first two banks inserted")
ax.axhline(0.99, color=C_LIM, lw=1.2, zorder=1, label="Controllability limit, 0.99")

# (dx, dy, ha, va): above the open marker, or beside it for the close pairs
top = {47: (-5, 0, "right", "center"), 42: (0, 6, "center", "bottom"),
       23: (0, 6, "center", "bottom"), 29: (0, 6, "center", "bottom"),
       21: (0, 6, "center", "bottom"), 44: (-5, 0, "right", "center"),
       59: (5, 0, "left", "center"), 1: (0, 6, "center", "bottom")}
for i in front:
    dx, dy, ha, va = top[i]
    ax.annotate(f"C8-{i}", (E[i], raw[i]["k_re12"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=8, ha=ha, va=va)
for i in two:
    ax.annotate(f"C8-{i}", (E[i], raw[i]["k_allre"]), xytext=(0, -7), textcoords="offset points",
                fontsize=8, ha="center", va="top")
ax.set_xlim(0, 6300)
ax.set_ylim(0.74, 1.12)
ax.set_xlabel("Cycle length [EFPD]")
ax.set_ylabel(r"Core $k_\mathrm{eff}$ with the banks inserted [-]")
ax.set_title("(a) Controllability margin", fontsize=10)
ax.grid(alpha=0.22, lw=0.6)
ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95)

# ---- (b) peaking in three rod configurations ---------------------------------------
x = np.array([E[i] for i in front])
for key, lab, col, mk, ls in (("peaking", "All rods out, the objective", C_FR, "o", "-"),
                              ("F_re12", "RE1 and RE2 inserted", C_TW, "s", "--"),
                              ("F_allre", "RE1 to RE4 inserted", "#C8102E", "^", ":")):
    bx.plot(x, [raw[i][key] for i in front], marker=mk, ls=ls, color=col, lw=1.5, ms=6.5,
            label=lab, zorder=3)
bx.axhline(2.0, color="#D55E00", lw=1.1, ls="--", zorder=1, label=r"$F_{\Delta H}$ limit, 2.0")
# below the curve, except the right member of each close pair (upper right)
off = {47: (0, -8, "center", "top"), 42: (4, 6, "left", "bottom"),
       23: (0, -8, "center", "top"), 29: (0, -8, "center", "top"),
       21: (0, -8, "center", "top"), 44: (-4, -6, "right", "top"),
       59: (2, 13, "left", "bottom"), 1: (0, -8, "center", "top")}
for i in front:
    dx, dy, ha, va = off[i]
    bx.annotate(f"C8-{i}", (E[i], F[i]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                ha=ha, va=va)
bx.set_xlim(1700, 6100)
bx.set_ylim(1.40, 2.66)
bx.set_xlabel("Cycle length [EFPD]")
bx.set_ylabel(r"Core $F_{\Delta H}$ [-]")
bx.set_title("(b) Peaking factor of the front", fontsize=10)
bx.grid(alpha=0.22, lw=0.6)
bx.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
print("front k_RE", [round(raw[i]["k_allre"], 3) for i in front])
print("two-bank k_RE12", [round(raw[i]["k_re12"], 3) for i in two])
