#!/usr/bin/env python3
"""make_c3_convergence_figure.py -- Campaign 3 convergence figure for Section 5.3.

Two panels, both read from the Campaign 3 checkpoint, no transport:

  (a) hypervolume against the active-learning iteration, with the design of
      experiments and the two infill blocks separated, and the per-iteration
      gain annotated. The stopping rule of the loop asks for three consecutive
      gains below 1 per cent, and Campaign 3 ends with three exact zeros.

  (b) what each phase contributed: designs that are feasible, and designs that
      stop at the burnup cap, out of the evaluations of that phase. Block 1
      walks into the feasible region below the cap, Block 2 returns to the cap.

usage (plot only, runs anywhere):
    python make_c3_convergence_figure.py CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(SRC))
hv = np.asarray(d["hv_history"], dtype=float)
recs, names = d["all_raw"], d["constraint_names"]
PHASES = [("Design of experiments", 0, 36), ("Block 1", 36, 54), ("Block 2", 54, 72)]

feas, capped, total = [], [], []
for _, lo, hi in PHASES:
    sub = recs[lo:hi]
    feas.append(sum(all(r[n] <= 0 for n in names) for r in sub))
    capped.append(sum(1 for r in sub if r.get("censored")
                      or abs(r.get("bu_eoc_mwd_kg", 0.0) - 75.0) < 0.1))
    total.append(len(sub))

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.5))

n_doe = PHASES[0][2]
batch = 6
it = n_doe + batch * np.arange(hv.size)       # evaluations completed at each record
lo, hi = hv.min(), hv.max()
span = hi - lo
ax1.set_ylim(lo - 0.12 * span, hi + 0.23 * span)       # headroom for the labels
ax1.plot(it, hv, "o-", color="#3f6d8c", lw=1.4, ms=5, zorder=3)
edges = [PHASES[0][2], PHASES[1][2]]                 # 36 and 54 evaluations
for xc in edges:
    ax1.axvline(xc, color="#9aa7b1", ls=":", lw=0.9, zorder=1)
ax1.set_xlim(n_doe - 0.55 * n_doe, it[-1] + 3)
for k, (lab, lo_e, hi_e) in enumerate(PHASES):
    # the design of experiments holds no recorded point, so its label is set
    # against the right edge of its own range instead of at its centre
    xpos, align = (hi_e - 1.0, "right") if k == 0 else (0.5 * (lo_e + hi_e), "center")
    ax1.text(xpos, 0.98, lab.replace(" of ", " of\n"),
             transform=ax1.get_xaxis_transform(),
             ha=align, va="top", fontsize=8, color="#404b54")
for i in range(1, hv.size):
    gain = 100.0 * (hv[i] - hv[i - 1]) / hv[i - 1]
    ax1.annotate(f"{gain:+.1f}%" if gain else "0%", (it[i], hv[i]),
                 textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
ax1.set_xlabel("Evaluations completed")
ax1.set_ylabel("Hypervolume")
ax1.set_title("(a) Convergence of the hypervolume", fontsize=9)
ax1.set_xticks(it)

x = np.arange(len(PHASES))
w = 0.38
ax2.bar(x - w / 2, feas, w, color="#3f6d8c", label="Feasible designs")
ax2.bar(x + w / 2, capped, w, color="#B23A48", label="Designs at the burnup cap")
for xi, (f, c, t) in enumerate(zip(feas, capped, total)):
    ax2.text(xi - w / 2, f + 0.4, f"{f}/{t}", ha="center", fontsize=8)
    ax2.text(xi + w / 2, c + 0.4, f"{c}/{t}", ha="center", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels([p[0].replace(" of ", " of\n") for p in PHASES])
ax2.set_ylabel("Designs")
ax2.set_title("(b) Feasibility and the burnup cap per phase", fontsize=9)
ax2.legend(fontsize=7.5, loc="upper center")
ax2.margins(y=0.2)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], dpi=170, bbox_inches="tight")
print("hypervolume", [round(v, 1) for v in hv])
print("feasible", feas, "capped", capped, "of", total)
