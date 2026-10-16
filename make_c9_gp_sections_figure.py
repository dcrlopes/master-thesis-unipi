#!/usr/bin/env python3
"""
make_c9_gp_sections_figure.py -- the Gaussian-process surrogate of Campaign 9
seen along each design variable, for the presentation. The counterpart, on
the real problem, of the one-dimensional illustration bg_gp.

The two objective surrogates (radial peaking factor, critical boron) are
refitted on the 60 evaluations of Campaign 9 as the loop fits them
(reactor_optimization.GPSurrogate, Matern 5/2 with ARD and a white-noise
term). Each column varies one design variable over its search range and
keeps the other three at the values of C9-27. The band is the 95 %
prediction interval, noise included. The points are the evaluated designs
whose other three variables are within 15 % of the range of those of C9-27,
so they are close to the section without being on it. There is no "true
function" here: OpenMC is the noisy evaluation and the truth is unknown.

Reads out_c9/optimization_checkpoint.json. No transport, seconds.
Usage: python make_c9_gp_sections_figure.py <prefix>    # <prefix>.pdf and .png
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from c9_acquisition_replay import _NoEvaluator, build
from reactor_optimization import ActiveLearningMOO

REF = 27                                        # the section passes through C9-27
NEAR = 0.15                                     # closeness in the other variables, fraction of the range
ck = json.load(open("out_c9/optimization_checkpoint.json"))
raw, names = ck["all_raw"], ck["design_variables"]
spec, cfg = build(ck, "asrun")
opt = ActiveLearningMOO(spec, _NoEvaluator(), cfg)
opt._seed_from_raw(raw)
sur = opt._new_surrogate().fit(opt.X, opt.F)
xl, xu = np.asarray(spec.design_space.xl, float), np.asarray(spec.design_space.xu, float)
span = xu - xl
X, F = np.asarray(opt.X, float), np.asarray(opt.F, float)
x0 = X[REF]
LABEL = {"enrich": "Enrichment [wt%]", "gd_wt": r"Gd$_2$O$_3$ in the poisoned rods [wt%]",
         "refl_thick": "Reflector thickness [cm]", "gd_pins": "Gadolinia rods per assembly (continuous)"}
OBJ = [(r"Radial peaking factor $F_{\Delta H}$ [-]", (1.40, 1.90)), (r"Critical boron $c_\mathrm{max}$ [ppm]", (0, 6500))]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True})
fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0), sharey="row")
for j, name in enumerate(names):
    g = np.linspace(xl[j], xu[j], 200)
    Xg = np.tile(x0, (len(g), 1)); Xg[:, j] = g
    mu, sd = sur.predict(Xg)
    others = [k for k in range(len(names)) if k != j]
    near = np.all(np.abs(X[:, others] - x0[others]) / span[others] <= NEAR, axis=1)
    near[REF] = False
    for r, (ylab, ylim) in enumerate(OBJ):
        ax = axes[r, j]
        ax.fill_between(g, mu[:, r] - 1.96 * sd[:, r], mu[:, r] + 1.96 * sd[:, r], color="#5b8ac0", alpha=0.22)
        ax.plot(g, mu[:, r], color="#173a5e", lw=2.0)
        ax.scatter(X[near, j], F[near, r], s=22, color="#B23A48", edgecolors="white", linewidths=0.6, zorder=4)
        ax.scatter([x0[j]], [F[REF, r]], s=70, marker="D", color="#009E73", edgecolors="k", linewidths=0.6, zorder=5)
        ax.axvline(x0[j], color="#009E73", lw=0.8, ls=":")
        ax.set_ylim(*ylim)
        ax.set_xlim(xl[j], xu[j])
        if j == 0:
            ax.set_ylabel(ylab)
        if r == 1:
            ax.set_xlabel(LABEL[name])
        else:
            ax.set_title(f"{int(near.sum())} evaluated designs near this section", fontsize=8, color="0.35")
    print(f"{name}: C9-{REF} at {x0[j]:.3f}, range {xl[j]:.2f} to {xu[j]:.2f}, {int(near.sum())} designs near the section")
h = [Line2D([], [], color="#173a5e", lw=2, label="GP posterior mean"),
     plt.Rectangle((0, 0), 1, 1, color="#5b8ac0", alpha=0.22, label="95 % prediction interval, noise included"),
     Line2D([], [], ls="", marker="o", color="#B23A48", mec="white", ms=6, label=f"Evaluated design within {NEAR:.0%} of the range of C9-{REF} in the other variables"),
     Line2D([], [], ls="", marker="D", color="#009E73", mec="k", ms=7, label=f"C9-{REF}, the design the sections pass through")]
fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"Campaign 9, surrogates fitted on the 60 evaluations: each variable varied through C9-{REF}, the other three fixed", fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(sys.argv[1] + ".pdf", bbox_inches="tight")
fig.savefig(sys.argv[1] + ".png", dpi=200, bbox_inches="tight")
print("written", sys.argv[1] + ".pdf")
