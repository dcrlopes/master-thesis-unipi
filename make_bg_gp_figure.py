#!/usr/bin/env python3
"""make_bg_gp_figure.py -- Figure 3.14 of the thesis: Gaussian-process
regression on noisy evaluations, a one-dimensional illustration.

Same synthetic data as the 8 Aug 2026 generator (theoretical_background_figures.py,
section 7, rng seed 3), with capitalised labels and "prediction interval" in
the legend. Plot only, no transport.

usage:
    python make_bg_gp_figure.py OUT.pdf
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444"})


def gp_fit(Xt, yt, Xs, ell=1.1, sf=1.0, sn=0.08):
    def K(A, Bm):
        d = A[:, None] - Bm[None, :]
        return sf ** 2 * np.exp(-0.5 * (d / ell) ** 2)
    Kt = K(Xt, Xt) + sn ** 2 * np.eye(len(Xt))
    Ks = K(Xs, Xt)
    L = np.linalg.cholesky(Kt)
    al = np.linalg.solve(L.T, np.linalg.solve(L, yt))
    mu = Ks @ al
    v = np.linalg.solve(L, Ks.T)
    var = sf ** 2 - np.sum(v ** 2, 0) + sn ** 2
    return mu, np.sqrt(np.maximum(var, 0))


rng = np.random.default_rng(3)
f = lambda x: 0.35 * np.sin(1.4 * x) + 0.08 * x
Xt = np.sort(rng.uniform(0, 9, 9))
yt = f(Xt) + rng.normal(0, 0.08, len(Xt))
Xs = np.linspace(-0.3, 9.6, 300)
mu, s = gp_fit(Xt, yt, Xs)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.fill_between(Xs, mu - 1.96 * s, mu + 1.96 * s, color="#5b8ac0", alpha=0.22,
                label="95% prediction interval, noise included")
ax.plot(Xs, mu, color="#173a5e", lw=2.0, label="GP posterior mean")
ax.plot(Xs, f(Xs), "--", color="#8a8a8a", lw=1.2, label="True function")
ax.errorbar(Xt, yt, yerr=0.08, fmt="o", ms=6, mfc="#B23A48", mec="white",
            ecolor="#B23A48", elinewidth=1.0, capsize=2.5,
            label="Noisy evaluations")
ax.annotate("Uncertainty grows away from the data:\nthe basis of active learning",
            xy=(9.25, float(mu[-4] + 1.96 * s[-4])), xytext=(5.4, 1.05),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.0))
ax.set_xlabel("Design variable (one-dimensional illustration)")
ax.set_ylabel("Objective")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Gaussian-process regression on noisy evaluations", fontsize=10)
fig.savefig(OUT, bbox_inches="tight")
print("written", OUT)
