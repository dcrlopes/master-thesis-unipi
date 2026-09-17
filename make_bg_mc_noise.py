#!/usr/bin/env python3
"""
make_bg_mc_noise.py -- Figure fig:bg-mc of the Theoretical Background chapter
(Monte Carlo statistics of the peaking estimator). Plotting only: no transport.

Panel (a), measured. Seed-to-seed standard deviation of the assembly F_dh
against the number of active histories, from the seed-replicated re-scores
  rank_front/ranking.csv   16000 and 256000 particles (Campaign 2 front)
  rank_c3/ranking.csv      64000 and 256000 particles (Campaign 3 finalists)
All at 120 batches with 30 inactive (results_timing/timing_rank_*_summary.json),
so N = 90 x particles. For each fidelity the s.d. is pooled over designs,
  s_pool^2 = sum_d sum_s (F_ds - mean_d)^2 / sum_d (n_d - 1),
with the approximate standard error s_pool / sqrt(2 dof). A 1/sqrt(N) law is
fitted to the measured points (slope fixed at -1/2, weights = dof), and the
free-slope fit is reported as a check. The Campaign 2 value 0.018 at
4000 x 60 (40 active batches), quoted in the results chapter, is drawn
separately: no seed-replicated runs at that fidelity are in the repository.

Panel (b), synthetic. A nearly flat field of 264 cells (the fuel pins of one
17x17 assembly, guide tubes removed) with true maximum 1.02 is sampled with
independent Gaussian per-cell noise, and F = max(q) / mean(q) is recorded for
each sample. Two noise levels: the mean relative error of the hottest pin at
256000 x 120 (measured) and the same at 16000 x 120 scaled by sqrt(9) to the
4000 x 60 history count.

Usage:  python make_bg_mc_noise.py [--out figs_bg] [--samples 20000]
Outputs: <out>/bg_mc_noise.pdf, <out>/bg_mc_noise.png, <out>/bg_mc_noise_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTIVE_BATCHES = 90                 # 120 batches, 30 inactive
C2_SD, C2_N = 0.018, 4000 * 40      # Campaign 2 table value, 4000 x 60 (20 inactive)
GUIDE_TUBES = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13), (5, 2), (5, 5), (5, 8),
               (5, 11), (5, 14), (8, 2), (8, 5), (8, 8), (8, 11), (8, 14), (11, 2),
               (11, 5), (11, 8), (11, 11), (11, 14), (13, 3), (13, 13), (14, 5),
               (14, 8), (14, 11)]


def read_runs(paths):
    """{particles: {(file, cand): [fdh, ...]}} and {particles: [rel_err, ...]}"""
    fdh, rel = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for p in paths:
        for r in csv.DictReader(open(p)):
            n = int(r["particles"])
            fdh[n][(p, r["cand"])].append(float(r["fdh"]))
            rel[n].append(float(r["rel_err"]))
    return fdh, rel


def pooled(groups):
    ss, dof, nd = 0.0, 0, 0
    for v in groups.values():
        if len(v) > 1:
            m = sum(v) / len(v)
            ss += sum((x - m) ** 2 for x in v)
            dof += len(v) - 1
            nd += 1
    s = math.sqrt(ss / dof)
    return s, dof, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs_bg")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260917)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fdh, rel = read_runs(["rank_front/ranking.csv", "rank_c3/ranking.csv"])
    pts = []
    for n in sorted(fdh):
        s, dof, nd = pooled(fdh[n])
        pts.append(dict(particles=n, N=n * ACTIVE_BATCHES, sd=s, dof=dof, designs=nd,
                        se=s / math.sqrt(2 * dof),
                        rel_err_hot=float(np.mean(rel[n]))))
    N = np.array([p["N"] for p in pts], float)
    sd = np.array([p["sd"] for p in pts], float)
    w = np.array([p["dof"] for p in pts], float)
    logc = float(np.sum(w * (np.log(sd) + 0.5 * np.log(N))) / np.sum(w))
    law = lambda x: math.exp(logc) / np.sqrt(x)
    slope_free = float(np.polyfit(np.log(N), np.log(sd), 1, w=np.sqrt(w))[0])
    c2_pred = float(law(C2_N))

    # ---- panel (b) synthetic field -------------------------------------------
    rng = np.random.default_rng(a.seed)
    ii, jj = np.meshgrid(np.arange(17), np.arange(17), indexing="ij")
    mask = np.ones((17, 17), bool)
    for r, c in GUIDE_TUBES:
        mask[r, c] = False
    shape = np.cos(np.pi * (ii - 8) / 34.0) * np.cos(np.pi * (jj - 8) / 34.0)
    q = shape[mask]
    q = q / q.mean()
    q = 1.0 + (q - 1.0) * (0.02 / (q.max() - 1.0))       # true max exactly 1.02
    true_max = float(q.max() / q.mean())
    rel16 = next(p["rel_err_hot"] for p in pts if p["particles"] == 16000)
    rel256 = next(p["rel_err_hot"] for p in pts if p["particles"] == 256000)
    noise = {"4000 x 60": rel16 * math.sqrt(16000 * 90 / C2_N), "256000 x 120": rel256}
    dist = {}
    for label, e in noise.items():
        x = q[None, :] * (1.0 + e * rng.standard_normal((a.samples, q.size)))
        dist[label] = x.max(axis=1) / x.mean(axis=1)

    # ---- figure --------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.1))

    xx = np.logspace(np.log10(1.0e5), np.log10(5.0e7), 50)
    ax.loglog(xx, law(xx), "-", color="#888888", lw=1.3,
              label=r"Fitted $\sigma \propto 1/\sqrt{N}$")
    ax.errorbar(N, sd, yerr=[p["se"] for p in pts], fmt="o", color="#173a5e", ms=6,
                capsize=3, label="Measured, pooled over designs")
    for p in pts:
        ax.annotate(f"{p['particles'] // 1000}k × 120\n{p['designs']} designs",
                    (p["N"], p["sd"]), textcoords="offset points", xytext=(8, 4), fontsize=7.5)
    ax.plot([C2_N], [C2_SD], "s", mfc="white", mec="#B23A48", mew=1.5, ms=6,
            label="Campaign 2 value, 4k × 60")
    ax.annotate("4k × 60", (C2_N, C2_SD), textcoords="offset points", xytext=(8, -12),
                fontsize=7.5, color="#B23A48")
    ax.set_xlabel("Active neutron histories $N$")
    ax.set_ylabel(r"Seed-to-seed s.d. of $F_{\Delta H}$")
    ax.set_title(r"(a) Statistical error against $1/\sqrt{N}$", fontsize=9)
    # explicit limits: some matplotlib versions autoscale a log axis down to
    # 10^1 when an errorbar collection is present
    ax.set_xlim(8.0e4, 6.0e7)
    ax.set_ylim(1.5e-3, 4.0e-2)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=7.8, loc="lower left")

    colors = {"4000 x 60": "#c0616b", "256000 x 120": "#5f9a7f"}
    edges = np.linspace(1.0, max(float(d.max()) for d in dist.values()) + 0.005, 160)
    bias = {}
    for label, d in dist.items():
        bias[label] = float(d.mean() - true_max)
        e = noise[label]
        bx.hist(d, bins=edges, density=True, alpha=0.65, color=colors[label],
                label=f"{label.replace('000 x', 'k ×')}: per-cell noise {100 * e:.1f}%, "
                      f"bias {bias[label]:+.3f}")
    bx.axvline(true_max, color="k", lw=1.5)
    bx.text(true_max - 0.002, bx.get_ylim()[1] * 0.55, "True maximum", rotation=90,
            fontsize=8, va="center", ha="right")
    bx.set_xlabel(r"Estimated $F_{\Delta H} = \max_i \hat{q}_i / \bar{q}$")
    bx.set_ylabel("Probability density")
    bx.set_title("(b) Maximum over 264 noisy cells, synthetic", fontsize=9)
    bx.legend(fontsize=7.6, loc="upper right")

    fig.tight_layout()
    fig.savefig(out / "bg_mc_noise.pdf", bbox_inches="tight")
    fig.savefig(out / "bg_mc_noise.png", dpi=170, bbox_inches="tight")

    summary = dict(points=pts, fit_logc=logc, slope_free=slope_free,
                   campaign2=dict(sd=C2_SD, N=C2_N, predicted_by_fit=c2_pred),
                   synthetic=dict(cells=int(q.size), true_max=true_max, samples=a.samples,
                                  seed=a.seed, noise=noise, bias=bias,
                                  sd_of_estimate={k: float(v.std(ddof=1)) for k, v in dist.items()}))
    (out / "bg_mc_noise_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
