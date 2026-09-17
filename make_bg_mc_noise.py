#!/usr/bin/env python3
"""
make_bg_mc_noise.py -- Figure fig:bg-mc of the Theoretical Background chapter
(Monte Carlo statistics of the peaking estimator). Plotting only: no transport.

Panel (a), measured. Seed-to-seed standard deviation of the assembly F_dh
against the number of active histories N, from the seed-replicated studies
  noise_test/noise_summary.csv  4000 x 60 (20 inactive): 3 designs x 5 seeds
                                (peaking_noise_test.py, Campaign 2 front,
                                per-design s.d. only)
  rank_front/ranking.csv        16000 and 256000 x 120 (30 inactive)
  rank_c3/ranking.csv           64000 and 256000 x 120 (30 inactive)
For each fidelity the s.d. is pooled over designs,
  s_pool^2 = sum_d (n_d - 1) s_d^2 / sum_d (n_d - 1),
with the approximate standard error s_pool / sqrt(2 dof). A 1/sqrt(N) law
(slope fixed at -1/2, weights = dof) is fitted to the three points at
120 batches; the free-slope fit is reported as a check. The 4000 x 60 point is
not used in the fit.

Paired check. The three noise_test designs are also in rank_front (matched on
the design vector through results_campaign2/figs/corrected_candidates.csv), so
their pooled 16k s.d. scaled by sqrt(N_16k / N_4k) gives a 1/sqrt(N)
prediction at 4000 x 60 for the same designs.

Panel (b), synthetic. A nearly flat field of 264 cells (the fuel pins of one
17x17 assembly, guide tubes removed) with true maximum 1.02 is sampled with
independent Gaussian per-cell noise, and F = max(q) / mean(q) is recorded for
each sample. The two noise levels are the measured mean relative errors of the
hottest pin at 4000 x 60 (noise_test) and at 256000 x 120.

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

N_4K = 4000 * 40                    # 60 batches, 20 inactive
N_SEEDS_4K = 5                      # peaking_noise_test.py --seeds 5
ACTIVE_120 = 90                     # 120 batches, 30 inactive
GUIDE_TUBES = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13), (5, 2), (5, 5), (5, 8),
               (5, 11), (5, 14), (8, 2), (8, 5), (8, 8), (8, 11), (8, 14), (11, 2),
               (11, 5), (11, 8), (11, 11), (11, 14), (13, 3), (13, 13), (14, 5),
               (14, 8), (14, 11)]
KEYS = ("enrich_inner", "enrich_outer", "gd_wt", "pitch", "refl_thick")


def pool(sds_and_n):
    dof = sum(n - 1 for _, n in sds_and_n)
    s = math.sqrt(sum((n - 1) * sd ** 2 for sd, n in sds_and_n) / dof)
    return s, dof, s / math.sqrt(2 * dof)


def per_design(path):
    """{particles: {cand: [fdh, ...]}} and {particles: [rel_err, ...]}"""
    fdh, rel = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for r in csv.DictReader(open(path)):
        n = int(r["particles"])
        fdh[n][r["cand"]].append(float(r["fdh"]))
        rel[n].append(float(r["rel_err"]))
    return fdh, rel


def sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs_bg")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260917)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- measured points ----------------------------------------------------
    groups, rels = defaultdict(list), defaultdict(list)
    front, front_rel = per_design("rank_front/ranking.csv")
    c3, c3_rel = per_design("rank_c3/ranking.csv")
    for src, rel in ((front, front_rel), (c3, c3_rel)):
        for p, cands in src.items():
            for v in cands.values():
                if len(v) > 1:
                    groups[p].append((sd(v), len(v)))
            rels[p] += rel[p]
    pts = []
    for p in sorted(groups):
        s, dof, se = pool(groups[p])
        pts.append(dict(label=f"{p // 1000}k × 120", particles=p, N=p * ACTIVE_120, sd=s,
                        dof=dof, designs=len(groups[p]), se=se,
                        rel_err_hot=float(np.mean(rels[p]))))

    nt = list(csv.DictReader(open("noise_test/noise_summary.csv")))
    s4, dof4, se4 = pool([(float(r["sd"]), N_SEEDS_4K) for r in nt])
    p4 = dict(label="4k × 60", particles=4000, N=N_4K, sd=s4, dof=dof4, designs=len(nt),
              se=se4, rel_err_hot=float(np.mean([float(r["rel_err"]) for r in nt])),
              per_design_sd=[float(r["sd"]) for r in nt])

    N = np.array([p["N"] for p in pts], float)
    S = np.array([p["sd"] for p in pts], float)
    W = np.array([p["dof"] for p in pts], float)
    logc = float(np.sum(W * (np.log(S) + 0.5 * np.log(N))) / np.sum(W))
    law = lambda x: math.exp(logc) / np.sqrt(x)
    slope_free = float(np.polyfit(np.log(N), np.log(S), 1, w=np.sqrt(W))[0])

    # ---- paired check on the same three designs -----------------------------
    cc = list(csv.DictReader(open("results_campaign2/figs/corrected_candidates.csv")))
    paired = []
    for r in nt:
        match = [c["cand"] for c in cc
                 if all(abs(float(c[k]) - float(r[k])) < 1e-9 for k in KEYS)]
        cand = match[0] if match else None
        v16 = front.get(16000, {}).get(cand, [])
        paired.append(dict(design=r["design"], cand=cand, sd_4k=float(r["sd"]),
                           sd_16k=sd(v16) if len(v16) > 1 else None, n_16k=len(v16)))
    ok = [q for q in paired if q["sd_16k"] is not None]
    s16p, dof16p, _ = pool([(q["sd_16k"], q["n_16k"]) for q in ok])
    pred4_paired = s16p * math.sqrt(16000 * ACTIVE_120 / N_4K)

    # ---- panel (b) synthetic field --------------------------------------------
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
    rel256 = next(p["rel_err_hot"] for p in pts if p["particles"] == 256000)
    noise = {"4k × 60": p4["rel_err_hot"], "256k × 120": rel256}
    dist = {}
    for label, e in noise.items():
        x = q[None, :] * (1.0 + e * rng.standard_normal((a.samples, q.size)))
        dist[label] = x.max(axis=1) / x.mean(axis=1)
    synth_sd = {k: float(v.std(ddof=1)) for k, v in dist.items()}
    synth_pred4 = synth_sd["256k × 120"] * math.sqrt(256000 * ACTIVE_120 / N_4K)

    # ---- figure --------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.1))

    xx = np.logspace(np.log10(1.0e5), np.log10(5.0e7), 50)
    ax.loglog(xx, law(xx), "-", color="#888888", lw=1.3,
              label=r"Fitted $\sigma \propto 1/\sqrt{N}$")
    ax.errorbar(N, S, yerr=[p["se"] for p in pts], fmt="o", color="#173a5e", ms=6,
                capsize=3, label="Measured, 120 batches")
    for p in pts:
        ax.annotate(f"{p['label']}\n{p['designs']} designs", (p["N"], p["sd"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7.5)
    ax.errorbar([p4["N"]], [p4["sd"]], yerr=[p4["se"]], fmt="s", mfc="white", mec="#B23A48",
                ecolor="#B23A48", mew=1.5, ms=6, capsize=3,
                label="Measured, 60 batches (not fitted)")
    ax.annotate(f"{p4['label']}\n{p4['designs']} designs", (p4["N"], p4["sd"]),
                textcoords="offset points", xytext=(8, -18), fontsize=7.5, color="#B23A48")
    ax.set_xlim(8.0e4, 6.0e7)
    ax.set_ylim(1.5e-3, 4.0e-2)
    ax.set_xlabel("Active neutron histories $N$")
    ax.set_ylabel(r"Seed-to-seed s.d. of $F_{\Delta H}$")
    ax.set_title(r"(a) Statistical error against $1/\sqrt{N}$", fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=7.8, loc="lower left")

    colors = {"4k × 60": "#c0616b", "256k × 120": "#5f9a7f"}
    edges = np.linspace(1.0, max(float(d.max()) for d in dist.values()) + 0.005, 160)
    bias = {}
    for label, d in dist.items():
        bias[label] = float(d.mean() - true_max)
        bx.hist(d, bins=edges, density=True, alpha=0.65, color=colors[label],
                label=f"{label}: per-cell noise {100 * noise[label]:.1f}%, "
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

    summary = dict(
        points=pts, point_4k=p4, fit_logc=logc, slope_free=slope_free,
        fit_prediction_4k=float(law(N_4K)),
        shortfall_4k_vs_fit=1.0 - p4["sd"] / float(law(N_4K)),
        paired=dict(designs=paired, pooled_sd_16k=s16p, dof_16k=dof16p,
                    prediction_4k=pred4_paired,
                    shortfall_4k=1.0 - p4["sd"] / pred4_paired),
        synthetic=dict(cells=int(q.size), true_max=true_max, samples=a.samples, seed=a.seed,
                       noise=noise, bias=bias, sd_of_estimate=synth_sd,
                       sqrtN_prediction_4k_from_256k=synth_pred4,
                       shortfall_4k=1.0 - synth_sd["4k × 60"] / synth_pred4))
    (out / "bg_mc_noise_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("point_4k", "slope_free", "fit_prediction_4k",
                                              "shortfall_4k_vs_fit", "paired")},
                     indent=1, ensure_ascii=False))
    print(json.dumps(summary["synthetic"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
