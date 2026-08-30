#!/usr/bin/env python3
"""c6_extra_figures.py -- two insight figures for the Campaign 6 archive.

convergence
    Cumulative hypervolume after every real evaluation, under the frozen
    in-campaign reference and under the full-range reference, with the five
    blocks shaded and the acquisition change of each block annotated. This
    is the figure that carries the campaign story: two metrics, one of
    which is blind below its reference point, and a search that only moves
    once the acquisition is fixed.

constraints
    Core k_eff at beginning of life against the cycle length, with the two
    reactivity limits drawn and every design classified by which constraint
    rejects it. Shows directly that the upper reactivity limit is the
    binding constraint at the long-cycle end of the front.

Usage:
    python c6_extra_figures.py --checkpoint out_c6/optimization_checkpoint.json
    python c6_extra_figures.py --csv c6_full114.csv --out c6_extra
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX, K_MIN, F_MAX = 1.35, 1.02, 2.0
REF_FROZEN = (6776.0, 2.0692)

BLOCKS = [(0, 42, "DOE\n(LHS design of experiments)"),
          (42, 60, "Block 2\nbatch collapse"),
          (60, 66, "Block 3\nconstraint-blind"),
          (66, 84, "Block 4\n+ feasibility margin"),
          (84, 114, "Block 5\nconvergence")]
BLOCK_FILL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]


def load_checkpoint(path):
    d = json.loads(Path(path).read_text())
    cn = d["constraint_names"]
    rows = []
    for i, x in enumerate(d["all_raw"]):
        rows.append(dict(pos=i,
                         efpd=float(x["cycle_length"]),
                         fdh=float(x["peaking"]),
                         k=float(x["keff_core_bol"]),
                         feas=all(float(x[c]) <= 0.0 for c in cn)))
    return rows


def load_csv(path):
    rows = []
    with open(path) as fh:
        for x in csv.DictReader(fh):
            rows.append(dict(pos=int(float(x["pos"])),
                             efpd=float(x["efpd"]),
                             fdh=float(x["fdh"]),
                             k=float(x["k_core"]),
                             feas=float(x["feas"]) == 1.0))
    return rows


def pareto(feas):
    def dom(a, b):
        ge = a["efpd"] >= b["efpd"] and a["fdh"] <= b["fdh"]
        gt = a["efpd"] > b["efpd"] or a["fdh"] < b["fdh"]
        return ge and gt
    return [a for a in feas if not any(dom(b, a) for b in feas if b is not a)]


def hypervolume(feas, ref_e, ref_f):
    pts = sorted((r for r in feas if r["efpd"] > ref_e and r["fdh"] < ref_f),
                 key=lambda r: -r["efpd"])
    area, prev = 0.0, ref_f
    for p in pts:
        if p["fdh"] < prev:
            area += (p["efpd"] - ref_e) * (prev - p["fdh"])
            prev = p["fdh"]
    return area


def fig_convergence(rows, out):
    n = len(rows)
    xs = np.arange(1, n + 1)
    hv_frozen = np.empty(n)
    hv_full = np.empty(n)
    for i in range(n):
        feas = [r for r in rows[:i + 1] if r["feas"]]
        hv_frozen[i] = hypervolume(feas, *REF_FROZEN)
        hv_full[i] = hypervolume(feas, 0.0, REF_FROZEN[1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.6), dpi=300,
                                   sharex=True)
    for ax, hv, lab in ((ax1, hv_frozen,
                         "Frozen in-campaign reference (6776 EFPD, 2.069)"),
                        (ax2, hv_full,
                         "Full-range reference (0 EFPD, 2.069)")):
        for (a, b, _), c in zip(BLOCKS, BLOCK_FILL):
            if a < n:
                ax.axvspan(a + 0.5, min(b, n) + 0.5, color=c, alpha=0.10,
                           zorder=0)
        ax.step(xs, hv, where="post", color="#26215C", lw=2.0, zorder=3)
        ax.set_ylabel("Hypervolume", fontsize=11.5)
        ax.grid(alpha=0.22, lw=0.6)
        ax.tick_params(labelsize=10.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(0.012, 0.06, lab, transform=ax.transAxes, va="bottom",
                fontsize=10.5, style="italic", color="#333333")

    # block labels in a strip above the top panel, outside the data area
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax1.transData, ax1.transAxes)
    for (a, b, name) in BLOCKS:
        if a < n:
            ax1.text((a + min(b, n)) / 2 + 0.5, 1.03, name, transform=tr,
                     ha="center", va="bottom", fontsize=8.6, color="#444444")

    ax1.annotate("first front improvement\nof the whole campaign",
                 xy=(67, hv_frozen[66]), xytext=(45, hv_frozen[66] * 1.12),
                 fontsize=9.5, ha="right", va="center",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="#444444"))
    ax1.annotate("stopping rule met:\nfive gains < 1%",
                 xy=(n, hv_frozen[-1]), xytext=(n - 2, hv_frozen[-1] * 0.80),
                 fontsize=9.5, ha="right", va="top",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="#444444"))
    ax2.annotate("blocks 2 and 3 improved the front here,\ninvisible to "
                 "the frozen reference above",
                 xy=(55, hv_full[54]), xytext=(60, hv_full[10] * 0.72),
                 fontsize=9.5, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="#444444"))

    ax2.set_xlabel("Real evaluations (OpenMC)", fontsize=12)
    ax2.set_xlim(0, n + 1)
    fig.suptitle("Campaign 6 convergence: archive hypervolume after every "
                 "evaluation", fontsize=14, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_convergence.{ext}", bbox_inches="tight")
        print(f"written -> {out}_convergence.{ext}")
    plt.close(fig)
    return hv_frozen, hv_full


def fig_constraints(rows, out):
    feas = [r for r in rows if r["feas"]]
    front = {r["pos"] for r in pareto(feas)}

    hi_k = [r for r in rows if r["k"] > K_MAX]
    lo_k = [r for r in rows if r["k"] < K_MIN]
    f_only = [r for r in rows
              if not r["feas"] and K_MIN <= r["k"] <= K_MAX]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    ax.axhspan(K_MAX, 1.47, color="#D55E00", alpha=0.08, zorder=0)
    ax.axhspan(0.78, K_MIN, color="#8B6CB5", alpha=0.08, zorder=0)
    ax.axhline(K_MAX, color="#D55E00", ls="--", lw=1.4, zorder=1)
    ax.axhline(K_MIN, color="#8B6CB5", ls="--", lw=1.4, zorder=1)
    ax.text(150, K_MAX + 0.004, r"$k_\mathrm{eff}^\mathrm{core}$ limit 1.35",
            color="#D55E00", fontsize=10, va="bottom")
    ax.text(150, K_MIN - 0.004, "floor 1.02", color="#8B6CB5", fontsize=10,
            va="top")

    ax.scatter([r["efpd"] for r in feas], [r["k"] for r in feas],
               s=46, c="#4292C6", alpha=0.85, edgecolors="none",
               label=f"Feasible ({len(feas)})", zorder=2)
    ax.scatter([r["efpd"] for r in hi_k], [r["k"] for r in hi_k],
               marker="x", s=52, c="#D55E00", lw=1.6,
               label=f"Rejected, k too high ({len(hi_k)})", zorder=3)
    ax.scatter([r["efpd"] for r in lo_k], [r["k"] for r in lo_k],
               marker="x", s=52, c="#8B6CB5", lw=1.6,
               label=f"Rejected, k too low ({len(lo_k)})", zorder=3)
    ax.scatter([r["efpd"] for r in f_only], [r["k"] for r in f_only],
               marker="x", s=46, c="#999999", lw=1.3,
               label=f"Rejected on peaking only ({len(f_only)})", zorder=2)

    fr = [r for r in feas if r["pos"] in front]
    ax.scatter([r["efpd"] for r in fr], [r["k"] for r in fr],
               s=150, facecolors="none", edgecolors="#C1272D",
               linewidths=1.8, label="On the Pareto front", zorder=4)

    long_front = [r for r in fr if r["efpd"] > 7000]
    kmin_lf = min(r["k"] for r in long_front)
    ax.annotate("long-cycle front designs ride the upper limit\n"
                f"(k between {kmin_lf:.3f} and "
                f"{max(r['k'] for r in long_front):.3f})",
                xy=(9100, 1.341), xytext=(4600, 1.415), fontsize=10,
                ha="left",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#444444"))

    ax.set_xlim(-150, 10600)
    ax.set_ylim(0.78, 1.47)
    ax.set_xlabel("Cycle length (EFPD)", fontsize=12.5)
    ax.set_ylabel(r"Core $k_\mathrm{eff}$ at beginning of life",
                  fontsize=12.5)
    ax.set_title("Campaign 6: reactivity constraint activity across the "
                 "archive", fontsize=14, pad=10)
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.93)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_constraints.{ext}", bbox_inches="tight")
        print(f"written -> {out}_constraints.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--csv")
    ap.add_argument("--out", default="c6_extra")
    args = ap.parse_args()

    rows = (load_checkpoint(args.checkpoint) if args.checkpoint
            else load_csv(args.csv))
    hv_frozen, hv_full = fig_convergence(rows, args.out)
    fig_constraints(rows, args.out)

    print("\ncheck against the campaign hv_history (iteration ends):")
    for i in (42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114):
        if i <= len(rows):
            print(f"  after {i:3d} evaluations: frozen {hv_frozen[i-1]:8.2f}"
                  f"   full-range {hv_full[i-1]:8.2f}")


if __name__ == "__main__":
    main()
