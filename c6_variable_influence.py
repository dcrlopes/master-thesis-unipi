#!/usr/bin/env python3
"""c6_variable_influence.py -- how the six design variables drive the two
objectives in the Campaign 6 archive.

Produces two figures.

influence
    Six panels, one per design variable, each plotting the variable against
    the cycle length, coloured by the radial enthalpy-rise factor. Pareto
    front designs are ringed. The Spearman rank correlation of the variable
    with each objective is printed in the panel, computed over the feasible
    designs only.

parallel
    A parallel-coordinates plot of the Pareto front designs, every variable
    normalised to its own box, lines coloured by cycle length. This is the
    design-diversity figure: it shows at a glance which variables separate
    the front designs and which are shared by all of them.

Both are written as PNG at 300 dpi for slides and PDF for LaTeX.

Usage:
    python c6_variable_influence.py --checkpoint out_c6/optimization_checkpoint.json
    python c6_variable_influence.py --csv c6_full114.csv --out c6_vars
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
from matplotlib.lines import Line2D

# name in the record, axis label, box lower bound, box upper bound
VARS = [
    ("e_in",   "Inner enrichment (wt% $^{235}$U)",      2.0,  17.17),
    ("e_out",  "Outer enrichment (wt% $^{235}$U)",      2.0,  17.17),
    ("gd_wt",  "Gadolinia content (wt% Gd$_2$O$_3$)",   0.0,   8.0),
    ("pitch",  "Lattice pitch (cm)",                    1.15,  1.43),
    ("refl",   "Reflector thickness (cm)",              2.0,  19.49),
    ("pins",   "Gadolinium pin count",                 12.0,  40.0),
]

SHORT = {"e_in": "Inner\nenrichment", "e_out": "Outer\nenrichment",
         "gd_wt": "Gadolinia", "pitch": "Pitch",
         "refl": "Reflector", "pins": "Gd pins"}

KEYMAP = {"e_in": "enrich_inner", "e_out": "enrich_outer",
          "gd_wt": "gd_wt", "pitch": "pitch",
          "refl": "refl_thick", "pins": "gd_pins_used"}


def load_checkpoint(path):
    d = json.loads(Path(path).read_text())
    cn = d["constraint_names"]
    rows = []
    for i, x in enumerate(d["all_raw"]):
        r = {"pos": i,
             "efpd": float(x["cycle_length"]),
             "fdh": float(x["peaking"]),
             "feas": all(float(x[c]) <= 0.0 for c in cn)}
        for short, full in KEYMAP.items():
            r[short] = float(x[full])
        rows.append(r)
    return rows


def load_csv(path):
    rows = []
    with open(path) as fh:
        for x in csv.DictReader(fh):
            r = {"pos": int(float(x["pos"])),
                 "efpd": float(x["efpd"]),
                 "fdh": float(x["fdh"]),
                 "feas": float(x["feas"]) == 1.0}
            for short in KEYMAP:
                r[short] = float(x[short])
            rows.append(r)
    return rows


def pareto(feas):
    def dom(a, b):
        ge = a["efpd"] >= b["efpd"] and a["fdh"] <= b["fdh"]
        gt = a["efpd"] > b["efpd"] or a["fdh"] < b["fdh"]
        return ge and gt
    return sorted((a for a in feas
                   if not any(dom(b, a) for b in feas if b is not a)),
                  key=lambda r: -r["efpd"])


def spearman(x, y):
    """Spearman rank correlation without a SciPy dependency."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties so repeated values do not bias the coefficient
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def fig_influence(rows, out):
    feas = [r for r in rows if r["feas"]]
    inf = [r for r in rows if not r["feas"]]
    front = {r["pos"] for r in pareto(feas)}
    # noqa: front used for ring drawing and for the corner auto-selection

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.4), dpi=300)
    norm = plt.Normalize(min(r["fdh"] for r in feas),
                         max(r["fdh"] for r in feas))
    sc = None

    for ax, (key, label, lo, hi) in zip(axes.ravel(), VARS):
        ax.scatter([r[key] for r in inf], [r["efpd"] for r in inf],
                   marker="x", s=26, c="#BBBBBB", lw=1.0, zorder=1)
        sc = ax.scatter([r[key] for r in feas], [r["efpd"] for r in feas],
                        c=[r["fdh"] for r in feas], cmap="viridis_r",
                        norm=norm, s=44, alpha=0.85, zorder=2,
                        edgecolors="none")
        fr = [r for r in feas if r["pos"] in front]
        ax.scatter([r[key] for r in fr], [r["efpd"] for r in fr],
                   s=132, facecolors="none", edgecolors="#C1272D",
                   linewidths=1.8, zorder=3)

        rho_e = spearman([r[key] for r in feas], [r["efpd"] for r in feas])
        rho_f = spearman([r[key] for r in feas], [r["fdh"] for r in feas])

        pad = 0.03 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        # corner auto-selection: put the rho box where it hides the least,
        # never on top of a Pareto-front ring if any corner avoids them
        xl_, xu_ = ax.get_xlim()
        yl_, yu_ = ax.get_ylim()
        fx = np.array([(r[key] - xl_) / (xu_ - xl_) for r in feas])
        fy = np.array([(r["efpd"] - yl_) / (yu_ - yl_) for r in feas])
        onf = np.array([r["pos"] in front for r in feas])
        corners = {  # x0, x1, y0, y1 in axes fraction, then anchor
            "tl": (0.02, 0.40, 0.78, 0.98, 0.035, 0.965, "top", "left"),
            "tr": (0.60, 0.98, 0.78, 0.98, 0.965, 0.965, "top", "right"),
            "bl": (0.02, 0.40, 0.02, 0.22, 0.035, 0.035, "bottom", "left"),
            "br": (0.60, 0.98, 0.02, 0.22, 0.965, 0.035, "bottom", "right"),
        }
        best, best_cost = None, None
        for name, (x0, x1, y0, y1, tx, ty, va, ha) in corners.items():
            inside = (fx >= x0) & (fx <= x1) & (fy >= y0) & (fy <= y1)
            cost = int(inside.sum()) + 10 * int((inside & onf).sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = (tx, ty, va, ha), cost
        tx, ty, va, ha = best
        ax.text(tx, ty,
                f"$\\rho$ vs EFPD  {rho_e:+.2f}\n$\\rho$ vs $F_{{\\Delta H}}$   {rho_f:+.2f}",
                transform=ax.transAxes, va=va, ha=ha, fontsize=10.5,
                family="monospace",
                bbox=dict(fc="white", ec="#CCCCCC", alpha=0.92, pad=4))
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Cycle length (EFPD)", fontsize=11)
        ax.grid(alpha=0.22, lw=0.6)
        ax.tick_params(labelsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # choose for each panel the corner whose box covers the fewest points,
    # weighting Pareto-front points ten times an ordinary feasible design
    handles = [
        Line2D([], [], ls="none", marker="o", ms=7, mfc="#4C6B8A",
               mec="none", label="Feasible design"),
        Line2D([], [], ls="none", marker="x", ms=7, color="#BBBBBB",
               label="Infeasible design"),
        Line2D([], [], ls="none", marker="o", ms=11, mfc="none",
               mec="#C1272D", mew=1.8, label="On the Pareto front"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.47, -0.005))
    fig.suptitle("Campaign 6: influence of each design variable on the "
                 "cycle length", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.915, bottom=0.11, top=0.90,
                        hspace=0.32, wspace=0.26)
    cax = fig.add_axes([0.935, 0.13, 0.013, 0.74])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$F_{\Delta H}$ (zoned core)", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_influence.{ext}", bbox_inches="tight")
        print(f"written -> {out}_influence.{ext}")
    plt.close(fig)


def fig_parallel(rows, out):
    feas = [r for r in rows if r["feas"]]
    front = pareto(feas)
    keys = [v[0] for v in VARS]

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    xs = np.arange(len(keys))
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(min(r["efpd"] for r in front),
                         max(r["efpd"] for r in front))

    # faint background: every feasible design
    for r in feas:
        y = [(r[k] - lo) / (hi - lo) for k, _, lo, hi in VARS]
        ax.plot(xs, y, color="#CCCCCC", lw=0.6, alpha=0.45, zorder=1)

    for r in front:
        y = [(r[k] - lo) / (hi - lo) for k, _, lo, hi in VARS]
        ax.plot(xs, y, color=cmap(norm(r["efpd"])), lw=2.1, alpha=0.92,
                zorder=3, solid_capstyle="round")
        ax.plot(xs, y, marker="o", ms=5, ls="none",
                color=cmap(norm(r["efpd"])), zorder=4)

    for i, (_, _, lo, hi) in enumerate(VARS):
        ax.axvline(i, color="#666666", lw=0.9, zorder=2)
        ax.text(i - 0.05, 0.0, f"{lo:g}", ha="right", va="center",
                fontsize=9, color="#666666")
        ax.text(i - 0.05, 1.0, f"{hi:g}", ha="right", va="center",
                fontsize=9, color="#666666")

    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=11.5)
    ax.set_yticks([])
    ax.set_ylim(-0.10, 1.11)
    ax.set_xlim(-0.45, len(keys) - 0.55)
    ax.set_ylabel("Each variable normalised to its own search box",
                  fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Campaign 6: the {len(front)} Pareto-front designs across "
                 f"the six design variables", fontsize=14, pad=16)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Cycle length (EFPD)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    ax.plot([], [], color="#CCCCCC", lw=0.9,
            label=f"All feasible designs ({len(feas)})")
    ax.legend(loc="lower left", fontsize=10, frameon=False,
              bbox_to_anchor=(0.0, -0.16))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_parallel.{ext}", bbox_inches="tight")
        print(f"written -> {out}_parallel.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--csv")
    ap.add_argument("--out", default="c6_vars")
    args = ap.parse_args()

    rows = (load_checkpoint(args.checkpoint) if args.checkpoint
            else load_csv(args.csv))
    feas = [r for r in rows if r["feas"]]
    front = pareto(feas)
    print(f"{len(rows)} designs, {len(feas)} feasible, "
          f"{len(front)} on the front")

    fig_influence(rows, args.out)
    fig_parallel(rows, args.out)

    print("\nSpearman rank correlation over the feasible designs:")
    print(f"  {'variable':<10}{'vs EFPD':>10}{'vs F_dH':>10}")
    for key, _, _, _ in VARS:
        re_ = spearman([r[key] for r in feas], [r["efpd"] for r in feas])
        rf_ = spearman([r[key] for r in feas], [r["fdh"] for r in feas])
        print(f"  {key:<10}{re_:>+10.3f}{rf_:>+10.3f}")

    print("\nRange spanned by the front designs:")
    for key, label, lo, hi in VARS:
        v = [r[key] for r in front]
        frac = (max(v) - min(v)) / (hi - lo)
        print(f"  {key:<10}{min(v):>8.2f} .. {max(v):<8.2f}"
              f"  ({frac:.0%} of the search box)")


if __name__ == "__main__":
    main()
