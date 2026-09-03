#!/usr/bin/env python
"""
fig_cost_vs_design.py -- figure for the computational-cost section.

Shows that the wall time of one Monte Carlo transport solve is not a constant
of the method but a function of the design, and that the dominant variable is
the lattice pitch.

Panel (a)  seconds per transport solve against pitch, Campaign 3 and
           Campaign 4 overlaid, with a least squares line per campaign and the
           Spearman rank correlation quoted.
Panel (b)  the same against reflector thickness, the second significant
           variable.
Panel (c)  standardised least squares coefficients for every design variable,
           both campaigns side by side, which separates a real partial effect
           from a correlation induced by the sampling.

input      cost_c3_joined.csv and cost_c4_joined.csv written by
           cost_vs_design.py
output     th_cost_vs_design.pdf and .png

usage
    python fig_cost_vs_design.py --c3 cost_c3_joined.csv --c4 cost_c4_joined.csv \\
        --out th_cost_vs_design

The PDF goes into the Overleaf images/ directory and is included with
\\includegraphics{images/th_cost_vs_design}.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARS = ["enrich_inner", "enrich_outer", "gd_wt", "pitch", "refl_thick"]
LABELS = {
    "enrich_inner": r"$e_\mathrm{in}$",
    "enrich_outer": r"$e_\mathrm{out}$",
    "gd_wt": r"$w_\mathrm{Gd}$",
    "pitch": r"$p$",
    "refl_thick": r"$t_\mathrm{refl}$",
}


def load(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty")
    out = {k: np.array([float(r[k]) for r in rows]) for k in VARS}
    out["t_solve"] = np.array([float(r["t_solve"]) for r in rows])
    return out


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for k in np.flatnonzero(counts > 1):
        r[inv == k] = r[inv == k].mean()
    return r


def spearman(a, b):
    ra, rb = rankdata(a) - rankdata(a).mean(), rankdata(b) - rankdata(b).mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def std_coeffs(d: dict):
    X = np.column_stack([d[v] for v in VARS])
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(A, d["t_solve"], rcond=None)
    pred = A @ beta
    ss_res = float(((d["t_solve"] - pred) ** 2).sum())
    ss_tot = float(((d["t_solve"] - d["t_solve"].mean()) ** 2).sum())
    return beta[1:], 1.0 - ss_res / ss_tot


def scatter_panel(ax, sets, var, xlabel):
    for (name, d, colour, marker) in sets:
        x, y = d[var], d["t_solve"]
        rho = spearman(x, y)
        ax.plot(x, y, marker, color=colour, ms=4.5, alpha=0.75,
                label=rf"{name}, $\rho={rho:+.3f}$")
        if len(x) > 2:
            b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(b, xs), "-", color=colour, lw=1.2, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("seconds per transport solve")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.25, lw=0.5)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c3", required=True, help="cost_c3_joined.csv")
    ap.add_argument("--c4", required=True, help="cost_c4_joined.csv")
    ap.add_argument("--out", default="th_cost_vs_design", help="output prefix")
    args = ap.parse_args()

    d3, d4 = load(args.c3), load(args.c4)
    sets = [("Campaign 3", d3, "#1f77b4", "o"),
            ("Campaign 4", d4, "#d62728", "s")]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    scatter_panel(axes[0], sets, "pitch", r"lattice pitch $p$ [cm]")
    axes[0].set_title("(a) pitch", fontsize=10, loc="left")

    scatter_panel(axes[1], sets, "refl_thick",
                  r"reflector thickness $t_\mathrm{refl}$ [cm]")
    axes[1].set_title("(b) reflector thickness", fontsize=10, loc="left")

    b3, r3 = std_coeffs(d3)
    b4, r4 = std_coeffs(d4)
    idx = np.arange(len(VARS))
    w = 0.38
    ax = axes[2]
    ax.barh(idx - w / 2, b3, height=w, color="#1f77b4",
            label=rf"Campaign 3, $R^2={r3:.3f}$")
    ax.barh(idx + w / 2, b4, height=w, color="#d62728",
            label=rf"Campaign 4, $R^2={r4:.3f}$")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_yticks(idx)
    ax.set_yticklabels([LABELS[v] for v in VARS])
    ax.invert_yaxis()
    ax.set_xlabel("seconds per solve, per standard deviation")
    ax.set_title("(c) standardised partial effects", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")
    print(f"C3: R^2 = {r3:.3f}   C4: R^2 = {r4:.3f}")
    for v, a, b in zip(VARS, b3, b4):
        print(f"   {v:<14s} C3 {a:+7.2f}   C4 {b:+7.2f}  s per sd")


if __name__ == "__main__":
    main()
