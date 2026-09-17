#!/usr/bin/env python3
"""c9_ceiling_two_pressures.py -- MTC boron ceiling at both pressures.

No transport. Reads the two tables written by mtc_front_table.py
  c9_post/mtc_ceiling_table.json        12.8 MPa
  c9_post/p155/mtc_ceiling_table.json   15.5 MPa
Each is a list of rows with keys idx, gd_wt, pins, inventory, ceiling, sigma,
demand, margin (as written by mtc_front_table.py).

Three panels.
  (a) ceiling against inventory at both pressures, with the unweighted
      logarithmic fit that mtc_front_table.py prints, and the demand
  (b) ceiling against Gd2O3 weight fraction, same data, since the two
      loading measures cannot be separated by these lattices
  (c) pressure shift of every lattice scanned at both pressures, with the
      weighted mean and its one-sigma band

USAGE (repository root)
  python c9_ceiling_two_pressures.py
  python c9_ceiling_two_pressures.py --out figs_c9
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})
C128, C155, CDEM, CGREY = "#0072B2", "#D55E00", "#009E73", "#444444"


def load(p):
    rows = json.loads(Path(p).read_text())
    return {int(r["idx"]): r for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", default="c9_post/mtc_ceiling_table.json")
    ap.add_argument("--high", default="c9_post/p155/mtc_ceiling_table.json")
    ap.add_argument("--out", default="c9_post")
    a = ap.parse_args()
    for p in (a.low, a.high):
        if not Path(p).exists():
            print(f"ABORT: {p} not found, run mtc_front_table.py first"); return 1
    lo, hi = load(a.low), load(a.high)
    both = sorted(set(lo) & set(hi), key=lambda i: lo[i]["inventory"])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for key, ax, lab, logx in (("inventory", axes[0], r"gadolinia inventory, Gd$_2$O$_3$ wt% $\times$ pins", True),
                               ("gd_wt", axes[1], r"Gd$_2$O$_3$ weight fraction (wt%)", False)):
        for tab, col, pr in ((lo, C128, "12.8"), (hi, C155, "15.5")):
            ids = sorted(tab, key=lambda i: tab[i][key])
            x = np.array([tab[i][key] for i in ids]); y = np.array([tab[i]["ceiling"] for i in ids])
            s = np.array([tab[i]["sigma"] for i in ids])
            ax.errorbar(x, y, yerr=s, fmt="o", ms=4.5, color=col, capsize=2, lw=1.0,
                        label=f"ceiling, {pr} MPa")
            if logx:
                k, q = np.polyfit(np.log(x), y, 1)
                xs = np.linspace(x.min() * 0.85, x.max() * 1.12, 200)
                ax.plot(xs, k * np.log(xs) + q, color=col, ls="--", lw=0.9)
            else:
                k, q = np.polyfit(x, y, 1)
                xs = np.linspace(0, x.max() * 1.05, 50)
                ax.plot(xs, k * xs + q, color=col, ls="--", lw=0.9)
            for i, xi, yi in zip(ids, x, y):
                ax.annotate(str(i), (xi, yi), textcoords="offset points", xytext=(4, 3),
                            fontsize=6, color=col)
        dem_ids = sorted(lo, key=lambda i: lo[i][key])
        ax.plot([lo[i][key] for i in dem_ids], [lo[i]["demand"] for i in dem_ids], "s",
                ms=4.5, mfc="white", mec=CDEM, mew=1.2, label="demand $c_{max}$")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab); ax.set_ylabel("boron concentration (ppm)")
    axes[0].set_title("(a) against inventory, log fit", fontsize=8, loc="left")
    axes[1].set_title("(b) against weight fraction, linear fit", fontsize=8, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")

    ax = axes[2]
    sh = np.array([hi[i]["ceiling"] - lo[i]["ceiling"] for i in both])
    ss = np.array([math.hypot(hi[i]["sigma"], lo[i]["sigma"]) for i in both])
    w = 1 / ss ** 2; m = float(np.sum(w * sh) / w.sum()); sm = float(w.sum() ** -0.5)
    chi2 = float(np.sum(w * (sh - m) ** 2)); dof = len(both) - 1
    y = np.arange(len(both))
    ax.axvspan(m - sm, m + sm, color=CGREY, alpha=0.15, lw=0)
    ax.axvline(m, color=CGREY, lw=1.0, ls="--",
               label=f"weighted mean {m:.0f} $\\pm$ {sm:.0f} ppm")
    ax.errorbar(sh, y, xerr=ss, fmt="o", color=C155, capsize=2, lw=1.0)
    ax.set_yticks(y, [f"C9-{i} ({lo[i]['inventory']:.1f})" for i in both], fontsize=7)
    ax.set_xlabel("ceiling at 15.5 minus 12.8 MPa (ppm)")
    ax.set_title(f"(c) pressure shift, $\\chi^2$ = {chi2:.1f} on {dof} dof", fontsize=8, loc="left")
    ax.set_ylim(-0.6, len(both) - 0.4 + 0.7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(axis="y", visible=False)
    for axx in axes:
        axx.spines["top"].set_visible(False); axx.spines["right"].set_visible(False)
    fig.tight_layout()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "c9_ceiling_two_pressures.pdf")
    fig.savefig(out / "c9_ceiling_two_pressures.png", dpi=300)
    plt.close(fig)
    for i, s_, e_ in zip(both, sh, ss):
        print(f"  C9-{i:<3d} shift {s_:+5.0f} +/- {e_:3.0f} ppm")
    print(f"  weighted mean {m:.0f} +/- {sm:.0f} ppm, chi2 {chi2:.2f} on {dof} dof")
    print(f"wrote {out}/c9_ceiling_two_pressures.pdf and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
