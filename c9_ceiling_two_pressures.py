#!/usr/bin/env python3
"""c9_ceiling_two_pressures.py -- MTC boron limit at both pressures.

No transport. Reads the two tables written by mtc_front_table.py
  c9_post/mtc_ceiling_table.json        12.8 MPa
  c9_post/p155/mtc_ceiling_table.json   15.5 MPa
Each is a list of rows with keys idx, gd_wt, pins, inventory, ceiling, sigma,
demand, margin (as written by mtc_front_table.py).

Three panels.
  (a) limit against inventory at both pressures, with the unweighted
      logarithmic fit that mtc_front_table.py prints, and the demand
  (b) limit against Gd2O3 weight fraction, same data, since the two
      loading measures cannot be separated by these lattices
  (c) pressure shift of every lattice scanned at both pressures, with the
      weighted mean and its one-sigma band

Design labels in (a) and (b) are placed by place_labels(): each label tries
a ring of offsets and takes the first whose box overlaps no marker, error
bar, fit line, legend or other label.

USAGE (repository root)
  python c9_ceiling_two_pressures.py
  python c9_ceiling_two_pressures.py --out ../thesis_tier3/images
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
from matplotlib.transforms import Bbox, offset_copy  # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})
C128, C155, CDEM, CGREY = "#0072B2", "#D55E00", "#009E73", "#444444"
MS = 4.5  # marker size in points


def load(p):
    rows = json.loads(Path(p).read_text())
    return {int(r["idx"]): r for r in rows}


# candidate offsets in points: right, above, left, below, diagonals, then farther out
_DIRS = [(1, 0.4), (1, -1), (-1, 0.4), (-1, -1), (0, 1.3), (0, -1.9), (1, 1.3), (-1, 1.3),
         (1, -1.9), (-1, -1.9)]
CANDIDATES = [(dx * r, dy * r) for r in (4, 7, 11, 16) for dx, dy in _DIRS]


def place_labels(fig, ax, labels, obstacles, extra_boxes=()):
    """labels: list of (text, x, y, colour) in data coordinates.
    obstacles: list of (x, y, radius_px) points in data coordinates.
    extra_boxes: display-space Bboxes that must stay free (legend)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    T = ax.transData
    obs = []
    for x, y, rad in obstacles:
        px, py = T.transform((x, y))
        obs.append(Bbox.from_extents(px - rad, py - rad, px + rad, py + rad))
    obs += list(extra_boxes)
    frame = ax.get_window_extent(r)
    placed = []
    for txt, x, y, col in labels:
        best = None
        for dx, dy in CANDIDATES:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
            tr = offset_copy(T, fig=fig, x=dx, y=dy, units="points")
            t = ax.text(x, y, txt, transform=tr, ha=ha, va=va, fontsize=6.5, color=col, zorder=6)
            bb = t.get_window_extent(r).expanded(1.08, 1.15)
            inside = (bb.x0 >= frame.x0 and bb.x1 <= frame.x1 and bb.y0 >= frame.y0 and bb.y1 <= frame.y1)
            hit = any(bb.overlaps(o) for o in obs) or any(bb.overlaps(p) for p in placed)
            if inside and not hit:
                best = (t, bb)
                break
            t.remove()
        if best is None:
            raise RuntimeError(f"no free position for label {txt} at ({x:.3g}, {y:.0f})")
        placed.append(best[1])


def line_points(ax, xs, ys, step=2):
    """Obstacles every few samples of a drawn line (data coordinates)."""
    return [(x, y, 2.5) for x, y in zip(xs[::step], ys[::step])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", default="c9_post/mtc_ceiling_table.json")
    ap.add_argument("--high", default="c9_post/p155/mtc_ceiling_table.json")
    ap.add_argument("--out", default="c9_post")
    ap.add_argument("--extra-low", default=None,
                    help="rows at 12.8 MPa drawn with open markers and kept out of the fits")
    ap.add_argument("--extra-high", default=None,
                    help="rows at 15.5 MPa drawn with open markers and kept out of the fits")
    a = ap.parse_args()
    for p in (a.low, a.high):
        if not Path(p).exists():
            print(f"ABORT: {p} not found, run mtc_front_table.py first"); return 1
    lo, hi = load(a.low), load(a.high)
    xlo = load(a.extra_low) if a.extra_low else {}
    xhi = load(a.extra_high) if a.extra_high else {}
    both = sorted(set(lo) & set(hi), key=lambda i: lo[i]["inventory"])
    xboth = sorted(set(xlo) & set(xhi), key=lambda i: xlo[i]["inventory"])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    rad_marker = MS / 2 * fig.dpi / 72 + 2.0
    for key, ax, lab, logx in (("inventory", axes[0], r"Gadolinia inventory, Gd$_2$O$_3$ wt% $\times$ pins", True),
                               ("gd_wt", axes[1], r"Gd$_2$O$_3$ weight fraction (wt%)", False)):
        labels, obstacles = [], []
        for tab, col, pr in ((lo, C128, "12.8"), (hi, C155, "15.5")):
            ids = sorted(tab, key=lambda i: tab[i][key])
            x = np.array([tab[i][key] for i in ids]); y = np.array([tab[i]["ceiling"] for i in ids])
            s = np.array([tab[i]["sigma"] for i in ids])
            ax.errorbar(x, y, yerr=s, fmt="o", ms=MS, color=col, capsize=2, lw=1.0,
                        label=f"MTC boron limit, {pr} MPa")
            if logx:
                k, q = np.polyfit(np.log(x), y, 1)
                xs = np.geomspace(x.min() * 0.85, x.max() * 1.12, 200)
                ys = k * np.log(xs) + q
            else:
                k, q = np.polyfit(x, y, 1)
                xs = np.linspace(0, x.max() * 1.05, 200)
                ys = k * xs + q
            ax.plot(xs, ys, color=col, ls="--", lw=0.9)
            obstacles += line_points(ax, xs, ys)
            for i, xi, yi, si in zip(ids, x, y, s):
                obstacles.append((xi, yi, rad_marker))
                obstacles += [(xi, yy, 3.0) for yy in np.linspace(yi - si, yi + si, 12)]
                labels.append((str(i), xi, yi, col))
        for tab, col in ((xlo, C128), (xhi, C155)):       # lattices outside the fit
            for i in sorted(tab, key=lambda i: tab[i][key]):
                xi, yi, si = tab[i][key], tab[i]["ceiling"], tab[i]["sigma"]
                ax.errorbar([xi], [yi], yerr=[si], fmt="o", ms=MS, mfc="white", mec=col,
                            color=col, capsize=2, lw=1.0, zorder=4)
                obstacles.append((xi, yi, rad_marker))
                obstacles += [(xi, yy, 3.0) for yy in np.linspace(yi - si, yi + si, 12)]
                labels.append((str(i), xi, yi, col))
        if xlo:
            ax.errorbar([], [], yerr=[], fmt="o", ms=MS, mfc="white", mec="0.35", color="0.35",
                        capsize=2, lw=1.0, label="Lattice scanned after the fit")
        dem_ids = sorted(list(lo) + list(xlo), key=lambda i: {**lo, **xlo}[i][key])
        dx_ = [{**lo, **xlo}[i][key] for i in dem_ids]; dy_ = [{**lo, **xlo}[i]["demand"] for i in dem_ids]
        ax.plot(dx_, dy_, "s", ms=MS, mfc="white", mec=CDEM, mew=1.2,
                label=r"Critical boron $c_\mathrm{max}$")
        obstacles += [(xx, yy, rad_marker) for xx, yy in zip(dx_, dy_)]
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab); ax.set_ylabel("Boron concentration (ppm)")
        ax.set_ylim(1250, 3600)
        if xlo:                                     # room for the labels of the extra lattices
            xmax = max(r[key] for r in list(lo.values()) + list(xlo.values()))
            ax.set_xlim(right=xmax * 1.45 if logx else xmax + 1.0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        extra = []
        if ax is axes[0]:
            leg = ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.5, loc="lower left")
            fig.canvas.draw()
            extra = [leg.get_window_extent(fig.canvas.get_renderer()).expanded(1.05, 1.1)]
        # labels sorted from the most crowded region so they get the first choice
        place_labels(fig, ax, labels, obstacles, extra)
    axes[0].set_title("(a) Against inventory, logarithmic fit", fontsize=8, loc="left")
    axes[1].set_title("(b) Against weight fraction, linear fit", fontsize=8, loc="left")

    ax = axes[2]
    sh = np.array([hi[i]["ceiling"] - lo[i]["ceiling"] for i in both])
    ss = np.array([math.hypot(hi[i]["sigma"], lo[i]["sigma"]) for i in both])
    w = 1 / ss ** 2; m = float(np.sum(w * sh) / w.sum()); sm = float(w.sum() ** -0.5)
    chi2 = float(np.sum(w * (sh - m) ** 2)); dof = len(both) - 1
    y = np.arange(len(both))
    ax.axvspan(m - sm, m + sm, color=CGREY, alpha=0.15, lw=0)
    ax.axvline(m, color=CGREY, lw=1.0, ls="--",
               label=f"Weighted mean {m:.0f} $\\pm$ {sm:.0f} ppm")
    ax.errorbar(sh, y, xerr=ss, fmt="o", color=C155, capsize=2, lw=1.0)
    xsh = [xhi[i]["ceiling"] - xlo[i]["ceiling"] for i in xboth]
    xss = [math.hypot(xhi[i]["sigma"], xlo[i]["sigma"]) for i in xboth]
    yx = np.arange(len(both), len(both) + len(xboth))
    if xboth:
        ax.errorbar(xsh, yx, xerr=xss, fmt="o", mfc="white", mec=C155, color=C155, capsize=2, lw=1.0)
    names = [f"C9-{i} ({lo[i]['inventory']:.1f})" for i in both] + \
            [f"C9-{i} ({xlo[i]['inventory']:.1f})" for i in xboth]
    ax.set_yticks(np.arange(len(names)), names, fontsize=7)
    ax.set_xlabel("MTC boron limit at 15.5 minus 12.8 MPa (ppm)")
    ax.set_title(f"(c) Pressure shift, $\\chi^2$ = {chi2:.1f} on {dof} degrees of freedom",
                 fontsize=8, loc="left")
    ax.set_ylim(-0.6, len(names) - 0.4 + 0.8)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
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
