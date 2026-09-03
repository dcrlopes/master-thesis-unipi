#!/usr/bin/env python3
"""c8_pareto_figure.py -- objective-space figure of the Campaign 8 archive.

Reads the campaign checkpoint and draws every evaluated design in the
objective plane, cycle length in EFPD against the core radial
enthalpy-rise factor F_dH. Feasible designs are filled markers coloured
by block, infeasible designs are crosses, the Pareto front of the
feasible set is joined by a step line, and the four feasible designs
that the first two regulating banks alone hold subcritical carry a
purple ring.

The refuelling requirement is drawn as three vertical lines at five
calendar years and capacity factors of 0.5, 0.8 and 1.0, that is 913,
1461 and 1826 EFPD. The band from 0.5 to 0.8 is the assumed duty of a
land-based prototype, the band from 0.8 to 1.0 is the assumed duty of the
boat at sea, and 0.8 is the reference case where the two meet.

Outputs <out>.png (300 dpi, for slides) and <out>.pdf (vector, for LaTeX),
and prints the front, the two-bank set and the hypervolume.

Usage on wks720, conda env openmc-env (matplotlib only, no OpenMC):
    python c8_pareto_figure.py \\
        --checkpoint out_c8/optimization_checkpoint.json \\
        --out figs_c8/c8_pareto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Campaign 8 blocks: DOE of 36, then four infill blocks of 6
BLOCK_EDGES = [0, 36, 42, 48, 54, 60, 200]
BLOCK_NAMES = ["DOE (block 1)", "Infill 1", "Infill 2", "Infill 3", "Infill 4"]
BLOCK_COLOR = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
BLOCK_MARK = ["o", "s", "D", "^", "v"]

F_LIMIT = 2.0                # screening bound of the campaign
SCREENS = [1.65, 1.50]       # licensing-style checks at candidate selection
T_CAL_YEARS = 5.0            # refuelling interval used for the lines
CF_LINES = [0.5, 0.8, 1.0]   # capacity factors drawn
CF_REF = 0.8                 # reference case, drawn heavier
C_PROTO = "#7FB3D5"          # band 0.5 to 0.8, prototype duty
C_BOAT = "#F1948A"           # band 0.8 to 1.0, at-sea duty
C_TWOBANK = "#6A3D9A"        # purple ring


def efpd_req(cf, t_cal=T_CAL_YEARS):
    return 365.25 * t_cal * cf


def load_checkpoint(path):
    d = json.loads(Path(path).read_text())
    cn = d["constraint_names"]
    rows = []
    for i, x in enumerate(d["all_raw"]):
        feas = all(float(x[c]) <= 0.0 for c in cn)
        rows.append(dict(pos=i, efpd=float(x["cycle_length"]),
                         fdh=float(x["peaking"]), feas=feas,
                         two_bank=feas and float(x.get("g_ctrl12", 1.0)) <= 0.0))
    ref = d.get("hv_ref")
    ref = (-float(ref[0]), float(ref[1])) if ref else (0.0, F_LIMIT)
    return rows, ref


def block_of(pos):
    for b in range(len(BLOCK_EDGES) - 1):
        if BLOCK_EDGES[b] <= pos < BLOCK_EDGES[b + 1]:
            return b
    return len(BLOCK_NAMES) - 1


def pareto(rows):
    feas = [r for r in rows if r["feas"]]

    def dom(a, b):
        ge = a["efpd"] >= b["efpd"] and a["fdh"] <= b["fdh"]
        gt = a["efpd"] > b["efpd"] or a["fdh"] < b["fdh"]
        return ge and gt

    return sorted((a for a in feas
                   if not any(dom(b, a) for b in feas if b is not a)),
                  key=lambda r: -r["efpd"])


def hypervolume(rows, ref_e, ref_f):
    pts = sorted((r for r in rows if r["feas"]
                  and r["efpd"] > ref_e and r["fdh"] < ref_f),
                 key=lambda r: -r["efpd"])
    area, prev = 0.0, ref_f
    for p in pts:
        if p["fdh"] < prev:
            area += (p["efpd"] - ref_e) * (prev - p["fdh"])
            prev = p["fdh"]
    return area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--out", default="figs_c8/c8_pareto")
    ap.add_argument("--title", default=None,
                    help="figure title; omitted by default for the thesis")
    ap.add_argument("--no-screens", action="store_true",
                    help="drop the 1.65 and 1.50 screening lines")
    args = ap.parse_args()

    rows, ref = load_checkpoint(args.checkpoint)
    front = pareto(rows)
    front_pos = {r["pos"] for r in front}
    two = [r for r in rows if r["two_bank"]]
    n_feas = sum(r["feas"] for r in rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=300)
    y_lo, y_hi = 1.30, 2.16

    # ---- refuelling requirement bands and lines ------------------------- #
    e05, e08, e10 = (efpd_req(cf) for cf in CF_LINES)
    ax.axvspan(e05, e08, color=C_PROTO, alpha=0.14, lw=0, zorder=0)
    ax.axvspan(e08, e10, color=C_BOAT, alpha=0.14, lw=0, zorder=0)
    for cf in CF_LINES:
        x = efpd_req(cf)
        heavy = abs(cf - CF_REF) < 1e-9
        ax.axvline(x, color="#333333", lw=1.6 if heavy else 0.9,
                   ls="-" if heavy else ":", zorder=1)
        ax.text(x, y_lo + 0.012, f"CF {cf:.1f}\n{x:.0f} d",
                fontsize=8.5, ha="center", va="bottom", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85))
    band_handles = [
        Patch(fc=C_PROTO, alpha=0.4, ec="none",
              label="5 years, prototype duty, CF 0.5 to 0.8"),
        Patch(fc=C_BOAT, alpha=0.4, ec="none",
              label="5 years, at-sea duty, CF 0.8 to 1.0"),
    ]

    # ---- peaking limit and screens -------------------------------------- #
    ax.axhspan(F_LIMIT, y_hi, color="#D55E00", alpha=0.06, zorder=0)
    ax.axhline(F_LIMIT, color="#D55E00", lw=1.4, ls="--", zorder=1)
    ax.text(2450, F_LIMIT + 0.012, r"$F_{\Delta H}$ limit (2.0)",
            color="#D55E00", fontsize=10, va="bottom")
    if not args.no_screens:
        for s in SCREENS:
            ax.axhline(s, color="#888888", lw=0.9, ls=":", zorder=1)
            ax.text(2450, s + 0.008, f"screen {s:.2f}", color="#666666",
                    fontsize=9, va="bottom", ha="left")

    # ---- hypervolume reference ------------------------------------------ #
    ax.plot(*ref, marker="*", ms=14, color="black", mfc="#F0E442",
            zorder=6, ls="none")
    ax.annotate("HV reference", ref, textcoords="offset points",
                xytext=(-8, -16), fontsize=8.5, color="#333333", ha="right")

    # ---- designs ---------------------------------------------------------- #
    inf = [r for r in rows if not r["feas"]]
    for b in range(len(BLOCK_NAMES)):
        pts = [r for r in inf if block_of(r["pos"]) == b]
        if pts:
            ax.scatter([r["efpd"] for r in pts], [r["fdh"] for r in pts],
                       marker="x", s=42, c=BLOCK_COLOR[b], lw=1.3,
                       alpha=0.85, zorder=2)
    ax.scatter([], [], marker="x", s=42, c="#444444", lw=1.3,
               label=f"Infeasible ({len(inf)}, colour = block)")

    seen = set()
    for r in rows:
        if not r["feas"]:
            continue
        b = block_of(r["pos"])
        lab = None
        if b not in seen:
            seen.add(b)
            n_b = sum(1 for q in rows if q["feas"] and block_of(q["pos"]) == b)
            lab = f"{BLOCK_NAMES[b]}, feasible ({n_b})"
        on_front = r["pos"] in front_pos
        ax.scatter(r["efpd"], r["fdh"], marker=BLOCK_MARK[b],
                   s=140 if on_front else 60, c=BLOCK_COLOR[b],
                   edgecolors="black" if on_front else "none",
                   linewidths=1.5 if on_front else 0.0,
                   alpha=1.0 if on_front else 0.8,
                   label=lab, zorder=5 if on_front else 3)

    # two-bank ring
    ax.scatter([r["efpd"] for r in two], [r["fdh"] for r in two],
               marker="o", s=330, facecolors="none", edgecolors=C_TWOBANK,
               linewidths=2.0, zorder=4,
               label=f"Two-bank controllable ({len(two)})")

    # Pareto step line
    fx = [r["efpd"] for r in front]; fy = [r["fdh"] for r in front]
    sx, sy = [], []
    for i in range(len(front)):
        sx.append(fx[i]); sy.append(fy[i])
        if i + 1 < len(front):
            sx.append(fx[i + 1]); sy.append(fy[i])
    ax.plot(sx, sy, color="#C1272D", lw=2.0, zorder=4,
            label=f"Pareto front ({len(front)} designs)")

    # named designs
    by_pos = {r["pos"]: r for r in rows}
    notes = {1:  ("design 1", (10, 10), "left"),
             47: ("design 47", (12, -14), "left"),
             53: ("design 53", (14, -16), "left"),
             31: ("design 31", (14, 12), "left"),
             21: ("design 21", (10, 12), "left")}
    for p, (txt, off, ha) in notes.items():
        if p in by_pos:
            r = by_pos[p]
            ax.annotate(txt, (r["efpd"], r["fdh"]), textcoords="offset points",
                        xytext=off, fontsize=9, ha=ha,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#444444"))

    # ---- axes ------------------------------------------------------------ #
    ax.set_xlim(0, 6400)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Cycle length (EFPD)", fontsize=13)
    ax.set_ylabel(r"Radial enthalpy-rise factor $F_{\Delta H}$ (core)",
                  fontsize=13)
    if args.title:
        ax.set_title(args.title, fontsize=14, pad=10)
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(band_handles + handles, [h.get_label() for h in band_handles]
              + labels, loc="upper right", fontsize=8.8, frameon=True,
              framealpha=0.94, borderpad=0.7, labelspacing=0.4)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"written -> {args.out}.{ext}")

    # ---- console report -------------------------------------------------- #
    print(f"\n{len(rows)} evaluations, {n_feas} feasible, front {len(front)}, "
          f"two-bank {sorted(r['pos'] for r in two)}")
    print("front:  " + ", ".join(f"{r['pos']} ({r['efpd']:.0f}, {r['fdh']:.3f})"
                                 for r in front))
    print(f"requirement lines at 5 years: " +
          ", ".join(f"CF {cf}: {efpd_req(cf):.0f}" for cf in CF_LINES))
    print(f"HV against checkpoint reference ({ref[0]:.0f}, {ref[1]:.4f}): "
          f"{hypervolume(rows, *ref):.1f}")


if __name__ == "__main__":
    main()
