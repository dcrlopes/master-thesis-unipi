#!/usr/bin/env python3
"""c6_pareto_figure.py -- objective-space figure of the Campaign 6 archive.

Reads the campaign checkpoint (preferred) or a CSV dump and draws every
evaluated design in the objective plane: cycle length in EFPD against the
zoned core radial enthalpy-rise factor F_dH. Feasible designs are coloured
by the block that produced them, infeasible designs are grey crosses, the
Pareto front of the feasible set is joined by a step line, and the F_dH
limit, the licensing-style screens and the frozen hypervolume reference
point are drawn for context.

Outputs <out>.png (300 dpi, for slides) and <out>.pdf (vector, for LaTeX),
and prints the front table and the hypervolume under both references.

Usage on the AWS box (checkpoint is authoritative):
    lab python c6_pareto_figure.py \
        --checkpoint out_c6/optimization_checkpoint.json --out c6_pareto
Usage from a CSV dump (columns as in c6_full66.csv):
    python c6_pareto_figure.py --csv c6_full66.csv --out c6_pareto
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

# block boundaries of Campaign 6: design of experiments, then infill blocks
BLOCK_EDGES = [0, 42, 60, 66, 84, 120]
BLOCK_NAMES = ["DOE (block 1)", "Block 2", "Block 3", "Block 4", "Block 5"]
BLOCK_COLOR = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
BLOCK_MARK = ["o", "s", "D", "^", "v"]

F_LIMIT = 2.0            # screening bound of the campaign
SCREENS = [1.65, 1.50]   # licensing-style checks at candidate selection
HV_REF = (6776.0, 2.0692)  # frozen in-campaign reference (EFPD, F_dH)


def load_checkpoint(path):
    d = json.loads(Path(path).read_text())
    cn = d["constraint_names"]
    rows = []
    for i, x in enumerate(d["all_raw"]):
        rows.append(dict(
            pos=i,
            efpd=float(x["cycle_length"]),
            fdh=float(x["peaking"]),
            feas=all(float(x[c]) <= 0.0 for c in cn),
        ))
    ref = d.get("hv_ref")
    ref = (-float(ref[0]), float(ref[1])) if ref else HV_REF
    return rows, ref


def load_csv(path):
    rows = []
    with open(path) as fh:
        for x in csv.DictReader(fh):
            rows.append(dict(pos=int(float(x["pos"])),
                             efpd=float(x["efpd"]),
                             fdh=float(x["fdh"]),
                             feas=float(x["feas"]) == 1.0))
    return rows, HV_REF


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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="optimization_checkpoint.json")
    src.add_argument("--csv", help="CSV dump with pos,efpd,fdh,...,feas")
    ap.add_argument("--out", default="c6_pareto", help="output file prefix")
    ap.add_argument("--title", default=None, help="override the title")
    args = ap.parse_args()

    rows, ref = (load_checkpoint(args.checkpoint) if args.checkpoint
                 else load_csv(args.csv))
    front = pareto(rows)
    front_pos = {r["pos"] for r in front}
    n_feas = sum(r["feas"] for r in rows)

    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=300)

    # constraint region and screens ---------------------------------------
    ax.axhspan(F_LIMIT, 2.85, color="#D55E00", alpha=0.06, zorder=0)
    ax.axhline(F_LIMIT, color="#D55E00", lw=1.4, ls="--", zorder=1)
    ax.text(1500, F_LIMIT + 0.015, r"$F_{\Delta H}$ limit (2.0)",
            color="#D55E00", fontsize=10, va="bottom")
    for s in SCREENS:
        ax.axhline(s, color="#888888", lw=0.9, ls=":", zorder=1)
        above = s >= 1.6
        ax.text(1500, s + (0.012 if above else -0.014),
                f"screen {s:.2f}", color="#666666", fontsize=9,
                va="bottom" if above else "top")

    # frozen hypervolume reference ----------------------------------------
    ax.plot(*ref, marker="*", ms=15, color="#000000", mfc="#F0E442",
            zorder=6, ls="none")
    ax.annotate("frozen HV\nreference", ref,
                textcoords="offset points", xytext=(14, 4),
                fontsize=9, color="#333333")

    # designs ---------------------------------------------------------------
    inf = [r for r in rows if not r["feas"]]
    ax.scatter([r["efpd"] for r in inf], [r["fdh"] for r in inf],
               marker="x", s=42, c="#9A9A9A", lw=1.3,
               label=f"infeasible ({len(inf)})", zorder=2)

    seen = set()
    for r in rows:
        if not r["feas"]:
            continue
        b = block_of(r["pos"])
        lab = None
        if b not in seen:
            seen.add(b)
            n_b = sum(1 for q in rows
                      if q["feas"] and block_of(q["pos"]) == b)
            lab = f"{BLOCK_NAMES[b]} feasible ({n_b})"
        on_front = r["pos"] in front_pos
        ax.scatter(r["efpd"], r["fdh"], marker=BLOCK_MARK[b],
                   s=150 if on_front else 62,
                   c=BLOCK_COLOR[b],
                   edgecolors="black" if on_front else "none",
                   linewidths=1.6 if on_front else 0.0,
                   alpha=1.0 if on_front else 0.75,
                   label=lab, zorder=5 if on_front else 3)

    # Pareto step line ------------------------------------------------------
    fx = [r["efpd"] for r in front]
    fy = [r["fdh"] for r in front]
    sx, sy = [], []
    for i in range(len(front)):
        sx.append(fx[i]); sy.append(fy[i])
        if i + 1 < len(front):
            sx.append(fx[i + 1]); sy.append(fy[i])
    ax.plot(sx, sy, color="#C1272D", lw=2.0, zorder=4,
            label=f"Pareto front ({len(front)} designs)")

    # annotations on the named designs -------------------------------------
    notes = {40: ("idx 40\n9035 EFPD, F=1.574", (-16, -30), "right"),
             2: ("idx 2\n9445 EFPD, F=1.655", (12, 12), "left"),
             59: ("idx 59\nF=1.404", (12, -4), "left")}
    for r in front:
        if r["pos"] in notes:
            txt, off, ha = notes[r["pos"]]
            ax.annotate(txt, (r["efpd"], r["fdh"]),
                        textcoords="offset points", xytext=off,
                        fontsize=9.5, ha=ha,
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color="#444444"))

    # axes ------------------------------------------------------------------
    ax.set_xlim(-150, 10600)
    ax.set_ylim(1.32, 2.80)
    ax.set_xlabel("Cycle length (EFPD)", fontsize=13)
    ax.set_ylabel(r"Radial enthalpy-rise factor $F_{\Delta H}$ (zoned core)",
                  fontsize=13)
    ttl = args.title or (f"Campaign 6 objective space: "
                         f"{len(rows)} evaluations, {n_feas} feasible")
    ax.set_title(ttl, fontsize=14, pad=10)
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.56, 0.99),
              fontsize=9.5, frameon=True,
              framealpha=0.92, borderpad=0.7, labelspacing=0.45)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"written -> {args.out}.{ext}")

    # console report --------------------------------------------------------
    print(f"\nPareto front ({len(front)} designs):")
    print("  pos   EFPD    F_dH    block")
    for r in front:
        print(f"  {r['pos']:3d} {r['efpd']:7.0f} {r['fdh']:7.4f}   "
              f"{BLOCK_NAMES[block_of(r['pos'])]}")
    hv_frozen = hypervolume(rows, ref[0], ref[1])
    hv_full = hypervolume(rows, 0.0, ref[1])
    print(f"\nHV frozen ref ({ref[0]:.0f}, {ref[1]:.4f}) : {hv_frozen:.2f}")
    print(f"HV full range (0, {ref[1]:.4f})    : {hv_full:.2f}")


if __name__ == "__main__":
    main()
