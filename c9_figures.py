#!/usr/bin/env python3
r"""
c9_figures.py
=============
The three figures of the Campaign 9 post-analysis, from the checkpoint
alone. No OpenMC. Seconds.

  c9_pareto.pdf      every evaluated design in (F_dH, c_max), feasible
                     designs filled, the front joined, the MTC ceiling as
                     a horizontal line, colour = gadolinia weight fraction
  c9_gd_tradeoff.pdf c_max against gadolinia for the feasible designs in
                     the enrichment band of the front, with the hump shown
                     as the difference between c_BOL and c_max
  c9_hump_vs_gd.pdf  hump against gadolinia, all designs, to show that the
                     low-gadolinia designs are the ones whose boron demand
                     rises after beginning of life

Usage
  python c9_figures.py --checkpoint out_c9/optimization_checkpoint.json \
      --manifest c9_post/c9_front.json --out figs_c9
"""
import argparse
import json
import pathlib
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--manifest", default="c9_post/c9_front.json")
    ap.add_argument("--out", default="figs_c9")
    ap.add_argument("--ceiling", type=float, default=2763.0)
    ap.add_argument("--ceiling-hi", type=float, default=2997.0,
                    help="second ceiling to draw dashed (15.5 MPa)")
    ap.add_argument("--f-max", type=float, default=1.65)
    ap.add_argument("--band", type=float, default=0.35,
                    help="half-width in wt% of the enrichment band around the front mean")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open(a.checkpoint))
    raw = d["all_raw"]
    m = json.load(open(a.manifest))
    k1, k2 = m["objectives"]
    feas, front = set(m["feasible"]), m["front"]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    F = np.array([x[k1] for x in raw]); C = np.array([x[k2] for x in raw])
    G = np.array([x["gd_wt"] for x in raw]); E = np.array([x["enrich"] for x in raw])
    isf = np.array([i in feas for i in range(len(raw))])

    # ---------------------------------------------------------------- 1
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sc = ax.scatter(F[~isf], C[~isf], s=22, c=G[~isf], cmap="viridis", vmin=0, vmax=8,
                    marker="x", alpha=0.55, label="infeasible")
    ax.scatter(F[isf], C[isf], s=34, c=G[isf], cmap="viridis", vmin=0, vmax=8,
               edgecolor="k", linewidth=0.5, label="feasible")
    fx = [raw[i][k1] for i in front]; fy = [raw[i][k2] for i in front]
    o = np.argsort(fx)
    ax.plot(np.array(fx)[o], np.array(fy)[o], "-", c="crimson", lw=1.4, zorder=2,
            label=f"Pareto front ({len(front)})")
    ax.axhline(a.ceiling, c="k", lw=1.0, ls="-")
    ax.text(ax.get_xlim()[1] if False else F[isf].max() + 0.005, a.ceiling + 40,
            f"MTC ceiling {a.ceiling:.0f} ppm (12.8 MPa)", fontsize=7.5, ha="right")
    ax.axhline(a.ceiling_hi, c="k", lw=0.8, ls="--")
    ax.text(F[isf].max() + 0.005, a.ceiling_hi + 40, f"{a.ceiling_hi:.0f} ppm (15.5 MPa)",
            fontsize=7.5, ha="right")
    ax.axvline(a.f_max, c="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"Hot channel factor $F_{\Delta H}$ (minimise)")
    ax.set_ylabel(r"Critical boron at the operating maximum $c_\mathrm{max}$ [ppm]")
    ax.set_ylim(0, 6300)
    cb = plt.colorbar(sc, ax=ax, pad=0.02); cb.set_label(r"Gd$_2$O$_3$ [wt%]")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "c9_pareto.pdf"); fig.savefig(out / "c9_pareto.png", dpi=160)

    # ---------------------------------------------------------------- 2
    e0 = np.mean([raw[i]["enrich"] for i in front]) if front else 4.3
    band = [i for i in range(len(raw)) if isf[i] and abs(E[i] - e0) <= a.band]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for i in band:
        x = raw[i]
        ax.plot([x["gd_wt"], x["gd_wt"]], [x.get("c_bol", x[k2]), x[k2]], c="gray", lw=0.8, zorder=1)
    ax.scatter([raw[i]["gd_wt"] for i in band], [raw[i].get("c_bol", raw[i][k2]) for i in band],
               s=26, facecolor="white", edgecolor="tab:blue", label=r"$c_\mathrm{BOL}$ (beginning of life)")
    ax.scatter([raw[i]["gd_wt"] for i in band], [raw[i][k2] for i in band],
               s=30, c="tab:blue", label=r"$c_\mathrm{max}$ (operating maximum)")
    fb = [i for i in band if i in front]
    ax.scatter([raw[i]["gd_wt"] for i in fb], [raw[i][k2] for i in fb], s=70,
               facecolor="none", edgecolor="crimson", lw=1.2, label="on the front")
    ax.axhline(a.ceiling, c="k", lw=1.0); ax.text(0.05, a.ceiling + 30, "MTC ceiling", fontsize=8)
    ax.set_xlabel(r"Gd$_2$O$_3$ weight fraction [wt%]")
    ax.set_ylabel("Critical boron [ppm]")
    ax.set_title(f"Feasible designs, {e0 - a.band:.2f} to {e0 + a.band:.2f} wt% ({len(band)} designs)", fontsize=9)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "c9_gd_tradeoff.pdf"); fig.savefig(out / "c9_gd_tradeoff.png", dpi=160)

    # ---------------------------------------------------------------- 3
    H = np.array([x.get("hump_core_pcm", 0.0) for x in raw])
    ok = H < 15000     # drop the subcritical duds whose leakage factor blows up
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(G[ok & ~isf], H[ok & ~isf], s=20, marker="x", c="gray", alpha=0.6, label="infeasible")
    ax.scatter(G[ok & isf], H[ok & isf], s=30, c="tab:blue", label="feasible")
    ax.axhline(400, c="k", lw=0.8, ls=":"); ax.text(7.2, 470, "400 pcm floor", fontsize=8, ha="right")
    ax.set_xlabel(r"Gd$_2$O$_3$ weight fraction [wt%]"); ax.set_ylabel("Core hump at the operating maximum [pcm]")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "c9_hump_vs_gd.pdf"); fig.savefig(out / "c9_hump_vs_gd.png", dpi=160)

    # band table for the text
    rows = sorted(band, key=lambda i: raw[i][k2])
    with open(out / "c9_gd_tradeoff.txt", "w") as f:
        f.write(f"feasible designs within {a.band} wt% of the front mean {e0:.2f} wt%, by c_max\n")
        f.write(f"{'id':>3} {'e':>5} {'Gd':>5} {'pin':>4} {'c_BOL':>6} {'c_max':>6} {'hump':>6} {'front':>5}\n")
        for i in rows:
            x = raw[i]
            f.write(f"{i:>3} {x['enrich']:5.2f} {x['gd_wt']:5.2f} {x['gd_pins_used']:4.0f} "
                    f"{x.get('c_bol', float('nan')):6.0f} {x[k2]:6.0f} {x.get('hump_core_pcm', 0):+6.0f} "
                    f"{'yes' if i in front else '':>5}\n")
    print(f"wrote {out}/c9_pareto.pdf, c9_gd_tradeoff.pdf, c9_hump_vs_gd.pdf, c9_gd_tradeoff.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
