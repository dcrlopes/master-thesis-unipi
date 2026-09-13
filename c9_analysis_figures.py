#!/usr/bin/env python3
r"""
c9_analysis_figures.py
======================
The three figures that carry the Campaign 9 argument, built from the
outputs of stages A to D. No OpenMC. Seconds.

  c9_two_formulations.pdf
      The Campaign 9 archive drawn in the Campaign 8 objective plane
      (cycle length, F_dH). The Campaign 9 front members are marked, and
      so is the set the Campaign 8 objectives would have selected from
      the same archive. This is the figure that shows, rather than
      asserts, that the original formulation would have discarded the
      designs the reformulation selects.

  c9_front_stability.pdf
      Pareto-membership frequency under the measurement noise, from
      c9_step0.json. Separates the front members that are robust from
      those that sit inside the noise band.

  c9_gd_mechanism.pdf
      Two panels sharing the gadolinia axis, over the feasible designs
      in the enrichment band of the front: the critical boron at the
      operating maximum, and the mid-cycle reactivity hump. The
      mechanism of the campaign in one figure.

Usage
  python c9_analysis_figures.py --checkpoint out_c9/optimization_checkpoint.json \
      --post c9_post --out figs_c9
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
    ap.add_argument("--post", default="c9_post")
    ap.add_argument("--out", default="figs_c9")
    ap.add_argument("--band", type=float, default=0.35)
    ap.add_argument("--efpd-req", type=float, default=1826.0)
    ap.add_argument("--f-max", type=float, default=1.65)
    ap.add_argument("--ceiling", type=float, default=2763.0)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    post = pathlib.Path(a.post)
    d = json.load(open(a.checkpoint)); raw = d["all_raw"]
    man = json.load(open(post / "c9_front.json"))
    front = man["front"]; feas = set(man["feasible"])
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    made = []

    # ------------------------------------------------ 1. two formulations
    rr = post / "c9_reverse_retro.json"
    if rr.is_file():
        R = json.load(open(rr))
        nd8 = set(R["nondominated_under_c8"]); feas8 = set(R["feasible_under_c8"])
        E = np.array([x["cycle_length"] for x in raw])
        F = np.array([x["peaking"] for x in raw])
        C = np.array([x["c_max"] for x in raw])
        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        oth = [i for i in range(len(raw)) if i not in feas8]
        ax.scatter(E[oth], F[oth], s=16, marker="x", c="lightgray", label="infeasible under C8 rules")
        f8 = [i for i in feas8 if i not in nd8 and i not in front]
        ax.scatter(E[f8], F[f8], s=28, c="0.55", label=f"feasible, dominated ({len(f8)})")
        n8 = [i for i in nd8 if i not in front]
        ax.scatter(E[n8], F[n8], s=60, marker="s", facecolor="none", edgecolor="tab:blue",
                   lw=1.3, label=f"Campaign 8 objectives would select ({len(nd8)})")
        fr = list(front)
        sc = ax.scatter(E[fr], F[fr], s=95, marker="o", c=C[fr], cmap="viridis_r",
                        vmin=1300, vmax=2100, edgecolor="crimson", lw=1.6, zorder=4,
                        label=f"Campaign 9 front ({len(fr)})")
        for i in fr:
            ax.annotate(str(i), (E[i], F[i]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.axvline(a.efpd_req, c="k", lw=0.9, ls="--")
        ax.text(a.efpd_req + 40, F[list(feas8)].max(), f"mission {a.efpd_req:.0f} EFPD",
                fontsize=7.5, rotation=90, va="top")
        ax.axhline(a.f_max, c="gray", lw=0.8, ls=":")
        ax.set_xlabel("Cycle length [EFPD]   (Campaign 8 objective, maximise)")
        ax.set_ylabel(r"$F_{\Delta H}$   (both campaigns, minimise)")
        cb = plt.colorbar(sc, ax=ax, pad=0.02); cb.set_label(r"$c_\mathrm{max}$ [ppm]")
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.grid(alpha=0.25)
        ax.set_title("The Campaign 9 archive in the Campaign 8 objective plane", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(out / "c9_two_formulations.pdf"); fig.savefig(out / "c9_two_formulations.png", dpi=160)
        made.append("c9_two_formulations")

    # ------------------------------------------------ 2. front stability
    s0 = post / "c9_step0.json"
    if s0.is_file():
        S = json.load(open(s0))["stability"]
        fq = S["membership_freq"]; nom = set(S["nominal_front"])
        ids = sorted(fq, key=lambda k: -fq[k]); v = [fq[i] for i in ids]
        col = ["crimson" if int(i) in nom else "0.7" for i in ids]
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        ax.bar(range(len(ids)), v, color=col, edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, fontsize=8)
        ax.axhline(0.5, c="k", lw=0.9, ls="--")
        ax.text(len(ids) - 0.4, 0.52, "50 %", fontsize=8, ha="right")
        ax.set_xlabel("Design"); ax.set_ylabel("Fraction of resamples on the front")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.25, axis="y")
        ax.set_title(f"Front membership under the measurement noise, "
                     f"HV ratio {S['hv_ratio_mean']:.3f} $\\pm$ {S['hv_ratio_sd']:.3f}", fontsize=9)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(fc="crimson", ec="k", label="on the nominal front"),
                           Patch(fc="0.7", ec="k", label="not on the nominal front")],
                  fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "c9_front_stability.pdf"); fig.savefig(out / "c9_front_stability.png", dpi=160)
        made.append("c9_front_stability")

    # ------------------------------------------------ 3. the mechanism
    e0 = float(np.mean([raw[i]["enrich"] for i in front]))
    band = [i for i in range(len(raw)) if i in feas and abs(raw[i]["enrich"] - e0) <= a.band]
    band.sort(key=lambda i: raw[i]["gd_wt"])
    G = [raw[i]["gd_wt"] for i in band]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.25, 1]))
    for i in band:
        a1.plot([raw[i]["gd_wt"]] * 2, [raw[i]["c_bol"], raw[i]["c_max"]], c="gray", lw=0.9, zorder=1)
    a1.scatter(G, [raw[i]["c_bol"] for i in band], s=26, facecolor="white",
               edgecolor="tab:blue", zorder=2, label=r"$c_\mathrm{BOL}$")
    a1.scatter(G, [raw[i]["c_max"] for i in band], s=32, c="tab:blue", zorder=3,
               label=r"$c_\mathrm{max}$, the objective")
    fb = [i for i in band if i in front]
    a1.scatter([raw[i]["gd_wt"] for i in fb], [raw[i]["c_max"] for i in fb], s=95,
               facecolor="none", edgecolor="crimson", lw=1.4, zorder=4, label="on the front")
    a1.axhline(a.ceiling, c="k", lw=1.0)
    a1.text(max(G), a.ceiling + 40, f"MTC ceiling {a.ceiling:.0f} ppm", fontsize=7.5, ha="right")
    a1.set_ylabel("Critical boron [ppm]"); a1.legend(fontsize=8, frameon=False); a1.grid(alpha=0.25)
    a1.set_title(f"Feasible designs at {e0 - a.band:.2f} to {e0 + a.band:.2f} wt% "
                 f"({len(band)} designs)", fontsize=9.5)
    H = [raw[i]["hump_core_pcm"] for i in band]
    a2.scatter(G, H, s=32, c="tab:red")
    a2.scatter([raw[i]["gd_wt"] for i in fb], [raw[i]["hump_core_pcm"] for i in fb], s=95,
               facecolor="none", edgecolor="crimson", lw=1.4)
    a2.axhline(400, c="k", lw=0.8, ls=":")
    a2.text(max(G), 520, "400 pcm noise floor", fontsize=7.5, ha="right")
    a2.set_xlabel(r"Gd$_2$O$_3$ weight fraction [wt%]")
    a2.set_ylabel("Hump at the operating\nmaximum [pcm]"); a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "c9_gd_mechanism.pdf"); fig.savefig(out / "c9_gd_mechanism.png", dpi=160)
    made.append("c9_gd_mechanism")

    # a small correlation table for the text
    g = np.array(G); c = np.array([raw[i]["c_max"] for i in band]); h = np.array(H)
    def sp(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        n = len(x); return 1 - 6 * ((rx - ry) ** 2).sum() / (n * (n * n - 1))
    txt = (f"band {e0-a.band:.2f} to {e0+a.band:.2f} wt%, {len(band)} feasible designs\n"
           f"Spearman rho, Gd against c_max : {sp(g, c):+.3f}\n"
           f"Spearman rho, Gd against hump  : {sp(g, h):+.3f}\n"
           f"c_max range {c.min():.0f} to {c.max():.0f} ppm over Gd {g.min():.2f} to {g.max():.2f} wt%\n"
           f"designs with hump > 400 pcm: {(h > 400).sum()} of {len(band)}, "
           f"their mean Gd {g[h>400].mean() if (h>400).any() else float('nan'):.2f} wt% "
           f"against {g[h<=400].mean():.2f} wt% for the rest\n")
    (pathlib.Path(a.post) / "c9_gd_correlations.txt").write_text(txt)
    print(txt)
    print("wrote " + ", ".join(f"{out}/{m}.pdf" for m in made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
