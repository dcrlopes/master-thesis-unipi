#!/usr/bin/env python3
"""
c8_post_figures.py -- figures, tables and numbers of the Campaign 8
post-analysis (run_c8_post.sh stages B, C, D, E, F, G and the archive).

Reads, relative to the working directory (the layout of run_c8_post.sh):
  out_c8/optimization_checkpoint.json      the archive (60 designs)
  boron_c8/runs.json                       boron sweep, 11 designs x 3 states x 4 ppm
  confirm3d_c8_0ppm/summary.json           stage D, ALL-RE at 0 ppm, 2D and 3D
  confirm3d_c8_0ppm/runs.json              stage D seed pairs
  confirm3d_c8/summary.json                stages A and C, 1000 ppm (optional here)
  confirm3d_c8_dc37/summary.json           stage G, refl override 3.969 (optional)
  confirm3d_c8_noparked/summary.json       stage E (optional)
  boron_c8_marginal/runs.json              stage F (optional)

Writes:
  figs_c8_post/c8_post_*.pdf and .png      eight figures
  figs_c8_post/c8_post_tables.tex          five booktabs tables
  figs_c8_post/c8_post_numbers.json        numbers for master_numbers.json

No OpenMC is needed. Any environment with numpy and matplotlib works,
openmc-env included. Run:
  python c8_post_figures.py --check      lists what is present, writes nothing
  python c8_post_figures.py              writes everything

Definitions used throughout (they match boron_worth.py and confirm3d.py):
  rho(k)   = 1e5 (k - 1) / k                                   [pcm]
  margin   = -rho(k_rodded)            positive = subcritical   [pcm]
  ALL-RE   = RE1 to RE4, the sixteen inner assemblies (zoning.RE_BANK_POSITIONS)
  RE12     = RE1 + RE2, eight assemblies (zoning.RE12_POSITIONS)
  g_ctrl   = k_ALLRE - 0.99  at 1000 ppm (margin 1000 pcm), constrained
  g_ctrl12 = k_RE12  - 0.99  at 1000 ppm, recorded only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

FRONT = [47, 42, 23, 29, 21, 44, 59, 1]
KEEP = [53, 31, 13]
ELEVEN = FRONT + KEEP
# The campaign screen is applied in delta-k by the evaluator:
#     g_ctrl = k_ALLRE - (1 - ctrl_margin_dk) = k_ALLRE - 0.99
# (run_optimization.py line 383, openmc_evaluator.py line 319, with
# --ctrl-margin 1000). Every margin plotted and tabulated here is
# -rho(k), the definition of boron_worth.py line 81, so the equivalent
# threshold on that axis is -rho(0.99) = 1010.1 pcm, not 1000.0. The
# 10 pcm difference changes no verdict (no design of the archive or of
# the post-analysis falls inside the band) but the line is drawn at the
# exact value so that a reader converting back finds no discrepancy.
K_SCREEN = 0.99          # the constraint threshold on k, dimensionless
MARGIN = -1e5 * (K_SCREEN - 1.0) / K_SCREEN     # 1010.1 pcm on the -rho axis
PPM = [0.0, 500.0, 1000.0, 1500.0]
CF_LINES = {"4 y at CF 0.8": 4 * 365.25 * 0.8, "5 y at CF 0.8": 5 * 365.25 * 0.8}

# beginning-of-life ALL-RE 3D hold-down classes for the zero-boron reading
CLASS_OK, CLASS_MARGINAL, CLASS_NO = "holds with margin", "subcritical, margin < 1000", "supercritical"


def rho(k: float) -> float:
    return 1e5 * (k - 1.0) / k


def load(path: str, required: bool = False):
    p = Path(path)
    if not p.exists():
        if required:
            sys.exit(f"FAIL: required file missing: {path}")
        return None
    return json.loads(p.read_text())


def check_tree() -> dict:
    files = {
        "archive": "out_c8/optimization_checkpoint.json",
        "boron_runs": "boron_c8/runs.json",
        "zero_summary": "confirm3d_c8_0ppm/summary.json",
        "zero_runs": "confirm3d_c8_0ppm/runs.json",
        "c1000_summary": "confirm3d_c8/summary.json",
        "dc37_summary": "confirm3d_c8_dc37/summary.json",
        "noparked_summary": "confirm3d_c8_noparked/summary.json",
        "marginal_runs": "boron_c8_marginal/runs.json",
    }
    print(f"cwd: {os.getcwd()}")
    for k, v in files.items():
        print(f"  {'ok ' if Path(v).exists() else '-- '} {v}")
    return files


# ----------------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------------
def archive_table(ck: dict) -> dict:
    A = ck["all_raw"]
    CN = ck["constraint_names"]
    out = {}
    for i, r in enumerate(A):
        feas = all(r[g] <= 0 for g in CN)
        out[i] = dict(
            enrich=r["enrich"], gd_wt=r["gd_wt"], refl=r["refl_thick"], pins=r["gd_pins_used"],
            e_max=r["e_max_zoned"], efpd=r["cycle_length"], F=r["peaking"], bu=r["bu_eoc_mwd_kg"],
            k_core=r["keff_core_bol"], k_allre=r["k_allre"], k_re12=r["k_re12"],
            g_ctrl=r["g_ctrl"], g_ctrl12=r["g_ctrl12"], g_kmax=r["g_kmax"], g_kmin=r["g_kmin"],
            feasible=feas, twobank=(feas and r["g_ctrl12"] <= 0),
            W16=rho(r["keff_core_bol"]) - rho(r["k_allre"]),
            W8=rho(r["keff_core_bol"]) - rho(r["k_re12"]),
            block=0 if i < 36 else 1 + (i - 36) // 6,
        )
    return out


def pareto(rows: dict, ids) -> list:
    ids = list(ids)
    front = []
    for i in ids:
        L, P = rows[i]["efpd"], rows[i]["F"]
        dom = any((rows[j]["efpd"] >= L and rows[j]["F"] <= P and (rows[j]["efpd"] > L or rows[j]["F"] < P))
                  for j in ids if j != i)
        if not dom:
            front.append(i)
    return sorted(front, key=lambda i: rows[i]["efpd"])


def boron_table(runs: dict) -> dict:
    """Per design: k by state and ppm (seed 0 of the sweep), margins, worths,
    and the boron concentration at which the eight- and sixteen-rod margins
    reach the campaign margin (linear interpolation between sweep points,
    linear extrapolation beyond 1500 ppm, flagged)."""
    designs = sorted({int(k.split("|")[0]) for k in runs})
    out = {}
    for d in designs:
        def k(st, p):
            key = f"{d}|{st}|{p}|0"
            return runs[key]["keff"] if key in runs else None
        if any(k(st, p) is None for st in ("ARO", "ARI", "RE12") for p in PPM):
            continue
        r_aro = {p: rho(k("ARO", p)) for p in PPM}
        m16 = {p: -rho(k("ARI", p)) for p in PPM}
        m8 = {p: -rho(k("RE12", p)) for p in PPM}
        w16 = {p: r_aro[p] - rho(k("ARI", p)) for p in PPM}
        w8 = {p: r_aro[p] - rho(k("RE12", p)) for p in PPM}
        wb = r_aro[0.0] - r_aro[1000.0]

        def ppm_for(m):
            xs, ys = PPM, [m[p] for p in PPM]
            if ys[0] >= MARGIN:          # already holds without boron
                return 0.0, False
            for i in range(3):
                if (ys[i] - MARGIN) * (ys[i + 1] - MARGIN) <= 0:
                    return xs[i] + (MARGIN - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]), False
            sl = (ys[3] - ys[2]) / (xs[3] - xs[2])
            return xs[3] + (MARGIN - ys[3]) / sl, True

        p8, ex8 = ppm_for(m8)
        p16, ex16 = ppm_for(m16)
        out[d] = dict(k={st: {p: k(st, p) for p in PPM} for st in ("ARO", "ARI", "RE12")},
                      F={st: {p: runs[f"{d}|{st}|{p}|0"]["fdh"] for p in PPM} for st in ("ARO", "ARI", "RE12")},
                      rho_aro=r_aro, m16=m16, m8=m8, w16=w16, w8=w8,
                      wb_pcm=wb, wb_per_ppm=wb / 1000.0,
                      share=wb / (wb + w16[1000.0]),
                      ppm8=max(p8, 0.0), ppm8_extrap=ex8, ppm16=max(p16, 0.0), ppm16_extrap=ex16,
                      ratio=w8[1000.0] / w16[1000.0])
    return out


def pooled_sigma(pairs) -> float:
    pairs = [p for p in pairs if len(p) == 2]
    if not pairs:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / (2 * len(pairs)))


def seed_pairs(runs: dict, state: str, mode: str, field: str) -> list:
    by = {}
    for key, rec in runs.items():
        f = key.split("|")
        if f[1] == state and f[2] == mode:
            by.setdefault(f[0], []).append(rec[field])
    return [v for v in by.values() if len(v) == 2]


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.framealpha": 0.9,
    })
    return plt


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    print(f"  wrote {name}.pdf/.png")


def fig_front(plt, rows, out, sigma_F):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    feas = [i for i in rows if rows[i]["feasible"]]
    infeas = [i for i in rows if not rows[i]["feasible"]]
    two = [i for i in feas if rows[i]["twobank"]]
    four = [i for i in feas if not rows[i]["twobank"]]
    front4 = pareto(rows, feas)
    front2 = pareto(rows, two)
    ax.scatter([rows[i]["efpd"] for i in infeas], [rows[i]["F"] for i in infeas], marker="x", s=22,
               color="0.55", linewidths=0.8, label="infeasible (35 fail $g_\\mathrm{ctrl}$, 6 fail $g_{k\\min}$)")
    ax.errorbar([rows[i]["efpd"] for i in four], [rows[i]["F"] for i in four], yerr=sigma_F, fmt="o", ms=5,
                color="tab:blue", ecolor="tab:blue", elinewidth=0.6, capsize=1.5, label="feasible, four regulating banks needed")
    ax.errorbar([rows[i]["efpd"] for i in two], [rows[i]["F"] for i in two], yerr=sigma_F, fmt="s", ms=6,
                color="tab:green", ecolor="tab:green", elinewidth=0.6, capsize=1.5, label="feasible, controllable with RE1 + RE2")
    ax.plot([rows[i]["efpd"] for i in front4], [rows[i]["F"] for i in front4], "-", color="tab:blue", lw=1.2,
            label="four-bank front (8 designs)")
    ax.plot([rows[i]["efpd"] for i in front2], [rows[i]["F"] for i in front2], "-", color="tab:green", lw=1.2,
            label="two-bank front (3 designs)")
    for i in front4 + front2:
        ax.annotate(str(i), (rows[i]["efpd"], rows[i]["F"]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    for lab, x in CF_LINES.items():
        ax.axvline(x, color="0.3", ls=":", lw=0.8)
        ax.text(x + 40, 1.795, lab + " (CF assumed)", rotation=90, va="top", fontsize=6.5, color="0.3")
    ax.axhline(1.65, color="tab:red", ls="--", lw=0.7)
    ax.text(7600, 1.653, "AP1000 design limit 1.65", ha="right", va="bottom", fontsize=7, color="tab:red")
    ax.set_xlabel("Cycle length, EFPD")
    ax.set_ylabel("$F_{\\Delta H}$, core, BOL, 1000 ppm")
    ax.set_xlim(-150, 8200)
    ax.set_ylim(1.49, 1.80)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Campaign 8 archive: 60 evaluations, 19 feasible, two controllability tiers")
    save(fig, out, "c8_post_front_two_tier")
    plt.close(fig)
    return front4, front2


def fig_margin_vs_boron(plt, rows, bt, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=False)
    enr = np.array([rows[d]["enrich"] for d in bt])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(enr.min(), enr.max())
    for ax, key, title in ((axs[0], "m16", "ALL-RE, sixteen assemblies (RE1 to RE4)"),
                           (axs[1], "m8", "RE1 + RE2, eight assemblies")):
        for d in bt:
            y = [bt[d][key][p] for p in PPM]
            ls = "-" if d in FRONT else "--"
            ax.plot(PPM, y, ls, marker="o", ms=3, lw=1.1, color=cmap(norm(rows[d]["enrich"])))
            ax.annotate(str(d), (PPM[-1], y[-1]), xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                        color=cmap(norm(rows[d]["enrich"])), va="center")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(MARGIN, color="k", lw=0.8, ls="--")
        ax.axvline(1000, color="0.4", lw=0.7, ls=":")
        ax.set_xlabel("Soluble boron, ppm")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-50, 1650)
    axs[0].set_ylabel("BOL hold-down margin $-\\rho$, pcm (2D, one seed)")
    axs[0].text(20, MARGIN + 250, f"screen $k \\leq {K_SCREEN}$ ({MARGIN:.0f} pcm)", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02)
    cb.set_label("Enrichment, wt% $^{235}$U (assembly base)")
    axs[1].text(0.02, 0.97, "solid: front, dashed: retained two-bank designs", transform=axs[1].transAxes,
                fontsize=6.5, va="top")
    save(fig, out, "c8_post_margin_vs_boron")
    plt.close(fig)


def fig_boron_required(plt, rows, bt, out):
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3))
    ds = sorted(bt, key=lambda d: rows[d]["enrich"])
    e = [rows[d]["enrich"] for d in ds]
    ax = axs[0]
    ax.plot(e, [bt[d]["ppm16"] for d in ds], "o", color="tab:blue", ms=5, label="sixteen CRAs (RE1 to RE4)")
    ax.plot(e, [bt[d]["ppm8"] for d in ds], "s", color="tab:green", ms=5, label="eight CRAs (RE1 + RE2)")
    for d in ds:
        if bt[d]["ppm8_extrap"]:
            ax.annotate("extrap.", (rows[d]["enrich"], bt[d]["ppm8"]), xytext=(0, 5), textcoords="offset points",
                        fontsize=5.5, ha="center", color="tab:green")
        ax.annotate(str(d), (rows[d]["enrich"], bt[d]["ppm8"]), xytext=(3, -8), textcoords="offset points", fontsize=6.5)
    ax.axhline(1000, color="0.3", ls=":", lw=0.8)
    ax.text(3.4, 1060, "screen at 1000 ppm", fontsize=7)
    ax.set_xlabel("Enrichment, wt% $^{235}$U")
    ax.set_ylabel("Boron for a 1000 pcm margin, ppm")
    ax.set_ylim(0, 3600)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) boron needed at BOL", fontsize=9)
    # worths across the archive
    ax = axs[1]
    E = np.array([rows[i]["enrich"] for i in rows])
    ax.scatter(E, [rows[i]["W16"] for i in rows], s=12, color="tab:blue", label="ALL-RE worth, 60 designs")
    ax.scatter(E, [rows[i]["W8"] for i in rows], s=12, color="tab:green", marker="s", label="RE1 + RE2 worth, 60 designs")
    ratio = np.array([rows[i]["W8"] / rows[i]["W16"] for i in rows])
    ax.text(0.03, 0.55, f"$W_8 / W_{{16}}$ = {ratio.mean():.3f} $\\pm$ {ratio.std():.3f}\n(min {ratio.min():.3f}, max {ratio.max():.3f})",
            transform=ax.transAxes, fontsize=7.5, va="top")
    ax.set_xlabel("Enrichment, wt% $^{235}$U")
    ax.set_ylabel("Bank worth at 1000 ppm, pcm")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(b) regulating-bank worth", fontsize=9)
    ax = axs[2]
    ax.plot(e, [bt[d]["wb_per_ppm"] for d in ds], "o", color="tab:red", ms=4)
    for d in ds:
        ax.annotate(str(d), (rows[d]["enrich"], bt[d]["wb_per_ppm"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Enrichment, wt% $^{235}$U")
    ax.set_ylabel("Boron worth 0 to 1000 ppm, pcm/ppm")
    ax.set_title("(c) boron worth, eleven designs", fontsize=9)
    ax.set_ylim(0, 10)
    fig.tight_layout()
    save(fig, out, "c8_post_boron_required_and_worths")
    plt.close(fig)


def fig_zero_boron(plt, rows, bt, zs, out):
    ds = sorted(zs, key=lambda d: rows[int(d)]["enrich"])
    x = np.arange(len(ds))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w, [bt[int(d)]["m16"][0.0] if int(d) in bt else np.nan for d in ds], w, color="0.75", label="2D, sweep, one seed")
    ax.bar(x, [zs[d]["ARI_margin2D_pcm"] for d in ds], w, color="tab:blue", label="2D, stage D, two seeds")
    ax.bar(x + w, [zs[d]["ARI_margin3D_pcm"] for d in ds], w, color="tab:orange", label="3D hardware, stage D, two seeds")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(MARGIN, color="k", lw=0.8, ls="--")
    ax.text(-0.45, MARGIN + 250, f"screen $k \\leq {K_SCREEN}$", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{rows[int(d)]['enrich']:.1f} wt%\n{'front' if int(d) in FRONT else 'two-bank'}" for d in ds], fontsize=7)
    ax.set_ylabel("ALL-RE hold-down margin at 0 ppm, BOL, pcm")
    ax.set_title("Zero-boron hold-down by the four regulating banks alone (stage D)")
    ax.legend(fontsize=7, loc="upper right")
    for xi, d in zip(x, ds):
        m3 = zs[d]["ARI_margin3D_pcm"]
        c = "tab:green" if m3 >= MARGIN else ("tab:orange" if m3 > 0 else "tab:red")
        ax.plot(xi + w, m3 + (250 if m3 >= 0 else -250), marker="v" if m3 >= 0 else "^", color=c, ms=4)
    save(fig, out, "c8_post_zero_boron_margin")
    plt.close(fig)


def fig_margins_1000(plt, rows, c1000, out):
    ds = sorted(c1000, key=lambda d: rows[int(d)]["enrich"])
    x = np.arange(len(ds)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 1.5 * w, [c1000[d]["ARI_margin2D_pcm"] for d in ds], w, color="tab:blue", label="ALL-RE, 2D")
    ax.bar(x - 0.5 * w, [c1000[d]["ARI_margin3D_pcm"] for d in ds], w, color="tab:blue", alpha=0.45, label="ALL-RE, 3D hardware")
    ax.bar(x + 0.5 * w, [c1000[d]["RE12_margin2D_pcm"] for d in ds], w, color="tab:green", label="RE1 + RE2, 2D")
    ax.bar(x + 1.5 * w, [c1000[d]["RE12_margin3D_pcm"] for d in ds], w, color="tab:green", alpha=0.45, label="RE1 + RE2, 3D hardware")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(MARGIN, color="k", lw=0.8, ls="--")
    ax.text(-0.45, MARGIN + 350, f"screen $k \\leq {K_SCREEN}$", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{rows[int(d)]['enrich']:.1f} wt%" for d in ds], fontsize=7)
    ax.set_ylabel("Hold-down margin at 1000 ppm, BOL, pcm")
    ax.set_title("1000 ppm confirmation, two seeds: the two-bank reading in 2D and in 3D")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    for xi, d in zip(x, ds):
        m = c1000[d]["RE12_margin3D_pcm"]
        if m >= MARGIN:
            ax.annotate("two-bank\nin 3D", (xi + 1.5 * w, m), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6, color="tab:green")
    save(fig, out, "c8_post_margins_1000_2d3d")
    plt.close(fig)


def fig_constraints(plt, rows, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.5))
    kc = np.array([rows[i]["k_core"] for i in rows])
    ka = np.array([rows[i]["k_allre"] for i in rows])
    k12 = np.array([rows[i]["k_re12"] for i in rows])
    feas = np.array([rows[i]["feasible"] for i in rows])
    only_ctrl = np.array([rows[i]["g_ctrl"] > 0 and rows[i]["g_kmax"] <= 0 and rows[i]["g_kmin"] <= 0 for i in rows])
    for ax, y, lab, c in ((axs[0], ka, "$k_\\mathrm{ALL\\text{-}RE}$ (sixteen assemblies)", "tab:blue"),
                          (axs[1], k12, "$k_\\mathrm{RE12}$ (eight assemblies)", "tab:green")):
        ax.scatter(kc[~feas], y[~feas], marker="x", color="0.55", s=20, label="infeasible")
        ax.scatter(kc[feas], y[feas], color=c, s=22, label="feasible")
        if ax is axs[0]:
            ax.scatter(kc[only_ctrl], y[only_ctrl], facecolors="none", edgecolors="tab:red", s=60,
                       label="rejected by $g_\\mathrm{ctrl}$ alone (12)")
        p = np.polyfit(kc, y, 1)
        xx = np.linspace(kc.min(), kc.max(), 50)
        ax.plot(xx, np.polyval(p, xx), "-", color="0.2", lw=0.8)
        ax.text(0.03, 0.95, f"fit: {p[0]:.3f} $k_\\mathrm{{core}}$ {p[1]:+.3f}\nresidual sd {1e5*np.std(y-np.polyval(p,kc)):.0f} pcm\n$k_\\mathrm{{core}}$ at 0.99: {(0.99-p[1])/p[0]:.3f}",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.axhline(0.99, color="tab:red", ls="--", lw=0.8)
        ax.axvline(1.166, color="k", ls=":", lw=0.8)
        ax.axvline(1.02, color="k", ls=":", lw=0.8)
        ax.set_xlabel("$k_\\mathrm{core}$ at BOL, all rods out, 1000 ppm")
        ax.set_ylabel(lab)
        ax.legend(fontsize=6.5, loc="lower right")
    axs[0].text(1.167, 0.72, "$k_{\\max}$ = 1.166", rotation=90, fontsize=7, va="bottom")
    axs[0].text(1.021, 0.72, "$k_{\\min}$ = 1.02", rotation=90, fontsize=7, va="bottom")
    axs[0].text(0.84, 0.993, "$g_\\mathrm{ctrl}$: $k \\leq 0.99$", fontsize=7, color="tab:red")
    axs[1].text(0.9, 0.993, "$g_\\mathrm{ctrl12}$ (recorded only)", fontsize=7, color="tab:red")
    axs[0].set_title("(a) the binding screen", fontsize=9)
    axs[1].set_title("(b) the two-bank reading", fontsize=9)
    fig.tight_layout()
    save(fig, out, "c8_post_constraint_structure")
    plt.close(fig)


def fig_core_map(plt, out):
    RE1 = {(2, 2), (2, 3), (3, 2), (3, 3)}
    RE2 = {(1, 1), (1, 4), (4, 1), (4, 4)}
    RE3 = {(1, 2), (2, 4), (4, 3), (3, 1)}
    RE4 = {(1, 3), (3, 4), (4, 2), (2, 1)}
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.0))
    col = {"RE1": "#1f77b4", "RE2": "#2ca02c", "RE3": "#9edae5", "RE4": "#c5b0d5", "SH": "#f0f0f0"}
    ring = {"C": "#c7e9c0", "M": "#fdd0a2", "P": "#dadaeb"}
    mult = {"C": 0.720, "M": 0.893, "P": 1.150}
    for ax, mode, title in ((axs[0], "banks", "(a) control-rod banks"),
                            (axs[1], "re12", "(b) two-bank state RE1 + RE2"),
                            (axs[2], "rings", "(c) enrichment rings, multipliers")):
        for i in range(6):
            for j in range(6):
                if (i, j) in {(0, 0), (0, 5), (5, 0), (5, 5)}:
                    continue
                pos = (i, j)
                if mode == "rings":
                    d = math.hypot(i - 2.5, j - 2.5)
                    z = "C" if d < 1.0 else ("M" if d < 2.3 else "P")
                    c, t = ring[z], f"{z}\n{mult[z]:.3f}"
                else:
                    b = "RE1" if pos in RE1 else "RE2" if pos in RE2 else "RE3" if pos in RE3 else "RE4" if pos in RE4 else "SH"
                    if mode == "re12" and b not in ("RE1", "RE2"):
                        c, t = "white", b if b == "SH" else "out"
                    else:
                        c, t = col[b], b
                ax.add_patch(plt.Rectangle((j, 5 - i), 1, 1, facecolor=c, edgecolor="k", lw=0.6))
                ax.text(j + 0.5, 5.5 - i, t, ha="center", va="center", fontsize=6.5)
        ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=9)
    fig.text(0.5, 0.01, "6x6-minus-corners map, 32 assemblies. ALL-RE = RE1 to RE4 (16). SH = shutdown, never inserted in the screens. "
             "Ring C = RE1 positions, ring M = RE2 to RE4, ring P = SH.", ha="center", fontsize=6.5)
    save(fig, out, "c8_post_core_banks_rings")
    plt.close(fig)


def fig_hv(plt, ck, out):
    hv = ck["hv_history"]
    ref = ck["hv_ref"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = list(range(len(hv)))
    ax.plot(x, hv, "o-", color="tab:blue")
    for xi, h in zip(x, hv):
        ax.annotate(f"{h:.1f}", (xi, h), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["DOE\n36"] + [f"block {i}\n+6" for i in range(1, len(hv))], fontsize=7.5)
    ax.set_ylabel("Hypervolume (feasible front)")
    ax.set_title(f"Campaign 8: reference point (EFPD {-ref[0]:.0f}, $F_{{\\Delta H}}$ {ref[1]:.3f})", fontsize=8.5)
    ax.set_ylim(min(hv) - 15, max(hv) + 15)
    save(fig, out, "c8_post_hv_history")
    plt.close(fig)


def fig_reflector(plt, rows, dc, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3))
    E = np.array([rows[i]["enrich"] for i in rows])
    R = np.array([rows[i]["refl"] for i in rows])
    F = np.array([rows[i]["F"] for i in rows])
    K = np.array([rows[i]["k_core"] for i in rows])
    feas = np.array([rows[i]["feasible"] for i in rows])
    cmap = plt.get_cmap("viridis"); norm = plt.Normalize(E.min(), E.max())
    for ax, y, lab in ((axs[0], F, "$F_{\\Delta H}$ (core, BOL, 1000 ppm)"), (axs[1], K, "$k_\\mathrm{core}$ (BOL, 1000 ppm)")):
        ax.scatter(R[~feas], y[~feas], c=E[~feas], cmap=cmap, norm=norm, marker="x", s=18)
        ax.scatter(R[feas], y[feas], c=E[feas], cmap=cmap, norm=norm, s=24, edgecolors="k", linewidths=0.4)
        ax.set_xlabel("Reflector thickness $t_\\mathrm{refl}$, cm")
        ax.set_ylabel(lab)
        ax.set_xlim(1.8, 5.9)
    X = np.column_stack([np.ones(len(E)), E, [rows[i]["gd_wt"] for i in rows], R, [rows[i]["pins"] for i in rows]])
    b = np.linalg.lstsq(X, F, rcond=None)[0]
    axs[0].text(0.03, 0.95, f"multiple regression slope in $t_\\mathrm{{refl}}$: {b[3]:+.4f} per cm\n({b[3]*3.66:+.3f} over the box, seed sigma about 0.01)",
                transform=axs[0].transAxes, fontsize=7, va="top")
    if dc:
        for d, r in dc.items():
            i = int(d)
            axs[0].annotate("", xy=(r["design"]["refl_thick"], r["ARO_2D"]["F"]), xytext=(rows[i]["refl"], rows[i]["F"]),
                            arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))
            axs[1].annotate("", xy=(r["design"]["refl_thick"], r["ARO_2D"]["k"]), xytext=(rows[i]["refl"], rows[i]["k_core"]),
                            arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))
            axs[1].annotate(str(i), (rows[i]["refl"], rows[i]["k_core"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
        fig.text(0.02, -0.02, "Red arrows: stage G, reflector thinned to 3.969 cm (3.7 cm downcomer), two-dimensional solve at 1000 ppm. Circles: feasible. Crosses: infeasible.", fontsize=6.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02); cb.set_label("Enrichment, wt%")
    save(fig, out, "c8_post_reflector_leverage")
    plt.close(fig)


# ----------------------------------------------------------------------------
# tables and numbers
# ----------------------------------------------------------------------------
def write_tables(rows, bt, zs, front4, front2, sig, dc, nop, marg, out: Path, c1000=None):
    L = []
    L.append("% c8_post_tables.tex -- generated by c8_post_figures.py. Requires booktabs.")
    # Table 1: two-tier candidates
    L.append(r"""
\begin{table}[htbp]
  \centering
  \footnotesize
  \caption{Campaign 8 candidates in two controllability tiers, archive values at 1000\,ppm. $W_{16}$ and $W_8$ are the beginning-of-life worths of the four regulating banks and of RE1 + RE2. Tier A is the feasible front. Tier B is the front of the designs that also satisfy $k_\mathrm{RE12} \leq 0.99$.}
  \label{tab:c8-post-tiers}
  \setlength{\tabcolsep}{3pt}\scriptsize
  \begin{tabular}{@{}c l r r r r r r r r r r r@{}}
    \toprule
    Tier & ID & $e$ & $e_\mathrm{max}$ & Gd & pins & $t_\mathrm{refl}$ & EFPD & $B_\mathrm{EOC}$ & $F_{\Delta H}$ & $k_\mathrm{core}$ & $W_{16}$ & $W_8$ \\
         &    & wt\% & wt\% & wt\% &  & cm &  & MWd/kg &  &  & pcm & pcm \\
    \midrule""")
    for tier, ids in (("A", front4), ("B", front2)):
        for i in ids:
            r = rows[i]
            L.append(f"    {tier} & {i} & {r['enrich']:.2f} & {r['e_max']:.2f} & {r['gd_wt']:.2f} & {r['pins']} & {r['refl']:.2f} & {r['efpd']:.0f} & {r['bu']:.1f} & {r['F']:.3f} & {r['k_core']:.4f} & {r['W16']:.0f} & {r['W8']:.0f} \\\\")
        if tier == "A":
            L.append("    \\midrule")
    L.append(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    # Table 2: boron sweep
    L.append(r"""
\begin{table}[htbp]
  \centering
  \footnotesize
  \caption{Boron sweep of the eleven designs, two-dimensional core, one seed, $150\,000$ particles and 200 batches. $W_B$ is the differential boron worth between 0 and 1000\,ppm. $M_{16}$ and $M_8$ are the hold-down margins under the four regulating banks and under RE1 + RE2, positive when subcritical. $c_8$ and $c_{16}$ are the boron concentrations at which the two margins reach the screen value $k \\leq 0.99$, that is 1010\,pcm on this axis, by linear interpolation between the sweep points (e: extrapolated beyond 1500\,ppm).}
  \label{tab:c8-post-boron}
  \setlength{\tabcolsep}{2.5pt}\scriptsize
  \begin{tabular}{@{}l l r r r r r r r r r@{}}
    \toprule
    ID & set & $W_B$ & $M_{16}(0)$ & $M_{16}(1000)$ & $M_8(1000)$ & $M_8(1500)$ & $c_{16}$ & $c_8$ & $W_8/W_{16}$ & share \\
       &     & pcm/ppm & pcm & pcm & pcm & pcm & ppm & ppm &  &  \\
    \midrule""")
    for d in sorted(bt, key=lambda d: rows[d]["enrich"]):
        b = bt[d]
        L.append(f"    {d} & {'front' if d in FRONT else 'two-bank'} & {b['wb_per_ppm']:.2f} & {b['m16'][0.0]:.0f} & {b['m16'][1000.0]:.0f} & {b['m8'][1000.0]:.0f} & {b['m8'][1500.0]:.0f} & {b['ppm16']:.0f}{'e' if b['ppm16_extrap'] else ''} & {b['ppm8']:.0f}{'e' if b['ppm8_extrap'] else ''} & {b['ratio']:.3f} & {b['share']:.3f} \\\\")
    L.append(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    # Table 3: zero boron 2D vs 3D
    L.append(r"""
\begin{table}[htbp]
  \centering
  \footnotesize
  \caption{Zero-boron hold-down by the four regulating banks alone at beginning of life, stage D, two seeds per solve. $L_\mathrm{ax}$ is $k_\mathrm{2D}/k_\mathrm{3D}$ for the rodded state with the benchmark rod stack (bottom 12.04\,cm unrodded, 30.48\,cm Ag-In-Cd, 77.48\,cm B$_4$C). The reading holds when the margin reaches the screen value $k \\leq 0.99$, that is 1010\,pcm. It is the necessary condition for boron-free operation, not its demonstration.}
  \label{tab:c8-post-zeroboron}
  \begin{tabular}{@{}l r r r r r r l@{}}
    \toprule
    ID & $e$ & $k_\mathrm{2D}$ & $k_\mathrm{3D}$ & $M_{16}^\mathrm{2D}$ & $M_{16}^\mathrm{3D}$ & $L_\mathrm{ax}$ & reading \\
       & wt\% &  &  & pcm & pcm &  &  \\
    \midrule""")
    for d in sorted(zs, key=lambda d: rows[int(d)]["enrich"]):
        z = zs[d]
        m3 = z["ARI_margin3D_pcm"]
        cls = CLASS_OK if m3 >= MARGIN else (CLASS_MARGINAL if m3 > 0 else CLASS_NO)
        L.append(f"    {d} & {rows[int(d)]['enrich']:.2f} & {z['ARI_2D']['k']:.4f} & {z['ARI_3Dhw']['k']:.4f} & {z['ARI_margin2D_pcm']:.0f} & {m3:.0f} & {z['ARI_Lax_hw']:.4f} & {cls} \\\\")
    L.append(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    # Table 4: seed statistics
    L.append(r"""
\begin{table}[htbp]
  \centering
  \footnotesize
  \caption{Seed-to-seed standard deviation of the core solves at $150\,000$ particles, 200 batches and 80 inactive batches, pooled over the replicate pairs of the post-analysis. The reported Monte Carlo standard deviation of $k$ is about 22\,pcm per solve.}
  \label{tab:c8-post-sigma}
  \begin{tabular}{@{}l l r r r@{}}
    \toprule
    State & Model & pairs & $\sigma_\mathrm{seed}(k)$, pcm & $\sigma_\mathrm{seed}(F_{\Delta H})$ \\
    \midrule""")
    for lab, s in sig.items():
        L.append(f"    {lab[0]} & {lab[1]} & {s['n']} & {s['sk']:.0f} & {s['sF']:.4f} \\\\")
    L.append(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    # Table 5: reflector / downcomer and parked rods
    if dc:
        L.append(r"""
\begin{table}[htbp]
  \centering
  \footnotesize
  \caption{Stage G, reflector thinned to 3.969\,cm so that the downcomer is 3.7\,cm at pitch 1.26\,cm, all rods out, 1000\,ppm, two seeds. The native column is the archive core solve ($100\,000 \times 170$). $\Delta\rho$ is the reactivity change of the two-dimensional core.}
  \label{tab:c8-post-downcomer}
  \setlength{\tabcolsep}{3pt}\scriptsize
  \begin{tabular}{@{}l r r r r r r r@{}}
    \toprule
    ID & $t_\mathrm{refl}$ & downcomer & $k_\mathrm{2D}$ & $k_\mathrm{2D}$ & $\Delta\rho$ & $F_{\Delta H}$ & $F_{\Delta H}$ \\
       & native, cm & native, cm & native & at 3.969 & pcm & native & at 3.969 \\
    \midrule""")
        for d, r in dc.items():
            i = int(d)
            L.append(f"    {i} & {rows[i]['refl']:.2f} & {7.669-rows[i]['refl']:.2f} & {rows[i]['k_core']:.4f} & {r['ARO_2D']['k']:.4f} & {rho(r['ARO_2D']['k'])-rho(rows[i]['k_core']):+.0f} & {rows[i]['F']:.3f} & {r['ARO_2D']['F']:.3f} \\\\")
        L.append(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    if c1000:
        L.append(r"""
\begin{table}[htbp]
  \centering
  \footnotesize
  \caption{Three-dimensional confirmation at 1000\,ppm, two seeds per solve, $150\,000 \times 200$. $L_\mathrm{ax} = k_\mathrm{2D}/k_\mathrm{3D}$ for each rod state. The margins are $-\rho$ of the rodded core, positive when subcritical. $F_{\Delta H}^\mathrm{3D}$ is axially integrated. The two-bank column applies the screen $k_\mathrm{RE12} \\leq 0.99$, that is 1010\,pcm. Design 13 was not run at 1000\,ppm.}
  \label{tab:c8-post-3d1000}
  \setlength{\tabcolsep}{3pt}\scriptsize
  \begin{tabular}{@{}l r r r r r r r r r r r@{}}
    \toprule
    ID & $e$ & $L_\mathrm{ax}^\mathrm{ARO}$ & $L_\mathrm{ax}^\mathrm{ALL\text{-}RE}$ & $L_\mathrm{ax}^\mathrm{RE12}$ & $M_{16}^\mathrm{2D}$ & $M_{16}^\mathrm{3D}$ & $M_8^\mathrm{2D}$ & $M_8^\mathrm{3D}$ & $F_{\Delta H}^\mathrm{2D}$ & $F_{\Delta H}^\mathrm{3D}$ & two-bank in 3D \\
       & wt\% &  &  &  & pcm & pcm & pcm & pcm &  &  &  \\
    \midrule""")
        for d in sorted(c1000, key=lambda d: rows[int(d)]["enrich"]):
            z = c1000[d]
            L.append(f"    {d} & {rows[int(d)]['enrich']:.2f} & {z['ARO_Lax_hw']:.4f} & {z['ARI_Lax_hw']:.4f} & {z['RE12_Lax_hw']:.4f} & {z['ARI_margin2D_pcm']:.0f} & {z['ARI_margin3D_pcm']:.0f} & {z['RE12_margin2D_pcm']:.0f} & {z['RE12_margin3D_pcm']:.0f} & {z['ARO_2D']['F']:.3f} & {z['ARO_3Dhw']['F']:.3f} & {'yes' if z['RE12_margin3D_pcm'] >= MARGIN else 'no'} \\\\")
        L.append(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    (out / "c8_post_tables.tex").write_text("\n".join(L) + "\n")
    print("  wrote c8_post_tables.tex")


def write_numbers(ck, rows, bt, zs, front4, front2, sig, dc, nop, marg, out: Path, c1000=None):
    A = ck["all_raw"]
    n = dict(
        ctrl_screen_k=K_SCREEN,
        ctrl_margin_pcm_delta_k=1000.0,
        ctrl_margin_pcm_reactivity=MARGIN,
        c8_n_eval=len(A), c8_n_feasible=sum(1 for i in rows if rows[i]["feasible"]),
        c8_n_twobank_feasible=sum(1 for i in rows if rows[i]["twobank"]),
        c8_front_ids=front4, c8_twobank_front_ids=front2,
        c8_hv_history=ck["hv_history"], c8_hv_ref=ck["hv_ref"],
        c8_rejected_only_gctrl=[i for i in rows if rows[i]["g_ctrl"] > 0 and rows[i]["g_kmax"] <= 0 and rows[i]["g_kmin"] <= 0],
        c8_rejected_only_gkmax=[i for i in rows if rows[i]["g_kmax"] > 0 and rows[i]["g_ctrl"] <= 0],
        c8_W8_over_W16_mean=float(np.mean([rows[i]["W8"] / rows[i]["W16"] for i in rows])),
        c8_W8_over_W16_sd=float(np.std([rows[i]["W8"] / rows[i]["W16"] for i in rows])),
        c8_W16_range_pcm=[float(min(rows[i]["W16"] for i in rows)), float(max(rows[i]["W16"] for i in rows))],
        c8_wall_total_h=sum(p["t_eval_s"] for p in ck["phase_log"]) / 3600,
        c8_t_eval_mean_min=float(np.mean([r["t_eval_s"] for r in A]) / 60),
        c8_t_ctrl_mean_min=float(np.mean([r["t_ctrl_s"] + r["t_ctrl12_s"] for r in A]) / 60),
        c8_ctrl_share_of_true_cost=float(np.mean([r["t_ctrl_s"] + r["t_ctrl12_s"] for r in A]) / np.mean([r["t_eval_s"] + r["t_ctrl_s"] + r["t_ctrl12_s"] for r in A])),
        boron=dict({str(d): dict(wb_per_ppm=b["wb_per_ppm"], ppm8=b["ppm8"], ppm8_extrap=b["ppm8_extrap"], ppm16=b["ppm16"],
                                 m16_0=b["m16"][0.0], m16_1000=b["m16"][1000.0], m8_1000=b["m8"][1000.0], m8_1500=b["m8"][1500.0],
                                 share=b["share"], ratio=b["ratio"]) for d, b in bt.items()}),
        zero_boron_3d=dict({d: dict(m2d=z["ARI_margin2D_pcm"], m3d=z["ARI_margin3D_pcm"], lax=z["ARI_Lax_hw"]) for d, z in zs.items()}),
        sigma_seed=sig,
    )
    if dc:
        n["downcomer37"] = {d: dict(k2d=r["ARO_2D"]["k"], F2d=r["ARO_2D"]["F"], k3d=r["ARO_3Dhw"]["k"], F3d=r["ARO_3Dhw"]["F"], lax=r["ARO_Lax_hw"]) for d, r in dc.items()}
    if nop:
        n["noparked_47"] = dict(lax=nop["47"]["ARO_Lax_hw"], k3d=nop["47"]["ARO_3Dhw"]["k"], F3d=nop["47"]["ARO_3Dhw"]["F"])
        if c1000 and "47" in c1000:
            n["parked_effect_47_pcm"] = 1e5 * (c1000["47"]["ARO_Lax_hw"] - nop["47"]["ARO_Lax_hw"])
    if c1000:
        n["confirm3d_1000"] = {d: dict(lax_aro=z["ARO_Lax_hw"], lax_allre=z["ARI_Lax_hw"], lax_re12=z["RE12_Lax_hw"],
                                       m16_2d=z["ARI_margin2D_pcm"], m16_3d=z["ARI_margin3D_pcm"],
                                       m8_2d=z["RE12_margin2D_pcm"], m8_3d=z["RE12_margin3D_pcm"],
                                       F2d=z["ARO_2D"]["F"], F3d=z["ARO_3Dhw"]["F"], k2d=z["ARO_2D"]["k"], k3d=z["ARO_3Dhw"]["k"])
                               for d, z in c1000.items()}
        n["twobank_in_3d_1000ppm"] = sorted([int(d) for d, z in c1000.items() if z["RE12_margin3D_pcm"] >= MARGIN])
        n["lax_aro_range"] = [min(z["ARO_Lax_hw"] for z in c1000.values()), max(z["ARO_Lax_hw"] for z in c1000.values())]
        n["lax_allre_range"] = [min(z["ARI_Lax_hw"] for z in c1000.values()), max(z["ARI_Lax_hw"] for z in c1000.values())]
        n["lax_re12_range"] = [min(z["RE12_Lax_hw"] for z in c1000.values()), max(z["RE12_Lax_hw"] for z in c1000.values())]
    if marg:
        by = {}
        for k, r in marg.items():
            by.setdefault(k.split("|")[0], []).append(r["keff"])
        n["marginal_rescore"] = {d: dict(k_mean=float(np.mean(v)), k_sd_pcm=float(1e5 * np.std(v, ddof=1)), n=len(v),
                                          g_ctrl=float(np.mean(v) - 0.99)) for d, v in by.items()}
    (out / "c8_post_numbers.json").write_text(json.dumps(n, indent=1))
    print("  wrote c8_post_numbers.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="list inputs, write nothing")
    ap.add_argument("--out", default="figs_c8_post")
    a = ap.parse_args()
    files = check_tree()
    if a.check:
        return 0
    ck = load(files["archive"], required=True)
    runs = load(files["boron_runs"], required=True)
    zs = load(files["zero_summary"], required=True)
    zr = load(files["zero_runs"])
    dc = load(files["dc37_summary"])
    nop = load(files["noparked_summary"])
    marg = load(files["marginal_runs"])
    c1000 = load(files["c1000_summary"])
    out = Path(a.out); out.mkdir(exist_ok=True)
    rows = archive_table(ck)
    bt = boron_table(runs)
    # seed statistics
    sig = {}
    if zr:
        for mode in ("2D", "3Dhw"):
            pk = seed_pairs(zr, "ARI", mode, "keff"); pf = seed_pairs(zr, "ARI", mode, "fdh")
            sig[("ALL-RE, 0 ppm", mode)] = dict(n=len(pk), sk=1e5 * pooled_sigma(pk), sF=pooled_sigma(pf))
    aro_runs = {}
    for p in ("confirm3d_c8_dc37/runs.json", "confirm3d_c8_noparked/runs.json", "confirm3d_c8/runs.json"):
        r = load(p)
        if r:
            aro_runs.update({f"{p}:{k}": v for k, v in r.items()})
    if aro_runs:
        for mode in ("2D", "3Dhw"):
            by = {}
            for key, rec in aro_runs.items():
                f = key.split(":")[1].split("|")
                if f[1] == "ARO" and f[2] == mode and f[9] == "150000":
                    by.setdefault(key.split(":")[0] + f[0], []).append(rec)
            pk = [[r["keff"] for r in v] for v in by.values() if len(v) == 2]
            pf = [[r["fdh"] for r in v] for v in by.values() if len(v) == 2]
            sig[("ARO, 1000 ppm", mode)] = dict(n=len(pk), sk=1e5 * pooled_sigma(pk), sF=pooled_sigma(pf))
    sigF = sig.get(("ARO, 1000 ppm", "2D"), {}).get("sF", 0.0083)
    sigma_archive = sigF * math.sqrt((150000 * 120) / (100000 * 110))   # scale to the archive fidelity 100000 x 170 (60 inactive)
    print(f"seed sigma F_dH (ARO 2D) {sigF:.4f} at 150000x200 -> about {sigma_archive:.4f} at the archive fidelity")
    plt = setup_mpl()
    front4, front2 = fig_front(plt, rows, out, sigma_archive)
    fig_margin_vs_boron(plt, rows, bt, out)
    fig_boron_required(plt, rows, bt, out)
    fig_zero_boron(plt, rows, bt, zs, out)
    if c1000:
        fig_margins_1000(plt, rows, c1000, out)
    fig_constraints(plt, rows, out)
    fig_core_map(plt, out)
    fig_hv(plt, ck, out)
    fig_reflector(plt, rows, dc, out)
    sig_tex = {k: v for k, v in sig.items()}
    write_tables(rows, bt, zs, front4, front2, sig_tex, dc, nop, marg, out, c1000)
    write_numbers(ck, rows, bt, zs, front4, front2, {f"{k[0]} {k[1]}": v for k, v in sig.items()}, dc, nop, marg, out, c1000)
    if c1000 is None:
        print("NOTE: confirm3d_c8/summary.json not found. The 1000 ppm three-state 3D table and the parked-rod "
              "comparison of stage E need it. Rerun this script when it is present.")
    else:
        missing = [d for d in ELEVEN if str(d) not in c1000]
        if missing:
            print(f"NOTE: confirm3d_c8 lacks the 1000 ppm confirmation of designs {missing}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
