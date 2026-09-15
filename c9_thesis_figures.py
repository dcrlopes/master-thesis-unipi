#!/usr/bin/env python3
"""
c9_thesis_figures.py -- every figure of the Campaign 9 archive, drawn from
out_c9/optimization_checkpoint.json and nothing else.

This script runs NO OpenMC and needs no cross sections. It needs only numpy
and matplotlib, so it runs on the laptop as well as on wks720.

Usage, from the repository root:

    python -c "import numpy, matplotlib; print('env ok')" && \
        python c9_thesis_figures.py --checkpoint out_c9/optimization_checkpoint.json --out figs_c9

Flags:
    --checkpoint PATH   archive to read (default out_c9/optimization_checkpoint.json)
    --out DIR           output directory (default figs_c9)
    --png               also write a 200 dpi PNG next to every PDF
    --numbers           print the numbers quoted in the captions and exit

Every figure is vector PDF at thesis column width, with no title inside the
axes, because the caption carries the title in LaTeX.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- constants --
# All read from meta.campaign9 and meta.limits of the archive, repeated here
# only as defaults for the annotation lines.
CEILING_PPM = 2763.0        # MTC ceiling at 12.8 MPa, design 47 lattice of C8
CEILING_HI_PPM = 2997.0     # MTC ceiling at 15.5 MPa
CLIP_PPM = 6000.0           # boron objective clip guard
EFPD_REQ = 1826.0           # five years at capacity factor 1.0
F_MAX = 1.65
N_DOE = 24
PROXY_A, PROXY_B = 23.292, -0.8496   # C8 proxy law w_B(e) = A e^B
BAND_C, BAND_H = 4.31, 0.35          # enrichment band of sec:res-c9-gd, wt%

# Pareto membership frequency of Table tab:c9-stability, 2000 perturbations
# with sigma_F = 0.010 and sigma_c = 10 ppm. Values are copied from the
# dissertation table so that figure and table cannot drift apart.
STABILITY = [(47, 1.00, True), (34, 1.00, True), (44, 0.59, True),
             (40, 0.54, True), (35, 0.45, True), (29, 0.31, False),
             (30, 0.24, False), (12, 0.21, False), (54, 0.10, False)]

# constraint set of Campaign 8, used to re-score the Campaign 9 archive
C8_CONS = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom", "g_ctrl"]

# colourblind-safe set (Okabe and Ito)
C_FEAS = "#0072B2"
C_INFEAS = "#999999"
C_FRONT = "#D55E00"
C_ACC = "#009E73"
C_WARN = "#CC79A7"
C_GREY = "#444444"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
})

W1, H1 = 5.2, 3.6      # single panel
W2, H2 = 6.9, 3.3      # two panels side by side


# -------------------------------------------------------------------- data --
def load(path):
    d = json.load(open(path))
    R = d["all_raw"]
    cons = d["constraint_names"]
    for i, r in enumerate(R):
        r["idx"] = i
        r["feasible"] = all(r[c] is not None and r[c] <= 0 for c in cons)
        r["phase"] = "DOE" if i < N_DOE else "infill"
        r["d_ppm"] = (r["c_max_ppm"] or 0.0) - (r["c_bol_ppm"] or 0.0)
    return d, R, cons


def pareto(rows, kx="peaking", ky="c_max"):
    """Non dominated set of a minimise/minimise pair."""
    out = []
    for a in rows:
        dominated = False
        for b in rows:
            if b is a:
                continue
            if b[kx] <= a[kx] and b[ky] <= a[ky] and (b[kx] < a[kx] or b[ky] < a[ky]):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return sorted(out, key=lambda r: r[kx])


def save(fig, out, name, png):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    if png:
        fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ----------------------------------------------------------------- figure 1 --
def fig_front(R, out, png):
    """Objective space, the headline result."""
    F = [r for r in R if r["feasible"]]
    I = [r for r in R if not r["feasible"]]
    front = pareto(F)

    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.axhspan(CLIP_PPM * 0.995, CLIP_PPM * 1.04, color=C_INFEAS, alpha=0.18, lw=0)
    ax.text(1.70, CLIP_PPM * 1.012, "clip guard, 6000 ppm", fontsize=7, color=C_GREY,
            ha="right", va="center")

    ax.scatter([r["peaking"] for r in I], [r["c_max_ppm"] for r in I],
               s=22, facecolors="none", edgecolors=C_INFEAS, linewidths=0.8,
               label=f"infeasible ({len(I)})", zorder=2)
    sc = ax.scatter([r["peaking"] for r in F], [r["c_max_ppm"] for r in F],
                    c=[r["cycle_length"] for r in F], cmap="viridis",
                    s=38, edgecolors="k", linewidths=0.4, zorder=3)
    ax.plot([r["peaking"] for r in front], [r["c_max_ppm"] for r in front],
            "-o", color=C_FRONT, ms=6, mfc="none", mew=1.6, lw=1.4,
            label=f"Pareto front ({len(front)})", zorder=4)

    ax.axhline(CEILING_PPM, color=C_WARN, ls="--", lw=1.2)
    ax.axhline(CEILING_HI_PPM, color=C_WARN, ls=":", lw=1.0)
    ax.text(1.735, CEILING_PPM * 0.955, "MTC ceiling 12.8 MPa", fontsize=7,
            color=C_WARN, ha="right", va="top")
    ax.text(1.735, CEILING_HI_PPM * 1.02, "15.5 MPa", fontsize=7,
            color=C_WARN, ha="right", va="bottom")
    ax.axvline(F_MAX, color=C_GREY, ls="-.", lw=1.0)
    ax.text(F_MAX - 0.004, 300, r"$F_{\Delta H}\leq 1.65$", fontsize=7,
            color=C_GREY, rotation=90, ha="right", va="bottom")

    for j, r in enumerate(front):
        off = (7, 7) if j % 2 == 0 else (7, -12)
        ax.annotate(str(r["idx"]), (r["peaking"], r["c_max_ppm"]),
                    textcoords="offset points", xytext=off, fontsize=7, color=C_FRONT)

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("cycle length (EFPD)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(r"core $F_{\Delta H}$ at BOL (dimensionless)")
    ax.set_ylabel(r"$c_{\max}$ (ppm)")
    ax.set_xlim(1.42, 1.74)
    ax.set_ylim(-200, CLIP_PPM * 1.06)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    tidy(ax)
    save(fig, out, "c9_front", png)


# ----------------------------------------------------------------- figure 2 --
def fig_gd_trade(R, out, png):
    """Boron demand against gadolinia, the physical mechanism."""
    F = [r for r in R if r["feasible"]]
    band = sorted([r for r in F if abs(r["enrich"] - BAND_C) <= BAND_H],
                  key=lambda r: r["gd_wt"])

    fig, axes = plt.subplots(1, 2, figsize=(W2, H2))

    ax = axes[0]
    sc = ax.scatter([r["gd_wt"] for r in F], [r["c_max_ppm"] for r in F],
                    c=[r["enrich"] for r in F], cmap="plasma", s=34,
                    edgecolors="k", linewidths=0.4, zorder=3)
    ax.axhline(CEILING_PPM, color=C_WARN, ls="--", lw=1.1)
    ax.axhline(CEILING_HI_PPM, color=C_WARN, ls=":", lw=1.0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("enrichment (wt%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(r"Gd$_2$O$_3$ content (wt%)")
    ax.set_ylabel(r"$c_{\max}$ (ppm)")
    gmax = max(r["gd_wt"] for r in F)
    ax.text(gmax, CEILING_PPM * 1.012, "MTC ceiling 12.8 MPa", fontsize=7,
            color=C_WARN, ha="right", va="bottom")
    ax.text(gmax, CEILING_HI_PPM * 1.015, "MTC ceiling 15.5 MPa", fontsize=7,
            color=C_WARN, ha="right", va="bottom")
    ax.set_title("(a) all feasible designs", fontsize=8, loc="left")
    tidy(ax)

    ax = axes[1]
    x = np.array([r["gd_wt"] for r in band])
    y = np.array([r["c_max_ppm"] for r in band])
    yb = np.array([r["c_bol_ppm"] for r in band])
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, color=C_GREY, ls="--", lw=1.0,
            label=f"fit, {m:.0f} ppm per wt%")
    ax.vlines(x, np.minimum(y, yb), np.maximum(y, yb), color=C_GREY, lw=0.8, alpha=0.6)
    ax.plot(x, yb, "o", ms=5, mfc="white", mec=C_ACC, mew=1.3, label=r"$c_\mathrm{BOL}$")
    ax.plot(x, y, "o", ms=5, color=C_FEAS, label=r"$c_{\max}$")
    for r in band:
        ax.annotate(str(r["idx"]), (r["gd_wt"], r["c_max_ppm"]),
                    textcoords="offset points", xytext=(4, 5), fontsize=6.5, color=C_GREY)
    ax.axhline(CEILING_PPM, color=C_WARN, ls="--", lw=1.1)
    ax.axhline(CEILING_HI_PPM, color=C_WARN, ls=":", lw=1.0)
    ax.text(x.max(), CEILING_PPM * 0.988, "12.8 MPa", fontsize=6.5,
            color=C_WARN, ha="right", va="top")
    ax.text(x.max(), CEILING_HI_PPM * 1.006, "15.5 MPa", fontsize=6.5,
            color=C_WARN, ha="right", va="bottom")
    ax.set_xlabel(r"Gd$_2$O$_3$ content (wt%)")
    ax.set_ylabel("boron concentration (ppm)")
    ax.set_title(rf"(b) band $e = {BAND_C} \pm {BAND_H}$ wt%, {len(band)} designs",
                 fontsize=8, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.margins(y=0.12)
    tidy(ax)

    fig.tight_layout()
    save(fig, out, "c9_gd_trade", png)


# ----------------------------------------------------------------- figure 3 --
def fig_hump(R, out, png):
    """The gadolinium hump and what it does to the objective."""
    F = [r for r in R if r["feasible"]]
    fig, axes = plt.subplots(1, 2, figsize=(W2, H2))

    ax = axes[0]
    ax.axhline(400, color=C_GREY, ls=":", lw=1.0)
    ax.text(0.15, 560, "noise floor 400 pcm", fontsize=7, color=C_GREY, ha="left")
    inb = [r for r in F if abs(r["enrich"] - BAND_C) <= BAND_H]
    outb = [r for r in F if abs(r["enrich"] - BAND_C) > BAND_H]
    ax.scatter([r["gd_wt"] for r in outb], [r["hump_core_pcm"] for r in outb],
               s=30, facecolors="none", edgecolors=C_FEAS, linewidths=1.0,
               label="outside the band")
    ax.scatter([r["gd_wt"] for r in inb], [r["hump_core_pcm"] for r in inb],
               s=34, color=C_FEAS, edgecolors="k", linewidths=0.4,
               label=f"$e = {BAND_C} \\pm {BAND_H}$ wt%")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.set_xlabel(r"Gd$_2$O$_3$ content (wt%)")
    ax.set_ylabel(r"core hump $\Delta\rho_\mathrm{hump}$ (pcm)")
    ax.set_title("(a) hump against gadolinia", fontsize=8, loc="left")
    tidy(ax)

    ax = axes[1]
    d = np.array([r["d_ppm"] for r in F])
    h = np.array([r["hump_core_pcm"] for r in F])
    ax.scatter(h, d, s=34, color=C_ACC, edgecolors="k", linewidths=0.4)
    ok = h > 0
    if ok.sum() >= 2:
        s = np.polyfit(h[ok], d[ok], 1)[0]
        xs = np.linspace(0, h.max() * 1.05, 20)
        ax.plot(xs, s * xs, color=C_GREY, ls="--", lw=1.0,
                label=f"slope {s:.3f} ppm per pcm")
        ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel(r"core hump $\Delta\rho_\mathrm{hump}$ (pcm)")
    ax.set_ylabel(r"$c_{\max}-c_\mathrm{BOL}$ (ppm)")
    ax.set_title("(b) penalty carried into the objective", fontsize=8, loc="left")
    tidy(ax)

    fig.tight_layout()
    save(fig, out, "c9_hump", png)


# ----------------------------------------------------------------- figure 4 --
def fig_learning(R, out, png):
    """Boron demand against evaluation order, DOE against infill."""
    idx = np.array([r["idx"] for r in R])
    c = np.array([r["c_max_ppm"] for r in R])
    feas = np.array([r["feasible"] for r in R])

    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.axvspan(-0.5, N_DOE - 0.5, color=C_INFEAS, alpha=0.14, lw=0)
    ax.text(N_DOE / 2, CLIP_PPM * 1.02, "DOE", fontsize=8, ha="center", color=C_GREY)
    ax.text((N_DOE + len(R)) / 2, CLIP_PPM * 1.02, "infill", fontsize=8,
            ha="center", color=C_GREY)

    ax.scatter(idx[~feas], c[~feas], s=24, facecolors="none", edgecolors=C_INFEAS,
               linewidths=0.8, label="infeasible")
    ax.scatter(idx[feas], c[feas], s=32, color=C_FEAS, edgecolors="k",
               linewidths=0.4, label="feasible")

    run = np.minimum.accumulate(np.where(feas, c, np.inf))
    run = np.where(np.isfinite(run), run, np.nan)
    ax.step(idx, run, where="post", color=C_FRONT, lw=1.4, label="best feasible so far")

    for a, b, lab in [(0, N_DOE, "DOE"), (N_DOE, len(R), "infill")]:
        m = np.mean(c[a:b])
        ax.hlines(m, a - 0.5, b - 0.5, color=C_ACC, ls="--", lw=1.2)
        ax.text(a + 0.5, m + 160, f"mean {m:.0f} ppm", fontsize=7, color=C_ACC, ha="left")

    ax.axhline(CEILING_PPM, color=C_WARN, ls="--", lw=1.1)
    ax.set_xlabel("evaluation index")
    ax.set_ylabel(r"$c_{\max}$ (ppm)")
    ax.set_ylim(-200, CLIP_PPM * 1.08)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5,
              bbox_to_anchor=(1.0, 0.93))
    tidy(ax)
    save(fig, out, "c9_learning", png)


# ----------------------------------------------------------------- figure 5 --
def fig_hv(d, out, png):
    """Hypervolume history and the stagnation that ended the campaign."""
    hv = np.array(d["hv_history"], dtype=float)
    it = np.arange(len(hv))
    gain = np.full_like(hv, np.nan)
    gain[1:] = 100.0 * (hv[1:] - hv[:-1]) / hv[:-1]

    fig, axes = plt.subplots(1, 2, figsize=(W2, H2))
    ax = axes[0]
    ax.plot(it, hv, "o-", color=C_FEAS, ms=5)
    ax.set_xlabel("iteration (0 is the DOE)")
    ax.set_ylabel("hypervolume (ppm $\\cdot$ dimensionless)")
    ax.set_title("(a) hypervolume", fontsize=8, loc="left")
    tidy(ax)

    ax = axes[1]
    ax.bar(it[1:], gain[1:], color=C_FEAS, width=0.6)
    ax.axhline(1.0, color=C_FRONT, ls="--", lw=1.1, label="1 % stopping rule")
    for i in it[1:]:
        ax.annotate(f"{gain[i]:.2f}", (i, gain[i]), textcoords="offset points",
                    xytext=(0, 3), fontsize=6.5, ha="center", color=C_GREY)
    ax.set_xlabel("iteration")
    ax.set_ylabel("hypervolume gain (%)")
    ax.set_title("(b) gain per iteration", fontsize=8, loc="left")
    ax.legend(frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, out, "c9_hv", png)


# ----------------------------------------------------------------- figure 6 --
def fig_design_space(R, out, png):
    """Where the infill sent each design variable."""
    specs = [
        ("enrich", "enrichment (wt%)", (2.0, 17.17)),
        ("gd_wt", r"Gd$_2$O$_3$ (wt%)", (0.0, 8.0)),
        ("refl_thick", "reflector thickness (cm)", (2.0, 5.66)),
        ("gd_pins", "gadolinia pins per assembly", (12.0, 40.0)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.6), sharex=True)
    for ax, (key, lab, (lo, hi)) in zip(axes.ravel(), specs):
        ax.axvspan(-0.5, N_DOE - 0.5, color=C_INFEAS, alpha=0.14, lw=0)
        f = [r for r in R if r["feasible"]]
        i = [r for r in R if not r["feasible"]]
        ax.scatter([r["idx"] for r in i], [r[key] for r in i], s=18,
                   facecolors="none", edgecolors=C_INFEAS, linewidths=0.7)
        ax.scatter([r["idx"] for r in f], [r[key] for r in f], s=26,
                   color=C_FEAS, edgecolors="k", linewidths=0.3)
        ax.axhline(lo, color=C_FRONT, ls=":", lw=1.0)
        ax.axhline(hi, color=C_FRONT, ls=":", lw=1.0)
        ax.set_ylabel(lab)
        ax.margins(y=0.10)
        tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("evaluation index")
    handles = [Line2D([], [], ls=":", color=C_FRONT, label="box bound"),
               Line2D([], [], marker="o", ls="", color=C_FEAS, label="feasible"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=C_INFEAS, label="infeasible")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, out, "c9_design_space", png)


# ----------------------------------------------------------------- figure 7 --
def fig_constraints(R, cons, out, png):
    """Which constraint rejected which design."""
    I = [r for r in R if not r["feasible"]]
    counts, sole = [], []
    for c in cons:
        v = [r for r in I if r[c] > 0]
        counts.append(len(v))
        sole.append(sum(1 for r in v if sum(1 for c2 in cons if r[c2] > 0) == 1))
    order = np.argsort(counts)
    labels = [cons[i].replace("_", r"\_") for i in order]
    labels = [f"$\\mathtt{{{l}}}$" for l in labels]

    fig, ax = plt.subplots(figsize=(W1, 3.0))
    y = np.arange(len(cons))
    ax.barh(y, [counts[i] for i in order], color=C_INFEAS, label="violated")
    ax.barh(y, [sole[i] for i in order], color=C_FRONT, label="only violation")
    for k, i in enumerate(order):
        if counts[i]:
            ax.text(counts[i] + 0.3, k, str(counts[i]), va="center", fontsize=7.5)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"designs rejected, out of {len(I)} infeasible")
    ax.set_xlim(0, max(counts) + 3)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    tidy(ax)
    save(fig, out, "c9_constraints", png)


# ----------------------------------------------------------------- figure 8 --
def fig_wb(R, out, png):
    """Measured differential boron worth against the Campaign 8 proxy law."""
    e = np.array([r["enrich"] for r in R])
    w = np.array([r["w_b_bol_pcm_per_ppm"] for r in R], dtype=float)
    ok = np.isfinite(w) & (w > 0)
    e, w = e[ok], w[ok]
    p = PROXY_A * e ** PROXY_B
    res = 100.0 * (w - p) / p
    rms = float(np.sqrt(np.mean(res ** 2)))
    bias = float(np.mean(res))

    fig, axes = plt.subplots(2, 1, figsize=(W1, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1]})
    ax = axes[0]
    xs = np.linspace(e.min(), e.max(), 200)
    ax.plot(xs, PROXY_A * xs ** PROXY_B, color=C_FRONT, lw=1.4,
            label=r"C8 proxy $w_B = 23.292\,e^{-0.8496}$")
    ax.scatter(e, w, s=30, color=C_FEAS, edgecolors="k", linewidths=0.4,
               label=f"C9 measured ({len(e)} designs)")
    ax.set_ylabel(r"$w_B$ (pcm per ppm)")
    ax.legend(frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.scatter(e, res, s=24, color=C_ACC, edgecolors="k", linewidths=0.3)
    ax.axhspan(-rms, rms, color=C_ACC, alpha=0.15, lw=0)
    ax.text(e.max(), rms + 1.2, f"rms {rms:.1f} %, bias {bias:+.1f} %",
            fontsize=7, ha="right", color=C_ACC)
    ax.set_xlabel("enrichment (wt%)")
    ax.set_ylabel("residual (%)")
    tidy(ax)

    fig.tight_layout()
    save(fig, out, "c9_wb_proxy", png)


# ----------------------------------------------------------------- figure 9 --
def fig_peaking_ratio(R, out, png):
    """Core against assembly peaking, the zoned proxy gap."""
    a = np.array([r["peaking_asm"] for r in R])
    c = np.array([r["peaking"] for r in R])
    ratio = c / a
    m, s = ratio.mean(), ratio.std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(W2, H2))
    ax = axes[0]
    ax.scatter(a, c, s=30, color=C_FEAS, edgecolors="k", linewidths=0.4)
    xs = np.linspace(a.min(), a.max(), 50)
    ax.plot(xs, m * xs, color=C_FRONT, lw=1.3, label=f"ratio {m:.3f}")
    ax.fill_between(xs, (m - s) * xs, (m + s) * xs, color=C_FRONT, alpha=0.15, lw=0)
    ax.set_xlabel(r"assembly $F_{\Delta H}$ (dimensionless)")
    ax.set_ylabel(r"core $F_{\Delta H}$ (dimensionless)")
    ax.legend(frameon=False)
    ax.set_title("(a) core against assembly", fontsize=8, loc="left")
    tidy(ax)

    ax = axes[1]
    ax.hist(ratio, bins=12, color=C_FEAS, edgecolor="white")
    ax.axvline(m, color=C_FRONT, lw=1.3, label=f"mean {m:.3f}")
    ax.axvline(m - s, color=C_FRONT, ls="--", lw=1.0)
    ax.axvline(m + s, color=C_FRONT, ls="--", lw=1.0, label=f"sd {s:.3f}")
    ax.set_xlabel("core to assembly ratio (dimensionless)")
    ax.set_ylabel("designs")
    ax.legend(frameon=False)
    ax.set_title("(b) spread of the ratio", fontsize=8, loc="left")
    tidy(ax)

    fig.tight_layout()
    save(fig, out, "c9_peaking_ratio", png)


# ---------------------------------------------------------------- figure 10 --
def fig_cost(R, out, png):
    """Where the eighteen hours went."""
    parts = [("t_deplete_s", "assembly depletion"),
             ("t_boron_s", "boron measurement"),
             ("t_core_bol_s", "core BOL solve"),
             ("t_ctrl12_s", "two-bank screen"),
             ("t_ctrl_s", "four-bank screen"),
             ("t_asm_bol_s", "assembly BOL solve")]
    tot = sum(r["t_eval_s"] for r in R) / 3600.0
    vals = [sum(r.get(k) or 0.0 for r in R) / 3600.0 for k, _ in parts]
    other = tot - sum(vals)

    fig, axes = plt.subplots(1, 2, figsize=(W2, H2))
    ax = axes[0]
    if abs(other) > 0.05:
        labs = [l for _, l in parts] + ["other"]
        v = vals + [other]
    else:
        labs = [l for _, l in parts]
        v = list(vals)
    o = np.argsort(v)
    ax.barh(np.arange(len(v)), [v[i] for i in o], color=C_FEAS)
    for k, i in enumerate(o):
        ax.text(v[i] + 0.12, k, f"{v[i]:.2f} h ({100*v[i]/tot:.1f} %)",
                va="center", fontsize=7)
    ax.set_yticks(np.arange(len(v)), [labs[i] for i in o], fontsize=7.5)
    ax.set_xlabel(f"wall time (h), total {tot:.2f} h")
    ax.set_xlim(0, max(v) * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) cost by component", fontsize=8, loc="left")
    tidy(ax)

    ax = axes[1]
    t = np.array([r["t_eval_s"] / 60.0 for r in R])
    nb = np.array([r["n_boron_solves"] for r in R])
    for n, col, lab in [(1, C_ACC, "one boron solve"), (2, C_FRONT, "two boron solves")]:
        m = nb == n
        ax.scatter(np.array([r["idx"] for r in R])[m], t[m], s=26, color=col,
                   edgecolors="k", linewidths=0.3, label=lab)
    ax.axvspan(-0.5, N_DOE - 0.5, color=C_INFEAS, alpha=0.14, lw=0)
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("evaluation wall time (min)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(b) cost per design", fontsize=8, loc="left")
    tidy(ax)

    fig.tight_layout()
    save(fig, out, "c9_cost", png)


# ---------------------------------------------------------------- figure 11 --
def fig_efpd(R, out, png):
    """The Campaign 8 objective seen from the Campaign 9 archive."""
    F = [r for r in R if r["feasible"]]
    I = [r for r in R if not r["feasible"]]
    front = pareto(F)

    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.scatter([r["cycle_length"] for r in I], [r["c_max_ppm"] for r in I],
               s=22, facecolors="none", edgecolors=C_INFEAS, linewidths=0.8,
               label="infeasible")
    ax.scatter([r["cycle_length"] for r in F], [r["c_max_ppm"] for r in F],
               s=34, color=C_FEAS, edgecolors="k", linewidths=0.4, label="feasible")
    ax.scatter([r["cycle_length"] for r in front], [r["c_max_ppm"] for r in front],
               s=70, facecolors="none", edgecolors=C_FRONT, linewidths=1.6,
               label="Pareto front")
    ax.axvline(EFPD_REQ, color=C_GREY, ls="-.", lw=1.1)
    ax.text(EFPD_REQ + 90, CLIP_PPM * 0.96, "mission floor 1826 EFPD",
            fontsize=7, color=C_GREY)
    ax.axhline(CEILING_PPM, color=C_WARN, ls="--", lw=1.1)
    ax.text(7600, CEILING_PPM * 0.94, "MTC ceiling", fontsize=7, color=C_WARN, ha="right")
    ax.set_xlabel("cycle length (EFPD)")
    ax.set_ylabel(r"$c_{\max}$ (ppm)")
    ax.set_ylim(-200, CLIP_PPM * 1.06)
    ax.legend(frameon=False, loc="lower right")
    tidy(ax)
    save(fig, out, "c9_efpd_boron", png)



# ---------------------------------------------------------------- figure 12 --
def fig_two_formulations(R, cons, out, png):
    """The Campaign 9 archive scored under the Campaign 8 objectives."""
    f8 = [r for r in R if all(r[c] <= 0 for c in C8_CONS)]
    f9 = [r for r in R if all(r[c] <= 0 for c in cons)]
    front9 = pareto(f9)
    ids9 = {r["idx"] for r in front9}

    # non dominated under maximise EFPD and minimise peaking
    nd8 = []
    for a in f8:
        if not any(b["peaking"] <= a["peaking"] and b["cycle_length"] >= a["cycle_length"]
                   and (b["peaking"] < a["peaking"] or b["cycle_length"] > a["cycle_length"])
                   for b in f8 if b is not a):
            nd8.append(a)

    fig, ax = plt.subplots(figsize=(W1, H1))
    rest = [r for r in R if r not in f8]
    ax.scatter([r["cycle_length"] for r in rest], [r["peaking"] for r in rest],
               marker="x", s=24, color=C_INFEAS, linewidths=0.9,
               label="infeasible under the C8 rules")
    ax.scatter([r["cycle_length"] for r in f8], [r["peaking"] for r in f8],
               s=26, color=C_INFEAS, alpha=0.55, edgecolors="none",
               label=f"C8 feasible ({len(f8)})")
    ax.scatter([r["cycle_length"] for r in nd8], [r["peaking"] for r in nd8],
               marker="s", s=58, facecolors="none", edgecolors=C_GREY, linewidths=1.2,
               label=f"C8 non dominated ({len(nd8)})")
    sc = ax.scatter([r["cycle_length"] for r in front9], [r["peaking"] for r in front9],
                    c=[r["c_max_ppm"] for r in front9], cmap="viridis_r",
                    s=62, edgecolors=C_FRONT, linewidths=1.6, zorder=5,
                    label="C9 Pareto front")
    ax.axvline(EFPD_REQ, color=C_GREY, ls="--", lw=1.1)
    ax.text(EFPD_REQ - 130, 1.86, "mission floor\n1826 EFPD", fontsize=7,
            color=C_GREY, ha="right", va="top")

    # inset on the cluster, where every C9 front member sits
    xs = [r["cycle_length"] for r in front9]
    ys = [r["peaking"] for r in front9]
    axi = ax.inset_axes([0.40, 0.52, 0.57, 0.45])
    near = [r for r in f8 if min(xs) - 120 <= r["cycle_length"] <= max(xs) + 120
            and min(ys) - 0.02 <= r["peaking"] <= max(ys) + 0.02]
    axi.scatter([r["cycle_length"] for r in near], [r["peaking"] for r in near],
                s=22, color=C_INFEAS, alpha=0.6, edgecolors="none")
    nd_near = [r for r in nd8 if r in near]
    axi.scatter([r["cycle_length"] for r in nd_near], [r["peaking"] for r in nd_near],
                marker="s", s=52, facecolors="none", edgecolors=C_GREY, linewidths=1.1)
    axi.scatter(xs, ys, c=[r["c_max_ppm"] for r in front9], cmap="viridis_r",
                s=52, edgecolors=C_FRONT, linewidths=1.4, zorder=5)
    for j, r in enumerate(sorted(front9, key=lambda q: q["peaking"])):
        off = (6, 5) if j % 2 == 0 else (6, -11)
        axi.annotate(f"C9-{r['idx']}", (r["cycle_length"], r["peaking"]),
                     textcoords="offset points", xytext=off, fontsize=6.5, color=C_FRONT)
    axi.axvline(EFPD_REQ, color=C_GREY, ls="--", lw=1.0)
    axi.tick_params(labelsize=6)
    axi.set_xlim(min(xs) - 140, max(xs) + 190)
    axi.set_ylim(min(ys) - 0.022, max(ys) + 0.022)
    axi.grid(alpha=0.2)
    ax.indicate_inset_zoom(axi, edgecolor=C_GREY, alpha=0.6, lw=0.8)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$c_{\max}$ of the C9 front (ppm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("cycle length (EFPD)")
    ax.set_ylabel(r"core $F_{\Delta H}$ at BOL (dimensionless)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=6.8, columnspacing=1.1, handletextpad=0.4)
    ax.set_ylim(1.46, 1.90)
    tidy(ax)
    save(fig, out, "c9_two_formulations", png)


# ---------------------------------------------------------------- figure 13 --
def fig_stability(out, png):
    """Pareto membership frequency under the measurement noise."""
    labs = [f"C9-{i}" for i, _, _ in STABILITY]
    fr = [f for _, f, _ in STABILITY]
    nom = [n for _, _, n in STABILITY]
    y = np.arange(len(labs))[::-1]

    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ax.barh(y, fr, color=[C_FRONT if n else C_INFEAS for n in nom])
    for yy, f in zip(y, fr):
        ax.text(f + 0.015, yy, f"{f:.2f}", va="center", fontsize=7.5)
    ax.axvline(0.5, color=C_GREY, ls=":", lw=1.0)
    ax.text(0.5, len(labs) - 0.2, "simple majority", fontsize=7, color=C_GREY, ha="center")
    ax.set_yticks(y, labs)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("frequency on the non dominated set, 2000 perturbations")
    handles = [Line2D([], [], marker="s", ls="", color=C_FRONT, label="on the nominal front"),
               Line2D([], [], marker="s", ls="", color=C_INFEAS, label="not on the nominal front")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    tidy(ax)
    save(fig, out, "c9_front_stability", png)


# ------------------------------------------------------------------ numbers --
def numbers(d, R, cons):
    F = [r for r in R if r["feasible"]]
    front = pareto(F)
    c = np.array([r["c_max_ppm"] for r in R])
    print(f"evaluations            {len(R)}")
    print(f"feasible               {len(F)}")
    print(f"front                  {[r['idx'] for r in front]}")
    print(f"below ceiling          {sum(1 for r in F if r['c_max_ppm'] <= CEILING_PPM)}")
    print(f"DOE mean c_max         {c[:N_DOE].mean():.0f} ppm")
    print(f"infill mean c_max      {c[N_DOE:].mean():.0f} ppm")
    print(f"hv last three          {d['hv_history'][-3:]}")
    print(f"total evaluator time   {sum(r['t_eval_s'] for r in R)/3600:.2f} h")
    print(f"boron time             {sum(r['t_boron_s'] for r in R)/3600:.2f} h")
    ratio = np.array([r["peaking"] / r["peaking_asm"] for r in R])
    print(f"core/assembly ratio    {ratio.mean():.3f} +/- {ratio.std(ddof=1):.3f}")
    e = np.array([r["enrich"] for r in R])
    w = np.array([r["w_b_bol_pcm_per_ppm"] for r in R], dtype=float)
    p = PROXY_A * e ** PROXY_B
    print(f"proxy rms              {100*np.sqrt(np.mean(((w-p)/p)**2)):.1f} %")
    print(f"infill refl range      {min(r['refl_thick'] for r in R[N_DOE:]):.3f} "
          f"to {max(r['refl_thick'] for r in R[N_DOE:]):.3f} cm")


# ---------------------------------------------------------------------- cli --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--out", default="figs_c9")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--numbers", action="store_true")
    a = ap.parse_args()

    path = pathlib.Path(a.checkpoint)
    if not path.exists():
        print(f"ABORT: {path} not found. Run from the repository root.")
        return 1
    d, R, cons = load(path)
    print(f"host archive : {d['meta'].get('host')}  started {d['meta'].get('started_utc')}")
    print(f"designs      : {len(R)}, constraints {cons}")

    if a.numbers:
        numbers(d, R, cons)
        return 0

    out = pathlib.Path(a.out)
    fig_front(R, out, a.png)
    fig_gd_trade(R, out, a.png)
    fig_hump(R, out, a.png)
    fig_learning(R, out, a.png)
    fig_hv(d, out, a.png)
    fig_design_space(R, out, a.png)
    fig_constraints(R, cons, out, a.png)
    fig_wb(R, out, a.png)
    fig_peaking_ratio(R, out, a.png)
    fig_cost(R, out, a.png)
    fig_efpd(R, out, a.png)
    fig_two_formulations(R, cons, out, a.png)
    fig_stability(out, a.png)
    print(f"\nAll figures in {out.resolve()}")
    numbers(d, R, cons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
