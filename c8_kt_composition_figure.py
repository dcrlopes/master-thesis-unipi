#!/usr/bin/env python3
"""c8_kt_composition_figure.py -- the two results of the Route B
burnup-and-composition check, in one two-panel figure.

Reads kt_burnup/summary.json, written by validate_ktarget_burnup.py, and
draws:

  (a) the BEGINNING-OF-LIFE RESIDUAL against the enrichment, with a
      weighted straight-line fit and its prediction band. This is the
      COMPOSITION dependence the single-reference-composition
      calibration table cannot carry. The vertical marker is the mean
      enrichment of the composition on which the table was built.

  (b) the DRIFT of the leakage factor from beginning of life to end of
      cycle, against the end-of-cycle burnup, with the
      inverse-variance-weighted mean of the ten designs and its one-sigma
      band. This is the BURNUP dependence, the assumption under test.

Outputs <out>.png (300 dpi, for slides) and <out>.pdf (vector, for
LaTeX), and prints the fit, the pooled drift and the chi-squared test.

Usage on wks720, from the repository root, in the openmc-env
environment (matplotlib and scipy only, no OpenMC):

    python c8_kt_composition_figure.py \
        --summary kt_burnup/summary.json \
        --checkpoint out_c8/optimization_checkpoint.json \
        --out figs_c8/c8_kt_composition

For the thesis PDF add --no-panel-titles, so the panel headings do not
duplicate the LaTeX caption. Keep them for slides.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the colourblind-safe palette used by the other campaign figures
C_FIT = "#C1272D"      # fitted line
C_PT = "#0072B2"       # measured points
C_REF = "#E69F00"      # calibration reference composition
C_MEAN = "#009E73"     # pooled mean of the drift
C_GREY = "#666666"

# composition on which ktarget_table_c8.json was built. Read from the
# table when it is available, this is only the fallback.
REF_ENRICH = (4.55, 4.05)


def load(summary_path, checkpoint_path=None):
    """One record per design, with everything the figure needs."""
    rows = json.loads(Path(summary_path).read_text())
    ck = None
    if checkpoint_path and Path(checkpoint_path).exists():
        ck = json.loads(Path(checkpoint_path).read_text())["all_raw"]

    out = []
    for r in rows:
        idx = r["idx"]
        # validate_ktarget_burnup writes the model keys, so the single
        # Campaign 8 enrichment arrives as enrich_inner. The checkpoint
        # is only a cross-check, never a substitute.
        enr = r.get("enrich_inner")
        if enr is None and ck is not None and isinstance(idx, int):
            enr = ck[idx].get("enrich")
        if enr is None:
            raise KeyError(f"design {idx}: no enrichment in the summary "
                           f"and no checkpoint to fall back on")
        kt = float(r["kt_table"])
        # the residual uncertainty follows the leakage-factor uncertainty
        sd_bol = 1e5 * float(r["lf_bol_sd"]) / kt
        out.append(dict(
            idx=idx,
            enrich=float(enr),
            refl=float(r["refl_thick"]),
            gd_wt=float(r.get("gd_wt", float("nan"))),
            b_eoc=float(r["b_eoc"]),
            r_bol=float(r["resid_bol_pcm"]),
            r_bol_sd=sd_bol,
            r_eoc=float(r["resid_eoc_pcm"]),
            drift=float(r["drift_pcm"]),
            drift_sd=float(r["drift_sd_pcm"]),
            defpd=float(r["defpd_days"]),
        ))
    out.sort(key=lambda d: d["enrich"])
    return out


def ref_enrichment(table_path):
    """Mean enrichment of the calibration composition, from the table."""
    p = Path(table_path)
    if not p.exists():
        return sum(REF_ENRICH) / 2.0
    d = json.loads(p.read_text()).get("design", {})
    e_in = float(d.get("enrich_inner", REF_ENRICH[0]))
    e_out = float(d.get("enrich_outer", REF_ENRICH[1]))
    return 0.5 * (e_in + e_out)


def ols_line(x, y):
    """Ordinary least squares y = a + b x, with the covariance matrix.

    Unweighted on purpose. The fit must be reproducible from the reported
    residuals alone, without depending on how the per-point uncertainty
    was propagated, and the residual scatter about the line is itself the
    honest estimate of the spread.
    """
    A = np.vstack([np.ones_like(x), x]).T
    XtXinv = np.linalg.inv(A.T @ A)
    beta = XtXinv @ (A.T @ y)
    resid = y - A @ beta
    dof = len(x) - 2
    s2 = float((resid ** 2).sum() / dof)
    return beta, s2 * XtXinv, s2, dof, resid


def weighted_line(x, y, sd):
    """Weighted least squares, printed as a cross-check only."""
    w = 1.0 / np.asarray(sd, float) ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    beta = cov @ (A.T @ np.diag(w) @ y)
    resid = y - A @ beta
    return beta, cov, float((w * resid ** 2).sum()), len(x) - 2


def band(x_grid, beta, cov):
    """One-sigma band of the fitted line at x_grid."""
    A = np.vstack([np.ones_like(x_grid), x_grid]).T
    var = np.einsum("ij,jk,ik->i", A, cov, A)
    return A @ beta, np.sqrt(var)


def pooled(values, sigmas):
    v = np.asarray(values, float)
    s = np.asarray(sigmas, float)
    w = 1.0 / s ** 2
    mean = float((w * v).sum() / w.sum())
    sd = float(1.0 / math.sqrt(w.sum()))
    chi2 = float((((v - mean) / s) ** 2).sum())
    dof = len(v) - 1
    try:
        from scipy import stats
        p = float(1.0 - stats.chi2.cdf(chi2, dof))
    except Exception:
        p = float("nan")
    return mean, sd, chi2, dof, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="kt_burnup/summary.json")
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json",
                    help="only used to recover the enrichment if the "
                         "summary does not carry it")
    ap.add_argument("--ktarget-table", default="ktarget_table_c8.json",
                    help="read the calibration composition from it")
    ap.add_argument("--out", default="figs_c8/c8_kt_composition",
                    help="output prefix, without extension")
    ap.add_argument("--annotate", type=int, nargs="*", default=None,
                    help="design indices to label in panel (a); default "
                         "is the two enrichment extremes")
    ap.add_argument("--no-panel-titles", action="store_true",
                    help="drop the panel headings, for the thesis PDF")
    args = ap.parse_args()

    rows = load(args.summary, args.checkpoint)
    e_ref = ref_enrichment(args.ktarget_table)

    e = np.array([r["enrich"] for r in rows])
    rb = np.array([r["r_bol"] for r in rows])
    rbsd = np.array([r["r_bol_sd"] for r in rows])
    b = np.array([r["b_eoc"] for r in rows])
    dr = np.array([r["drift"] for r in rows])
    drsd = np.array([r["drift_sd"] for r in rows])

    beta, cov, s2, doff, resid = ols_line(e, rb)
    a_fit, b_fit = float(beta[0]), float(beta[1])
    sa, sb = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    wbeta, wcov, wchi2, wdof = weighted_line(e, rb, rbsd)
    pearson = float(np.corrcoef(e, rb)[0, 1])
    zero_at = -a_fit / b_fit if b_fit != 0 else float("nan")

    d_mean, d_sd, d_chi2, d_dof, d_p = pooled(dr, drsd)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=300)

    # ---- panel (a): composition dependence ------------------------------ #
    xg = np.linspace(e.min() - 0.6, e.max() + 0.6, 200)
    yg, yse = band(xg, beta, cov)
    axa.axhline(0.0, color="black", lw=0.9, zorder=1)
    axa.fill_between(xg, yg - yse, yg + yse, color=C_FIT, alpha=0.13,
                     lw=0, zorder=2)
    axa.plot(xg, yg, color=C_FIT, lw=1.9, zorder=3,
             label=(rf"$r_\mathrm{{BOL}} = {a_fit:+.0f} {b_fit:+.1f}\,e$"
                    "\n"
                    rf"Pearson $r = {pearson:+.3f}$"))
    axa.axvline(e_ref, color=C_REF, lw=1.4, ls="--", zorder=2)
    axa.annotate(f"calibration composition, {e_ref:.2f} wt%",
                 xy=(e_ref, 0.02), xycoords=("data", "axes fraction"),
                 textcoords="offset points", xytext=(8, 2),
                 fontsize=8.5, color="#8a6200", va="bottom", rotation=90)
    axa.errorbar(e, rb, yerr=rbsd, fmt="o", ms=7.0, color=C_PT,
                 ecolor=C_PT, elinewidth=1.1, capsize=3.0, mec="black",
                 mew=0.7, zorder=5, label="Campaign 8 designs (10)")

    todo = args.annotate
    if todo is None:
        todo = [rows[0]["idx"], rows[-1]["idx"]]
    for r in rows:
        if r["idx"] in todo:
            up = r["r_bol"] > 0
            axa.annotate(f"design {r['idx']}",
                         (r["enrich"], r["r_bol"]),
                         textcoords="offset points",
                         xytext=(10, 12 if up else -20),
                         fontsize=9.0, color="#333333",
                         arrowprops=dict(arrowstyle="-", lw=0.7,
                                         color="#888888"))

    axa.set_xlabel("Enrichment $e$ (wt% $^{235}$U)", fontsize=12)
    axa.set_ylabel("Beginning-of-life residual $r_\\mathrm{BOL}$ (pcm)",
                   fontsize=12)
    if not args.no_panel_titles:
        axa.set_title("(a) composition dependence", fontsize=12, pad=8)
    axa.legend(loc="upper right", fontsize=9.0, frameon=True,
               framealpha=0.92, borderpad=0.6)

    # ---- panel (b): burnup dependence ----------------------------------- #
    axb.axhspan(d_mean - d_sd, d_mean + d_sd, color=C_MEAN, alpha=0.16,
                lw=0, zorder=1)
    axb.axhline(d_mean, color=C_MEAN, lw=1.8, zorder=3,
                label=(rf"pooled mean ${d_mean:+.0f} \pm {d_sd:.0f}$ pcm"
                       "\n"
                       rf"$\chi^2/\nu = {d_chi2/d_dof:.2f}$, "
                       rf"$p = {d_p:.2f}$"))
    axb.axhline(0.0, color="black", lw=0.9, ls=":", zorder=2)
    axb.errorbar(b, dr, yerr=drsd, fmt="s", ms=6.5, color=C_PT,
                 ecolor=C_PT, elinewidth=1.1, capsize=3.0, mec="black",
                 mew=0.7, zorder=5, label="measured drift, 3 seeds")
    # greedy declutter: designs 44 and 59 sit almost on top of each other
    placed = []
    for r in sorted(rows, key=lambda q: q["b_eoc"]):
        dx, dy = 7, 5
        for px, py in placed:
            if (abs(r["b_eoc"] - px) < 0.04 * (b.max() - b.min())
                    and abs(r["drift"] - py) < 0.07 * (dr.max() - dr.min())):
                dx, dy = 7, -13
                break
        axb.annotate(str(r["idx"]), (r["b_eoc"], r["drift"]),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=8.0, color=C_GREY)
        placed.append((r["b_eoc"], r["drift"]))

    axb.set_xlabel("End-of-cycle burnup $B_\\mathrm{EOC}$ (MWd/kgHM)",
                   fontsize=12)
    axb.set_ylabel("Drift $\\mathrm{LF}_\\mathrm{EOC} - "
                   "\\mathrm{LF}_\\mathrm{BOL}$ (pcm)", fontsize=12)
    if not args.no_panel_titles:
        axb.set_title("(b) burnup dependence", fontsize=12, pad=8)
    axb.legend(loc="lower right", fontsize=9.0, frameon=True,
               framealpha=0.92, borderpad=0.6)

    for ax in (axa, axb):
        ax.grid(alpha=0.22, lw=0.6)
        ax.tick_params(labelsize=10.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"written -> {args.out}.{ext}")

    # ---- console report -------------------------------------------------- #
    print("\n(a) composition dependence of the leakage factor")
    print(f"  OLS fit       r_BOL = {a_fit:+.1f} ({sa:.1f}) "
          f"{b_fit:+.2f} ({sb:.2f}) * e   [pcm, e in wt%]")
    print(f"  weighted fit  r_BOL = {wbeta[0]:+.1f} {wbeta[1]:+.2f} * e "
          f"(cross-check, chi2/dof {wchi2/wdof:.2f})")
    print(f"  Pearson r                     : {pearson:+.3f}")
    print(f"  rms residual about the line   : {resid.std(ddof=2):.0f} pcm")
    print(f"  residual zero crossing        : {zero_at:.2f} wt%")
    print(f"  calibration composition       : {e_ref:.2f} wt%, "
          f"predicted residual {a_fit + b_fit*e_ref:+.0f} pcm")
    for name in ("refl", "b_eoc", "gd_wt"):
        x = np.array([r[name] for r in rows])
        if np.all(np.isfinite(x)):
            print(f"  Pearson r_BOL vs {name:6s}       : "
                  f"{np.corrcoef(x, rb)[0,1]:+.3f}")

    print("\n(b) burnup dependence of the leakage factor")
    print(f"  pooled drift                  : {d_mean:+.1f} +/- "
          f"{d_sd:.1f} pcm  ({abs(d_mean)/d_sd:.2f} sigma)")
    print(f"  chi2                          : {d_chi2:.2f} on {d_dof} "
          f"dof, chi2/dof {d_chi2/d_dof:.2f}, p = {d_p:.3f}")
    print(f"  two-sigma bound on the mean   : {2*d_sd:.0f} pcm")
    worst = max(rows, key=lambda r: abs(r["drift"]) / r["drift_sd"])
    print(f"  largest |drift|/sigma         : {abs(worst['drift'])/worst['drift_sd']:.2f} "
          f"(design {worst['idx']})")
    print(f"  cycle-length correction range : {min(r['defpd'] for r in rows):+.0f} "
          f"to {max(r['defpd'] for r in rows):+.0f} d")


if __name__ == "__main__":
    main()
