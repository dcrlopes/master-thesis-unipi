#!/usr/bin/env python3
"""
analyze_kt_enrich.py
====================
Post-processing of the schema-4 table written by sweep_ktarget_enrich.py.

The sweep reports one planar fit over all 42 nodes. That fit answers the
question it was built for, namely the enrichment slope in the pcm per
weight per cent convention of validate_ktarget_burnup.py, but it hides
three things the thesis needs.

  1. Whether a plane is adequate. The residual root mean square of the
     plane is compared with a quadratic in enrichment.
  2. The honest uncertainty on the enrichment slope. The seven nodes of
     one enrichment row share a single assembly solve, so their errors
     are correlated and the 39 degrees of freedom of the plane are
     optimistic. The slope is refitted on the six row means, where the
     shared assembly error is carried once.
  3. Whether the reflector slope depends on enrichment, that is whether
     the surface has an interaction term the bilinear table would keep
     and the plane would discard.

Usage, from the repository root, no OpenMC needed:
    python analyze_kt_enrich.py --table kt_enrich/ktarget_table_c8_enrich.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def wls(A, y, sd=None):
    """Least squares with optional per-point sigma. Returns beta, sigma_beta,
    residual rms and degrees of freedom."""
    A = np.asarray(A, float); y = np.asarray(y, float)
    if sd is None:
        W = np.eye(len(y))
    else:
        W = np.diag(1.0 / np.asarray(sd, float) ** 2)
    XtWXi = np.linalg.inv(A.T @ W @ A)
    beta = XtWXi @ (A.T @ W @ y)
    r = y - A @ beta
    dof = len(y) - A.shape[1]
    if sd is None:
        s2 = float((r ** 2).sum() / dof)
        cov = s2 * XtWXi
        rms = math.sqrt(s2)
    else:
        cov = XtWXi
        rms = math.sqrt(float((r ** 2).sum() / dof))
    return beta, np.sqrt(np.diag(cov)), rms, dof, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="kt_enrich/ktarget_table_c8_enrich.json")
    ap.add_argument("--c8-table", default="ktarget_table_c8.json")
    a = ap.parse_args()

    t = json.loads(Path(a.table).read_text())
    e = np.asarray(t["enrich_wt"], float)
    r = np.asarray(t["refl_thick_cm"], float)
    Z = np.asarray(t["k_target_2d"], float)          # [i_enrich][j_refl]
    SD = np.asarray(t["k_target_sd"], float)
    kinf = np.asarray(t["k_inf_assembly"], float)
    lf0 = float(Z.mean())

    def pcm(x):
        return 1e5 * x / lf0

    print("=" * 74)
    print("1. ROW MEANS AND LOCAL ENRICHMENT SLOPE")
    print("=" * 74)
    print(f"{'e[wt%]':>7} {'k_inf':>9} {'mean LF':>10} {'sd LF':>8} "
          f"{'local slope [pcm/wt%]':>24}")
    rowmean = Z.mean(axis=1)
    rowsd = SD.mean(axis=1) / math.sqrt(Z.shape[1])
    for i in range(len(e)):
        loc = ""
        if i:
            s = pcm((rowmean[i] - rowmean[i - 1]) / (e[i] - e[i - 1]))
            loc = f"{s:+9.1f}  ({e[i-1]:g} to {e[i]:g})"
        print(f"{e[i]:>7.2f} {kinf[i]:>9.5f} {rowmean[i]:>10.6f} "
              f"{rowsd[i]:>8.6f} {loc:>24}")

    print("\n" + "=" * 74)
    print("2. IS A PLANE ADEQUATE?")
    print("=" * 74)
    E = np.repeat(e, len(r)); T = np.tile(r, len(e))
    L = Z.reshape(-1); LSD = SD.reshape(-1)
    one = np.ones_like(E)
    models = {
        "plane      a + b e + c t      ": np.vstack([one, E, T]).T,
        "quadratic  a + b e + q e^2 + c t": np.vstack([one, E, E ** 2, T]).T,
        "with inter a + b e + c t + d e t": np.vstack([one, E, T, E * T]).T,
    }
    for name, A in models.items():
        beta, sb, rms, dof, res = wls(A, L)
        print(f"{name}  rms {pcm(rms):5.0f} pcm on {dof:2d} dof")
        if "quadratic" in name:
            print(f"    quadratic term q = {beta[2]:+.3e} +/- {sb[2]:.3e} "
                  f"per wt%^2  ({abs(beta[2]/sb[2]):.1f} sigma)")
        if "inter" in name:
            print(f"    interaction  d = {beta[3]:+.3e} +/- {sb[3]:.3e} "
                  f"per wt% per cm  ({abs(beta[3]/sb[3]):.1f} sigma)")
    print(f"mean single-solve uncertainty {pcm(LSD.mean()):.0f} pcm")

    print("\n" + "=" * 74)
    print("3. ENRICHMENT SLOPE ON THE SIX ROW MEANS (correlated errors)")
    print("=" * 74)
    A = np.vstack([np.ones_like(e), e]).T
    beta, sb, rms, dof, res = wls(A, rowmean)
    print(f"linear    slope {pcm(beta[1]):+.1f} +/- {pcm(sb[1]):.1f} pcm/wt%"
          f"   rms {pcm(rms):.0f} pcm on {dof} dof")
    A2 = np.vstack([np.ones_like(e), e, e ** 2]).T
    b2, s2, rms2, dof2, _ = wls(A2, rowmean)
    print(f"quadratic rms {pcm(rms2):.0f} pcm on {dof2} dof, "
          f"q = {b2[2]:+.3e} +/- {s2[2]:.3e} ({abs(b2[2]/s2[2]):.1f} sigma)")
    for x in (4.0, 6.0, 8.0, 11.0):
        d = pcm(b2[1] + 2 * b2[2] * x)
        print(f"    local slope of the quadratic at {x:5.1f} wt%: {d:+7.1f} pcm/wt%")
    print("plane value reported by the sweep : "
          f"{t['plane_fit']['enrich_slope_pcm_per_wt']:+.1f} +/- "
          f"{t['plane_fit']['enrich_slope_sd_pcm_per_wt']:.1f} pcm/wt%")
    print("validate_ktarget_burnup on designs: -62.5 +/- 5.0 pcm/wt%")

    print("\n" + "=" * 74)
    print("4. REFLECTOR SLOPE ROW BY ROW")
    print("=" * 74)
    Ar = np.vstack([np.ones_like(r), r]).T
    print(f"{'e[wt%]':>7} {'slope [pcm/cm]':>16} {'range 2.0-5.66 [pcm]':>22}")
    for i in range(len(e)):
        b, s, _, _, _ = wls(Ar, Z[i])
        print(f"{e[i]:>7.2f} {pcm(b[1]):>+10.1f} +/- {pcm(s[1]):<4.1f} "
              f"{pcm(b[1]) * (r[-1] - r[0]):>20.0f}")
    try:
        c8 = json.loads(Path(a.c8_table).read_text())
        s8 = c8["fit_slope_per_cm"]
        print(f"Campaign 8 table (two rings, 4.55 and 4.05 wt%): "
              f"{pcm(s8):+.1f} pcm/cm, range {pcm(s8) * (r[-1] - r[0]):.0f} pcm")
    except Exception as exc:
        print(f"(Campaign 8 table not read: {exc})")


if __name__ == "__main__":
    main()
