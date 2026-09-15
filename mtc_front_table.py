#!/usr/bin/env python3
r"""
mtc_front_table.py -- refit every moderator temperature coefficient scan with
its uncertainty, and tabulate each design's boron demand against its own
measured ceiling.

The scan script reports the crossing by linear interpolation between the two
bracketing points and stores no uncertainty. That is enough when the margin
is large and useless when it is not. This script refits the whole set of
points of each scan by weighted least squares, using the sd column of the
report, and propagates the fit covariance into the crossing.

    MTC(c) = m c + b          crossing  c* = -b / m

    dc*/dm = b / m^2
    dc*/db = -1 / m
    var(c*) = (dc*/dm)^2 var(m) + (dc*/db)^2 var(b)
              + 2 (dc*/dm)(dc*/db) cov(m, b)

Terms:
    MTC   moderator temperature coefficient, pcm per K
    c     soluble boron concentration, ppm
    m     fitted slope, pcm per K per ppm
    b     fitted intercept, pcm per K
    c*    crossing concentration, the ceiling of that lattice, ppm

It needs numpy only, so it runs anywhere. No OpenMC, no cross sections.

Usage, from the repository root:

    python -c "import numpy; print('env ok')" && \
        python mtc_front_table.py --checkpoint out_c9/optimization_checkpoint.json \
          --glob 'mtc_c9_*_core3d' --pressure 12.8 --out c9_post

Flags:
    --checkpoint PATH  archive that holds the boron demand of each design
    --glob PAT         directories to read (default mtc_c9_*_core3d)
    --pressure P       keep only scans at this pressure, blank keeps all
    --out DIR          where to write the table, LaTeX and JSON
    --figure           also write ceiling_vs_inventory.pdf
    --k-sigma K        verdict is unresolved inside K sigma (default 2.0)
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-+][\d.]+)\s+([\d.]+)\s")


def read_scan(d):
    """Return the points of one scan directory, or None if unreadable."""
    rep, summ = d / "report.txt", d / "summary.json"
    if not rep.exists():
        return None
    pts = []
    for line in rep.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if m:
            pts.append((float(m.group(1)), float(m.group(4)), float(m.group(5))))
    if len(pts) < 2:
        return None
    meta = {}
    if summ.exists():
        try:
            meta = json.loads(summ.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    quoted = None
    m = re.search(r"CROSSING CONCENTRATION:\s*(\d+)", rep.read_text(encoding="utf-8"))
    if m:
        quoted = float(m.group(1))
    return dict(dir=d.name, points=sorted(pts), meta=meta, quoted=quoted)


def crossing(points, mode="local", inflate=True):
    """Weighted straight-line fit of MTC against boron, with the crossing.

    mode "local" keeps the three points nearest the sign change, because the
    coefficient is mildly curved in boron and a line through the whole sweep
    averages that curvature into the crossing. mode "global" keeps them all.
    inflate scales the uncertainty by sqrt(chi2/dof) when that exceeds one,
    which widens the error bar when the straight line does not describe the
    points, instead of reporting a precision the fit has not earned.
    """
    c = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    s = np.array([p[2] for p in points], float)
    if mode == "local" and len(c) > 3:
        neg = np.where(y < 0)[0]
        pos = np.where(y >= 0)[0]
        if len(neg) and len(pos):
            lo, hi = neg[-1], pos[0]
            order = np.argsort(np.abs(c - 0.5 * (c[lo] + c[hi])))[:3]
            keep = np.sort(order)
            c, y, s = c[keep], y[keep], s[keep]
    s = np.where(s > 0, s, np.nanmedian(s[s > 0]) if np.any(s > 0) else 1.0)
    if len(c) < 3:                      # two points, no degrees of freedom
        m = (y[1] - y[0]) / (c[1] - c[0])
        b = y[0] - m * c[0]
        cstar = -b / m
        sig = float(np.hypot(*s) / abs(m))
        return cstar, sig, m, b, float("nan"), len(c)
    W = np.diag(1.0 / s ** 2)
    A = np.vstack([c, np.ones_like(c)]).T
    cov = np.linalg.inv(A.T @ W @ A)
    m, b = cov @ (A.T @ W @ y)
    cstar = -b / m
    dm, db = b / m ** 2, -1.0 / m
    var = dm ** 2 * cov[0, 0] + db ** 2 * cov[1, 1] + 2 * dm * db * cov[0, 1]
    resid = y - (m * c + b)
    chi2 = float(np.sum((resid / s) ** 2) / (len(c) - 2))
    sig = float(np.sqrt(max(var, 0.0)))
    if inflate and chi2 > 1.0:
        sig *= float(np.sqrt(chi2))
    return float(cstar), sig, float(m), float(b), chi2, len(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--glob", default="mtc_c9_*_core3d")
    ap.add_argument("--pressure", default="12.8")
    ap.add_argument("--out", default="c9_post")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--k-sigma", type=float, default=2.0)
    ap.add_argument("--fit", choices=["local", "global"], default="local")
    ap.add_argument("--no-inflate", action="store_true")
    a = ap.parse_args()

    ck_path = pathlib.Path(a.checkpoint)
    if not ck_path.exists():
        print(f"ABORT: {ck_path} not found. Run from the repository root.")
        return 1
    ck = json.loads(ck_path.read_text(encoding="utf-8"))
    arc = ck["all_raw"]
    cons = ck.get("constraint_names", [])

    scans = [s for s in (read_scan(d) for d in sorted(pathlib.Path(".").glob(a.glob)))
             if s]
    if a.pressure:
        scans = [s for s in scans
                 if abs(s["meta"].get("pressure_mpa", -1) - float(a.pressure)) < 1e-6]
    if not scans:
        print(f"ABORT: no readable scan matched {a.glob} at {a.pressure} MPa")
        return 1

    rows = []
    for s in scans:
        idx = s["meta"].get("idx")
        if idx is None:
            m = re.search(r"_d(\d+)_", s["dir"])
            idx = int(m.group(1)) if m else None
        if idx is None or idx >= len(arc):
            print(f"  skipped {s['dir']}, no design index")
            continue
        r = arc[idx]
        cstar, sig, m, b, chi2, n = crossing(s["points"], a.fit, not a.no_inflate)
        need = r.get("c_max_ppm") or r.get("c_bol_ppm")   # absent in C8 archives
        margin = None if need is None else cstar - need
        feasible = bool(cons) and all(r.get(g) is not None and r[g] <= 0 for g in cons)
        rows.append(dict(
            idx=idx, dir=s["dir"], gd_wt=r["gd_wt"], pins=float(r["gd_pins_used"]),
            inventory=r["gd_wt"] * float(r["gd_pins_used"]),
            enrich=r["enrich"], ceiling=cstar, sigma=sig, quoted=s["quoted"],
            slope=m, chi2=chi2, n_points=n,
            demand=None if need is None else float(need), margin=margin,
            k_sigma=(margin / sig) if (margin is not None and sig > 0) else None,
            resolved=(abs(margin) > a.k_sigma * sig)
                     if (margin is not None and sig > 0) else None,
            feasible=feasible))

    # Several scans of one lattice are independent measurements of the same
    # quantity. Combine them by weighted mean so the fit statistics below see
    # one row per design, and report the consistency of the merge.
    merged, seen = [], {}
    for x in sorted(rows, key=lambda r: r["idx"]):
        if x["idx"] in seen:
            y = seen[x["idx"]]
            w1, w2 = 1.0 / y["sigma"] ** 2, 1.0 / x["sigma"] ** 2
            c = (y["ceiling"] * w1 + x["ceiling"] * w2) / (w1 + w2)
            sg = (w1 + w2) ** -0.5
            chi = ((y["ceiling"] - c) / y["sigma"]) ** 2 + \
                  ((x["ceiling"] - c) / x["sigma"]) ** 2
            print(f"  merged {y['n_scans'] + 1} scans of design {x['idx']}: "
                  f"{c:.0f} +/- {sg:.0f} ppm, chi2 {chi:.2f} on "
                  f"{y['n_scans']} dof")
            y.update(ceiling=c, sigma=sg, n_scans=y["n_scans"] + 1,
                     merge_chi2=float(chi))
            if y["demand"] is not None:
                y["margin"] = c - y["demand"]
                y["k_sigma"] = y["margin"] / sg
                y["resolved"] = abs(y["margin"]) > a.k_sigma * sg
        else:
            x["n_scans"], x["merge_chi2"] = 1, None
            seen[x["idx"]] = x
            merged.append(x)
    rows = merged
    rows.sort(key=lambda x: x["inventory"])

    L = []
    P = L.append
    P("=" * 96)
    P(f"MTC ceiling of each lattice against its own boron demand, {a.pressure} MPa")
    P("=" * 96)
    where = ("the three points nearest the sign change" if a.fit == "local"
             else "every point of the scan")
    P(f"ceiling refitted by weighted least squares over {where}, uncertainty")
    P("propagated from the fit covariance" + (", inflated by sqrt(chi2/dof)"
      " where that exceeds one" if not a.no_inflate else "") + ".")
    P("'quoted' is the two-point interpolation printed by mtc_scan.py, kept")
    P("for comparison.")
    P("")
    P("  id  Gd wt%  pins  invent  ceiling +/- sd  quoted  demand   margin   k_sig  verdict")
    P("  " + "-" * 92)
    for x in rows:
        if x["margin"] is None:
            v, dem, mar, ks = "no demand in archive", "     -", "      -", "     -"
        else:
            v = "clears" if x["margin"] > 0 else "fails"
            if x["resolved"] is False:
                v = "UNRESOLVED"
            dem = f"{x['demand']:6.0f}"
            mar = f"{x['margin']:+7.0f}"
            ks = f"{x['k_sigma']:+6.1f}"
        q = f"{x['quoted']:6.0f}" if x["quoted"] else "     -"
        P(f"  {x['idx']:2d}  {x['gd_wt']:6.2f}  {x['pins']:4.0f}  {x['inventory']:6.1f}  "
          f"{x['ceiling']:7.0f} +/- {x['sigma']:3.0f}  {q}  {dem}  {mar}  {ks}  {v}")
    P("")

    inv = np.array([x["inventory"] for x in rows])
    ce = np.array([x["ceiling"] for x in rows])
    if len(rows) >= 4:
        def spearman(u, v):
            ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v))
            return float(np.corrcoef(ru, rv)[0, 1])
        gd = np.array([x["gd_wt"] for x in rows])
        pn = np.array([x["pins"] for x in rows])
        P(f"  Spearman ceiling against inventory {spearman(inv, ce):+.3f}, "
          f"against Gd wt% {spearman(gd, ce):+.3f}, against pins {spearman(pn, ce):+.3f}")
        k, q = np.polyfit(np.log(inv), ce, 1)
        res = ce - (k * np.log(inv) + q)
        P(f"  log fit  ceiling = {k:.0f} ln(inventory) + {q:.0f}, "
          f"rms {np.sqrt(np.mean(res ** 2)):.0f} ppm, max {np.max(np.abs(res)):.0f} ppm")
        kl, ql = np.polyfit(inv, ce, 1)
        rl = ce - (kl * inv + ql)
        P(f"  linear in inventory rms {np.sqrt(np.mean(rl ** 2)):.0f} ppm, "
          f"linear in Gd wt% rms "
          f"{np.sqrt(np.mean((ce - np.polyval(np.polyfit(gd, ce, 1), gd)) ** 2)):.0f} ppm")
        P("")
        unres = [x["idx"] for x in rows if x["resolved"] is False]
        P(f"  designs whose verdict is inside {a.k_sigma:g} sigma: "
          f"{unres if unres else 'none'}")
        P(f"  designs that fail their own ceiling: "
          f"{[x['idx'] for x in rows if x['margin'] is not None and x['margin'] < 0]}")

    txt = "\n".join(L)
    print(txt)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mtc_ceiling_table.txt").write_text(txt + "\n", encoding="utf-8")
    (out / "mtc_ceiling_table.json").write_text(json.dumps(rows, indent=1),
                                                encoding="utf-8")

    T = ["\\begin{table}[htbp]", "  \\centering",
         "  \\caption[Boron ceiling of each candidate lattice]{Moderator "
         f"temperature coefficient ceiling of each candidate lattice at "
         f"\\SI{{{a.pressure}}}{{MPa}} in three dimensions, against the boron "
         "demand of the same design. The ceiling is the zero crossing of a "
         "weighted straight-line fit over " + where + ", with the uncertainty "
         "propagated from the fit covariance" + (" and inflated by "
         "$\\sqrt{\\chi^2/\\nu}$ where that exceeds unity"
         if not a.no_inflate else "") + ".}",
         "  \\label{tab:c9-ceilings}",
         "  \\begin{tabular}{rrrrrrr}", "    \\toprule",
         "    id & Gd (wt\\%) & pins & inventory & ceiling (ppm) & "
         "demand (ppm) & margin (ppm) \\\\", "    \\midrule"]
    for x in rows:
        if x["margin"] is None:
            T.append(f"    {x['idx']} & {x['gd_wt']:.2f} & {x['pins']:.0f} & "
                     f"{x['inventory']:.1f} & ${x['ceiling']:.0f} \\pm {x['sigma']:.0f}$ "
                     f"& -- & -- \\\\")
        else:
            T.append(f"    {x['idx']} & {x['gd_wt']:.2f} & {x['pins']:.0f} & "
                     f"{x['inventory']:.1f} & ${x['ceiling']:.0f} \\pm {x['sigma']:.0f}$ & "
                     f"{x['demand']:.0f} & ${x['margin']:+.0f}$ \\\\")
    T += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    (out / "mtc_ceiling_table.tex").write_text("\n".join(T) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/mtc_ceiling_table.txt, .json and .tex")

    if a.figure:
        make_figure(rows, out)
    return 0


def make_figure(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})
    C_CEIL, C_DEM, C_GREY, C_BAD = "#0072B2", "#009E73", "#444444", "#D55E00"

    inv = np.array([x["inventory"] for x in rows])
    ce = np.array([x["ceiling"] for x in rows])
    sg = np.array([x["sigma"] for x in rows])
    rows = [x for x in rows if x["demand"] is not None]
    if len(rows) < 2:
        print("  figure skipped, fewer than two designs with a demand")
        return
    dem = np.array([x["demand"] for x in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

    ax = axes[0]
    xs = np.linspace(inv.min() * 0.85, inv.max() * 1.12, 200)
    k, q = np.polyfit(np.log(inv), ce, 1)
    ax.plot(xs, k * np.log(xs) + q, color=C_GREY, ls="--", lw=1.0,
            label=f"{k:.0f} ln(inventory) + {q:.0f}")
    ax.errorbar(inv, ce, yerr=sg, fmt="o", ms=5, color=C_CEIL, capsize=2,
                lw=1.0, label="measured ceiling")
    ax.plot(inv, dem, "s", ms=5, mfc="white", mec=C_DEM, mew=1.3, label="demand")
    ax.vlines(inv, np.minimum(ce, dem), np.maximum(ce, dem),
              color=C_GREY, lw=0.7, alpha=0.5)
    for x in rows:
        ax.annotate(f"{x['idx']}", (x["inventory"], x["ceiling"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=6.5,
                    color=C_GREY)
    ax.set_xscale("log")
    ax.set_xlabel(r"gadolinia inventory, Gd$_2$O$_3$ wt% $\times$ pins")
    ax.set_ylabel("boron concentration (ppm)")
    ax.set_title("(a) ceiling and demand", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[1]
    mar = np.array([x["margin"] for x in rows])
    col = [C_BAD if m < 0 else C_CEIL for m in mar]
    y = np.arange(len(rows))
    ax.barh(y, mar, color=col, xerr=sg, error_kw=dict(lw=0.8, capsize=2, ecolor=C_GREY))
    ax.axvline(0, color=C_GREY, lw=1.0)
    ax.set_yticks(y, [f"{x['idx']} ({x['gd_wt']:.1f} wt%)" for x in rows], fontsize=7)
    ax.set_xlabel("margin, ceiling minus demand (ppm)")
    ax.set_title("(b) operating margin", fontsize=8, loc="left")
    ax.grid(axis="y", visible=False)

    for axx in axes:
        axx.spines["top"].set_visible(False)
        axx.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "c9_ceiling_inventory.pdf")
    fig.savefig(out / "c9_ceiling_inventory.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out}/c9_ceiling_inventory.pdf")


if __name__ == "__main__":
    sys.exit(main())
