#!/usr/bin/env python3
r"""
c9_axial_rescore.py -- rescore every Campaign 9 design with the axial burnup
correction, and draw the Pareto front before and after it.

METHOD (default, on wks720)
    For each archived design the assembly depletion history is read from the
    campaign's own case directory (openmc_runs_c9/case_NNNN) with
    zoning.read_k_history, and the end of cycle is found with the campaign's
    own crossing function, core_geometry.eoc_crossing_burnup, against

        k_target(B) = k_target x exp(rho_A(B) / 1e5)

    with rho_A(B) from axial_correction.json (c9_axial_correction.py). Every
    design is first rescored with rho_A = 0, which must reproduce its archived
    cycle length: that checks the case-to-design mapping and the reader.

    The other constraints and both objectives are taken from the archive. The
    boron requirement is left at its archived value (the 3D runs changed it
    only for C9-1, by +41 ppm).

    The seven designs depleted directly in 3D are printed next to their
    rescored value. Four of them (C9-47, 35, 27, 1) were used to build the
    correction. C9-24, C9-16 and C9-11 were not, so they test it.

METHOD (--gd-regression, the recommended correction)
    The rescored cycle is archive x (0.9025 - 0.0201 Gd), the regression of
    the 3D cycle length on the gadolinia content over the seven designs
    depleted in 3D (leave-one-out error 71 d rms, 107 d maximum, 2.6 to
    6.9 wt% gadolinia). The measured 3D value is used where it exists.
    On the assembly histories this beats the burnup-indexed table (276 d rms)
    and a table aligned on each design's gadolinium hump (169 d rms). Designs
    whose corrected margin is within 71 d of the floor are flagged.

OUTPUTS (in --out)
    c9_axial_rescore.json, c9_axial_rescore.txt
    c9_front_2d_3d.pdf/.png

USAGE (repository root)
    python c9_axial_rescore.py --correction figs_c9_axial/axial_correction.json --out figs_c9_axial
    python c9_axial_rescore.py --gd-regression --out figs_c9_axial
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

FLOOR, CEIL = 1826.0, 2763.0
MEASURED_3D = {47: 1573.3, 35: 1641.4, 27: 1956.1, 1: 2271.2, 24: 1915.3, 16: 1908.7, 11: 1839.0}
USED_IN_CORRECTION = {47, 35, 27, 1}
GD_FIT = (0.9025, -0.0201)          # ratio = a + b Gd, seven 3D designs
F_MAX = 1.65                        # peaking constraint of Campaign 9

C_OLD = "#D55E00"
C_NEW = "#0072B2"
C_GREY = "#999999"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "pdf.fonttype": 42,
})


def nondominated(pts):
    """pts: {idx: (F, c)} -> sorted list of non-dominated idx (minimise both)."""
    out = []
    for i, (f, c) in pts.items():
        if not any((f2 <= f and c2 <= c) and (f2 < f or c2 < c) for j, (f2, c2) in pts.items() if j != i):
            out.append(i)
    return sorted(out, key=lambda i: pts[i][0])


def fast_k_history(case, spec):
    """(burnup [MWd/kgHM], k) of one campaign case, read with h5py from the
    two datasets that matter, /time and /eigenvalues, instead of loading every
    nuclide through openmc.deplete.Results. Same merge rules as
    zoning.read_k_history: all chunks, zero-filled entries dropped, repeated
    restart states removed, t in days, bu = t x spec / 1000. Any unexpected
    layout raises, and the caller falls back to zoning.read_k_history."""
    import glob
    import h5py
    chunks = sorted(glob.glob(str(Path(case) / "dep_*" / "depletion_results.h5")))
    if not chunks:
        raise FileNotFoundError(case)
    pairs = []
    for ch in chunks:
        with h5py.File(ch, "r") as f:
            t = np.asarray(f["time"])[:, 0] / 86400.0
            ev = np.asarray(f["eigenvalues"])
            k = ev[:, 0] if ev.ndim == 2 else ev[:, 0, 0]      # (steps, 2) on OpenMC 0.15.3
        real = k > 0.0
        pairs.extend(zip(t[real], k[real]))
    pairs.sort()
    t_out, k_out = [], []
    for t, kk in pairs:
        if t_out and abs(t - t_out[-1]) < 1e-6:
            continue
        t_out.append(t); k_out.append(kk)
    return np.asarray(t_out) * spec / 1000.0, np.asarray(k_out)


def rescore_archive(ckpt, corr, workdir):
    import time
    import reactor_model as rm
    import zoning as zn
    import core_geometry as cg
    spec = rm.core_specific_power_w_per_g(rm.Operating(), rm.Geometry17x17())
    grid = np.array(corr["burnup_mwd_kg"]); rho = np.array(corr["rho_A_pcm"])
    out = {}
    t0 = time.time()
    n = len(ckpt["all_raw"])
    for i, r in enumerate(ckpt["all_raw"]):
        case = Path(workdir) / f"case_{i:04d}"
        reader = "h5py"
        try:
            try:
                bu, k = fast_k_history(case, spec)
            except (KeyError, IndexError, ValueError):
                reader = "openmc (slow)"
                bu, k = zn.read_k_history(case, spec)
        except Exception as exc:
            out[i] = dict(status=f"no history ({exc.__class__.__name__})")
            print(f"  [{i + 1:2d}/{n}] C9-{i:<3d} no history ({exc.__class__.__name__})", flush=True)
            continue
        kt = float(r["k_target"])
        b0 = cg.eoc_crossing_burnup(bu, k, kt)
        e0 = b0 * 1000.0 / spec if b0 is not None else None
        mapping_ok = e0 is not None and abs(e0 - float(r["cycle_length"])) < 0.5
        kc = np.asarray(k) / np.exp(np.interp(bu, grid, rho, left=0.0, right=rho[-1]) / 1e5)
        b1 = cg.eoc_crossing_burnup(bu, kc, kt)
        e1 = (b1 * 1000.0 / spec) if b1 is not None else None
        out[i] = dict(status="ok" if mapping_ok else "MAPPING MISMATCH",
                      efpd_archive=float(r["cycle_length"]), efpd_check=e0,
                      efpd_corrected=e1, bu_last=float(bu[-1]), reader=reader)
        print(f"  [{i + 1:2d}/{n}] C9-{i:<3d} archive {r['cycle_length']:7.1f}  reread "
              f"{'--' if e0 is None else f'{e0:7.1f}'}  corrected {'--' if e1 is None else f'{e1:7.1f}'}  "
              f"{out[i]['status']}  ({reader}, {time.time() - t0:.0f} s)", flush=True)
    return out


def project(ckpt):
    out = {}
    for i, r in enumerate(ckpt["all_raw"]):
        if i in MEASURED_3D:
            e, src = MEASURED_3D[i], "measured 3D"
        else:
            e, src = float(r["cycle_length"]) * (GD_FIT[0] + GD_FIT[1] * float(r["gd_wt"])), "projected"
        out[i] = dict(status=src, efpd_archive=float(r["cycle_length"]), efpd_corrected=e)
    return out


def classify(ckpt, res):
    raw, cn = ckpt["all_raw"], ckpt["constraint_names"]
    others = [c for c in cn if c != "g_efpd"]
    old = {i: (r["peaking"], r["c_max"]) for i, r in enumerate(raw) if all(r[c] <= 0 for c in cn)}
    new = {}
    for i, r in enumerate(raw):
        e = res.get(i, {}).get("efpd_corrected")
        if e is not None and e >= FLOOR and all(r[c] <= 0 for c in others):
            new[i] = (r["peaking"], r["c_max"])
    return old, new, nondominated(old), nondominated(new)


def place_labels(fig, ax, xy, texts, colours, obstacles, polyline, orders=None):
    """First candidate offset whose box is clear of every point, front and label."""
    CAND = [(6, 5), (-34, 5), (6, -13), (-34, -13), (10, -4), (-38, -4), (-14, 10), (-14, -18),
            (6, 16), (-34, 16), (6, -24), (-34, -24), (22, 9), (-50, 9), (22, -17), (-50, -17)]
    import numpy as np
    r = fig.canvas.get_renderer()
    obst = np.vstack([ax.transData.transform(obstacles), ax.transData.transform(polyline)])
    # the limit annotation already on the axes is an obstacle like any other
    taken = [x.get_window_extent(r) for x in list(ax.texts)]
    frame = ax.get_window_extent()
    orders = orders or [None] * len(texts)
    for (x, y), s, c, order in zip(xy, texts, colours, orders):
        a = ax.annotate(s, (x, y), xytext=(order or CAND)[0], textcoords="offset points",
                        fontsize=7, color=c)
        cands = order or CAND
        best, best_score = cands[0], None
        for dx, dy in cands:
            a.xyann = (dx, dy)
            b = a.get_window_extent(r).expanded(1.2, 1.3)
            hit = ((obst[:, 0] > b.x0) & (obst[:, 0] < b.x1)
                   & (obst[:, 1] > b.y0) & (obst[:, 1] < b.y1)).any()
            inside = frame.contains(b.x0, b.y0) and frame.contains(b.x1, b.y1)
            # a label over another label is worse than a label over a point
            score = (4 * sum(b.overlaps(o) for o in taken) + 2 * int(hit)
                     + (0 if inside else 8))
            if best_score is None or score < best_score:
                best, best_score = (dx, dy), score
            if score == 0:
                break
        a.xyann = best
        taken.append(a.get_window_extent(r).expanded(1.2, 1.3))


def figure(ckpt, res, old, new, f_old, f_new, out, provisional):
    raw = ckpt["all_raw"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.6))

    infeas = [i for i in range(len(raw)) if i not in old and i not in new]
    only_old = [i for i in old if i not in new]
    meas = [i for i in MEASURED_3D if i in old or i in new]
    xo = [old[i][0] for i in f_old]; yo = [old[i][1] for i in f_old]
    xn = [new[i][0] for i in f_new]; yn = [new[i][1] for i in f_new]
    labelled = sorted(set(f_old) | set(f_new))

    # the window of panel (b): the two fronts with room for the labels
    fx = xo + xn
    fy = yo + yn
    ZOOM = (min(fx) - 0.016, max(fx) + 0.022, min(fy) - 280, max(fy) + 250)

    for panel in (ax, bx):
        panel.axhline(CEIL, color="#CC79A7", ls="--", lw=1.1)
        panel.plot([raw[i]["peaking"] for i in infeas], [raw[i]["c_max"] for i in infeas], "x",
                   color="#B0B0B0", ms=5, mew=1.0,
                   label=f"Infeasible on the campaign ({len(infeas)})")
        panel.plot([old[i][0] for i in only_old], [old[i][1] for i in only_old], "o", mfc="none",
                   mec=C_GREY, ms=5, mew=1.1,
                   label="Feasible on the assembly, misses the floor in 3D")
        panel.plot([new[i][0] for i in new], [new[i][1] for i in new], "o", mfc="none",
                   mec=C_NEW, ms=5, label="Meets the floor with the axial correction")
        panel.step(xo, yo, where="post", color=C_OLD, lw=1.3)
        panel.plot(xo, yo, "s", color=C_OLD, ms=6,
                   label="Front, assembly cycle length (campaign)")
        panel.step(xn, yn, where="post", color=C_NEW, lw=1.3)
        panel.plot(xn, yn, "o", color=C_NEW, ms=6.5,
                   label="Front, axially corrected cycle length")
        panel.plot([raw[i]["peaking"] for i in meas], [raw[i]["c_max"] for i in meas], "o",
                   mfc="none", mec="black", ms=10, mew=0.8, label="Depleted directly in 3D")
        panel.set_xlabel(r"Radial peaking $F_{\Delta H}$ (campaign)")
        panel.set_ylabel(r"$c_\mathrm{max}$ [ppm]")

    # ---- (a) the whole archive, no design labels -------------------------
    xs = [r["peaking"] for r in raw]
    ys = [r["c_max"] for r in raw]
    ax.set_xlim(min(xs) - 0.014, max(xs) + 0.012)
    ax.set_ylim(min(ys) - 260, max(ys) + 320)
    ax.text(0.99, CEIL + 60, "MTC boron limit 2763 ppm", color="#CC79A7", fontsize=7,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2],
                           facecolor="none", edgecolor="0.25", ls="--", lw=1.0, zorder=5))
    ax.annotate("Window of panel (b)", xy=(ZOOM[1], ZOOM[3]), xytext=(1.66, 3450),
                fontsize=8.5, color="0.25", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0))
    ax.axvline(F_MAX, color="#555555", ls="-.", lw=1.0, zorder=2)
    ax.text(F_MAX - 0.006, 0.02, r"$F_{\Delta H} \leq 1.65$", color="#555555", fontsize=7.5,
            rotation=90, ha="right", va="bottom", transform=ax.get_xaxis_transform())
    # A design measured in 3D that meets the corrected floor without reaching the
    # front is named in panel (b) where the window holds it, and here where it
    # does not, so that every such design is named exactly once.
    off_front = [i for i in sorted(new) if i in MEASURED_3D and i not in f_new]
    in_window = [i for i in off_front
                 if ZOOM[0] <= raw[i]["peaking"] <= ZOOM[1]
                 and ZOOM[2] <= raw[i]["c_max"] <= ZOOM[3]]
    outside = [i for i in off_front if i not in in_window]
    place_labels(fig, ax, [(raw[i]["peaking"], raw[i]["c_max"]) for i in outside],
                 [f"C9-{i}" for i in outside], [C_NEW] * len(outside),
                 [(r["peaking"], r["c_max"]) for r in raw],
                 [(F_MAX, min(ys) + (max(ys) - min(ys)) * s / 199.0) for s in range(200)]
                 + [(min(xs) + (max(xs) - min(xs)) * s / 199.0, CEIL) for s in range(200)])
    ax.set_title("(a) The whole archive, 60 designs", fontsize=10)

    # ---- (b) the region of the two fronts, with every front member named --
    bx.set_xlim(ZOOM[0], ZOOM[1])
    bx.set_ylim(ZOOM[2], ZOOM[3])
    bx.text(0.99, CEIL + 25, "MTC boron limit", color="#CC79A7", fontsize=7, va="bottom",
            ha="right", transform=bx.get_yaxis_transform())
    bx.set_title("(b) The region of the two fronts", fontsize=10)

    seg = []
    for xx, yy in ((xo, yo), (xn, yn)):
        for (x0, y0), (x1, y1) in zip(list(zip(xx, yy))[:-1], list(zip(xx, yy))[1:]):
            seg += [(x0 + (x1 - x0) * s / 40.0, y0) for s in range(41)]
            seg += [(x1, y0 + (y1 - y0) * s / 40.0) for s in range(41)]
    seg += [(ZOOM[0] + (ZOOM[1] - ZOOM[0]) * s / 199.0, CEIL) for s in range(200)]
    named = labelled + [i for i in in_window if i not in labelled]
    ABOVE = [(-11, 11), (-11, 17), (-11, -19), (5, 11), (-28, 11)]
    place_labels(fig, bx, [(raw[i]["peaking"], raw[i]["c_max"]) for i in named],
                 [f"C9-{i}" for i in named],
                 [C_OLD if i in f_old else C_NEW for i in named],
                 [(r["peaking"], r["c_max"]) for r in raw], seg,
                 orders=[None if i in labelled else ABOVE for i in named])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    stem = "c9_front_2d_3d" if provisional else "c9_front_2d_3d_table"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"{stem}.{ext}", **kw)
    plt.close(fig)
    return stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--correction", default="figs_c9_axial/axial_correction.json")
    ap.add_argument("--workdir", default="openmc_runs_c9")
    ap.add_argument("--gd-regression", "--projection", dest="projection", action="store_true",
                    help="correct with the gadolinia regression (recommended), no case directories needed")
    ap.add_argument("--out", default="figs_c9_axial")
    a = ap.parse_args(argv)
    ckpt = json.loads(Path(a.checkpoint).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.projection:
        res = project(ckpt)
    else:
        res = rescore_archive(ckpt, json.loads(Path(a.correction).read_text()), a.workdir)
        bad = [i for i, v in res.items() if v["status"] != "ok"]
        print(f"mapping check: {len(res) - len(bad)} of {len(res)} designs reproduce their archived cycle length")
        if bad:
            print("  NOT reproduced:", {i: res[i]["status"] for i in bad})
    old, new, f_old, f_new = classify(ckpt, res)
    raw = ckpt["all_raw"]
    L = [f"=== Campaign 9 rescored with the axial burnup correction "
         f"({'gadolinia regression, measured 3D where available' if a.projection else 'burnup-indexed table on the archive histories, NOT recommended'})",
         f"feasible on the assembly (campaign): {len(old)}   meets the floor with the correction: {len(new)}",
         f"front, assembly cycle length : {['C9-%d' % i for i in f_old]}",
         f"front, corrected cycle length: {['C9-%d' % i for i in f_new]}", "",
         "designs depleted directly in 3D, rescored against measured:" if not a.projection else
         "designs depleted directly in 3D (projection mode uses their measured value as is):"]
    for i, m in MEASURED_3D.items():
        e = res.get(i, {}).get("efpd_corrected")
        tag = "used to build the correction" if i in USED_IN_CORRECTION else "independent test"
        L.append(f"  C9-{i:<3} rescored {e if e is None else round(e):>6} d   measured 3D {m:6.0f} d   "
                 f"difference {'--' if e is None else f'{e - m:+.0f}'} d   ({tag})")
    L += ["", f"{'design':>7} {'enr':>5} {'Gd':>5} {'F_dH':>6} {'c_max':>6} {'archive':>7} {'corrected':>9} {'margin':>7}"]
    for i in sorted(set(old) | set(new), key=lambda j: -(res[j].get("efpd_corrected") or 0)):
        r = raw[i]; e = res[i].get("efpd_corrected")
        flag = ""
        if res[i].get("status") != "measured 3D" and a.projection:
            if e is not None and abs(e - FLOOR) < 71.0:
                flag += "  within the regression error of the floor"
            if not (2.6 <= r["gd_wt"] <= 6.9):
                flag += "  Gd outside the fitted 2.6-6.9 wt%"
        L.append(f"  C9-{i:<3} {r['enrich']:5.2f} {r['gd_wt']:5.2f} {r['peaking']:6.3f} {r['c_max']:6.0f} "
                 f"{r['cycle_length']:7.0f} {e if e is None else round(e):>9} "
                 f"{'--' if e is None else f'{e - FLOOR:+.0f}':>7}" + ("  front" if i in f_new else "")
                 + ("  (3D)" if res[i].get("status") == "measured 3D" else "") + flag)
    txt = "\n".join(L)
    (out / "c9_axial_rescore.txt").write_text(txt + "\n")
    (out / "c9_axial_rescore.json").write_text(json.dumps(dict(
        provisional=a.projection, results={str(k): v for k, v in res.items()},
        front_assembly=f_old, front_corrected=f_new), indent=1))
    stem = figure(ckpt, res, old, new, f_old, f_new, out, a.projection)
    print(txt)
    print(f"\nwrote {out}/c9_axial_rescore.txt/.json and {out}/{stem}.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
