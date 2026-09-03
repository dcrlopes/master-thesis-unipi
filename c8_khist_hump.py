#!/usr/bin/env python3
"""
c8_khist_hump.py -- rebuild k_inf(B) of every Campaign 8 design from the
depletion files that the evaluator left on disk, and locate the mid-cycle
maximum (the gadolinium burnout hump) against which the beginning-of-life
hold-down margins of boron_worth.py and confirm3d.py must be read.

Why. Every controllability margin of the post-analysis is a beginning-of-life
number. If k_inf rises after the gadolinium burns out before it falls, the
minimum margin over the cycle sits at that hump, not at BOL, and the
boron-free sentence of the thesis needs the hump, not the BOL value.

What it reads.  openmc_runs_c8/case_NNNN/dep_MM/depletion_results.h5, the
chunked single-assembly depletion of openmc_evaluator._cycle_length
(reflective assembly, 1000 ppm, 4000 x 60 transport). The case directory is
matched to the archive record by k_bol (the first eigenvalue of dep_00
equals record['k_bol'] to machine precision), never by the directory
number, because n_calls may be offset after a resume.

How the history is stitched. get_keff() is read from every chunk with its
time stamps. Restart chunks may be chunk-local or cumulative depending on
the OpenMC version. Keeping (time, k) pairs, dropping duplicate times and
sorting by time is correct in both cases. Burnup follows from time through
the specific power, which is recovered from the record itself:
    spec_power = bu_eoc_mwd_kg * 1000 / cycle_length   [W/gHM]
and cross-checked against the campaign value 9.9827 W/gHM.

What it reports, per design.
  k_bol        xenon-free BOL eigenvalue, record['k_bol']
  k_xe         first equilibrium-xenon point, k at 0.5 MWd/kgHM
  k_peak, B_peak   maximum of k over the operational trajectory (B >= 0.5)
  hump_vs_bol  1e5 (k_peak - k_bol) / k_bol, pcm. Negative means the
               xenon-free BOL screen bounds the whole cycle. Positive means
               the BOL margins overstate the minimum margin by this amount.
  hump_vs_xe   1e5 (k_peak - k_xe) / k_xe, pcm, the classical Gd hump.
  crossing     EOC burnup where k falls through k_target, for closure
               against record['bu_eoc_mwd_kg'].

Fidelity caveat. The depletion transport is 4000 x 60 with 20 inactive,
so each k carries about 150 to 250 pcm of statistical noise. The hump is
resolved only if it exceeds about 400 pcm. The script prints the
per-design noise estimate (sd of the residual around a smoothing spline is
not attempted, the OpenMC k uncertainty of each state is used instead).

USAGE (wks720, openmc-env, campaign8 branch)
  python -c "import numpy, openmc; print('env ok')" && python c8_khist_hump.py --selftest
  python -c "import numpy, openmc; print('env ok')" && \
      python c8_khist_hump.py --checkpoint out_c8/optimization_checkpoint.json \
          --workdir openmc_runs_c8 --designs 47 42 23 29 21 44 59 1 53 31 13 --dry-run
  python -c "import numpy, openmc; print('env ok')" && \
      python c8_khist_hump.py --checkpoint out_c8/optimization_checkpoint.json \
          --workdir openmc_runs_c8 --designs 47 42 23 29 21 44 59 1 53 31 13 \
          --out khist_c8

No transport is run. Reading eleven designs takes under a minute.
Flags: --checkpoint path, --workdir path, --designs list (default: all
feasible), --out directory, --dry-run (match only), --selftest (no OpenMC),
--spec-power to override the recovered value.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SPEC_POWER_CAMPAIGN = 9.9827      # W/gHM, Campaign 8, 48 MWth over 4808 kgHM
XE_POINT = 0.5                    # MWd/kgHM, first equilibrium-xenon state


def rho_pcm(k: float) -> float:
    return 1e5 * (k - 1.0) / k


# --------------------------------------------------------------------------
# pure functions (tested by --selftest, no OpenMC)
# --------------------------------------------------------------------------
def stitch(chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """chunks: list of (time_days, k, k_sd) arrays, one per dep_MM in order.
    Returns (time_days, k, k_sd) with duplicate times removed (last write
    wins) and sorted by time. Works for chunk-local and cumulative files."""
    seen = {}
    for t, k, s in chunks:
        for ti, ki, si in zip(np.asarray(t, float), np.asarray(k, float), np.asarray(s, float)):
            seen[round(float(ti), 6)] = (float(ki), float(si))
    times = np.array(sorted(seen))
    k = np.array([seen[t][0] for t in times])
    s = np.array([seen[t][1] for t in times])
    return times, k, s


def hump(bu: np.ndarray, k: np.ndarray, k_target: float) -> dict:
    """Locate the operational maximum and the EOC crossing.
    The BOL point (bu == 0) is xenon-free and excluded from the peak search,
    exactly as openmc_evaluator._cycle_length does."""
    bu = np.asarray(bu, float)
    k = np.asarray(k, float)
    k_bol = float(k[0])
    op = bu > 1e-9
    if op.sum() == 0:
        raise ValueError("history has only the BOL point")
    i_xe = int(np.argmin(np.abs(bu - XE_POINT)))
    k_xe = float(k[i_xe])
    j = int(np.argmax(k[op]))
    k_peak = float(k[op][j])
    b_peak = float(bu[op][j])
    # last downward crossing of k_target, linear interpolation
    crossing = None
    for i in range(len(bu) - 1, 0, -1):
        if k[i - 1] > k_target >= k[i]:
            crossing = float(bu[i - 1] + (k[i - 1] - k_target) * (bu[i] - bu[i - 1]) / (k[i - 1] - k[i]))
            break
    return dict(k_bol=k_bol, k_xe=k_xe, b_xe=float(bu[i_xe]), k_peak=k_peak, b_peak=b_peak,
                hump_vs_bol_pcm=1e5 * (k_peak - k_bol) / k_bol,
                hump_vs_xe_pcm=1e5 * (k_peak - k_xe) / k_xe,
                bol_bounds_cycle=bool(k_peak <= k_bol),
                crossing_mwd_kg=crossing)


def selftest() -> int:
    print("selftest (no OpenMC):")
    # stitching: chunk-local restart (chunk 1 repeats the last state of chunk 0)
    c0 = (np.array([0, 10, 30]), np.array([1.20, 1.17, 1.18]), np.full(3, 2e-4))
    c1 = (np.array([30, 70, 110]), np.array([1.18, 1.19, 1.16]), np.full(3, 2e-4))
    t, k, s = stitch([c0, c1])
    assert t.tolist() == [0, 10, 30, 70, 110] and k.tolist() == [1.20, 1.17, 1.18, 1.19, 1.16], (t, k)
    # stitching: cumulative restart (chunk 1 carries the whole history)
    c1c = (np.array([0, 10, 30, 70, 110]), np.array([1.20, 1.17, 1.18, 1.19, 1.16]), np.full(5, 2e-4))
    t2, k2, _ = stitch([c0, c1c])
    assert t2.tolist() == t.tolist() and k2.tolist() == k.tolist()
    print("  stitching ok for chunk-local and cumulative restart files")
    # hump with a real Gd rise: peak above the xenon point but below the xenon-free BOL
    bu = np.array([0.0, 0.5, 1.5, 3.5, 7.5, 13.5, 17.5, 21.5])
    kk = np.array([1.190, 1.160, 1.165, 1.172, 1.178, 1.150, 1.100, 1.060])
    h = hump(bu, kk, 1.083)
    assert abs(h["k_peak"] - 1.178) < 1e-12 and abs(h["b_peak"] - 7.5) < 1e-12
    assert h["bol_bounds_cycle"] and h["hump_vs_bol_pcm"] < 0 and h["hump_vs_xe_pcm"] > 0
    assert abs(h["crossing_mwd_kg"] - (17.5 + (1.100 - 1.083) * 4.0 / 0.040)) < 1e-9, h
    print("  hump: peak 1.178 at 7.5 MWd/kg, below the xenon-free BOL, crossing 19.2 MWd/kg")
    # hump above BOL
    kk2 = kk.copy(); kk2[4] = 1.195
    h2 = hump(bu, kk2, 1.083)
    assert not h2["bol_bounds_cycle"] and h2["hump_vs_bol_pcm"] > 0
    print("  hump above the xenon-free BOL detected")
    print("selftest OK")
    return 0


# --------------------------------------------------------------------------
# OpenMC reading
# --------------------------------------------------------------------------
def read_case(case: Path):
    import openmc.deplete
    chunks = []
    for d in sorted(case.glob("dep_*")):
        f = d / "depletion_results.h5"
        if not f.exists():
            continue
        res = openmc.deplete.Results(str(f))
        try:
            t, karr = res.get_keff(time_units="d")
        except TypeError:                      # older signature, seconds
            t, karr = res.get_keff()
            t = np.asarray(t, float) / 86400.0
        karr = np.asarray(karr, float)
        chunks.append((np.asarray(t, float), karr[:, 0], karr[:, 1] if karr.shape[1] > 1 else np.zeros(len(t))))
    if not chunks:
        return None
    return stitch(chunks)


def first_k_of_case(case: Path):
    """Cheap match key: the first eigenvalue of dep_00."""
    import openmc.deplete
    f = case / "dep_00" / "depletion_results.h5"
    if not f.exists():
        return None
    _t, karr = openmc.deplete.Results(str(f)).get_keff()
    return float(np.asarray(karr, float)[0, 0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--workdir", default="openmc_runs_c8")
    ap.add_argument("--designs", type=int, nargs="*", default=None, help="archive indices (default: all feasible)")
    ap.add_argument("--out", default="khist_c8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--spec-power", type=float, default=None, help="W/gHM, override the value recovered from the record")
    ap.add_argument("--match-tol", type=float, default=1e-6)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    try:
        import openmc  # noqa: F401
    except ImportError:
        sys.exit("FAIL: openmc not importable. You are probably in (base). Run: conda activate openmc-env")
    ck = json.loads(Path(a.checkpoint).read_text())
    A = ck["all_raw"]
    CN = ck["constraint_names"]
    sched = ck["meta"].get("schedule", {})
    designs = a.designs if a.designs else [i for i, r in enumerate(A) if all(r[g] <= 0 for g in CN)]
    wd = Path(a.workdir)
    cases = sorted(wd.glob("case_*"))
    if not cases:
        sys.exit(f"FAIL: no case_* directories under {wd}")
    print(f"archive {len(A)} records, {len(cases)} case directories, schedule {sched}")

    # match cases to records by k_bol
    keys = {}
    for c in cases:
        kb = first_k_of_case(c)
        if kb is not None:
            keys[c] = kb
    match = {}
    for i in designs:
        kb = A[i]["k_bol"]
        hits = [c for c, v in keys.items() if abs(v - kb) < a.match_tol]
        if len(hits) == 1:
            match[i] = hits[0]
        print(f"  design {i:>3}: k_bol {kb:.6f} -> {hits[0].name if len(hits) == 1 else ('NO MATCH' if not hits else 'AMBIGUOUS ' + str([h.name for h in hits]))}"
              + (f"   (directory number {'agrees' if hits and hits[0].name == f'case_{i:04d}' else 'DIFFERS'})" if hits else ""))
    if a.dry_run:
        return 0

    out = Path(a.out); out.mkdir(exist_ok=True)
    results = {}
    rows = []
    for i, case in match.items():
        r = A[i]
        sp = a.spec_power or (r["bu_eoc_mwd_kg"] * 1000.0 / r["cycle_length"] if r["cycle_length"] > 0 else SPEC_POWER_CAMPAIGN)
        if abs(sp - SPEC_POWER_CAMPAIGN) / SPEC_POWER_CAMPAIGN > 0.01:
            print(f"  WARNING design {i}: recovered specific power {sp:.4f} W/gHM differs from {SPEC_POWER_CAMPAIGN} by more than 1 percent")
        t, k, s = read_case(case)
        bu = t * sp / 1000.0
        if abs(k[0] - r["k_bol"]) > a.match_tol:
            print(f"  WARNING design {i}: stitched first k {k[0]:.6f} != record k_bol {r['k_bol']:.6f}")
        h = hump(bu, k, r["k_target"])
        noise = float(np.median(s) * 1e5) if np.any(s > 0) else float("nan")
        closure = (h["crossing_mwd_kg"] - r["bu_eoc_mwd_kg"]) if h["crossing_mwd_kg"] is not None else None
        results[str(i)] = dict(case=case.name, spec_power=sp, bu=bu.tolist(), k=k.tolist(), k_sd=s.tolist(),
                               k_target=r["k_target"], bu_eoc_record=r["bu_eoc_mwd_kg"], efpd_record=r["cycle_length"],
                               noise_pcm_per_state=noise, closure_mwd_kg=closure, **h)
        rows.append((i, r["enrich"], r["gd_wt"], r["gd_pins_used"], h, noise, closure, len(k)))
        print(f"  design {i:>3} ({case.name}): {len(k)} states, k_bol {h['k_bol']:.4f}, k_xe {h['k_xe']:.4f}, "
              f"peak {h['k_peak']:.4f} at {h['b_peak']:.1f} MWd/kg, hump vs BOL {h['hump_vs_bol_pcm']:+.0f} pcm, "
              f"vs Xe {h['hump_vs_xe_pcm']:+.0f} pcm, noise ~{noise:.0f} pcm, "
              f"crossing {h['crossing_mwd_kg'] if h['crossing_mwd_kg'] is None else round(h['crossing_mwd_kg'], 2)} "
              f"vs record {r['bu_eoc_mwd_kg']:.2f}")
    (out / "khist.json").write_text(json.dumps(results, indent=1))

    # LaTeX table
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \footnotesize",
         r"  \caption{Mid-cycle maximum of $k_\infty$ of the depleting assembly (1000\,ppm, $4000 \times 60$ transport) against the xenon-free beginning-of-life value that the controllability screens use. A negative hump means the beginning-of-life margin bounds the whole cycle. Noise is the Monte Carlo standard deviation of one depletion state.}",
         r"  \label{tab:c8-khist-hump}",
         r"  \begin{tabular}{@{}l r r r r r r r r r l@{}}", r"    \toprule",
         r"    ID & $e$ & Gd & pins & $k_\mathrm{BOL}$ & $k_\mathrm{Xe}$ & $k_\mathrm{peak}$ & $B_\mathrm{peak}$ & hump vs BOL & hump vs Xe & noise \\",
         r"       & wt\% & wt\% &  &  &  &  & MWd/kg & pcm & pcm & pcm \\", r"    \midrule"]
    for i, e, gd, pins, h, noise, closure, n in sorted(rows, key=lambda x: x[1]):
        L.append(f"    {i} & {e:.2f} & {gd:.2f} & {pins} & {h['k_bol']:.4f} & {h['k_xe']:.4f} & {h['k_peak']:.4f} & {h['b_peak']:.1f} & {h['hump_vs_bol_pcm']:+.0f} & {h['hump_vs_xe_pcm']:+.0f} & {noise:.0f} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / "khist_table.tex").write_text("\n".join(L) + "\n")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        cmap = plt.get_cmap("viridis")
        es = [r[1] for r in rows]; norm = plt.Normalize(min(es), max(es))
        for i, e, gd, pins, h, noise, closure, n in rows:
            d = results[str(i)]
            ax.plot(d["bu"], d["k"], "-", marker=".", ms=3, lw=1.0, color=cmap(norm(e)), label=f"{i}")
            ax.plot(h["b_peak"], h["k_peak"], "v", color=cmap(norm(e)), ms=5)
            ax.axhline(d["k_target"], color="0.5", lw=0.5, ls=":")
        ax.set_xlabel("Burnup, MWd/kgHM"); ax.set_ylabel("$k_\\infty$ of the depleting assembly (1000 ppm)")
        ax.set_title("Campaign 8 candidates: k history, peak marked, dotted lines at $k_\\mathrm{target}$", fontsize=9)
        ax.legend(fontsize=7, ncol=3, title="design")
        fig.savefig(out / "c8_post_khist.pdf", bbox_inches="tight"); fig.savefig(out / "c8_post_khist.png", dpi=300, bbox_inches="tight")
        print(f"wrote {out}/khist.json, khist_table.tex, c8_post_khist.pdf")
    except Exception as ex:  # matplotlib absent
        print(f"wrote {out}/khist.json and khist_table.tex (no figure: {ex})")
    n_pos = sum(1 for r in rows if not r[4]["bol_bounds_cycle"])
    print(f"\n{n_pos} of {len(rows)} designs have a mid-cycle maximum above the xenon-free BOL. "
          "For those, subtract the hump from every BOL margin of the boron and confirmation studies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
