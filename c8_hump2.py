#!/usr/bin/env python3
"""
c8_hump2.py
===========
Gadolinium burnout hump of the Campaign 8 designs, rebuilt from the
depletion files on disk. No transport is run. Replacement for
c8_khist_hump.py, which failed on 4 September 2026 with a
ZeroDivisionError because it treated a chunked restart file as a
cumulative one.

Why the merge needs care, from the files of design 47:

    dep_00/depletion_results.h5   6 slots, t = 0 to 1352.25 d,
                                  k[0] = 1.187907 = the archive k_bol
    dep_01/depletion_results.h5   8 slots, the first 5 zero-filled
                                  placeholders for the steps of the
                                  previous chunk, slot 5 the restart
                                  state at 1352.25 d with the same k as
                                  the last slot of dep_00, then two new
                                  steps at 1752.92 and 2153.58 d

A chunk file is therefore cumulative in SHAPE and chunk-local in
CONTENT. This script merges by dropping every slot whose eigenvalue is
zero or not finite, then de-duplicating by time, then sorting. The
duplicated restart point is kept once and its two copies are compared,
which is a free consistency check on the restart.

Burnup is obtained from the time axis and the specific power of each
record, recovered as bu_eoc_mwd_kg / cycle_length * 1000, which
reproduces the 9.9834 W/gHM of the campaign on every design.

    python c8_hump2.py --selftest
    python c8_hump2.py --checkpoint out_c8/optimization_checkpoint.json \\
        --workdir openmc_runs_c8 --designs 47 42 23 29 21 44 59 1 53 31 13 \\
        --dry-run
    python c8_hump2.py --checkpoint out_c8/optimization_checkpoint.json \\
        --workdir openmc_runs_c8 --designs 47 42 23 29 21 44 59 1 53 31 13 \\
        --out khist_c8

Outputs in <out>/:
    khist.json        one record per design, every number the text needs
    khist_table.tex   booktabs table, caption and label included
    c8_post_khist.pdf and .png   k against burnup, all designs, one panel

Reading the result, per design:
    hump_vs_bol_pcm  <= 0  the xenon-free BOL screen bounds the whole
                           cycle and every BOL margin of the
                           post-analysis stands as it is
    hump_vs_bol_pcm  >  0  subtract it from every BOL margin before the
                           candidate decision. For the boron-free
                           reading of design 47 the test is
                           (0 ppm four-bank 3D margin) - hump >= 1000 pcm
    crossing_bu      must reproduce bu_eoc_mwd_kg of the archive within
                     one depletion step, otherwise the case match is wrong

The depletion solve noise is 150 to 250 pcm per state, so a hump below
about 400 pcm is not resolved and should be reported as not resolved
rather than as a number.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ZERO_TOL = 1e-12
TIME_TOL = 1e-6          # days, for de-duplicating the restart overlap


# --------------------------------------------------------------------------
# reading and merging
# --------------------------------------------------------------------------
def read_chunk(path: str):
    """Return (time_days, keff) of one depletion_results.h5, unfiltered."""
    import openmc.deplete as dep
    r = dep.Results(path)
    try:
        t, k = r.get_keff(time_units="d")
    except Exception:
        t, k = r.get_eigenvalue()
    t = np.asarray(t, dtype=float).ravel()
    k = np.asarray(k, dtype=float)
    if k.ndim == 2 and k.shape[-1] == 2:      # (value, std dev) pairs
        k = k[:, 0]
    return t, k.ravel()


def merge_chunks(chunks):
    """
    chunks: list of (name, time, keff) in chunk order.

    Drops zero-filled and non-finite placeholders, de-duplicates the
    restart overlap, sorts by time. Returns (time, keff, notes) where
    notes is a list of strings describing what was dropped and how well
    the duplicated points agreed.
    """
    notes = []
    seen = {}                                  # rounded time -> (k, source)
    for name, t, k in chunks:
        good = np.isfinite(k) & (np.abs(k) > ZERO_TOL)
        n_drop = int(np.sum(~good))
        if n_drop:
            notes.append(f"{name}: dropped {n_drop} placeholder slot(s)")
        for ti, ki in zip(t[good], k[good]):
            key = round(float(ti) / max(TIME_TOL, 1e-9))
            if key in seen:
                k_prev, src = seen[key]
                d_pcm = 1e5 * (ki - k_prev) / k_prev
                notes.append(f"overlap at t = {ti:.2f} d, {src} vs {name}, "
                             f"{d_pcm:+.1f} pcm")
            else:
                seen[key] = (float(ki), name)
    if not seen:
        raise ValueError("no usable depletion points after filtering")
    times = np.array(sorted(seen), dtype=float) * max(TIME_TOL, 1e-9)
    keffs = np.array([seen[round(float(t) / max(TIME_TOL, 1e-9))][0] for t in times])
    order = np.argsort(times)
    return times[order], keffs[order], notes


def case_dirs(workdir: str):
    return sorted(d for d in glob.glob(os.path.join(workdir, "case_*")) if os.path.isdir(d))


def chunk_files(case_dir: str):
    return sorted(glob.glob(os.path.join(case_dir, "dep_*", "depletion_results.h5")))


def match_cases(records, workdir, designs):
    """
    Match each design index to a case directory by the archived k_bol,
    which is the first eigenvalue of its first chunk. Directory numbers
    can be offset after a resume, so the number is only reported, never
    trusted.
    """
    firsts = {}
    for d in case_dirs(workdir):
        files = chunk_files(d)
        if not files:
            continue
        try:
            t, k = read_chunk(files[0])
        except Exception as e:
            print(f"  {d}: unreadable first chunk ({type(e).__name__})")
            continue
        good = np.isfinite(k) & (np.abs(k) > ZERO_TOL)
        if not good.any():
            continue
        firsts[d] = float(k[good][0])

    out = {}
    print(f"archive {len(records)} records, {len(firsts)} readable case directories")
    for i in designs:
        kb = float(records[i]["k_bol"])
        hits = [(d, abs(v - kb)) for d, v in firsts.items() if abs(v - kb) < 1e-6]
        if len(hits) != 1:
            near = sorted(((abs(v - kb), d) for d, v in firsts.items()))[:3]
            print(f"  design {i:>3}: k_bol {kb:.6f} -> {len(hits)} match(es). "
                  f"Closest: " + ", ".join(f"{d} ({dv:.2e})" for dv, d in near))
            out[i] = None
            continue
        d = hits[0][0]
        num = int(Path(d).name.split("_")[-1])
        agree = "directory number agrees" if num == i else f"DIFFERENT directory number ({num})"
        print(f"  design {i:>3}: k_bol {kb:.6f} -> {Path(d).name}   ({agree})")
        out[i] = d
    return out


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------
def analyse(bu, k, k_bol_record, k_target):
    """
    bu: burnup in MWd/kgHM, k: eigenvalue, both sorted and cleaned.

    Returns a dict. The hump is the maximum of k after the first point,
    reported against two references:
      - the xenon-free beginning of life, k[0], which is the state the
        control screen of the campaign evaluated;
      - the first depleted point, k[1], the closest available proxy for
        equilibrium xenon, since the campaign never solved a xenon state.
    """
    k = np.asarray(k, dtype=float)
    bu = np.asarray(bu, dtype=float)
    k0 = float(k[0])
    if not math.isfinite(k0) or abs(k0) < ZERO_TOL:
        raise ValueError("first eigenvalue is zero or not finite after merging")

    j = int(np.argmax(k[1:])) + 1 if len(k) > 1 else 0
    k_peak, bu_peak = float(k[j]), float(bu[j])
    k_xe = float(k[1]) if len(k) > 1 else k0

    crossing = None
    for a in range(len(k) - 1):
        if (k[a] - k_target) * (k[a + 1] - k_target) <= 0 and k[a] != k[a + 1]:
            f = (k[a] - k_target) / (k[a] - k[a + 1])
            crossing = float(bu[a] + f * (bu[a + 1] - bu[a]))
            break

    return dict(
        n_points=int(len(k)),
        k_bol=k0,
        k_bol_record=float(k_bol_record),
        k_bol_minus_record_pcm=1e5 * (k0 - k_bol_record) / k_bol_record,
        k_peak=k_peak,
        bu_peak_mwd_kg=bu_peak,
        hump_vs_bol_pcm=1e5 * (k_peak - k0) / k0,
        hump_vs_xenon_pcm=1e5 * (k_peak - k_xe) / k_xe,
        k_first_depleted=k_xe,
        k_target=float(k_target),
        crossing_bu_mwd_kg=crossing,
        bu_last_mwd_kg=float(bu[-1]),
    )


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def write_table(res, path: Path):
    rows = []
    for i, r in res.items():
        if r is None or "analysis" not in r:
            continue
        a, rec = r["analysis"], r["record"]
        hump = a["hump_vs_bol_pcm"]
        hump_s = f"{hump:+.0f}" if abs(hump) >= 400 else r"$<$ noise"
        cross = a["crossing_bu_mwd_kg"]
        cross_s = f"{cross:.2f}" if cross is not None else "not reached"
        dev_s = (f"{cross - rec['bu_eoc_mwd_kg']:+.2f}" if cross is not None else "--")
        rows.append(f"    {i} & {rec['enrich']:.2f} & {rec['gd_wt']:.2f} & "
                    f"{rec['gd_pins_used']:.0f} & {a['k_bol']:.4f} & "
                    f"{a['k_peak']:.4f} & {a['bu_peak_mwd_kg']:.2f} & {hump_s} & "
                    f"{cross_s} & {dev_s} \\\\")
    body = "\n".join(rows)
    path.write_text(
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\caption[Gadolinium burnout hump of the Campaign 8 working set]{"
        "Gadolinium burnout hump of the Campaign 8 working set, reconstructed "
        "from the depletion histories. The hump is the peak eigenvalue after "
        "beginning of life, referred to the xenon-free beginning of life. "
        "Entries below the depletion solve noise of about \\SI{400}{pcm} are "
        "reported as unresolved. The last column is the crossing of the Route "
        "B target minus the archived end-of-cycle burnup.}\n"
        "  \\label{tab:c8-hump}\n"
        "  \\begin{tabular}{cccccccccc}\n"
        "    \\toprule\n"
        "    Design & $e$ & $w_\\mathrm{Gd}$ & $n_\\mathrm{Gd}$ & $k_\\mathrm{BOL}$ &\n"
        "      $k_\\mathrm{peak}$ & $B_\\mathrm{peak}$ & hump & $B_\\mathrm{cross}$ &\n"
        "      $\\Delta B$ \\\\\n"
        "    {[--]} & [wt\\%] & [wt\\%] & [rods] & [--] & [--] & [MWd/kgHM] &\n"
        "      [pcm] & [MWd/kgHM] & [MWd/kgHM] \\\\\n"
        "    \\midrule\n" + body + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n", encoding="utf-8")


def write_figure(res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, r in sorted(res.items()):
        if r is None or "analysis" not in r:
            continue
        bu, k = np.array(r["curve"]["bu"]), np.array(r["curve"]["k"])
        line, = ax.plot(bu, k, lw=1.2, label=f"{i} ({r['record']['enrich']:.1f} wt\\%)"
                        .replace("\\", ""))
        a = r["analysis"]
        ax.plot([a["bu_peak_mwd_kg"]], [a["k_peak"]], "o", ms=3.5, color=line.get_color())
        if a["crossing_bu_mwd_kg"] is not None:
            ax.plot([a["crossing_bu_mwd_kg"]], [a["k_target"]], "x", ms=5,
                    color=line.get_color())
    ax.set_xlabel("Burnup [MWd/kgHM]")
    ax.set_ylabel(r"$k_\infty$ of the reflective assembly [--]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, title="design (enrichment)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# self-test, no OpenMC and no files
# --------------------------------------------------------------------------
def selftest() -> int:
    # Exactly the structure observed in case_0047 on 4 September 2026.
    t0 = np.array([0., 50.08, 150.25, 350.58, 751.25, 1352.25])
    k0 = np.array([1.187907, 1.162877, 1.166631, 1.163092, 1.157410, 1.131726])
    t1 = np.array([0., 0., 0., 0., 0., 1352.25, 1752.92, 2153.58])
    k1 = np.array([0., 0., 0., 0., 0., 1.131726, 1.096361, 1.067362])
    t, k, notes = merge_chunks([("dep_00", t0, k0), ("dep_01", t1, k1)])
    assert len(t) == 8, (len(t), t)
    assert abs(k[0] - 1.187907) < 1e-9, k[0]
    assert abs(t[-1] - 2153.58) < 1e-6
    assert not np.any(k == 0.0)
    assert any("dropped 5" in n for n in notes), notes
    assert any("overlap" in n for n in notes), notes

    # The chunks in reverse order must give the same history.
    t_r, k_r, _ = merge_chunks([("dep_01", t1, k1), ("dep_00", t0, k0)])
    assert np.allclose(t_r, t) and np.allclose(k_r, k)

    # A genuinely cumulative second file (no placeholders) must not duplicate.
    t2 = np.concatenate([t0, [1752.92]])
    k2 = np.concatenate([k0, [1.096361]])
    t_c, k_c, _ = merge_chunks([("dep_00", t0, k0), ("dep_01", t2, k2)])
    assert len(t_c) == 7, len(t_c)

    bu = t * 9.983354024602374 / 1000.0
    a = analyse(bu, k, 1.187907, 1.0827269091968406)
    assert abs(a["k_bol_minus_record_pcm"]) < 1e-6
    assert a["hump_vs_bol_pcm"] < 0                      # peak is at BOL here
    assert a["hump_vs_xenon_pcm"] > 0                    # but above the first depleted point
    assert a["crossing_bu_mwd_kg"] is not None
    assert abs(a["crossing_bu_mwd_kg"] - 19.38) < 0.5, a["crossing_bu_mwd_kg"]

    # A history with a real hump must be found and located.
    bu_h = np.array([0., 0.5, 2.0, 5.0, 9.0, 15.0, 22.0])
    k_h = np.array([1.050, 1.062, 1.081, 1.075, 1.050, 1.010, 0.960])
    h = analyse(bu_h, k_h, 1.050, 1.0)
    assert abs(h["bu_peak_mwd_kg"] - 2.0) < 1e-9
    assert abs(h["hump_vs_bol_pcm"] - 1e5 * (1.081 - 1.050) / 1.050) < 1e-6

    print("selftest OK")
    print("  merged the observed case_0047 structure into 8 points, "
          "first k 1.187907, last t 2153.58 d")
    print("  crossing of the Route B target reproduces "
          f"{a['crossing_bu_mwd_kg']:.2f} MWd/kgHM against the archived 19.38")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--workdir", default="openmc_runs_c8")
    ap.add_argument("--designs", type=int, nargs="*",
                    default=[47, 42, 23, 29, 21, 44, 59, 1, 53, 31, 13])
    ap.add_argument("--out", default="khist_c8")
    ap.add_argument("--dry-run", action="store_true",
                    help="match the cases and list the chunk files, read nothing else")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--noise-pcm", type=float, default=400.0,
                    help="humps below this are reported as unresolved")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    records = json.load(open(a.checkpoint))["all_raw"]
    matched = match_cases(records, a.workdir, a.designs)

    if a.dry_run:
        print()
        for i, d in matched.items():
            if d is None:
                print(f"  design {i:>3}: NO MATCH")
                continue
            files = chunk_files(d)
            print(f"  design {i:>3}: {len(files)} chunk file(s)")
            for f in files:
                t, k = read_chunk(f)
                mask = np.isfinite(k) & (np.abs(k) > ZERO_TOL)
                n_ok = int(np.sum(mask))
                span = (f"t {t[mask].min():.1f} to {t[mask].max():.1f} d"
                        if n_ok else "no usable slot")
                print(f"      {os.path.relpath(f, d)}  {len(k)} slot(s), "
                      f"{n_ok} usable, {span}")
        bad = [i for i, d in matched.items() if d is None]
        print("\nevery design matched exactly one case" if not bad
              else f"\nSTOP: designs without a unique match: {bad}")
        return 1 if bad else 0

    res, failed = {}, []
    for i, d in matched.items():
        if d is None:
            failed.append(i)
            continue
        rec = records[i]
        files = chunk_files(d)
        chunks = []
        for f in files:
            t, k = read_chunk(f)
            chunks.append((os.path.basename(os.path.dirname(f)), t, k))
        try:
            t, k, notes = merge_chunks(chunks)
            spec_power = rec["bu_eoc_mwd_kg"] / rec["cycle_length"] * 1000.0   # W/gHM
            bu = t * spec_power / 1000.0
            an = analyse(bu, k, rec["k_bol"], rec["k_target"])
        except Exception as e:
            print(f"  design {i}: FAILED, {type(e).__name__}: {e}")
            failed.append(i)
            continue
        an["specific_power_w_per_g"] = spec_power
        an["resolved"] = bool(abs(an["hump_vs_bol_pcm"]) >= a.noise_pcm)
        res[i] = dict(case=os.path.basename(d), notes=notes, analysis=an,
                      record={key: rec[key] for key in
                              ("enrich", "gd_wt", "gd_pins_used", "refl_thick",
                               "k_bol", "k_target", "bu_eoc_mwd_kg", "cycle_length",
                               "keff_core_bol", "peaking")},
                      curve=dict(bu=bu.tolist(), k=k.tolist(), t_days=t.tolist()))

    print(f"\n{'idx':>4} {'pts':>4} {'kBOL-rec':>9} {'k_peak':>8} {'B_peak':>7} "
          f"{'hump':>8} {'vs Xe':>8} {'B_cross':>8} {'dB_arch':>8}")
    for i, r in res.items():
        an, rec = r["analysis"], r["record"]
        c = an["crossing_bu_mwd_kg"]
        dv = (c - rec["bu_eoc_mwd_kg"]) if c is not None else float("nan")
        print(f"{i:>4} {an['n_points']:>4} {an['k_bol_minus_record_pcm']:>+9.1f} "
              f"{an['k_peak']:>8.4f} {an['bu_peak_mwd_kg']:>7.2f} "
              f"{an['hump_vs_bol_pcm']:>+8.0f} {an['hump_vs_xenon_pcm']:>+8.0f} "
              f"{'n/a' if c is None else f'{c:>8.2f}'} {dv:>+8.2f}")
    print("\nhump and vs Xe in pcm, B in MWd/kgHM. dB_arch is the crossing minus the")
    print(f"archived end-of-cycle burnup. Humps under {a.noise_pcm:.0f} pcm are inside the")
    print("depletion solve noise and must be reported as unresolved.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "khist.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_table(res, out / "khist_table.tex")
    try:
        write_figure(res, out / "c8_post_khist")
    except Exception as e:
        print(f"figure not written ({type(e).__name__}: {e}), the json and table are")
    print(f"\nwrote {out}/khist.json, {out}/khist_table.tex, {out}/c8_post_khist.pdf")
    if failed:
        print(f"FAILED designs: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
