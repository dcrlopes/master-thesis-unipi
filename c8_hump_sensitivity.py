#!/usr/bin/env python3
"""
c8_hump_sensitivity.py -- Campaign 8: how robust is each controllability
verdict to the choice of hump estimator?

Two independent estimates of the mid-cycle multiplication-factor maximum
exist for every design in the working set.

  single shot   kt_burnup/summary.json, a fresh depletion with no restart
                chunks, written by validate_ktarget_burnup.py.  This is the
                estimate c8_reactivity_swing.py uses for the margins it
                reports, through the field hump_pcm.

  archive       khist_c8/khist.json, the original campaign depletion read
                back from openmc_runs_c8 by c8_hump2.py, carried into the
                swing record as chunked_hump_vs_bol_pcm.

The margins at the operating maximum are

    margin_peak = margin_bol - max(LF_bol * hump, 0)

so a different hump moves every margin whose hump is positive, and leaves
untouched every margin whose hump is negative under both estimates.  This
script reports both margin sets side by side and names the designs whose
verdict against the 1010.1 pcm screen depends on the choice.

Reads swing_c8/swing.json only.  No OpenMC, no transport, seconds to run.

USAGE
    python c8_hump_sensitivity.py --selftest
    python c8_hump_sensitivity.py
    python c8_hump_sensitivity.py --swing swing_c8/swing.json --out swing_c8
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

SCREEN_PCM = 1010.1          # k <= 0.99, the measured control screen g_ctrl
STATES = (("ARI", "four banks"), ("RE12", "two banks"))


# --------------------------------------------------------------- preflight --
def preflight(swing_path):
    """Light check only.  This script runs no transport, so it does not
    require openmc-env, but it must be run from the repository root so the
    relative paths resolve and the outputs land beside the other results."""
    print("=" * 62)
    print(" PREFLIGHT")
    print("=" * 62)
    print(f"  host        : {socket.gethostname()}")
    print(f"  cwd         : {os.getcwd()}")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  conda env   : {os.environ.get('CONDA_DEFAULT_ENV', 'none')}")
    print(f"  swing file  : {swing_path}")
    if not Path(swing_path).is_file():
        sys.exit(f"FAIL: {swing_path} not found. Run c8_reactivity_swing.py first, "
                 "from ~/master-thesis-unipi.")
    print("  no transport required, environment not enforced")
    print("preflight OK\n")


# ------------------------------------------------------------------- core --
def margins(rec):
    """Return the two margin sets for one design record.

    Keys consumed, all written by c8_reactivity_swing.py:
      hump_pcm                    single-shot assembly hump
      chunked_hump_vs_bol_pcm     archive assembly hump, absent if the
                                  cross-check did not run
      lf_bol                      leakage factor at beginning of life
      <state>_margin_bol_3d_pcm   three-dimensional margin at 1000 ppm, BOL
    """
    lf = rec.get("lf_bol")
    h_ss = rec.get("hump_pcm")
    h_ar = rec.get("chunked_hump_vs_bol_pcm")
    out = {"lf_bol": lf, "hump_ss_pcm": h_ss, "hump_archive_pcm": h_ar}
    if lf is None or h_ss is None:
        return out
    out["hump_core_ss_pcm"] = lf * h_ss
    out["hump_core_archive_pcm"] = None if h_ar is None else lf * h_ar
    for st, _ in STATES:
        bol = rec.get(f"{st}_margin_bol_3d_pcm")
        if bol is None:
            continue
        out[f"{st}_peak_ss_pcm"] = bol - max(lf * h_ss, 0.0)
        if h_ar is not None:
            out[f"{st}_peak_archive_pcm"] = bol - max(lf * h_ar, 0.0)
    return out


def selftest():
    """Two synthetic designs, one whose verdict flips and one whose does not."""
    flip = {"lf_bol": 1.08, "hump_pcm": 1800.0, "chunked_hump_vs_bol_pcm": 2000.0,
            "ARI_margin_bol_3d_pcm": 2990.0}
    m = margins(flip)
    assert abs(m["ARI_peak_ss_pcm"] - (2990.0 - 1944.0)) < 1e-9
    assert abs(m["ARI_peak_archive_pcm"] - (2990.0 - 2160.0)) < 1e-9
    assert m["ARI_peak_ss_pcm"] > SCREEN_PCM > m["ARI_peak_archive_pcm"]

    stable = {"lf_bol": 1.08, "hump_pcm": -1600.0, "chunked_hump_vs_bol_pcm": -1800.0,
              "RE12_margin_bol_3d_pcm": 1330.0}
    m = margins(stable)
    assert m["RE12_peak_ss_pcm"] == m["RE12_peak_archive_pcm"] == 1330.0

    absent = {"lf_bol": 1.08, "hump_pcm": 500.0, "ARI_margin_bol_3d_pcm": 2000.0}
    m = margins(absent)
    assert "ARI_peak_archive_pcm" not in m
    print("selftest OK")
    return 0


def cross_check_stats(rows):
    """Mean, scatter and reduced chi-squared of the archive minus single-shot
    peak difference, against an assumed per-solve noise."""
    d = [r["peak_diff_pcm"] for r in rows if r.get("peak_diff_pcm") is not None]
    if len(d) < 2:
        return None
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = var ** 0.5
    return {"n": n, "mean_pcm": mean, "sd_pcm": sd, "sem_pcm": sd / n ** 0.5}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swing", default="swing_c8/swing.json",
                    help="record written by c8_reactivity_swing.py")
    ap.add_argument("--out", default="swing_c8",
                    help="directory for hump_sensitivity.txt and .json")
    ap.add_argument("--screen", type=float, default=SCREEN_PCM,
                    help="control screen in pcm, default 1010.1 (k <= 0.99)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the internal consistency checks and exit")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    preflight(a.swing)
    swing = json.loads(Path(a.swing).read_text())

    rows = []
    for key, rec in swing.items():
        m = margins(rec)
        m["idx"] = int(key)
        kp_ss = rec.get("k_peak")
        kp_ar = rec.get("chunked_k_peak")
        m["peak_diff_pcm"] = (rec.get("chunked_minus_single_shot_peak_pcm")
                              if kp_ss and kp_ar else None)
        rows.append(m)
    rows.sort(key=lambda r: r["idx"])

    L = []
    P = L.append
    P("=== C8 hump sensitivity: control margins under two hump estimators ===")
    P(f"screen: {a.screen:.1f} pcm, equivalent to k <= 0.99")
    P("")
    P(f"{'idx':>4} {'hump ss':>9} {'hump arc':>9} | "
      f"{'ARI ss':>9} {'ARI arc':>9} | {'RE12 ss':>9} {'RE12 arc':>9} | flips")
    for r in rows:
        flips = []
        for st, _ in STATES:
            s, c = r.get(f"{st}_peak_ss_pcm"), r.get(f"{st}_peak_archive_pcm")
            if s is not None and c is not None and (s >= a.screen) != (c >= a.screen):
                flips.append(st)

        def f(x):
            return "      n/a" if x is None else f"{x:>+9.0f}"

        P(f"{r['idx']:>4} {f(r.get('hump_ss_pcm'))} {f(r.get('hump_archive_pcm'))} | "
          f"{f(r.get('ARI_peak_ss_pcm'))} {f(r.get('ARI_peak_archive_pcm'))} | "
          f"{f(r.get('RE12_peak_ss_pcm'))} {f(r.get('RE12_peak_archive_pcm'))} | "
          f"{','.join(flips) if flips else '-'}")

    P("")
    P("ss = single-shot depletion (kt_burnup), arc = campaign archive (khist_c8).")
    P("ARI = four regulating banks, RE12 = two regulating banks. All at 1000 ppm,")
    P("three-dimensional, reduced by the positive part of the hump on the core.")
    P("")
    for st, lab in STATES:
        for tag, lb in (("ss", "single shot"), ("archive", "archive")):
            fail = [r["idx"] for r in rows
                    if r.get(f"{st}_peak_{tag}_pcm") is not None
                    and r[f"{st}_peak_{tag}_pcm"] < a.screen]
            P(f"  below the screen, {lab:<10} {lb:<11}: {fail or 'none'}")
    P("")
    flipped = [r["idx"] for r in rows
               if any((r.get(f"{st}_peak_ss_pcm") is not None
                       and r.get(f"{st}_peak_archive_pcm") is not None
                       and (r[f"{st}_peak_ss_pcm"] >= a.screen)
                       != (r[f"{st}_peak_archive_pcm"] >= a.screen))
                      for st, _ in STATES)]
    P(f"designs whose verdict depends on the hump estimator: {flipped or 'none'}")
    P("These must be reported as marginal, not as qualified or rejected.")

    st = cross_check_stats(rows)
    if st:
        P("")
        P("cross-check on the peak, archive minus single shot:")
        P(f"  n = {st['n']}, mean {st['mean_pcm']:+.0f} pcm, "
          f"standard error {st['sem_pcm']:.0f} pcm, scatter {st['sd_pcm']:.0f} pcm")
        P("  a mean within one standard error of zero indicates no systematic")
        P("  offset between the two depletion routes.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hump_sensitivity.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "hump_sensitivity.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}/hump_sensitivity.txt and {out}/hump_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
