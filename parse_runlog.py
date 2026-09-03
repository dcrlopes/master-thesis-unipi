#!/usr/bin/env python3
"""
parse_runlog.py -- Extract the useful information from the (very verbose) run.log
produced by run_optimization.py + OpenMC, and summarise it in a few lines.

What it reports:
  * OpenMC version / date / OpenMP thread count
  * DOE (design of experiments) size and every active-learning iteration with its
    hypervolume (HV)
  * the per-case summary table ([case NNNN] lines) -> optional CSV export
  * depletion time-step schedule of the first case (in days = EFPD at full power)
  * transport statistics: number of solves, mean/total transport time, mean
    Monte Carlo k-effective uncertainty in pcm (1e-5)
  * warning/error census (collapsed by message type)
  * total wall time, seconds per evaluation, and a time/cost projection for
    additional evaluations (--project), optionally at a different particle count
    (--particle-scale) and hourly price (--hourly-rate)

Only the Python standard library is required.

Examples:
  python parse_runlog.py run.log
  python parse_runlog.py run.log --csv cases.csv
  python parse_runlog.py run.log --project 90 --particle-scale 4 --hourly-rate 1.9
"""

import argparse
import csv
import re
import statistics as st
import sys
from collections import Counter

RE_CASE = re.compile(
    r"\[case (\d+)\] e=\(\s*([\d.]+)/\s*([\d.]+)\) Gd=([\d.]+) p=([\d.]+) "
    r"refl=\s*([\d.]+) k_target=([\d.]+) -> EFPD=\s*([\d.]+)"
    r"(?:\(CEN\))?\s+F_dh=([\d.]+) "
    r"k_bol=([\d.]+)")
RE_STAGE1 = re.compile(r"\[Stage 1\] (\d+) real evaluations done\. HV=([\d.]+)")
RE_STAGE2 = re.compile(
    r"\[Stage 2\] iter (\d+)/(\d+): \+(\d+) real evals \(total (\d+)\), HV=([\d.]+)")
RE_KCOMB = re.compile(r"Combined k-effective\s*=\s*([\d.]+) \+/- ([\d.]+)")
RE_ELAPSED = re.compile(r"Total time elapsed\s*=\s*([\d.eE+-]+) seconds")
RE_RATE = re.compile(r"Calculation Rate \(active\)\s*=\s*([\d.eE+-]+)")
RE_DONE = re.compile(r"Done in ([\d.]+)s, (\d+) total real evaluations")
RE_DEPLETE = re.compile(r"\[openmc.deplete\] t=([\d.eE+-]+)(?: s)?, dt=([\d.eE+-]+)")
RE_WARN = re.compile(r"(\w+Warning): (.+)")
RE_HEADER = {
    "version": re.compile(r"^\s*Version \|\s*(.+)$"),
    "date": re.compile(r"^\s*Date/Time \|\s*(.+)$"),
    "threads": re.compile(r"^\s*OpenMP Threads \|\s*(\d+)"),
}

CASE_FIELDS = ["case", "enrich_inner", "enrich_outer", "gd_wt", "pitch",
               "refl_thick", "k_target", "EFPD", "F_dh", "k_bol"]


def normalise_warning(msg):
    """Collapse numeric details so identical warning types group together."""
    return re.sub(r"[\d.]+", "#", msg)[:90]


def parse(path):
    cases, stages, ksig, elapsed, rates, dts = [], [], [], [], [], []
    header, done, warnings = {}, None, Counter()
    n_err = 0
    seen_t0 = 0
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_CASE.search(line)
            if m:
                g = m.groups()
                cases.append(dict(zip(CASE_FIELDS,
                                      [int(g[0])] + [float(x) for x in g[1:]])))
                continue
            m = RE_STAGE1.search(line)
            if m:
                stages.append(("DOE", int(m.group(1)), float(m.group(2))))
                continue
            m = RE_STAGE2.search(line)
            if m:
                stages.append((f"iter {m.group(1)}/{m.group(2)}",
                               int(m.group(4)), float(m.group(5))))
                continue
            m = RE_KCOMB.search(line)
            if m:
                ksig.append(float(m.group(2)) * 1e5)  # -> pcm
                continue
            m = RE_ELAPSED.search(line)
            if m:
                elapsed.append(float(m.group(1)))
                continue
            m = RE_RATE.search(line)
            if m:
                rates.append(float(m.group(1)))
                continue
            m = RE_DEPLETE.search(line)
            if m:
                t = float(m.group(1))
                if t == 0.0:
                    seen_t0 += 1
                if seen_t0 == 1:  # collect the schedule of the FIRST case only
                    dts.append(float(m.group(2)) / 86400.0)  # seconds -> days
                continue
            m = RE_DONE.search(line)
            if m:
                done = (float(m.group(1)), int(m.group(2)))
            m = RE_WARN.search(line)
            if m:
                warnings[f"{m.group(1)}: {normalise_warning(m.group(2))}"] += 1
            if "Traceback" in line or re.search(r"\bERROR\b", line):
                n_err += 1
            for key, rx in RE_HEADER.items():
                if key not in header:
                    mh = rx.match(line)
                    if mh:
                        header[key] = mh.group(1).strip()
    return dict(cases=cases, stages=stages, ksig=ksig, elapsed=elapsed,
                rates=rates, dts=dts, header=header, done=done,
                warnings=warnings, n_err=n_err)


def summarise(d, args):
    print("=" * 74)
    print(f"RUN LOG SUMMARY: {args.logfile}")
    print("=" * 74)
    h = d["header"]
    print(f"OpenMC {h.get('version','?')}   started {h.get('date','?')}   "
          f"{h.get('threads','?')} OpenMP threads")

    if d["stages"]:
        print("-" * 74)
        print("OPTIMIZATION PROGRESS")
        for name, total, hv in d["stages"]:
            print(f"  {name:<12} total evals={total:<4} HV={hv:.1f}")

    cases = d["cases"]
    if cases:
        print("-" * 74)
        efpd = [c["EFPD"] for c in cases]
        ceiling = max(efpd)
        n_ceil = sum(1 for e in efpd if abs(e - ceiling) < 0.51)
        print(f"CASES: {len(cases)}   EFPD min/median/max = "
              f"{min(efpd):.0f}/{st.median(efpd):.0f}/{ceiling:.0f}   "
              f"({n_ceil} cases at the {ceiling:.0f} EFPD schedule ceiling)")
        fdh = [c["F_dh"] for c in cases]
        print(f"       F_dh min/median/max = {min(fdh):.3f}/"
              f"{st.median(fdh):.3f}/{max(fdh):.3f}")
        kt = sorted(set(round(c["k_target"], 4) for c in cases))
        print(f"       interpolated k_target range: {kt[0]:.4f} - {kt[-1]:.4f}")

    if d["dts"]:
        print("-" * 74)
        sched = [x for x in d["dts"] if x > 0]
        print(f"DEPLETION SCHEDULE (first case): {len(sched)} steps, "
              f"total {sum(sched):.1f} EFPD")
        print("  dt [d]: " + ", ".join(f"{x:.0f}" for x in sched))
        print(f"  (largest step {max(sched):.0f} EFPD -> end-of-cycle "
              f"interpolation happens across steps this coarse)")

    if d["elapsed"]:
        print("-" * 74)
        n = len(d["elapsed"])
        print(f"TRANSPORT: {n} solves   mean {st.mean(d['elapsed']):.1f} s   "
              f"total {sum(d['elapsed'])/3600:.2f} h")
        if cases:
            print(f"           transport solves per case: {n / len(cases):.1f}")
        if d["rates"]:
            print(f"           mean active rate: {st.mean(d['rates']):.0f} "
                  f"particles/s")
        if d["ksig"]:
            print(f"           Monte Carlo sigma(k): mean {st.mean(d['ksig']):.0f} "
                  f"pcm  (min {min(d['ksig']):.0f} / max {max(d['ksig']):.0f})")

    if d["warnings"]:
        print("-" * 74)
        print("WARNINGS (collapsed by type):")
        for msg, cnt in d["warnings"].most_common(8):
            print(f"  {cnt:>6} x {msg}")
    print("-" * 74)
    print(f"ERROR/Traceback lines: {d['n_err']}")

    if d["done"]:
        wall, nev = d["done"]
        per = wall / nev
        print("-" * 74)
        print(f"WALL TIME: {wall:.0f} s = {wall/3600:.2f} h for {nev} evaluations "
              f"-> {per:.0f} s/eval ({per/60:.1f} min/eval)")
        if args.project:
            frac_tr = (sum(d["elapsed"]) / wall) if d["elapsed"] else 0.85
            scale = frac_tr * args.particle_scale + (1 - frac_tr)
            t_add = args.project * per * scale
            print(f"PROJECTION: +{args.project} evals at particle-scale "
                  f"x{args.particle_scale:g} (transport is {100*frac_tr:.0f}% of "
                  f"wall time -> effective x{scale:.2f}):")
            print(f"  ~{t_add/3600:.1f} h"
                  + (f"  ~${t_add/3600*args.hourly_rate:.0f} at "
                     f"${args.hourly_rate}/h" if args.hourly_rate else ""))
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", help="path to run.log")
    ap.add_argument("--csv", metavar="FILE",
                    help="also export the per-case table to this CSV file")
    ap.add_argument("--project", type=int, default=None, metavar="N",
                    help="project wall time/cost for N additional real evaluations")
    ap.add_argument("--particle-scale", type=float, default=1.0,
                    help="multiply the particle count by this factor in the "
                         "projection (e.g. 4 for 4000 -> 16000); only the "
                         "transport share of the wall time is scaled")
    ap.add_argument("--hourly-rate", type=float, default=None,
                    help="instance price in $/h to convert the projection to cost")
    args = ap.parse_args()

    d = parse(args.logfile)
    summarise(d, args)

    if args.csv and d["cases"]:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CASE_FIELDS)
            w.writeheader()
            w.writerows(d["cases"])
        print(f"per-case table -> {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
