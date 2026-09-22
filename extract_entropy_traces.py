#!/usr/bin/env python3
"""extract_entropy_traces.py -- Shannon-entropy traces for the source-convergence figure.

Walks the given campaign workdirs (the --workdir of run_optimization.py, for
example openmc_runs_c9) for the BOL statepoints of openmc_evaluator:

    case_XXXX/bol/statepoint.*.h5        reflective single-assembly solve
    case_XXXX/core_bol/statepoint.*.h5   finite-core solve

Depletion chunks (case_XXXX/dep_NN) and any other folder are skipped. For each
statepoint it reads the per-batch Shannon entropy and applies the convergence
sentinel of openmc_evaluator (3-point smoothed entropy inside mean +/- 3 sd of
the late active batches). It writes entropy_traces.json with

    summary   one record per statepoint (path, campaign, kind, conv, ...)
    stats     per campaign and kind: count, median and max conv, and how many
              solves converged after the end of the inactive batches
    chosen    full traces of three representative cases
                assembly       the median-convergence assembly solve
                core_typical   the median-convergence core solve
                core_slowest   the latest-converging core solve

Reads files only, runs no transport.

RUN (wks720, conda env openmc-env):
    python -u extract_entropy_traces.py openmc_runs openmc_runs_c3 \
        openmc_runs_c5 openmc_runs_c8 openmc_runs_c9 --out entropy_traces.json
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import openmc

FOLDER_KIND = {"bol": "assembly", "core_bol": "core"}


def converged_batch(H, n_inactive):
    """Same rule as openmc_evaluator (core_bol sentinel)."""
    tail = H[n_inactive + (len(H) - n_inactive) // 2:]
    mu, sd = float(tail.mean()), float(tail.std(ddof=1))
    Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
    Hs[0], Hs[-1] = H[0], H[-1]
    bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
    return int(bad[-1]) + 2 if len(bad) else 1


def group_stats(recs):
    conv = [r["conv"] for r in recs]
    late = [r for r in recs if r["conv"] > r["n_inactive"]]
    return dict(n=len(recs), n_inactive=sorted({r["n_inactive"] for r in recs}),
                conv_median=st.median(conv), conv_max=max(conv),
                n_after_inactive=len(late),
                after_inactive=[r["path"] for r in late])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="entropy_traces.json")
    args = ap.parse_args()

    records, traces, n_skip = [], {}, 0
    for root in args.roots:
        if not Path(root).is_dir():
            raise SystemExit(f"not a directory: {root}")
        for sp_path in sorted(Path(root).rglob("statepoint.*.h5")):
            kind = FOLDER_KIND.get(sp_path.parent.name)
            if kind is None:
                continue                      # depletion chunks, other studies
            try:
                with openmc.StatePoint(str(sp_path)) as sp:
                    H = np.asarray(getattr(sp, "entropy", []), dtype=float)
                    n_inact = int(sp.n_inactive)
                    keff = float(sp.keff.nominal_value)
            except Exception as exc:          # unreadable file
                print(f"skip {sp_path}: {exc}")
                n_skip += 1
                continue
            if H.size < n_inact + 4:          # no entropy mesh or too short
                print(f"skip {sp_path}: {H.size} entropy values, {n_inact} inactive")
                n_skip += 1
                continue
            conv = converged_batch(H, n_inact)
            records.append(dict(path=str(sp_path), campaign=Path(root).name, kind=kind,
                                n_batches=int(H.size), n_inactive=n_inact,
                                conv=conv, keff=keff))
            traces[str(sp_path)] = H.tolist()
            print(f"{kind:8s} conv {conv:4d} / inactive {n_inact:3d}  {sp_path}", flush=True)

    if not records:
        raise SystemExit(f"no BOL statepoints with entropy found under {args.roots}")

    core = sorted((r for r in records if r["kind"] == "core"), key=lambda r: r["conv"])
    asm = sorted((r for r in records if r["kind"] == "assembly"), key=lambda r: r["conv"])
    chosen = {}
    if asm:
        chosen["assembly"] = dict(asm[len(asm) // 2])
    if core:
        chosen["core_typical"] = dict(core[len(core) // 2])
        chosen["core_slowest"] = dict(core[-1])
    for r in chosen.values():
        r["entropy"] = traces[r["path"]]

    stats = {}
    for camp in dict.fromkeys(r["campaign"] for r in records):
        for kind in ("assembly", "core"):
            g = [r for r in records if r["campaign"] == camp and r["kind"] == kind]
            if g:
                stats[f"{camp}/{kind}"] = group_stats(g)
    for kind, g in (("assembly", asm), ("core", core)):
        if g:
            stats[f"ALL/{kind}"] = group_stats(g)

    Path(args.out).write_text(json.dumps(dict(summary=records, stats=stats, chosen=chosen),
                                         indent=1))
    print(f"\n{len(records)} statepoints ({len(core)} core, {len(asm)} assembly), "
          f"{n_skip} skipped -> {args.out}\n")
    print(f"{'group':32s} {'n':>4s} {'inactive':>10s} {'median':>7s} {'max':>5s} {'late':>5s}")
    for k, s in stats.items():
        print(f"{k:32s} {s['n']:4d} {str(s['n_inactive']):>10s} "
              f"{s['conv_median']:7.1f} {s['conv_max']:5d} {s['n_after_inactive']:5d}")
    print()
    for k, r in chosen.items():
        print(f"{k}: conv {r['conv']} of {r['n_batches']} batches "
              f"({r['n_inactive']} inactive), {r['path']}")


if __name__ == "__main__":
    main()
