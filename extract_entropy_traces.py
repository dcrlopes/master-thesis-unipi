#!/usr/bin/env python3
"""extract_entropy_traces.py -- Shannon-entropy traces for the source-convergence figure.

Walks the given directories for OpenMC statepoints, reads the per-batch
Shannon entropy of each, and applies the convergence sentinel of
openmc_evaluator (3-point smoothed entropy inside mean +/- 3 sd of the late
active batches). It writes entropy_traces.json with a summary of every
statepoint and the full trace of three representative cases:

    assembly       a reflective single-assembly solve (path without "core")
    core_typical   the core solve with the median convergence batch
    core_slowest   the core solve with the latest convergence batch

Reads files only, runs no transport.

RUN (wks720, conda env openmc-env):
    python -u extract_entropy_traces.py out_c4 rescore_c4 out_c6 out_c8 out_c9 \
        --out entropy_traces.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openmc


def converged_batch(H, n_inactive):
    """Same rule as openmc_evaluator (core_bol sentinel)."""
    tail = H[n_inactive + (len(H) - n_inactive) // 2:]
    mu, sd = float(tail.mean()), float(tail.std(ddof=1))
    Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
    Hs[0], Hs[-1] = H[0], H[-1]
    bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
    return int(bad[-1]) + 2 if len(bad) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="entropy_traces.json")
    args = ap.parse_args()

    records, traces = [], {}
    for root in args.roots:
        for sp_path in sorted(Path(root).rglob("statepoint.*.h5")):
            try:
                with openmc.StatePoint(str(sp_path)) as sp:
                    H = np.asarray(getattr(sp, "entropy", []), dtype=float)
                    n_inact = int(sp.n_inactive)
                    keff = float(sp.keff.nominal_value)
            except Exception as exc:          # unreadable or no entropy mesh
                print(f"skip {sp_path}: {exc}")
                continue
            if H.size < n_inact + 4:
                continue
            kind = "core" if "core" in str(sp_path).lower() else "assembly"
            conv = converged_batch(H, n_inact)
            records.append(dict(path=str(sp_path), kind=kind, n_batches=int(H.size),
                                n_inactive=n_inact, conv=conv, keff=keff))
            traces[str(sp_path)] = H.tolist()
            print(f"{kind:8s} conv {conv:4d} / inactive {n_inact:3d}  {sp_path}")

    core = sorted((r for r in records if r["kind"] == "core"), key=lambda r: r["conv"])
    asm = sorted((r for r in records if r["kind"] == "assembly"), key=lambda r: r["conv"])
    chosen = {}
    if asm:
        chosen["assembly"] = asm[0]
    if core:
        chosen["core_typical"] = core[len(core) // 2]
        chosen["core_slowest"] = core[-1]
    for r in chosen.values():
        r["entropy"] = traces[r["path"]]

    Path(args.out).write_text(json.dumps(dict(summary=records, chosen=chosen), indent=1))
    print(f"\n{len(records)} statepoints, {len(core)} core, {len(asm)} assembly -> {args.out}")
    for k, r in chosen.items():
        print(f"{k}: conv {r['conv']} of {r['n_batches']} batches, {r['path']}")


if __name__ == "__main__":
    main()
