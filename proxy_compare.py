#!/usr/bin/env python3
"""
proxy_compare.py -- compare cheaper cycle-length proxies with the eight-layer
core depletion at 20 000 x 160, on the designs that have it. No transport.

Reference   mean over the seeds of every eight-layer core depletion at
            20 000 x 160 found in */runs.json
Proxies     campaign    assembly depletion stored in the checkpoint
            lofi        eight-layer core depletion at 10 000 x 100 (lofi_dep_core3d*)
            asm3d_*     single-assembly depletions of c9_dep_asm3d.py (asm3d_*)

For each proxy, on the designs it shares with the reference:
    n, mean wall time, bias and scatter of (proxy - reference), Spearman rank
    correlation, scatter after a one-factor calibration fitted leave-one-out,
    wrong feasibility decisions against 1826 d (all designs, and designs whose
    reference margin is larger than 2 seed standard deviations).

usage (repository root):  python proxy_compare.py [--checkpoint out_c9a/optimization_checkpoint.json]
"""
import argparse
import glob
import json
import os
import statistics as st

from zoning import spearman

REQ, SEED_SD = 1826.0, 15.9          # d; pooled seed s.d. of the reference, 20 depletions


def load(pattern, keep):
    """{design: [(efpd, wall_min), ...]} over the runs.json files matching `pattern`."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        try:
            R = json.load(open(f))
        except Exception:
            continue
        for k, r in R.items():
            if isinstance(r, dict) and k.startswith("d") and "efpd" in r and keep(f, r):
                out.setdefault(int(k[1:]), []).append((r["efpd"], r.get("wall_s", 0.0) / 60.0))
    return out


def is_ref(f, r):
    tr = r.get("transport") or {}
    return (r.get("layers") == 8 and tr.get("particles") == 20000 and tr.get("batches") == 160
            and r.get("model") != "assembly3d" and not os.path.dirname(f).startswith(("gdstudy", "lofi", "asm3d")))


def metrics(name, proxy, ref, wall):
    ids = sorted(set(proxy) & set(ref))
    if len(ids) < 3:
        return f"{name:22s} n {len(ids)}: fewer than 3 common designs"
    p = [proxy[i] for i in ids]; r = [ref[i] for i in ids]
    d = [a - b for a, b in zip(p, r)]
    # one-factor calibration, leave-one-out: reference ~ c x proxy
    loo = []
    for j in range(len(ids)):
        c = sum(r[k] for k in range(len(ids)) if k != j) / sum(p[k] for k in range(len(ids)) if k != j)
        loo.append(c * p[j] - r[j])
    c_all = sum(r) / sum(p)
    wrong = [i for i, a, b in zip(ids, p, r) if (c_all * a >= REQ) != (b >= REQ)]
    wrong_clear = [i for i in wrong if abs(ref[i] - REQ) > 2 * SEED_SD]
    return (f"{name:22s} n {len(ids):2d}  wall {wall:5.1f} min  bias {st.mean(d):+7.1f} d  scatter {st.stdev(d):5.1f} d  "
            f"Spearman {spearman(p, r):+.3f}  calibrated (x{c_all:.3f}) LOO rms {(sum(x * x for x in loo) / len(loo)) ** 0.5:5.1f} d  "
            f"wrong decisions {len(wrong)} {wrong}, of which clear {len(wrong_clear)} {wrong_clear}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out_c9a/optimization_checkpoint.json")
    a = ap.parse_args()
    raw = json.load(open(a.checkpoint))["all_raw"]
    refs = load("*/runs.json", is_ref)
    ref = {i: st.mean(e for e, _ in v) for i, v in refs.items()}
    print(f"reference: {len(ref)} designs, eight-layer core depletion at 20 000 x 160")
    for i in sorted(ref):
        print(f"  C9-{i:<2d} {ref[i]:7.1f} d  ({len(refs[i])} seed(s))  margin {ref[i] - REQ:+6.1f} d")
    lines = [metrics("campaign assembly", {i: raw[i]["cycle_length"] for i in ref}, ref,
                     st.mean(raw[i]["t_deplete_s"] for i in ref) / 60.0)]
    groups = {"lofi core, 8 layers": "lofi_dep_core3d*/runs.json"}
    for d in sorted(glob.glob("asm3d_*")):
        if os.path.exists(os.path.join(d, "runs.json")):
            groups[d] = os.path.join(d, "runs.json")
    for name, pat in groups.items():
        runs = load(pat, lambda f, r: True)
        if runs:
            lines.append(metrics(name, {i: st.mean(e for e, _ in v) for i, v in runs.items()}, ref,
                                 st.mean(w for v in runs.values() for _, w in v)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
