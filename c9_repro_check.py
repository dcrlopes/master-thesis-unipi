#!/usr/bin/env python3
r"""
c9_repro_check.py -- is a Campaign 9 assembly depletion bit-reproducible
with the campaign seed, and if not, why?

WHAT WAS FOUND WITHOUT TRANSPORT (26 Sep, c9_dep_replicas/runs.json against
kh_c9/k_histories.json)
    The rerun of each front design with the campaign seed gives the same
    beginning-of-life k_inf as the archive to the last one or two digits
    (1e-15 to 1e-14), and then diverges from the first depletion step on,
    by amounts of the size of the statistical noise (|z| < 2.6 on every
    step). Same OpenMC 0.15.3, same host, same 64 threads, same schedule,
    same target.
    Hypothesis: the tallies are accumulated by several threads, so the sum
    order and its last bits change from run to run; the reaction rates that
    feed the depletion therefore differ in the last bits, the depleted
    nuclide densities differ in the last bits, and the particle histories
    of the next step diverge completely. With one thread the sums are
    ordered and the run should be bit-identical.

WHAT IT RUNS
    The assembly depletion of the campaign evaluator (OpenMCEvaluator.
    _cycle_length, configured from the checkpoint as c9_dep_replicas.py
    does) for one design with its campaign seed, the campaign transport
    4000 x 60 / 20, and a SHORTENED schedule: the first two steps of the
    beginning-of-life block (0.5 and 1.0 MWd/kgHM), so three transport
    solves. It stores every k_inf with full precision.

USAGE (wks720, conda env openmc-env, repository root)
    python c9_repro_check.py --design 47 --threads 64 --tag t64a
    python c9_repro_check.py --design 47 --threads 64 --tag t64b
    python c9_repro_check.py --design 47 --threads 1 --particles 1000 --tag t1a
    python c9_repro_check.py --design 47 --threads 1 --particles 1000 --tag t1b
      (the single-thread pair runs at 1000 x 30 / 10: it only has to agree
       with itself, and at 4000 x 60 it would take about three hours each)
    python c9_repro_check.py --compare                 # table of all tags + archive

EXPECTED IF THE HYPOTHESIS HOLDS
    t64a vs t64b : step 0 equal to ~1e-15, steps 1 and 2 differ by ~100 pcm
    t1a  vs t1b  : every step bit-identical
    t64 vs archive: as t64a vs t64b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

OUT = Path("repro_c9")


def run(a):
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import openmc
    from c9_dep_replicas import CAMPAIGN, design_of, load_ckpt, make_evaluator
    ckpt, _raw, _meta = load_ckpt(a.checkpoint)       # load_ckpt returns (checkpoint, all_raw, meta)
    design, spec = design_of(ckpt, a.design)
    work = OUT / "work" / a.tag
    transport = dict(CAMPAIGN)
    if a.particles:
        transport.update(particles=a.particles, batches=a.batches, inactive=a.inactive)
    ev = make_evaluator(ckpt, spec, work, transport)
    ev.bol_steps = [0.5, 1.0]           # the first two steps of the campaign BOL block
    ev.max_burnup = 1.6                 # stop after them (must exceed their sum)
    t0 = time.time()
    ev._cycle_length(design, work / "case")
    k = [float(v) for v in ev._last_k_hist]
    bu = [float(v) for v in ev._last_bu_hist]
    import openmc_evaluator as oe
    res = dict(tag=a.tag, design=a.design, threads=a.threads, transport=transport, seed=int(oe._design_seed(design)),
               openmc=openmc.__version__, bu=bu, k=[repr(v) for v in k],
               wall_s=time.time() - t0)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[{a.tag}] threads {a.threads}, seed {res['seed']}, bu {bu}")
    print(f"[{a.tag}] k " + "  ".join(res["k"]) + f"   ({res['wall_s'] / 60:.1f} min)")


def compare(a):
    rows = {}
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        rows[r["tag"]] = [float(x) for x in r["k"]]
    kh = json.loads(Path("kh_c9/k_histories.json").read_text())[str(a.design)]
    rows["archive"] = [float(x) for x in kh["k"][:3]]
    # only runs at the campaign transport are comparable with the archive
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("transport", {}).get("particles") != 4000:
            print(f"note: {r['tag']} ran at {r['transport']}, compare it only with its own pair")
    tags = list(rows)
    print("step  " + "  ".join(f"{t:>20s}" for t in tags))
    for i in range(3):
        print(f"{i:4d}  " + "  ".join(f"{rows[t][i]:20.16f}" if i < len(rows[t]) else " " * 20 for t in tags))
    print("\npairwise difference per step, pcm (0 = bit-identical)")
    for x in range(len(tags)):
        for y in range(x + 1, len(tags)):
            p, q = rows[tags[x]], rows[tags[y]]
            n = min(len(p), len(q))
            d = [p[i] - q[i] for i in range(n)]
            txt = "  ".join("0 (identical)" if v == 0 else f"{1e5 * v:+.3g}" for v in d)
            print(f"  {tags[x]:>8s} vs {tags[y]:<8s}: {txt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--design", type=int, default=47)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--particles", type=int, default=None,
                    help="reduced transport for the single-thread pair, which only has to agree with itself")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--inactive", type=int, default=10)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare(a)
        return 0
    if not a.tag:
        ap.error("--tag is required for a run")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
