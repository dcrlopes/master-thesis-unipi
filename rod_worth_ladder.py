#!/usr/bin/env python3
"""
rod_worth_ladder.py -- control rod worth and rodded peaking of the zoned core.

Solves the zoned champion core (2D, fully inserted rods only, which is the
axially uniform state 2D represents fairly) for a ladder of rodded-assembly
counts and reports, for every state, BOTH the reactivity worth and the
radial enthalpy-rise peaking factor, because rods that hold the core down
while pushing F above its limit do not make the design controllable.

Rodded patterns are chosen symmetric, filling from the outside in, because
shutdown banks at the periphery cost the least peaking. The stuck case
removes the single highest-worth assembly of the winning count (measured,
one extra solve per candidate position on the symmetric orbit).

Usage:
  python rod_worth_ladder.py --checkpoint out_c5/optimization_checkpoint.json \
      --idx 7 --m-center 0.72 --m-periphery 1.250 --absorber B4C \
      --ladder 0,8,16,24,32 --seeds 2 --threads 64 --out rods_idx7

Flags: --absorber B4C or AIC; --ladder comma list of rodded counts;
--seeds transport seeds per state; --core-particles/batches/inactive
override the core solve fidelity; --out output directory.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "64")
import numpy as np

import reactor_model as rm
import zoning as zn
from openmc_evaluator import _design_seed

# symmetric fill order, outside in: corners of the P ring first, then P
# edges, then the M ring, then C. (row, col) on the 6x6 map, corners empty.
ORDER = [(0, 1), (0, 4), (1, 0), (1, 5), (4, 0), (4, 5), (5, 1), (5, 4),
         (0, 2), (0, 3), (2, 0), (2, 5), (3, 0), (3, 5), (5, 2), (5, 3),
         (1, 1), (1, 4), (4, 1), (4, 4),
         (1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3),
         (2, 2), (2, 3), (3, 2), (3, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--m-center", type=float, required=True)
    ap.add_argument("--m-periphery", type=float, required=True)
    ap.add_argument("--absorber", choices=("B4C", "AIC"), default="B4C")
    ap.add_argument("--ladder", default="0,8,16,24,32")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--core-particles", type=int, default=100000)
    ap.add_argument("--core-batches", type=int, default=170)
    ap.add_argument("--core-inactive", type=int, default=70)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", default="rods")
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    ck = json.loads(Path(args.checkpoint).read_text())
    dv = ck["design_variables"]
    design = {k: float(ck["all_raw"][args.idx][k]) for k in dv}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rmap = zn.ring_map()
    nC, nM, nP = zn.ring_counts(rmap)
    m_m = (32 - nC * args.m_center - nP * args.m_periphery) / nM
    zdes = zn.zone_designs(design, args.m_center, m_m, args.m_periphery)
    design_map = zn.design_map_for(rmap, zdes)
    print(f"multipliers C/M/P = {args.m_center:.3f}/{m_m:.4f}/"
          f"{args.m_periphery:.3f}  (balance-solved middle)")
    op, geo = rm.Operating(), rm.Geometry17x17()
    seed0 = _design_seed(design, salt="rods")

    def state(rodded_positions, tag):
        ks, fs = [], []
        for i in range(args.seeds):
            r = zn.core_bol_solve(
                design, design_map, op, geo,
                particles=args.core_particles, batches=args.core_batches,
                inactive=args.core_inactive, seed=seed0 + 7919 * i,
                case=out / f"{tag}_s{i}",
                rodded_map=((set(rodded_positions), args.absorber)
                            if rodded_positions else None))
            ks.append(r["keff"])
            fs.append(r["fdh_core"])
        return (float(np.mean(ks)), float(np.std(ks, ddof=1)) if args.seeds > 1
                else 0.0, float(np.mean(fs)), float(np.std(fs, ddof=1))
                if args.seeds > 1 else 0.0)

    ladder = [int(x) for x in args.ladder.split(",")]
    res = {}
    k0 = None
    for n in ladder:
        pos = ORDER[:n]
        k, ksd, F, Fsd = state(pos, f"N{n:02d}")
        if n == 0:
            k0 = k
        worth = (k0 - k) * 1e5 if k0 is not None else float("nan")
        res[n] = dict(k=k, k_sd=ksd, F=F, F_sd=Fsd, worth_pcm=worth,
                      positions=pos)
        print(f"N={n:2d} rodded: k={k:.5f}+/-{ksd:.5f}  "
              f"worth={worth:7.0f} pcm  F={F:.4f}+/-{Fsd:.4f}", flush=True)

    (out / f"ladder_idx{args.idx}_{args.absorber}.json").write_text(
        json.dumps(dict(idx=args.idx, absorber=args.absorber,
                        multipliers=dict(C=args.m_center, M=None,
                                         P=args.m_periphery),
                        ladder=res), indent=2, default=str))
    print(f"\nwrote {out}/ladder_idx{args.idx}_{args.absorber}.json")
    print("next: pick the smallest N whose worth exceeds the cold excess plus"
          " 1000 pcm, then rerun with --ladder N and remove one assembly at a"
          " time from its pattern for the stuck-rod worth.")


if __name__ == "__main__":
    main()
