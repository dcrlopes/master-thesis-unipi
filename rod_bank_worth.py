#!/usr/bin/env python3
"""rod_bank_worth.py -- control rod BANKS for the 32-assembly core, inspired
by the NuScale-like benchmark, plus a regulating-bank controllability screen.

WHY A NEW SCRIPT AND NOT AN EDIT OF rod_worth_ladder.py
-------------------------------------------------------
The ladder's outside-in ORDER already produced the authority measurements in
rods_pos86/71/74/107, which the thesis will cite. Changing its ORDER would
make those logs irreproducible from the committed code. This script is the
operational counterpart: same solver, same conventions, bank semantics.

THE SIX BANKS (position lists, (row, col) on the 6x6 map, corners empty)
------------------------------------------------------------------------
Adapted from the NuScale-like benchmark (16 CRAs of 37 FAs, regulating
inboard, shutdown outboard), extended so EVERY assembly carries a CRA:
the four RE banks are the complete inner sixteen (C and M rings) and the
three SH banks are the complete outer sixteen (P ring), 32 CRAs of 32:

    RE1  inner ring (C)                 (2,2) (2,3) (3,2) (3,3)
    RE2  M-ring diagonals               (1,1) (1,4) (4,1) (4,4)
    RE3  M-ring edges, orbit A          (1,2) (2,4) (4,3) (3,1)
    RE4  M-ring edges, orbit B          (1,3) (3,4) (4,2) (2,1)
    SH3  P-ring edge mids, orbit A      (0,2) (2,5) (5,3) (3,0)
    SH4  P-ring edge mids, orbit B      (0,3) (3,5) (5,2) (2,0)
    SH5  P-ring corner-adjacent, both orbits (8 CRAs)
         (0,1) (1,5) (5,4) (4,0) (0,4) (4,5) (5,1) (1,0)

Every bank is closed under 90-degree rotation (SH5 is the union of the two
P-corner orbits), so every bank pattern and every prefix of the operational
sequence is exactly four-fold symmetric, and no assembly is left unrodded.

OPERATIONAL SEQUENCE
    RE1 -> RE2 -> RE3 -> RE4 -> SH3 -> SH4 -> SH5
Regulating banks first, inboard to outboard. The controllability criterion
is that the four RE banks alone hold the core subcritical by the operating
margin, because SH3 and SH4 are reserved for scram.

MODES
    --check                geometry self-test and ASCII map, no transport
    --sequence             the six-step operational insertion ladder
    --banks RE1,RE2        one state with exactly those banks inserted
    --screen I,J,K,...     controllability screen over archive designs:
                           unrodded plus ALL-RE per design, verdict per
                           design against --margin (pcm, default 1000)

Every transport state reports k, the worth in both conventions, and the
rodded F_dH, because rods that hold the core down while destroying the
radial power shape do not make a design controllable.

EXAMPLES
    lab python rod_bank_worth.py --check
    lab python -u rod_bank_worth.py --checkpoint out_c6/optimization_checkpoint.json \\
        --idx 86 --m-center 0.72 --m-periphery 1.15 --absorber B4C \\
        --sequence --seeds 2 --out banks_pos86
    lab python -u rod_bank_worth.py --checkpoint out_c6/optimization_checkpoint.json \\
        --screen 107,59,85,110,26,7,40,86 --m-center 0.72 --m-periphery 1.15 \\
        --absorber B4C --seeds 2 --out banks_screen
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "64")
import numpy as np

BANKS = {
    "RE1": [(2, 2), (2, 3), (3, 2), (3, 3)],
    "RE2": [(1, 1), (1, 4), (4, 1), (4, 4)],
    "RE3": [(1, 2), (2, 4), (4, 3), (3, 1)],
    "RE4": [(1, 3), (3, 4), (4, 2), (2, 1)],
    "SH3": [(0, 2), (2, 5), (5, 3), (3, 0)],
    "SH4": [(0, 3), (3, 5), (5, 2), (2, 0)],
    "SH5": [(0, 1), (1, 5), (5, 4), (4, 0),
            (0, 4), (4, 5), (5, 1), (1, 0)],
}
SEQUENCE = ["RE1", "RE2", "RE3", "RE4", "SH3", "SH4", "SH5"]
RE_BANKS = ["RE1", "RE2", "RE3", "RE4"]
CORE = [(r, c) for r in range(6) for c in range(6)
        if (r, c) not in [(0, 0), (0, 5), (5, 0), (5, 5)]]


def rot(p):
    r, c = p
    return (c, 5 - r)


def self_check() -> None:
    print("[check] bank geometry")
    seen = []
    for name, cells in BANKS.items():
        assert len(cells) in (4, 8), f"{name} has {len(cells)} cells"
        assert all(p in CORE for p in cells), f"{name} leaves the core map"
        u = set(cells)
        assert all(rot(p) in u for p in u), \
            f"{name} is not closed under 90-degree rotation"
        seen += cells
    assert len(seen) == len(set(seen)) == 32, \
        "banks must cover all 32 assemblies exactly once"
    n_cra = sum(len(c) for c in BANKS.values())
    print(f"    7 banks, {n_cra} CRAs of 32 assemblies, full coverage,")
    print(f"    every bank closed under 90-degree rotation")
    print("    map (bank name, or -- for no rod, .. outside the core):")
    grid = [[".." for _ in range(6)] for _ in range(6)]
    for r, c in CORE:
        grid[r][c] = "--"
    for name, cells in BANKS.items():
        for r, c in cells:
            grid[r][c] = name[:2] + name[-1] if len(name) > 3 else name
    for row in grid:
        print("      " + " ".join(f"{x:>3s}" for x in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--idx", type=int)
    ap.add_argument("--m-center", type=float)
    ap.add_argument("--m-periphery", type=float)
    ap.add_argument("--absorber", choices=["B4C", "AIC"], default="B4C")
    ap.add_argument("--sequence", action="store_true",
                    help="operational ladder RE1->RE2->RE3->RE4->SH3->SH4")
    ap.add_argument("--banks", default=None,
                    help="comma list of banks for ONE state, e.g. RE1,RE2")
    ap.add_argument("--screen", default=None,
                    help="comma list of archive indices for the RE-only "
                         "controllability screen")
    ap.add_argument("--margin", type=float, default=1000.0,
                    help="required subcriticality under ALL-RE, in pcm of "
                         "reactivity (default 1000)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--core-particles", type=int, default=100000)
    ap.add_argument("--core-batches", type=int, default=170)
    ap.add_argument("--core-inactive", type=int, default=60)
    ap.add_argument("--threads", type=int,
                    default=int(os.environ["OMP_NUM_THREADS"]))
    ap.add_argument("--out", default="banks_out")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        self_check()
        return
    self_check()

    for req in ("checkpoint", "m_center", "m_periphery"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_','-')} is required for "
                             f"transport modes")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import reactor_model as rm            # after thread env, like the ladder
    import zoning as zn
    from openmc_evaluator import _design_seed

    ck = json.loads(Path(args.checkpoint).read_text())
    dv = ck["design_variables"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    op, geo = rm.Operating(), rm.Geometry17x17()

    rmap = zn.ring_map()
    nC, nM, nP = zn.ring_counts(rmap)
    m_m = (32 - nC * args.m_center - nP * args.m_periphery) / nM
    print(f"multipliers C/M/P = {args.m_center:.3f}/{m_m:.4f}/"
          f"{args.m_periphery:.3f}  (balance-solved middle)")

    def make_state(design, design_map, seed0):
        def state(positions, tag):
            ks, fs = [], []
            for i in range(args.seeds):
                r = zn.core_bol_solve(
                    design, design_map, op, geo,
                    particles=args.core_particles,
                    batches=args.core_batches,
                    inactive=args.core_inactive, seed=seed0 + 7919 * i,
                    case=out / f"{tag}_s{i}",
                    rodded_map=((set(positions), args.absorber)
                                if positions else None))
                ks.append(r["keff"])
                fs.append(r["fdh_core"])
            return (float(np.mean(ks)),
                    float(np.std(ks, ddof=1)) if args.seeds > 1 else 0.0,
                    float(np.mean(fs)),
                    float(np.std(fs, ddof=1)) if args.seeds > 1 else 0.0)
        return state

    def rho_pcm(k0, k):
        return 1e5 * (1.0 / k - 1.0 / k0)

    def load_design(idx):
        d = {k: float(ck["all_raw"][idx][k]) for k in dv}
        zdes = zn.zone_designs(d, args.m_center, m_m, args.m_periphery)
        return d, zn.design_map_for(rmap, zdes)

    results = {"banks": {k: v for k, v in BANKS.items()},
               "absorber": args.absorber, "margin_pcm": args.margin,
               "multipliers": dict(C=args.m_center, M=m_m,
                                   P=args.m_periphery), "states": []}

    if args.screen:
        idxs = [int(x) for x in args.screen.split(",")]
        re_pos = [p for b in RE_BANKS for p in BANKS[b]]
        print(f"[screen] ALL-RE (16 CRAs) controllability, margin "
              f"{args.margin:.0f} pcm, {len(idxs)} designs")
        for idx in idxs:
            design, dmap = load_design(idx)
            st = make_state(design, dmap, _design_seed(design, salt="banks"))
            k0, k0s, F0, F0s = st([], f"i{idx}_ARO")
            kr, krs, Fr, Frs = st(re_pos, f"i{idx}_ALLRE")
            rho = rho_pcm(k0, kr)
            need = 1e5 * (1.0 - 1.0 / k0)          # excess in reactivity pcm
            marg = -1e5 * (1.0 - 1.0 / kr)         # subcritical margin, pcm
            verdict = "CONTROLLABLE" if marg >= args.margin else \
                      ("subcritical, thin margin" if kr < 1.0
                       else "NOT controllable by RE banks")
            print(f"  idx {idx:>3}: k0={k0:.5f}  k(ALL-RE)={kr:.5f}  "
                  f"RE worth={rho:8.0f} pcm  excess={need:7.0f} pcm  "
                  f"margin={marg:7.0f} pcm  F(ALL-RE)={Fr:.3f}  {verdict}")
            results["states"].append(dict(
                mode="screen", idx=idx, k0=k0, k0_sd=k0s, F0=F0,
                k_allre=kr, k_sd=krs, F_allre=Fr, F_sd=Frs,
                re_worth_pcm=rho, excess_pcm=need, margin_pcm=marg,
                verdict=verdict))
    else:
        if args.idx is None:
            raise SystemExit("--idx required for --sequence or --banks")
        design, dmap = load_design(args.idx)
        st = make_state(design, dmap, _design_seed(design, salt="banks"))
        k0, k0s, F0, F0s = st([], "ARO")
        print(f"ARO           : k={k0:.5f}+/-{k0s:.5f}  F={F0:.4f}"
              f"+/-{F0s:.4f}")
        results["states"].append(dict(mode="ARO", idx=args.idx, k=k0,
                                      k_sd=k0s, F=F0, F_sd=F0s))
        steps = ([",".join(SEQUENCE[:i + 1]) for i in range(len(SEQUENCE))]
                 if args.sequence else [args.banks])
        for spec in steps:
            names = [s.strip() for s in spec.split(",")]
            bad = [n for n in names if n not in BANKS]
            if bad:
                raise SystemExit(f"unknown banks {bad}, choose from "
                                 f"{sorted(BANKS)}")
            pos = [p for n in names for p in BANKS[n]]
            k, ks, F, Fs = st(pos, "B_" + "_".join(names))
            print(f"{spec:14s}: k={k:.5f}+/-{ks:.5f}  "
                  f"worth={rho_pcm(k0, k):8.0f} pcm  dk={1e5*(k0-k):8.0f} "
                  f"pcm  F={F:.4f}+/-{Fs:.4f}")
            results["states"].append(dict(mode="banks", idx=args.idx,
                                          banks=names, k=k, k_sd=ks, F=F,
                                          F_sd=Fs,
                                          rho_pcm=rho_pcm(k0, k)))
    path = out / f"banks_{args.absorber}.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
