#!/usr/bin/env python3
"""verify_zoned_evaluator.py -- prove the zoned evaluator before Campaign 6.

Three tiers, increasing cost. Run on wks720 in the openmc-env environment.

TIER 1  policy arithmetic (no OpenMC import, instant)
        The frozen map is m_C = 0.720, m_M balanced, m_P from leu_policy.
        Checks the balance closes at exactly 1, the ring counts are
        4 / 12 / 16, and the periphery at the top of the search box lands
        exactly on the LEU cap.

TIER 2  model assembly (imports openmc, builds in memory, no transport,
        seconds)
        Builds one zoned 32-assembly core for a mid-box probe design and
        confirms the design_map covers every fuel position with the three
        ring designs.

TIER 3  champion re-solve (--solve, one BOL transport, minutes)
        Rebuilds an archived Campaign 5 design (default idx 54) and runs
        the zoned core BOL solve at the frozen map. The zoned study
        measured F = 1.4863 for idx54 at this same map, so agreement
        within Monte Carlo noise proves the evaluator path reproduces the
        study that justified it.

USAGE
    python verify_zoned_evaluator.py               # tiers 1 and 2
    python verify_zoned_evaluator.py --tier1-only  # no openmc import
    python verify_zoned_evaluator.py --solve       # add tier 3, idx 54
    python verify_zoned_evaluator.py --solve --idx 58 --particles 100000
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path


def tier1() -> None:
    import leu_policy as _leu
    n_c, n_m, n_p = 4, 12, 16
    m_c = 0.720
    m_p = _leu.M_P_DESIGN
    m_m = (n_c + n_m + n_p - n_c * m_c - n_p * m_p) / n_m
    bal = (n_c * m_c + n_m * m_m + n_p * m_p) / (n_c + n_m + n_p)
    top = _leu.E_SEARCH_MAX * m_p
    print("[tier 1] frozen-map arithmetic")
    print(f"    m_C = {m_c:.4f}   m_M = {m_m:.4f} (balanced)   "
          f"m_P = {m_p:.4f} (from leu_policy)")
    print(f"    ring counts assumed  : {n_c} / {n_m} / {n_p}")
    print(f"    core-average multiplier = {bal:.12f}")
    assert abs(bal - 1.0) < 1e-12, "fissile balance does not close"
    print(f"    periphery at box top : E_SEARCH_MAX x m_P = "
          f"{_leu.E_SEARCH_MAX:.4f} x {m_p:.4f} = {top:.4f} wt%")
    assert abs(top - _leu.LEU_CAP_WTPC) < 1e-9, \
        "box top does not map onto the LEU cap"
    print(f"    equals the LEU cap {_leu.LEU_CAP_WTPC} wt%: ok")
    if m_p == 1.0:
        print("    WARNING: m_P = 1.0. The evaluator will zone with a "
              "peripheral multiplier of 1, which is NOT the Campaign 6 "
              "intent. Regenerate leu_policy.py with --m-p 1.15.")


def tier2() -> None:
    import zoning as zn
    print("\n[tier 2] frozen map against the REAL ring map, then one "
          "zoned model build")
    rmap, m_c, m_m, m_p = zn.evaluator_multipliers()
    counts = zn.ring_counts(rmap)
    print(f"    ring counts from ring_map(): {counts[0]} / {counts[1]} / "
          f"{counts[2]}")
    assert counts == (4, 12, 16), \
        f"ring counts {counts} differ from the (4, 12, 16) the study used"
    print(f"    evaluator_multipliers(): m_C = {m_c:.4f}   "
          f"m_M = {m_m:.4f}   m_P = {m_p:.4f}")

    probe = dict(enrich_inner=10.0, enrich_outer=12.0, gd_wt=4.0,
                 pitch=1.26, refl_thick=10.0, gd_pins=16)
    dmap = zn.evaluator_design_map(probe)
    n_pos = sum(1 for _ in dmap)
    zones = {d["zone"] for d in dmap.values()}
    print(f"    design_map covers {n_pos} fuel positions, zones {sorted(zones)}")
    assert n_pos == 32, f"expected 32 fuel positions, got {n_pos}"
    assert zones == {"C", "M", "P"}, f"unexpected zone set {zones}"
    e_top = max(d["enrich_outer"] for d in dmap.values())
    print(f"    highest as-built enrichment of the probe: {e_top:.4f} wt% "
          f"(= 12.0 x {m_p:.3f})")

    import reactor_model as rm
    # same construction refine_zoning.py and rescore_zoned_core.py use
    geo, op = rm.Geometry17x17(), rm.Operating()
    m = rm.make_core_model(probe, op, geo, design_map=dmap)
    model = m[0] if isinstance(m, tuple) else m
    n_mats = len(model.materials) if model.materials else "auto"
    print(f"    zoned model assembled in memory (materials: {n_mats}). ok")


def tier3(idx: int, particles: int, batches: int, inactive: int) -> None:
    import zoning as zn
    import reactor_model as rm
    print(f"\n[tier 3] zoned BOL re-solve of archived design idx {idx}")
    rows = {r["idx"]: r for r in
            csv.DictReader(open("campaign5/c5_full.csv"))}
    if str(idx) not in rows:
        sys.exit(f"ABORT: idx {idx} not in campaign5/c5_full.csv")
    r = rows[str(idx)]
    design = dict(enrich_inner=float(r["enrich_inner"]),
                  enrich_outer=float(r["enrich_outer"]),
                  gd_wt=float(r["gd_wt"]), pitch=float(r["pitch"]),
                  refl_thick=float(r["refl_thick"]),
                  gd_pins=float(r["gd_pins"]))
    dmap = zn.evaluator_design_map(design)
    geo, op = rm.Geometry17x17(), rm.Operating()
    out = zn.core_bol_solve(design, dmap, op, geo, particles=particles,
                            batches=batches, inactive=inactive, seed=1,
                            case=Path("verify_zoned_ev") / f"idx{idx}")
    print(f"    F_dH(zoned) = {out['fdh_core']:.4f}   "
          f"k(zoned) = {out['keff']:.5f}   wall {out['wall_s']:.0f} s")
    ref = {"54": 1.4863, "58": 1.4495, "7": 1.4977, "35": 1.5613}
    if str(idx) in ref:
        print(f"    zoned-study value at the same map: F = {ref[str(idx)]}")
        print(f"    difference: {out['fdh_core'] - ref[str(idx)]:+.4f} "
              f"(expect within Monte Carlo noise, order 0.02)")
        print(f"    unzoned C5 value for comparison  : "
              f"F_core = {float(r['F_core']):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1-only", action="store_true",
                    help="run only the arithmetic checks, no openmc import")
    ap.add_argument("--solve", action="store_true",
                    help="add the tier 3 transport re-solve")
    ap.add_argument("--idx", type=int, default=54,
                    help="archived Campaign 5 design for tier 3 "
                         "(default 54)")
    ap.add_argument("--particles", type=int, default=200000,
                    help="particles per batch for tier 3 (default 200000, "
                         "the confirmation-run fidelity)")
    ap.add_argument("--batches", type=int, default=170,
                    help="total batches for tier 3 (default 170)")
    ap.add_argument("--inactive", type=int, default=60,
                    help="inactive batches for tier 3 (default 60)")
    args = ap.parse_args()

    tier1()
    if args.tier1_only:
        return
    tier2()
    if args.solve:
        tier3(args.idx, args.particles, args.batches, args.inactive)


if __name__ == "__main__":
    main()
