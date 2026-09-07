#!/usr/bin/env python3
"""
valgrid_make_list.py -- the exhaustive grid of the Step 3 verification,
written as an --eval-list file for run_optimization.py, together with the
--freeze flags that define the slice.

THE SLICE
    Two live variables, enrichment and gadolinia weight fraction, on a
    regular grid. The other two Campaign 8 variables are frozen at the
    values of one anchor design from the Campaign 8 archive (default
    design 47, the central candidate): refl_thick at its archived value,
    gd_pins at its SNAPPED value (gd_pins_used), so the frozen count is
    exactly a rung of the Strategy-A ladder and no rounding ambiguity
    enters the evaluator.

THE GRID
    enrich  n_enr points from --enr-lo to --enr-hi (default 7 on 3.0 to
            7.0 wt%, step 0.667 wt%)
    gd_wt   n_gd  points from 0 to 8 wt% Gd2O3 (default 5, step 2 wt%)
    The default 7 x 5 = 35 nodes bracket the feasible band of the archive
    at 12 gadolinia rods and 4.3 cm of reflector: the k_min floor near
    3.4 wt%, the k_max / g_ctrl ceiling near 6 wt%, over the whole Gd
    range. The box is deliberately wider than the feasible band so the
    enumerated front sees its own boundaries.

ORDER
    Coarse first: the 3 x 3 sub-grid of corners, mid-points and centre is
    evaluated before the remaining nodes, so an interrupted run still
    yields a coarse front. Within a level, ascending enrichment then
    ascending gadolinia.

OUTPUT (default directory valgrid/)
    grid_list.json    {"designs": [...], "frozen": {...}, "grid": {...},
                       "anchor": {...}}   consumed by --eval-list
    freeze.txt        the exact --freeze flags for the shell driver
    box.txt           the exact --enr-box-low / --enr-box-high flags

    python valgrid_make_list.py --check          print, write nothing
    python valgrid_make_list.py                  write valgrid/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def coarse_first_order(n_e: int, n_g: int):
    """Indices (i, j) over an n_e x n_g grid, level 0 first (the 3x3
    corner / mid / centre sub-grid, or whatever exists), then the rest."""
    def marks(n):
        return sorted({0, n // 2, n - 1})
    lvl0 = {(i, j) for i in marks(n_e) for j in marks(n_g)}
    order = [(i, j) for i in range(n_e) for j in range(n_g) if (i, j) in lvl0]
    order += [(i, j) for i in range(n_e) for j in range(n_g)
              if (i, j) not in lvl0]
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--anchor", type=int, default=47,
                    help="archive index of the design whose refl_thick and "
                         "gd_pins_used are frozen (default 47)")
    ap.add_argument("--enr-lo", type=float, default=3.0)
    ap.add_argument("--enr-hi", type=float, default=7.0)
    ap.add_argument("--n-enr", type=int, default=7)
    ap.add_argument("--gd-lo", type=float, default=0.0)
    ap.add_argument("--gd-hi", type=float, default=8.0)
    ap.add_argument("--n-gd", type=int, default=5)
    ap.add_argument("--out", default="valgrid")
    ap.add_argument("--check", action="store_true", help="print, write nothing")
    a = ap.parse_args()

    ck = json.loads(Path(a.checkpoint).read_text())
    raw = ck["all_raw"]
    if not 0 <= a.anchor < len(raw):
        raise SystemExit(f"anchor {a.anchor} not in the archive of {len(raw)}")
    r = raw[a.anchor]
    refl = float(r["refl_thick"])
    pins = int(r["gd_pins_used"])
    frozen = {"refl_thick": round(refl, 6), "gd_pins": float(pins)}
    # the archive's own box for the two live variables, for the record
    box_gd = ck.get("meta", {}).get("enrichment_policy", {})
    enr = np.linspace(a.enr_lo, a.enr_hi, a.n_enr)
    gd = np.linspace(a.gd_lo, a.gd_hi, a.n_gd)
    order = coarse_first_order(a.n_enr, a.n_gd)
    designs = [{"enrich": round(float(enr[i]), 6),
                "gd_wt": round(float(gd[j]), 6),
                "grid_ij": [int(i), int(j)]} for i, j in order]
    payload = {
        "designs": designs,
        "frozen": frozen,
        "grid": {"enrich": enr.round(6).tolist(), "gd_wt": gd.round(6).tolist(),
                 "n_enr": a.n_enr, "n_gd": a.n_gd, "order": "coarse-first"},
        "anchor": {"index": a.anchor, "checkpoint": a.checkpoint,
                   "enrich": r["enrich"], "gd_wt": r["gd_wt"],
                   "refl_thick": refl, "gd_pins": r["gd_pins"],
                   "gd_pins_used": pins, "cycle_length": r["cycle_length"],
                   "peaking": r["peaking"]},
        "c8_enrichment_policy": box_gd,
    }
    freeze_flags = " ".join(f"--freeze {k}={v!r}" for k, v in frozen.items())
    box_flags = f"--enr-box-low {a.enr_lo:g} --enr-box-high {a.enr_hi:g}"

    print(f"anchor design {a.anchor}: enrich {r['enrich']:.4f} wt%, "
          f"gd_wt {r['gd_wt']:.4f}, refl_thick {refl:.4f} cm, "
          f"gd_pins_used {pins}, {r['cycle_length']:.0f} EFPD, "
          f"F_dH {r['peaking']:.4f}")
    print(f"frozen : {freeze_flags}")
    print(f"box    : {box_flags}")
    print(f"grid   : {a.n_enr} x {a.n_gd} = {len(designs)} nodes")
    print(f"enrich : {enr.round(4).tolist()}")
    print(f"gd_wt  : {gd.round(4).tolist()}")
    print(f"first 9: {[(d['enrich'], d['gd_wt']) for d in designs[:9]]}")
    if a.check:
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid_list.json").write_text(json.dumps(payload, indent=2))
    (out / "freeze.txt").write_text(freeze_flags + "\n")
    (out / "box.txt").write_text(box_flags + "\n")
    print(f"wrote {out / 'grid_list.json'}, {out / 'freeze.txt'}, "
          f"{out / 'box.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
