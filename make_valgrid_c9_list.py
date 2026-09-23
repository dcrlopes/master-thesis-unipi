#!/usr/bin/env python3
"""
make_valgrid_c9_list.py -- the enumeration node list for the Campaign 9
verification of the surrogate-assisted search (Section 4.14.3 of the thesis,
which so far rests on Campaign 8 alone).

SLICE
  live      enrich, gd_wt            the same pair as the Campaign 8 slice,
                                     so the two verifications are comparable
  frozen    refl_thick, gd_pins      at the values of the anchor design
  anchor    C9-47, the best-ranked design of Campaign 9

BOX
  The grid box is deliberately wider than the feasible band of the Campaign 9
  archive (enrich 4.18 to 7.89 wt%, gd_wt 0.14 to 6.91 wt%), so the enumerated
  front is bounded by its own constraint boundaries on every side and not by
  the edge of the box.

Writes valgrid_c9/grid_list.json in the format slice_space.run_eval_list reads,
and valgrid_c9/box.txt and freeze.txt with the flags the search must repeat.
Reads the checkpoint only: no transport.

USAGE
  python make_valgrid_c9_list.py [--ckpt out_c9/optimization_checkpoint.json]
                                 [--anchor 47] [--n-enr 7] [--n-gd 5]
                                 [--enr-low 3.5] [--enr-high 8.0]
                                 [--gd-low 0.0] [--gd-high 8.0]
                                 [--out valgrid_c9]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="out_c9/optimization_checkpoint.json")
ap.add_argument("--anchor", type=int, default=47)
ap.add_argument("--n-enr", type=int, default=7)
ap.add_argument("--n-gd", type=int, default=5)
ap.add_argument("--enr-low", type=float, default=3.5)
ap.add_argument("--enr-high", type=float, default=8.0)
ap.add_argument("--gd-low", type=float, default=0.0)
ap.add_argument("--gd-high", type=float, default=8.0)
ap.add_argument("--out", default="valgrid_c9")
a = ap.parse_args()

ck = json.loads(Path(a.ckpt).read_text())
raw, con = ck["all_raw"], ck.get("constraint_names", [])
anchor = raw[a.anchor]

feas = [r for r in raw if all(float(r.get(c, 0.0)) <= 1e-9 for c in con)]
band = {v: (min(float(r[v]) for r in feas), max(float(r[v]) for r in feas))
        for v in ("enrich", "gd_wt")}
assert a.enr_low <= band["enrich"][0] and a.enr_high >= band["enrich"][1], \
    f"the enrichment box {a.enr_low}-{a.enr_high} does not bracket the feasible band {band['enrich']}"

step = lambda lo, hi, n: [round(lo + (hi - lo) * i / (n - 1), 6) for i in range(n)]
enr, gd = step(a.enr_low, a.enr_high, a.n_enr), step(a.gd_low, a.gd_high, a.n_gd)

# coarse-first order: the corners and the middle before the rest, so an
# interrupted run still covers the box
nodes = [{"enrich": e, "gd_wt": g, "grid_ij": [i, j]}
         for i, e in enumerate(enr) for j, g in enumerate(gd)]
mid = ((a.n_enr - 1) / 2, (a.n_gd - 1) / 2)
nodes.sort(key=lambda d: -((d["grid_ij"][0] - mid[0]) ** 2 + (d["grid_ij"][1] - mid[1]) ** 2))

frozen = {"refl_thick": float(anchor["refl_thick"]), "gd_pins": float(anchor["gd_pins"])}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "grid_list.json").write_text(json.dumps({
    "designs": nodes,
    "frozen": frozen,
    "grid": {"enrich": enr, "gd_wt": gd, "n_enr": a.n_enr, "n_gd": a.n_gd,
             "order": "coarse-first"},
    "anchor": {"index": a.anchor, "checkpoint": a.ckpt,
               **{k: float(anchor[k]) for k in ("enrich", "gd_wt", "refl_thick", "gd_pins")},
               **{k: float(anchor[k]) for k in ("peaking", "c_max", "cycle_length") if k in anchor}},
    "feasible_band_c9": band,
    "limits": ck["meta"].get("limits"),
}, indent=1))

(out / "box.txt").write_text(f"--enr-box-low {a.enr_low:g} --enr-box-high {a.enr_high:g}\n")
(out / "freeze.txt").write_text(
    f"--freeze refl_thick={frozen['refl_thick']:.6f} --freeze gd_pins={frozen['gd_pins']:.6f}\n")

print(f"anchor C9-{a.anchor}: enrich {anchor['enrich']:.3f}, gd_wt {anchor['gd_wt']:.3f}, "
      f"refl_thick {frozen['refl_thick']:.3f}, gd_pins {frozen['gd_pins']:.3f}")
print(f"feasible band of the Campaign 9 archive: enrich {band['enrich'][0]:.2f} to {band['enrich'][1]:.2f}, "
      f"gd_wt {band['gd_wt'][0]:.2f} to {band['gd_wt'][1]:.2f}")
print(f"grid {a.n_enr} x {a.n_gd} = {len(nodes)} nodes")
print(f"  enrich {enr}")
print(f"  gd_wt  {gd}")
print(f"wrote {out}/grid_list.json, box.txt, freeze.txt")
