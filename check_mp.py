import csv
from pathlib import Path
import zoning as zn
import reactor_model as rm

r = {x["idx"]: x for x in csv.DictReader(open("campaign5/c5_full.csv"))}["54"]
design = dict(enrich_inner=float(r["enrich_inner"]),
              enrich_outer=float(r["enrich_outer"]),
              gd_wt=float(r["gd_wt"]), pitch=float(r["pitch"]),
              refl_thick=float(r["refl_thick"]), gd_pins=float(r["gd_pins"]))
geo, op = rm.Geometry17x17(), rm.Operating()
rmap = zn.ring_map()
counts = zn.ring_counts(rmap)
for mp in (1.15, 1.25):
    mc, mm, mpp = zn.balanced_multipliers(0.720, mp, counts)
    dmap = zn.design_map_for(rmap, zn.zone_designs(design, mc, mm, mpp))
    out = zn.core_bol_solve(design, dmap, op, geo, particles=200000,
                            batches=170, inactive=60, seed=1,
                            case=Path("verify_mp") / f"mp{mp}")
    maxe = zn.max_zoned_enrichment(design, mpp)
    print(f"m_C=0.720 m_M={mm:.4f} m_P={mpp:.3f}: "
          f"F={out['fdh_core']:.4f}  k={out['keff']:.5f}  "
          f"max_enr={maxe:.3f} wt%", flush=True)
