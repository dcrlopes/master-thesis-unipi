import argparse, json
from pathlib import Path
import zoning as zn
import reactor_model as rm

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="out_c6/optimization_checkpoint.json")
ap.add_argument("--pos", type=int, required=True)
ap.add_argument("--particles", type=int, default=200000)
ap.add_argument("--batches", type=int, default=230)
ap.add_argument("--inactive", type=int, default=120)
ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()

r = json.load(open(a.checkpoint))["all_raw"][a.pos]
design = {k: float(r[k]) for k in
          ("enrich_inner", "enrich_outer", "gd_wt", "pitch",
           "refl_thick", "gd_pins")}
rep = float(r["peaking"])
print(f"position {a.pos}: reported peaking {rep:.4f}, "
      f"entropy_conv {r['core_entropy_conv']}", flush=True)
print(f"re-solving at inactive={a.inactive}, batches={a.batches}", flush=True)

geo, op = rm.Geometry17x17(), rm.Operating()
dmap = zn.evaluator_design_map(design)
vals = []
for s in range(1, a.seeds + 1):
    out = zn.core_bol_solve(design, dmap, op, geo, particles=a.particles,
                            batches=a.batches, inactive=a.inactive, seed=s,
                            case=Path("recheck_entropy") / f"pos{a.pos}_s{s}")
    vals.append(out["fdh_core"])
    print(f"  seed {s}: F={out['fdh_core']:.4f}  k={out['keff']:.5f}", flush=True)

m = sum(vals) / len(vals)
print(f"\nmean F = {m:.4f}   reported = {rep:.4f}   shift = {m - rep:+.4f}")
