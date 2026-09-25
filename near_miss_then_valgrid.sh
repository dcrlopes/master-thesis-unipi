#!/usr/bin/env bash
# After the axial run: deplete at eight layers every near miss (infeasible on
# g_efpd alone, by less than one leave-one-out rms in days, below the MTC
# boron limit, and non-dominated against the current front if it were
# feasible), refit and rescore, then start the valgrid run with EQUAL=1.
cd ~/master-thesis-unipi
echo "$(date '+%F %T') waiting for the axial runner 719667"
while kill -0 719667 2>/dev/null; do sleep 60; done
NM=$(python - <<'PY'
import json
import axial_ratio_model as arm
ck = json.load(open("out_c9a/optimization_checkpoint.json")); raw = ck["all_raw"]
model = arm.load("axial_ratio_model.json"); con = ck["constraint_names"]
others = [c for c in con if c != "g_efpd"]
ok = lambda r: all(float(r.get(c, 0)) <= 1e-9 for c in others) and float(r["c_max"]) <= 2763
front = [(float(r["peaking"]), float(r["c_max"])) for r in raw if ok(r) and float(r["g_efpd"]) <= 0]
dom = lambda f, c: any(g <= f and d <= c and (g < f or d < c) for g, d in front)
print(" ".join(str(i) for i, r in enumerate(raw)
      if ok(r) and 0 < float(r["g_efpd"]) <= model["s_loo"] * float(r["cycle_length"])
      and not dom(float(r["peaking"]), float(r["c_max"])) and arm.measured_lookup(model, r) is None))
PY
)
echo "$(date '+%F %T') near misses: ${NM:-none}"
if [ -n "$NM" ]; then
  python -u c9_dep_core3d.py --checkpoint out_c9a/optimization_checkpoint.json \
    --designs $NM --layers 8 --threads 64 --out c9a_dep_core3d_nm 2>&1 | tee -a c9a_dep_core3d_nm.log
  python c9_dep_core3d.py --checkpoint out_c9a/optimization_checkpoint.json --out c9a_dep_core3d_nm --analyse >> c9a_dep_core3d_nm.log 2>&1
  python axial_ratio_model.py --fit gd_wt hump_asm_pcm --kappa 1.0 --out axial_ratio_model.json
  python seed_c9_axial.py --inplace out_c9a/optimization_checkpoint.json --model axial_ratio_model.json
  bash run_c9_axial.sh --status
fi
echo "$(date '+%F %T') starting the valgrid run"
EQUAL=1 bash run_valgrid_c9.sh >> valgrid_c9.log 2>&1 < /dev/null
