#!/usr/bin/env bash
# run_c8_confirm.sh -- the Campaign 8 candidate confirmation on wks720, in
# sequence, detached. Three stages, each resumable and each skipped if its
# summary already exists.
#
#   Stage 1  boron worth, 8 candidates x 3 states x 4 concentrations   (~2.7 h)
#   Stage 2  seed replicates for sigma(F) and the two marginal designs  (~1.5 h,
#            reuses the 1000 ppm seed-0 solves of stage 1)
#   Stage 3  three-dimensional confirmation, water-slab axial model,
#            5 designs x 2 seeds                                         (~2.1 h)
#
# What it does NOT do, and why: the hardware-stack model (nozzles, grids,
# plenum, core barrel) is not written yet, so stage 3 is the same 15 cm
# water-slab model as the axial study, which is the basis the campaign
# target was corrected with. The 3.7 cm downcomer sensitivity is purely
# geometric in the two-dimensional model (the barrel and the water are not
# in the transport model) and needs no run: see the note printed at the end.
#
# USAGE
#   chmod +x run_c8_confirm.sh
#   setsid nohup ./run_c8_confirm.sh > c8_confirm.log 2>&1 < /dev/null &
#   tail -f c8_confirm.log

set -u
REPO="$HOME/master-thesis-unipi"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
ENVNAME="openmc-env"
CKPT="out_c8/optimization_checkpoint.json"
THREADS=32

FRONT="47 23 21 44 1"            # front representatives
TWO="53 13 31"                   # two-bank subset
MARGINAL="54 11"                 # fail g_ctrl by 17 and 63 pcm
AXIAL="47,23,21,53,31"           # three-dimensional confirmation set

cd "$REPO" || { echo "ABORT: cannot cd $REPO"; exit 1; }
if [ -f "$CONDA_SH" ]; then . "$CONDA_SH"; conda activate "$ENVNAME" || exit 1
else echo "ABORT: $CONDA_SH not found"; exit 1; fi

echo "$(date -Is) environment check"
python -c "import numpy, openmc; print('  env ok', openmc.__version__)" || exit 1
python -c "import zoning as zn; zn.RE12_POSITIONS" 2>/dev/null || { echo "ABORT: campaign8 branch needed"; exit 1; }
[ -f "$CKPT" ] || { echo "ABORT: $CKPT missing"; exit 1; }
python boron_worth.py --selftest > /dev/null || { echo "ABORT: boron_worth selftest"; exit 1; }
python apply_axial_c8.py --check | grep -q "APPLIED\|ready" || { echo "ABORT: apply_axial_c8 anchor"; exit 1; }
python apply_axial_c8.py > /dev/null || exit 1        # idempotent
nproc; echo "$(date -Is) start, uptime: $(uptime -p)"

# ---------------------------------------------------------------- stage 1
if [ ! -f boron_c8/summary.json ]; then
  echo "$(date -Is) STAGE 1 boron worth"
  python -u boron_worth.py --checkpoint "$CKPT" --designs $FRONT $TWO \
      --ppm 0 500 1000 1500 --states ARO ARI RE12 --seeds 1 \
      --threads $THREADS --out boron_c8 || { echo "stage 1 failed"; exit 1; }
else echo "$(date -Is) STAGE 1 already done"; fi

# ---------------------------------------------------------------- stage 2
# Same output directory: the (design, state, 1000 ppm, seed 0) solves are
# read back from boron_c8/runs.json, only seeds 1 and 2 are new.
if [ ! -f boron_c8/seeds_done ]; then
  echo "$(date -Is) STAGE 2 seed replicates"
  python -u boron_worth.py --checkpoint "$CKPT" --designs $FRONT $TWO $MARGINAL \
      --ppm 1000 --states ARO ARI RE12 --seeds 3 \
      --threads $THREADS --out boron_c8 || { echo "stage 2 failed"; exit 1; }
  touch boron_c8/seeds_done
else echo "$(date -Is) STAGE 2 already done"; fi

# ---------------------------------------------------------------- stage 3
if [ ! -f axial_c8/axial_B4C.json ]; then
  echo "$(date -Is) STAGE 3 three-dimensional confirmation (water-slab axial model)"
  python -u axial_leakage_study.py --checkpoint "$CKPT" --idx "$AXIAL" \
      --m-center 0.72 --m-periphery 1.15 --absorber B4C \
      --h-active 120 --axial-refl 15 --seeds 2 --threads $THREADS \
      --out axial_c8 || { echo "stage 3 failed"; exit 1; }
else echo "$(date -Is) STAGE 3 already done"; fi

# --------------------------------------------------- sigma(F) from stage 2
python - << 'PYEOF'
import json, statistics as st
runs = json.load(open("boron_c8/runs.json"))
by = {}
for key, rec in runs.items():
    idx, state, ppm, seed = key.split("|")
    if float(ppm) == 1000.0:
        by.setdefault((idx, state), []).append(rec)
print("\nsigma(F_dH) and k over seeds at 1000 ppm (n = seeds found):")
print(f"{'design':>6} {'state':>5} {'n':>2} {'k mean':>8} {'sd_k pcm':>9} {'F mean':>7} {'sd_F':>6}")
for (idx, state), recs in sorted(by.items(), key=lambda t: (int(t[0][0]), t[0][1])):
    ks = [r["keff"] for r in recs]; fs = [r["fdh"] for r in recs]
    sdk = 1e5 * st.stdev(ks) if len(ks) > 1 else float("nan")
    sdf = st.stdev(fs) if len(fs) > 1 else float("nan")
    print(f"{idx:>6} {state:>5} {len(recs):>2} {st.mean(ks):8.5f} {sdk:9.0f} {st.mean(fs):7.3f} {sdf:6.3f}")
PYEOF

echo
echo "Downcomer 3.7 cm: geometric only in the 2D model. Reflector budget at"
echo "pitch 1.26 becomes 3.969 cm. Of the twelve candidates only 42, 1, 13"
echo "and 12 fit. The physical sensitivity (barrel return) needs the barrel"
echo "in the transport model, which is the hardware-stack applier still to write."
echo "$(date -Is) all stages complete"
