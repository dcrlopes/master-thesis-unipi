#!/usr/bin/env bash
# run_c8_resume.sh -- launch or RESUME Campaign 8, safely from @reboot.
#
# WHY THIS EXISTS
#   The bare launch command must NEVER go in a crontab. run_optimization.py
#   without --resume starts a FRESH campaign and overwrites out_c8, so a
#   reboot twelve hours in would destroy twelve hours of transport. This
#   wrapper decides for itself:
#
#     no checkpoint            -> fresh launch, all 4 infill blocks
#     checkpoint with N evals  -> --resume with only the blocks still owed
#     60 evals reached         -> does nothing and says so
#
#   It also holds a lock directory, so the accidental double-launch that
#   happened in Campaign 7 cannot repeat, and it verifies the environment
#   and every applier before spending a single core-second.
#
# USAGE
#   ./run_c8_resume.sh                         # foreground, prints progress
#   setsid nohup ./run_c8_resume.sh > c8_resume.log 2>&1 < /dev/null &
#
# CRONTAB (one line, survives a reboot)
#   @reboot sleep 60 && /home/diogo/master-thesis-unipi/run_c8_resume.sh >> \
#           /home/diogo/master-thesis-unipi/c8_resume.log 2>&1
#
#   `sleep 60` lets the filesystem and network settle before conda is
#   sourced, exactly as the axial entry does.
#
# NOTE ON RESUMING
#   The checkpoint is written only at the END of the DOE stage and after
#   every infill block. A reboot inside a block loses that block, never
#   more. The DOE block is the long one, roughly 10.5 h.

set -u

REPO="/home/diogo/master-thesis-unipi"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
ENVNAME="openmc-env"
LOCK="$REPO/.c8_resume.lock"

OUT="out_c8"
WORKDIR="openmc_runs_c8"
CKPT="$OUT/optimization_checkpoint.json"
TABLE="ktarget_table_c8.json"
LOG="out_c8_run.log"

N_INIT=36            # design of experiments
N_INFILL=6           # designs per infill block
ITERS_TOTAL=4        # infill blocks
N_TARGET=$((N_INIT + ITERS_TOTAL * N_INFILL))     # 60

K_BASIS="core"
K_MIN=1.02
K_MAX=1.166          # 3D-derived: (0.99 + 0.14307) * L_ax 1.0289
ENR_MAX=16
ENR_BOX_LOW=3.0
CTRL_MARGIN=1000
CTRL_ABSORBER="B4C"
MIN_SEP=0.14
FEAS_KAPPA=1.5
NSGA_POP=300
NSGA_GEN=400

cd "$REPO" || { echo "ABORT: cannot cd to $REPO"; exit 1; }

# --- single instance -------------------------------------------------------
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "$(date -Is) another run holds $LOCK, exiting"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# --- environment -----------------------------------------------------------
# cron gives a minimal shell with no conda, so source it explicitly
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    . "$CONDA_SH"
    conda activate "$ENVNAME" || { echo "ABORT: cannot activate $ENVNAME"; exit 1; }
else
    echo "ABORT: $CONDA_SH not found, adjust CONDA_SH in this script"
    exit 1
fi

python -c "import numpy, openmc" 2>/dev/null \
    || { echo "ABORT: openmc not importable in $ENVNAME"; exit 1; }
[ -f "$TABLE" ] || { echo "ABORT: $TABLE not found"; exit 1; }

# --- the appliers must all be in place -------------------------------------
python - <<'PYEOF' || { echo "ABORT: the Campaign 8 patches are not all applied"; exit 1; }
import sys, json
import zoning as zn
import reactor_optimization as ro
ok = True
def bad(msg):
    global ok
    print(f"  MISSING: {msg}")
    ok = False
if not hasattr(zn, "RE12_POSITIONS"):
    bad("zoning.RE12_POSITIONS (apply_ctrl12.py)")
names = ro.example_reactor_problem().design_space.names
if names != ["enrich", "gd_wt", "refl_thick", "gd_pins"]:
    bad(f"design space is {names} (apply_c8_space.py)")
import core_geometry as cg
if abs(cg.VESSEL_CLEARANCE_CM - 7.08) > 1e-9:
    bad(f"VESSEL_CLEARANCE_CM is {cg.VESSEL_CLEARANCE_CM} (apply_c8_space.py)")
src = open("run_optimization.py").read()
if '"re12_positions"' not in src:
    bad("ctrl_screen re12 provenance (apply_ctrl12_meta.py)")
if "_smoke_core" not in src:
    bad("smoke core fidelity (apply_smoke_core.py)")
t = json.load(open("ktarget_table_c8.json"))
if "pitch_cm" in t or "axial_leakage_factor" not in t:
    bad("ktarget_table_c8.json is not the 1-D axial-corrected table")
sys.exit(0 if ok else 1)
PYEOF

# --- how many blocks are still owed? ---------------------------------------
if [ -f "$CKPT" ]; then
    N_DONE=$(python -c "import json,sys; print(len(json.load(open('$CKPT'))['all_raw']))") \
        || { echo "ABORT: cannot read $CKPT"; exit 1; }
else
    N_DONE=0
fi

if [ "$N_DONE" -ge "$N_TARGET" ]; then
    echo "$(date -Is) campaign complete: $N_DONE of $N_TARGET evaluations, nothing to do"
    exit 0
fi

echo "$(date -Is) start, uptime: $(uptime -p)"
echo "  evaluations on disk: $N_DONE of $N_TARGET"

COMMON=(--k-basis "$K_BASIS" --k-min "$K_MIN" --k-max "$K_MAX"
        --enr-max "$ENR_MAX" --enr-box-low "$ENR_BOX_LOW"
        --ctrl-margin "$CTRL_MARGIN" --ctrl-absorber "$CTRL_ABSORBER"
        --ktarget-table "$TABLE"
        --n-infill "$N_INFILL"
        --infill-min-sep "$MIN_SEP" --feas-kappa "$FEAS_KAPPA"
        --nsga-pop "$NSGA_POP" --nsga-gen "$NSGA_GEN"
        --workdir "$WORKDIR" --out "$OUT")

if [ "$N_DONE" -eq 0 ]; then
    echo "  fresh launch: DOE $N_INIT + $ITERS_TOTAL x $N_INFILL"
    python -u run_optimization.py \
        --n-init "$N_INIT" --iters "$ITERS_TOTAL" \
        "${COMMON[@]}" >> "$LOG" 2>&1
else
    # blocks still owed, rounding up a partially lost block
    REMAIN=$(( (N_TARGET - N_DONE + N_INFILL - 1) / N_INFILL ))
    echo "  resuming: $REMAIN infill block(s) still owed"
    python -u run_optimization.py \
        --resume "$CKPT" --iters "$REMAIN" \
        "${COMMON[@]}" >> "$LOG" 2>&1
fi

rc=$?
echo "$(date -Is) run_optimization exited rc=$rc"
exit $rc
