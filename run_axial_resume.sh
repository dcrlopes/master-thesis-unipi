#!/usr/bin/env bash
# run_axial_resume.sh -- run axial_leakage_study.py design by design, skipping
# any design already finished, so an @reboot crontab entry can relaunch it
# after a reboot without losing or repeating work.
#
# WHY
#   axial_leakage_study.py writes its JSON only after ALL designs finish, so a
#   reboot loses the whole run. This wrapper invokes it once per design into
#   its own output directory. A design whose JSON exists is skipped, so the
#   same command can be issued any number of times and only ever does the
#   work that remains.
#
# USAGE
#   ./run_axial_resume.sh                 # foreground, prints progress
#   setsid nohup ./run_axial_resume.sh > axial_resume.log 2>&1 < /dev/null &
#
# CRONTAB (survives a reboot)
#   @reboot /home/diogo/master-thesis-unipi/run_axial_resume.sh >> \
#           /home/diogo/master-thesis-unipi/axial_resume.log 2>&1
#
# A lock directory prevents two copies running at once, which would halve the
# speed and interleave the case directories.

set -u

REPO="/home/diogo/master-thesis-unipi"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
ENVNAME="openmc-env"
DESIGNS="31 1 22 69 59 86"
CKPT="out_c6/optimization_checkpoint.json"
OUTROOT="axial_designs"
LOCK="$REPO/.axial_resume.lock"

M_C=0.72
M_P=1.15
H_ACTIVE=120
AXIAL_REFL=15
SEEDS=2
THREADS=64

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
python -c "import zoning as zn, inspect, sys; \
    sys.exit(0 if 'h_active' in inspect.signature(zn.core_bol_solve).parameters else 1)" \
    || { echo "ABORT: core3d not applied (run apply_core3d.py)"; exit 1; }
[ -f "$CKPT" ] || { echo "ABORT: $CKPT not found"; exit 1; }

echo "$(date -Is) start, uptime: $(uptime -p)"
mkdir -p "$OUTROOT"

# --- one design at a time --------------------------------------------------
for IDX in $DESIGNS; do
    OUT="$OUTROOT/d$IDX"
    if [ -f "$OUT/axial_B4C.json" ]; then
        echo "$(date -Is) design $IDX already done, skipping"
        continue
    fi
    echo "$(date -Is) design $IDX starting"
    python -u axial_leakage_study.py \
        --checkpoint "$CKPT" \
        --idx "$IDX" \
        --m-center "$M_C" --m-periphery "$M_P" \
        --h-active "$H_ACTIVE" --axial-refl "$AXIAL_REFL" \
        --seeds "$SEEDS" --threads "$THREADS" \
        --out "$OUT"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "$(date -Is) design $IDX FAILED rc=$rc, stopping"
        exit $rc
    fi
    echo "$(date -Is) design $IDX done"
done

echo "$(date -Is) all designs complete"
python - <<'PY'
import json, glob
states = []
for p in sorted(glob.glob("axial_designs/d*/axial_B4C.json")):
    d = json.load(open(p))
    states += d["states"]
    meta = d
if states:
    meta["states"] = sorted(states, key=lambda s: s["idx"])
    json.dump(meta, open("axial_merged_B4C.json", "w"), indent=2)
    print(f"merged {len(states)} designs -> axial_merged_B4C.json")
    L = [s["extras"]["L_ax"] for s in states]
    print(f"L_ax: min {min(L):.4f}  max {max(L):.4f}  "
          f"mean {sum(L)/len(L):.4f}")
    for s in meta["states"]:
        e = s["extras"]
        print(f"  idx {s['idx']:>3}: k2D={e['k_2d']:.5f} k3D={s['k0']:.5f} "
              f"dk_ax={e['dk_axial_pcm']:6.0f} pcm  L_ax={e['L_ax']:.4f}  "
              f"margin3D={s['states']['ALLRE']['margin_pcm']:6.0f} pcm")
PY
