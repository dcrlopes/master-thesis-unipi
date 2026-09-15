#!/usr/bin/env bash
# MTC ceiling for every candidate on either peaking front.
#   2D peaking front (stage 8) : 34 35 44 47 58
#   3D peaking front (stage 8) : 12 34 40 47
#   union                      : 12 34 35 40 44 47 58
# 47 (2874 ppm) and 12 (2037 ppm) are already measured, so five remain.
# Identical window, boron grid, seeds and fidelity as the champion scan,
# so every crossing in the table is obtained the same way.
set -u
CKPT=out_c9/optimization_checkpoint.json
P=12.8 ; TLO=547 ; THI=567 ; LV=core3d ; THREADS=64
cd ~/master-thesis-unipi || exit 1
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || exit 1
hostname ; git branch --show-current ; date '+start %F %H:%M'
for IDX in 34 40 44 35 58; do
  TAG="mtc_c9_d${IDX}_p${P/./}_${LV}"
  [ -f "$TAG/report.txt" ] && { echo "  $TAG already done"; continue; }
  echo "=== $TAG  $(date '+%H:%M')"
  python -u mtc_scan.py --checkpoint "$CKPT" --idx "$IDX" \
    --pressure "$P" --t-lo "$TLO" --t-hi "$THI" --level "$LV" \
    --boron 0,1000,2000,3000 --seeds 2 --threads "$THREADS" \
    --out "$TAG" 2>&1 | tee "$TAG.log" || echo "FAIL $TAG"
done
date '+end %F %H:%M'
grep -H CROSSING mtc_c9_*p128_core3d/report.txt
