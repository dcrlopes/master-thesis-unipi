#!/usr/bin/env bash
# MTC ceiling on C9-57 and C9-12, to separate the gadolinia loading from the
# pin count in the shift measured on the champion C9-47.
#   C9-47  Gd 4.42 wt%, 20 pins  -> 2874 ppm (already measured)
#   C9-57  Gd 0.78 wt%, 20 pins  -> loading at fixed pin count
#   C9-12  Gd 0.14 wt%, 32 pins  -> pin count at near-fixed loading
# Same window, boron grid, seeds and fidelity as stage 5, so the three
# numbers are directly comparable.
set -u
CKPT=out_c9/optimization_checkpoint.json
P=12.8 ; TLO=547 ; THI=567 ; LV=core3d ; THREADS=64
cd ~/master-thesis-unipi || exit 1
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || exit 1
hostname ; git branch --show-current ; date '+start %F %H:%M'
for IDX in 57 12; do
  TAG="mtc_c9_d${IDX}_p${P/./}_${LV}"
  [ -f "$TAG/report.txt" ] && { echo "  $TAG already done"; continue; }
  echo "=== $TAG"
  python -u mtc_scan.py --checkpoint "$CKPT" --idx "$IDX" \
    --pressure "$P" --t-lo "$TLO" --t-hi "$THI" --level "$LV" \
    --boron 0,1000,2000,3000 --seeds 2 --threads "$THREADS" \
    --out "$TAG" 2>&1 | tee "$TAG.log" || echo "FAIL $TAG"
done
date '+end %F %H:%M'
echo "=== all crossings at ${P} MPa, ${LV}"
grep -H CROSSING mtc_c9_*p128_core3d/report.txt
