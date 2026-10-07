#!/usr/bin/env bash
# Ceiling at 15.5 MPa on C9-34 and C9-44, so the operability claim at the
# higher pressure rests on three lattices rather than on the champion alone.
# The champion crossed at 3244 ppm there, and these two carry a smaller
# gadolinia inventory, so their crossings are expected somewhat below it.
set -u
cd ~/master-thesis-unipi || exit 1
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || exit 1
hostname ; git branch --show-current ; date '+start %F %H:%M'
for IDX in 34 44; do
  TAG="mtc_c9_d${IDX}_p155_core3d"
  [ -f "$TAG/report.txt" ] && { echo "  $TAG already done"; continue; }
  echo "=== $TAG  $(date '+%H:%M')"
  python -u mtc_scan.py --checkpoint out_c9/optimization_checkpoint.json \
    --idx "$IDX" --pressure 15.5 --t-lo 570 --t-hi 590 --level core3d \
    --boron 2000,2800,3400,4000 --seeds 2 --threads 64 \
    --out "$TAG" 2>&1 | tee "$TAG.log" || echo "FAIL $TAG"
done
date '+end %F %H:%M'
grep -H CROSSING mtc_c9_*p155_core3d/report.txt
