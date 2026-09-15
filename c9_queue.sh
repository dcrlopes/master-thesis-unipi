#!/usr/bin/env bash
# 1. redo the 15.5 MPa 3D scan on the champion with a wider boron grid.
#    the first attempt found no crossing below 3000 ppm and the retry was
#    killed with the ssh session, so the directory may be absent or partial
# 2. then the MTC ceiling of every candidate on either peaking front
set -u
cd ~/master-thesis-unipi || exit 1
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || exit 1
hostname ; git branch --show-current ; date '+queue start %F %H:%M'

TAG=mtc_c9_p155_core3d
if [ ! -f "$TAG/report.txt" ] || grep -q "NO CROSSING" "$TAG/report.txt"; then
  echo "=== $TAG, wider grid  $(date '+%H:%M')"
  rm -rf "$TAG"
  python -u mtc_scan.py --checkpoint out_c9/optimization_checkpoint.json --idx 47 \
    --pressure 15.5 --t-lo 570 --t-hi 590 --level core3d \
    --boron 2000,3000,4000 --seeds 2 --threads 64 \
    --out "$TAG" 2>&1 | tee "$TAG.log" || echo "FAIL $TAG"
else
  echo "  $TAG already has a crossing, skipping"
fi

bash mtc_front_c9.sh
date '+queue end %F %H:%M'
