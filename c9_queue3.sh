#!/usr/bin/env bash
# c9_queue3.sh -- two jobs in order, then the figures.
#
# 1. MTC ceiling at 15.5 MPa on C9-58 and C9-12, so the inventory relation
#    is tested at the higher pressure over five lattices spanning 4.5 to
#    88.4 in inventory, the same span that made the 12.8 MPa result hold.
#    Already measured at 15.5 MPa: C9-47, C9-34, C9-44.
#    Expected crossings, from the 12.8 MPa values plus the 71 to 396 ppm
#    pressure shift measured on the other three:
#      C9-58  2483 at 12.8  ->  roughly 2550 to 2900
#      C9-12  2091 at 12.8  ->  roughly 2160 to 2500
#    Each design gets its own grid so the bracket stays narrow.
#
# 2. Axial power shape, assembly and pin maps, and axial rod worth on the
#    hardware model for the seven candidates on either peaking front.
#    A smoke run goes first and the full run is skipped if it fails.
#
# 3. Figures, from the separate script, seconds, no transport.
#
# Cost: 32 solves for stage 1 (about 2.0 h), 28 for stage 2 (about 1.7 h).
#
# LAUNCH (detached, survives the ssh session)
#   cd ~/master-thesis-unipi
#   pgrep -af "mtc_scan|confirm3d|axial_shape|openmc" || \
#     setsid nohup bash c9_queue3.sh > c9_queue3.log 2>&1 < /dev/null &
#   sleep 20 && pgrep -af "c9_queue3|mtc_scan" | head -3
set -u
cd ~/master-thesis-unipi || exit 1

CKPT=out_c9/optimization_checkpoint.json
THREADS=64
FRONT="47 44 34 40 35 58 12"

echo "=== preflight $(date '+%F %H:%M')"
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || exit 1
hostname; git branch --show-current
[ -f "$CKPT" ] || { echo "FAIL: $CKPT not found"; exit 1; }
for f in mtc_scan.py axial_shape_c9.py axial_figures_c9.py; do
  [ -f "$f" ] || { echo "FAIL: $f not found"; exit 1; }
done
python axial_shape_c9.py --selftest || { echo "FAIL: axial selftest"; exit 1; }

# --------------------------------------------------------------- stage 1 ---
echo
echo "=== stage 1, MTC ceiling at 15.5 MPa  $(date '+%F %H:%M')"
scan_155 () {   # $1 = design index, $2 = boron grid
  local IDX="$1" GRID="$2" TAG
  TAG="mtc_c9_d${IDX}_p155_core3d"
  if [ -f "$TAG/report.txt" ]; then echo "  $TAG already done"; return 0; fi
  echo "--- $TAG, grid $GRID  $(date '+%H:%M')"
  python -u mtc_scan.py --checkpoint "$CKPT" --idx "$IDX" \
    --pressure 15.5 --t-lo 570 --t-hi 590 --level core3d \
    --boron "$GRID" --seeds 2 --threads "$THREADS" \
    --out "$TAG" 2>&1 | tee "$TAG.log" || echo "FAIL $TAG"
}
scan_155 58 2200,2600,2900,3200
scan_155 12 1800,2200,2500,2800

echo
echo "--- every 15.5 MPa crossing so far"
grep -H CROSSING mtc_c9_*p155_core3d/report.txt 2>/dev/null

# --------------------------------------------------------------- stage 2 ---
echo
echo "=== stage 2, axial shape  $(date '+%F %H:%M')"
if [ -f axial_c9/summary.json ]; then
  echo "  axial_c9 already has a summary, the run resumes from its cache"
fi
echo "--- smoke, one design, one state, one seed, about a minute"
if python -u axial_shape_c9.py --checkpoint "$CKPT" --designs 47 --states ARO \
     --seeds 1 --threads "$THREADS" --out axial_c9_smoke --smoke \
     > axial_c9_smoke.log 2>&1; then
  echo "  smoke OK"
  python axial_figures_c9.py --out axial_c9_smoke --champion 47 \
    >> axial_c9_smoke.log 2>&1 || echo "  smoke figures failed, see axial_c9_smoke.log"
  echo "--- full run, 7 designs, ARO and RE12, 2 seeds  $(date '+%H:%M')"
  python -u axial_shape_c9.py --checkpoint "$CKPT" --designs $FRONT \
    --states ARO RE12 --seeds 2 --threads "$THREADS" --out axial_c9 \
    2>&1 | tee axial_c9.log || echo "FAIL axial_c9"
else
  echo "  SMOKE FAILED, the full axial run is skipped. See axial_c9_smoke.log"
  tail -20 axial_c9_smoke.log
fi

# --------------------------------------------------------------- stage 3 ---
echo
echo "=== stage 3, figures  $(date '+%F %H:%M')"
if [ -d axial_c9 ]; then
  python axial_figures_c9.py --out axial_c9 --champion 47 --png \
    2>&1 | tee axial_c9_figures.log || echo "FAIL figures"
fi

echo
echo "=== queue end $(date '+%F %H:%M')"
echo "--- 15.5 MPa ceilings"
grep -H CROSSING mtc_c9_*p155_core3d/report.txt 2>/dev/null
echo "--- refit both pressures"
python mtc_front_table.py --checkpoint "$CKPT" --glob 'mtc_c9_*p155_core3d' \
  --pressure 15.5 --out /tmp/p155 2>/dev/null | sed -n '6,16p'
python mtc_front_table.py --checkpoint "$CKPT" --glob 'mtc_c9_*_core3d*' \
  --pressure 12.8 --out c9_post 2>/dev/null | sed -n '6,18p'
