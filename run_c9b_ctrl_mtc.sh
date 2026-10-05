#!/usr/bin/env bash
# run_c9b_ctrl_mtc.sh -- control-rod and MTC measurements of C9-63, C9-68 and
# C9-94, the three continuation designs that satisfy the minimum cycle length
# in 8 layers, with the settings used for the final front (C9-69, C9-70).
#
#   rods   three-dimensional core with the four regulating banks (ARI) and
#          with RE1 + RE2 (RE12), 1000 ppm, 150 000 x 200 (80 inactive),
#          2 seeds. The all-rods-out solves are already cached in
#          confirm3d_c9a_rest.                              about 26 min per design
#   mtc    MTC boron limit on the three-dimensional core at 12.8 MPa,
#          0, 1000, 2000 and 3000 ppm, 547 to 567 K, 2 seeds.
#                                                           about 30 min per design
#   all    rods, then mtc.                                   about 3 h for the three
#
# One job at a time, 64 threads. Finished solves are cached and skipped.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9b_ctrl_mtc.sh preflight
#   setsid nohup bash run_c9b_ctrl_mtc.sh all > c9b_ctrl_mtc.log 2>&1 < /dev/null &
#   bash run_c9b_ctrl_mtc.sh status
set -u -o pipefail

THREADS=${THREADS:-64}
CKPT=out_c9a/optimization_checkpoint.json
DESIGNS=${DESIGNS:-"63 68 94"}
ROD_OUT=confirm3d_c9a_rest
REPORT=c9b_ctrl_mtc_report.txt

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
busy()  { pgrep -af "run_c9b_layers.sh|c9_dep_core3d.py|confirm3d.py|mtc_scan.py|run_optimization.py" || true; }
tag()   { echo "mtc_c9b_d$1_p128_core3d"; }

report() {
  { python - "$ROD_OUT/summary.json" $DESIGNS <<'PY'
import json, os, sys
S = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
for d in sys.argv[2:]:
    r = S.get(d, {})
    if "RE12_margin3D_pcm" in r:
        print(f"C9-{d}: two-bank margin 3D {r['RE12_margin3D_pcm']:.0f} pcm, four-bank margin 3D {r['ARI_margin3D_pcm']:.0f} pcm, F_3D all rods out {r['ARO_3Dhw']['F']:.3f}")
    else:
        print(f"C9-{d}: rodded solves not finished")
PY
  for d in $DESIGNS; do
    f="$(tag "$d")/report.txt"
    [ -f "$f" ] && echo "C9-$d MTC: $(grep -h CROSSING "$f" | head -1)" || echo "C9-$d MTC: not finished"
  done; } | tee "$REPORT"
}

preflight() {
  [ "$(hostname)" = "wks720" ]                || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"
  for f in "$CKPT" confirm3d.py mtc_scan.py "$ROD_OUT/runs.json"; do [ -e "$f" ] || die "missing $f (run from ~/master-thesis-unipi)"; done
  b=$(busy); [ -z "$b" ] || die "another transport job is running, one 64-thread job at a time: $b"
}

rods() {
  echo "$(stamp) rodded solves of designs $DESIGNS"
  python -u confirm3d.py --checkpoint "$CKPT" --designs $DESIGNS --states ARO ARI RE12 \
    --boron-ppm 1000 --seeds 2 --threads "$THREADS" --out "$ROD_OUT" || die "confirm3d failed"
}

mtc() {
  for d in $DESIGNS; do
    t=$(tag "$d")
    [ -f "$t/report.txt" ] && { echo "$(stamp) $t already done"; continue; }
    echo "$(stamp) $t"
    python -u mtc_scan.py --checkpoint "$CKPT" --idx "$d" --pressure 12.8 --t-lo 547 --t-hi 567 \
      --level core3d --boron 0,1000,2000,3000 --seeds 2 --threads "$THREADS" --out "$t" || die "$t failed"
  done
}

finish() {
  echo "$(stamp) finished"; report
  echo "to send the results back:"
  echo "  git add -f $ROD_OUT/runs.json $ROD_OUT/summary.json mtc_c9b_d*_p128_core3d/report.txt mtc_c9b_d*_p128_core3d/summary.json $REPORT && git commit -m 'C9-63, 68, 94: rodded 3D solves and MTC boron limit' && git push"
}

case "${1:-}" in
status)    echo "$(stamp) host $(hostname)"; b=$(busy); [ -n "$b" ] && echo "running: $b" || echo "running: nothing"; report ;;
preflight) preflight; echo "$(stamp) preflight ok, designs: $DESIGNS" ;;
rods)      preflight; rods; finish ;;
mtc)       preflight; mtc; finish ;;
all)       preflight; rods; mtc; finish ;;
*)         die "usage: bash run_c9b_ctrl_mtc.sh preflight | rods | mtc | all | status" ;;
esac
