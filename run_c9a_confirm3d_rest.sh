#!/usr/bin/env bash
# run_c9a_confirm3d_rest.sh -- three-dimensional core solve of the 34 designs
# of the Campaign 9 continuation that do not have one (C9-60 to C9-95 without
# C9-69 and C9-70), so that every evaluated design has a three-dimensional
# peaking factor. Queued: it waits until no other transport job is running,
# for example the C9-69 seeds of run_c9_d69_seeds.sh, and then starts.
#
# Same settings as confirm3d_c9_all and confirm3d_c9a: confirm3d.py, all rods
# out, 1000 ppm, 150 000 x 200 (80 inactive), 2 seeds, 2D and 3D solve each.
#
# TIME    about 13 min per design (4 solves of about 197 s, the mean of the
#         240 solves of confirm3d_c9_all), about 7.4 h for the 34 designs,
#         after the wait.
#         Finished solves are cached in confirm3d_c9a_rest/runs.json and are
#         skipped on a relaunch.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9a_confirm3d_rest.sh --preflight
#   setsid nohup bash run_c9a_confirm3d_rest.sh > c9a_confirm3d_rest.log 2>&1 < /dev/null &
#   bash run_c9a_confirm3d_rest.sh --status
set -u -o pipefail

THREADS=${THREADS:-64}
CKPT=out_c9a/optimization_checkpoint.json
OUT=confirm3d_c9a_rest
DESIGNS=$(seq 60 95 | grep -vx -e 69 -e 70 | tr '\n' ' ')
N=$(echo $DESIGNS | wc -w)

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
# every transport job of this repository; this script's own name matches none of them
busy()  { pgrep -af "run_c9_d69_seeds.sh|c9_dep_core3d.py|confirm3d.py|run_optimization.py|axial_shape_c9.py|mtc_scan.py|boron_worth.py" || true; }

report() {
  [ -f "$OUT/summary.json" ] || { echo "no $OUT/summary.json yet"; return 0; }
  python - "$OUT/summary.json" "$N" <<'PY'
import json, sys
S = json.load(open(sys.argv[1]))
done = sorted((int(k) for k, v in S.items() if "ARO_3Dhw" in v))
print(f"designs with a three-dimensional peaking factor: {len(done)} of {sys.argv[2]}")
for i in done:
    r = S[str(i)]
    print(f"  C9-{i}: F_3D {r['ARO_3Dhw']['F']:.3f}   F_2D {r['ARO_2D']['F']:.3f}")
PY
}

case "${1:-}" in
--status)
  echo "$(stamp) host $(hostname)"
  b=$(busy); [ -n "$b" ] && echo "running: $b" || echo "running: nothing"
  report; exit 0 ;;
esac

# ------------------------------------------------------------ preflight ----
[ "$(hostname)" = "wks720" ]                || die "not on wks720"
[ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"
for f in "$CKPT" confirm3d.py; do [ -e "$f" ] || die "missing $f (run from ~/master-thesis-unipi)"; done
[ "$N" -eq 34 ] || die "expected 34 designs, found $N"
python confirm3d.py --checkpoint "$CKPT" --designs 60 --states ARO --out "$OUT" --dry-run > /dev/null || die "dry run failed"
echo "$(stamp) designs ($N): $DESIGNS"
if [ "${1:-}" = "--preflight" ]; then
  b=$(busy); [ -n "$b" ] && echo "would wait for: $b" || echo "nothing is running, would start at once"
  echo "$(stamp) preflight ok"; exit 0
fi

# ----------------------------------------------------------------- wait ----
# one 64-thread job at a time: start after two idle checks one minute apart
idle=0
while [ "$idle" -lt 2 ]; do
  if [ -n "$(busy)" ]; then idle=0; else idle=$((idle + 1)); fi
  [ "$idle" -lt 2 ] && sleep 60
done
echo "$(stamp) no other transport job, starting"

# ------------------------------------------------------------------ run ----
python -u confirm3d.py --checkpoint "$CKPT" --designs $DESIGNS --states ARO \
  --boron-ppm 1000 --seeds 2 --threads "$THREADS" --out "$OUT" || die "confirm3d failed"
echo "$(stamp) finished"
report
echo "to send the results back:"
echo "  git add -f $OUT/runs.json $OUT/summary.json && git commit -m 'C9 continuation: three-dimensional core solve of the other 34 designs' && git push"
