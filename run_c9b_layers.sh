#!/usr/bin/env bash
# run_c9b_layers.sh -- eight-layer depletion of the continuation designs that
# are feasible under the constraint of their stage and are not dominated by
# the final front with the three-dimensional peaking factor: C9-63, C9-68,
# C9-79 and C9-94. None of them has an eight-layer depletion yet.
#
# Same settings as the final front: c9_dep_core3d.py, 8 layers, 20 000 x 160
# (60 inactive), relative end of cycle. One depletion at a time, 64 threads.
#
#   first   one depletion per design, default seed           about 0.9 h each
#   seeds   seeds 2 to 5 of each design, salts core3d-seedN   about 3.6 h each
#
# Run "seeds" only on the designs that satisfy 1826 d in "first":
#   DESIGNS="63 68" bash run_c9b_layers.sh seeds
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9b_layers.sh preflight
#   setsid nohup bash run_c9b_layers.sh first > c9b_first.log 2>&1 < /dev/null &
#   setsid nohup bash run_c9b_layers.sh seeds > c9b_seeds.log 2>&1 < /dev/null &
#   bash run_c9b_layers.sh status
set -u -o pipefail

THREADS=${THREADS:-64}
CKPT=out_c9a/optimization_checkpoint.json
DESIGNS=${DESIGNS:-"63 68 79 94"}
REPORT=c9b_layers_report.txt

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
busy()  { pgrep -af "run_c9_d69_seeds.sh|run_c9a_confirm3d_rest.sh|c9_dep_core3d.py|confirm3d.py|run_optimization.py|mtc_scan.py" || true; }
dep()   { python -u c9_dep_core3d.py --checkpoint "$CKPT" --layers 8 --threads "$THREADS" "$@"; }

report() {
  python - $DESIGNS <<'PY' | tee "$REPORT"
import json, os, statistics, sys
REQ = 1826.0
for d in sys.argv[1:]:
    files = ["c9b_dep_core3d/runs.json"] + [f"c9b_dep_core3d_seed{k}/runs.json" for k in (2, 3, 4, 5)]
    v = []
    for f in files:
        if os.path.exists(f):
            r = json.load(open(f)).get(f"d{d}")
            if r and "efpd" in r:
                v.append(r["efpd"])
    if not v:
        print(f"C9-{d}: no depletion yet"); continue
    m = statistics.mean(v)
    line = f"C9-{d}: {len(v)} seed(s) " + ", ".join(f"{x:.1f}" for x in v) + f" d; mean {m:.1f} d; margin over {REQ:.0f} d {m - REQ:+.1f} d"
    if len(v) > 1:
        se = statistics.stdev(v) / len(v) ** 0.5
        line += f"; standard error {se:.1f} d; {(m - REQ) / se:.1f} standard errors"
    print(line)
PY
}

preflight() {
  [ "$(hostname)" = "wks720" ]                || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"
  for f in "$CKPT" c9_dep_core3d.py; do [ -e "$f" ] || die "missing $f (run from ~/master-thesis-unipi)"; done
  b=$(busy); [ -z "$b" ] || die "another transport job is running, one 64-thread job at a time: $b"
  python c9_dep_core3d.py --checkpoint "$CKPT" --designs $DESIGNS --layers 8 --out c9b_dep_core3d --dry-run || die "dry run failed"
}

case "${1:-}" in
status)
  echo "$(stamp) host $(hostname)"; b=$(busy); [ -n "$b" ] && echo "running: $b" || echo "running: nothing"
  report ;;
preflight)
  preflight; echo "$(stamp) preflight ok, designs: $DESIGNS" ;;
first)
  preflight
  dep --designs $DESIGNS --out c9b_dep_core3d || die "first depletion failed"
  echo "$(stamp) finished"; report
  echo "to send the results back:"
  echo "  git add -f c9b_dep_core3d/runs.json $REPORT && git commit -m 'C9-63, 68, 79, 94: eight-layer depletion' && git push" ;;
seeds)
  preflight
  for k in 2 3 4 5; do
    echo "$(stamp) seed $k of designs $DESIGNS"
    dep --designs $DESIGNS --salt "core3d-seed$k" --out "c9b_dep_core3d_seed$k" || die "seed $k failed"
  done
  echo "$(stamp) finished"; report
  echo "to send the results back:"
  echo "  git add -f c9b_dep_core3d_seed*/runs.json $REPORT && git commit -m 'C9 continuation candidates: eight-layer seeds 2 to 5' && git push" ;;
*)
  die "usage: bash run_c9b_layers.sh preflight | first | seeds | status" ;;
esac
