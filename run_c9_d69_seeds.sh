#!/usr/bin/env bash
# run_c9_d69_seeds.sh -- seeds 2 to 5 of the eight-layer depletion of C9-69,
# so that the three members of the final front (C9-27, C9-69, C9-70) have the
# same five seeds. Seed 1 is the run of c9f_dep_core3d (1895.2 d).
#
# Same settings as the replicates of C9-27 and C9-70: c9_dep_core3d.py,
# 8 layers, 20 000 x 160 (60 inactive), relative end of cycle, salts
# core3d-seed2 to core3d-seed5. One depletion at a time, 64 threads.
#
# TIME    about 0.9 h per seed on an idle wks720, about 3.6 h in all.
#         A finished seed is cached in its runs.json and skipped on a relaunch.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9_d69_seeds.sh --preflight
#   setsid nohup bash run_c9_d69_seeds.sh > c9_d69_seeds.log 2>&1 < /dev/null &
#   bash run_c9_d69_seeds.sh --status
#   cat c9_d69_seeds_report.txt
set -u -o pipefail

THREADS=${THREADS:-64}
CKPT=out_c9a/optimization_checkpoint.json
FIRST=c9f_dep_core3d/runs.json            # seed 1 of C9-69
SEEDS="2 3 4 5"
REPORT=c9_d69_seeds_report.txt

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
outdir() { echo "c9a_dep_core3d_d69_seed$1"; }
busy()  { pgrep -af "c9_dep_core3d.py|run_optimization.py|axial_shape_c9.py|confirm3d" | grep -v "run_c9_d69_seeds" || true; }

report() {
  python - "$FIRST" $(for k in $SEEDS; do echo "$(outdir "$k")/runs.json"; done) <<'PY' | tee "$REPORT"
import json, os, statistics, sys
REQ = 1826.0
v = [json.load(open(f))["d69"]["efpd"] for f in sys.argv[1:] if os.path.exists(f) and "d69" in json.load(open(f))]
print(f"C9-69, eight-layer cycle length, {len(v)} seed(s): " + ", ".join(f"{x:.1f}" for x in v) + " d")
if len(v) > 1:
    m, sd = statistics.mean(v), statistics.stdev(v)
    se = sd / len(v) ** 0.5
    print(f"mean {m:.1f} d, seed standard deviation {sd:.1f} d, standard error {se:.1f} d")
    print(f"margin over {REQ:.0f} d: {m - REQ:+.1f} d, {(m - REQ) / se:.1f} standard errors")
PY
}

case "${1:-}" in
--status)
  echo "$(stamp) host $(hostname)"
  b=$(busy); [ -n "$b" ] && echo "running: $b" || echo "running: nothing"
  for k in $SEEDS; do [ -f "$(outdir "$k")/runs.json" ] && echo "seed $k: runs.json present" || echo "seed $k: not finished"; done
  report; exit 0 ;;
esac

# ------------------------------------------------------------ preflight ----
[ "$(hostname)" = "wks720" ]                || die "not on wks720"
[ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"
for f in "$CKPT" "$FIRST" c9_dep_core3d.py; do [ -e "$f" ] || die "missing $f (run from ~/master-thesis-unipi)"; done
b=$(busy); [ -z "$b" ] || die "another transport job is running, one 64-thread job at a time: $b"
python c9_dep_core3d.py --checkpoint "$CKPT" --designs 69 --layers 8 --salt core3d-seed2 \
  --out "$(outdir 2)" --dry-run || die "dry run failed"
[ "${1:-}" = "--preflight" ] && { echo "$(stamp) preflight ok"; exit 0; }

# ------------------------------------------------------------------ run ----
for k in $SEEDS; do
  echo "$(stamp) seed $k of C9-69"
  python -u c9_dep_core3d.py --checkpoint "$CKPT" --designs 69 --layers 8 \
    --threads "$THREADS" --salt "core3d-seed$k" --out "$(outdir "$k")" || die "seed $k failed"
done
echo "$(stamp) all seeds finished"
report
echo "to send the results back:"
echo "  git add -f c9a_dep_core3d_d69_seed*/runs.json $REPORT && git commit -m 'C9-69: eight-layer seeds 2 to 5' && git push"
