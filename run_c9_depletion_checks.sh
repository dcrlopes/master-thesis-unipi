#!/usr/bin/env bash
# run_c9_depletion_checks.sh -- the three depletion checks of the Campaign 9
# front, one after the other, with a cost gate before each core depletion.
#
#   check 1   c9_dep_replicas.py   replicas and production fidelity of the
#                                  assembly depletion, five front designs
#   check 2   c9_dep_core2d.py     depletion of the zoned 2D core, C9-47
#   check 3   c9_dep_core3d.py     depletion of the 3D hardware core, C9-47
#
# TIME (wks720, 64 threads)
#   check 1   about 7.7 h, projected from the archived depletion times
#   check 2   unknown: its --estimate (about 10 min) projects it, and the
#             stage runs only if the projection fits the budget
#   check 3   unknown: same rule, against what is left of the budget
#   budget    MAX_HOURS for checks 2 and 3 TOGETHER, default 24 h
#
# STAGES (each skipped when its marker exists, so the script can be relaunched)
#   P  preflight       W  wait for a PID       R1 A1  check 1 run, analyse
#   E2 R2 A2  estimate, run, analyse check 2   E3 R3 A3  the same for check 3
#   F  final report    c9_depletion_checks_report.txt
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9_depletion_checks.sh --preflight
#   setsid nohup bash run_c9_depletion_checks.sh > c9_dep.log 2>&1 < /dev/null &
#   setsid nohup bash run_c9_depletion_checks.sh 252615 > c9_dep.log 2>&1 < /dev/null &   # wait for that PID first
#   bash run_c9_depletion_checks.sh --status
#   rm .c9_dep_markers/R2                     # force a stage to rerun
#
# SETTINGS (environment variables, all optional)
#   MAX_HOURS=24  THREADS=64  LAYERS=8  DESIGNS1="47 44 34 40 35"
#   PARTICLES2=20000 BATCHES2=160 INACTIVE2=60   the 2D core depletion transport
#   PARTICLES3=20000 BATCHES3=160 INACTIVE3=60   the 3D core depletion transport
#   FORCE=1       run checks 2 and 3 even if the projection exceeds the budget
#   SKIP3=1       stop after check 2
set -u -o pipefail

MAX_HOURS=${MAX_HOURS:-24}
THREADS=${THREADS:-64}
LAYERS=${LAYERS:-8}
DESIGNS1=${DESIGNS1:-"47 44 34 40 35"}
PARTICLES2=${PARTICLES2:-20000}; BATCHES2=${BATCHES2:-160}; INACTIVE2=${INACTIVE2:-60}
PARTICLES3=${PARTICLES3:-20000}; BATCHES3=${BATCHES3:-160}; INACTIVE3=${INACTIVE3:-60}
FORCE=${FORCE:-0}
SKIP3=${SKIP3:-0}
MARK=.c9_dep_markers
REPORT=c9_depletion_checks_report.txt
OUT1=c9_dep_replicas; OUT2=c9_dep_core2d; OUT3=c9_dep_core3d
CKPT=out_c9/optimization_checkpoint.json

mkdir -p "$MARK"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
hr()    { printf '%.0s-' $(seq 1 74); echo; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
envok() { python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"; }
hours_from() { grep -o "PROJECTED_HOURS [0-9.]*" "$1" | tail -1 | awk '{print $2}'; }

# ---------------------------------------------------------------- status --
status() {
  echo "time        : $(stamp)   host $(hostname)"
  # every fork of this status call shares its argv, so exclude by the flag
  pid=$(pgrep -f "^bash run_c9_depletion_checks.sh" | while read -r p; do
          ps -o args= -p "$p" 2>/dev/null | grep -q -- "--status" || echo "$p"; done | head -1)
  echo "runner      : $( [ -n "$pid" ] && echo "ALIVE (pid $pid)" || echo "NOT RUNNING" )"
  st=""; for s in P W R1 A1 E2 R2 A2 E3 R3 A3 F; do done_ $s && st="$st $s"; done
  echo "stages done :${st:- none}"
  n1=$(python -c "import json,sys; print(len(json.load(open('$OUT1/runs.json'))))" 2>/dev/null || echo 0)
  echo "check 1     : $n1 of $(( $(echo $DESIGNS1 | wc -w) * 5 )) runs in $OUT1/runs.json"
  for k in 2 3; do
    eval "o=\$OUT$k"
    e=$( [ -f "$o/estimate_d47.json" ] && python -c "import json; print(round(json.load(open('$o/estimate_d47.json'))['projected_h'],1), 'h projected')" 2>/dev/null || echo "no projection yet")
    r=$(python -c "import json; print(len(json.load(open('$o/runs.json'))))" 2>/dev/null || echo 0)
    echo "check $k     : $e, $r design(s) finished"
  done
  [ -f "$REPORT" ] && echo "report      : $REPORT"
}

# ------------------------------------------------------------- preflight --
stage_P() {
  hr; echo " STAGE P. Preflight"; hr
  [ "$(hostname)" = "wks720" ]                || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  envok
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] || die "OpenMC is not 0.15.3"
  [ "$(git branch --show-current)" = "main" ]  || die "not on main"
  echo "  head        : $(git log --oneline -1)"
  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] || die "OPENMC_CROSS_SECTIONS unset"
  [ -n "${OPENMC_CHAIN_FILE:-}" ]     || die "OPENMC_CHAIN_FILE unset"
  [ -f "$OPENMC_CHAIN_FILE" ]         || die "chain file not found: $OPENMC_CHAIN_FILE"
  for f in "$CKPT" ktarget_table_c8.json dep_common.py c9_dep_replicas.py c9_dep_core2d.py \
           c9_dep_core3d.py test_dep_common.py test_dep_builders.py fake_openmc.py \
           openmc_evaluator.py hardware3d.py zoning.py \
           axial_shape_c9.py campaign_timing.py boron_objective.py; do
    [ -f "$f" ] && echo "  found       : $f" || die "missing $f"
  done
  echo "  budget      : $MAX_HOURS h for checks 2 and 3, threads $THREADS of $(nproc), layers $LAYERS"
  echo "  disk free   : $(df -h . | awk 'NR==2{print $4}')"
  python dep_common.py --selftest       || die "dep_common selftest failed"
  python test_dep_common.py | tail -1   || die "test_dep_common failed"
  python test_dep_builders.py | tail -1 || die "test_dep_builders failed"
  python c9_dep_replicas.py --selftest  || die "c9_dep_replicas selftest failed"
  python c9_dep_core2d.py --selftest    || die "c9_dep_core2d selftest failed"
  python c9_dep_core3d.py --selftest    || die "c9_dep_core3d selftest failed"
  python c9_dep_replicas.py --designs $DESIGNS1 --dry-run --out "$OUT1" | tail -2
  python c9_dep_core2d.py --dry-run --out "$OUT2" | tail -1
  python c9_dep_core3d.py --layers "$LAYERS" --dry-run --out "$OUT3" | tail -1
  echo "  other jobs  : $(pgrep -xc openmc || true) openmc solver process(es)"
  echo "preflight OK"; mark P
}

# ------------------------------------------------------------------ wait --
stage_W() {
  local pid="$1"
  done_ W && return 0
  kill -0 "$pid" 2>/dev/null || die "PID $pid is not running. Check it with: pgrep -af 'python -u'"
  echo "$(stamp) waiting for PID $pid: $(ps -o args= -p "$pid" | cut -c1-90)"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  echo "$(stamp) PID $pid has ended"; sleep 60; mark W
}

# ----------------------------------------------------------------- check 1 --
stage_R1() {
  done_ R1 && { echo "[R1] already done"; return 0; }
  hr; echo " STAGE R1. Check 1, replicas and fidelity, designs $DESIGNS1  ($(stamp))"; hr
  envok
  python -u c9_dep_replicas.py --designs $DESIGNS1 --threads "$THREADS" --out "$OUT1" \
    || die "check 1 failed. Relaunch: finished runs are cached in $OUT1/runs.json"
  mark R1
}
stage_A1() {
  done_ A1 && return 0
  python c9_dep_replicas.py --designs $DESIGNS1 --analyse --out "$OUT1" || die "check 1 analysis failed"
  mark A1
}

# ----------------------------------------------------------------- check 2 --
stage_E2() {
  done_ E2 && { echo "[E2] already done: projected $(hours_from "$OUT2/estimate.log") h"; return 0; }
  hr; echo " STAGE E2. Check 2 estimate  ($(stamp))"; hr
  envok
  python -u c9_dep_core2d.py --estimate --threads "$THREADS" --out "$OUT2" \
      --particles "$PARTICLES2" --batches "$BATCHES2" --inactive "$INACTIVE2" | tee "$OUT2/estimate.log" \
    || die "check 2 estimate failed"
  mark E2
}
gate() {   # gate <hours> <budget> <label>
  local h="$1" b="$2" l="$3"
  [ -n "$h" ] || die "$l: no projection found"
  if awk -v h="$h" -v b="$b" 'BEGIN{exit !(h > b)}'; then
    if [ "$FORCE" = "1" ]; then echo "$l: projected $h h exceeds $b h, FORCE=1 so running anyway"
    else die "$l: projected $h h exceeds the remaining budget of $b h. Relaunch with FORCE=1 or a larger MAX_HOURS, or lower the transport settings."; fi
  else echo "$l: projected $h h within the remaining budget of $b h"; fi
}
stage_R2() {
  done_ R2 && { echo "[R2] already done"; return 0; }
  hr; echo " STAGE R2. Check 2, zoned 2D core depletion of C9-47  ($(stamp))"; hr
  envok
  python -u c9_dep_core2d.py --threads "$THREADS" --out "$OUT2" \
      --particles "$PARTICLES2" --batches "$BATCHES2" --inactive "$INACTIVE2" \
    || die "check 2 failed. Relaunch: a finished design is cached in $OUT2/runs.json"
  mark R2
}
stage_A2() {
  done_ A2 && return 0
  python c9_dep_core2d.py --analyse --out "$OUT2" || die "check 2 analysis failed"; mark A2
}

# ----------------------------------------------------------------- check 3 --
stage_E3() {
  done_ E3 && { echo "[E3] already done: projected $(hours_from "$OUT3/estimate.log") h"; return 0; }
  hr; echo " STAGE E3. Check 3 estimate, $LAYERS layers  ($(stamp))"; hr
  envok
  python -u c9_dep_core3d.py --estimate --threads "$THREADS" --out "$OUT3" --layers "$LAYERS" \
      --particles "$PARTICLES3" --batches "$BATCHES3" --inactive "$INACTIVE3" | tee "$OUT3/estimate.log" \
    || die "check 3 estimate failed"
  mark E3
}
stage_R3() {
  done_ R3 && { echo "[R3] already done"; return 0; }
  hr; echo " STAGE R3. Check 3, 3D core depletion of C9-47, $LAYERS layers  ($(stamp))"; hr
  envok
  python -u c9_dep_core3d.py --threads "$THREADS" --out "$OUT3" --layers "$LAYERS" \
      --particles "$PARTICLES3" --batches "$BATCHES3" --inactive "$INACTIVE3" \
    || die "check 3 failed. Relaunch: a finished design is cached in $OUT3/runs.json"
  mark R3
}
stage_A3() {
  done_ A3 && return 0
  python c9_dep_core3d.py --analyse --out "$OUT3" --layers "$LAYERS" || die "check 3 analysis failed"; mark A3
}

# ------------------------------------------------------------------ report --
stage_F() {
  hr; echo " STAGE F. Report  ($(stamp))"; hr
  {
    echo "=== Depletion checks of the Campaign 9 front, $(stamp), $(git log --oneline -1)"
    echo "=== budget $MAX_HOURS h for checks 2 and 3, threads $THREADS"
    for o in "$OUT1" "$OUT2" "$OUT3"; do
      echo; [ -f "$o/summary.txt" ] && cat "$o/summary.txt" || echo "($o: no summary, stage not run)"
    done
  } > "$REPORT"
  cat "$REPORT"; mark F
  echo "$(stamp) DONE. Report: $REPORT"
}

# -------------------------------------------------------------------- main --
case "${1:-}" in
  --preflight) stage_P; exit 0 ;;
  --status)    status; exit 0 ;;
  ""|[0-9]*)
    stage_P
    [ -n "${1:-}" ] && stage_W "$1"
    mkdir -p "$OUT1" "$OUT2" "$OUT3"
    stage_R1; stage_A1
    stage_E2; H2=$(hours_from "$OUT2/estimate.log"); gate "$H2" "$MAX_HOURS" "check 2"
    stage_R2; stage_A2
    if [ "$SKIP3" != "1" ]; then
      stage_E3; H3=$(hours_from "$OUT3/estimate.log")
      LEFT=$(awk -v b="$MAX_HOURS" -v h="$H2" 'BEGIN{printf "%.2f", b - h}')
      gate "$H3" "$LEFT" "check 3"
      stage_R3; stage_A3
    fi
    stage_F ;;
  *) die "unknown option $1" ;;
esac
