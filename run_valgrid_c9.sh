#!/usr/bin/env bash
# run_valgrid_c9.sh -- Campaign 9 verification of the surrogate-assisted search
# against exhaustive enumeration on a two-variable slice, the Campaign 9
# counterpart of the Campaign 8 exercise reported in Section 4.14.3.
#
# WHY
#   Section 4.14 answers three questions. The first two, surrogate accuracy and
#   front stability, are answered on the archive for both campaigns at no
#   transport cost. The third, whether the search recovers the front that an
#   enumeration finds, rests on Campaign 8 alone, whose formulation differs
#   from the one that produced the candidate designs of this work: Campaign 9
#   minimises the critical boron concentration and the peaking factor under a
#   cycle-length floor, with the control screen active.
#
# THE SLICE
#   live      enrich, gd_wt                 the same pair as the Campaign 8 slice
#   frozen    refl_thick, gd_pins           at the values of C9-47
#   box       enrich 3.5 to 8.0 wt%, gd_wt 0 to 8 wt%, wider than the feasible
#             band of the Campaign 9 archive (4.18 to 7.89 and 0.14 to 6.91)
#   problem   the Campaign 9 flags below, identical to run_c9.sh
#
# TIME (wks720, 64 threads, from the 60 Campaign 9 evaluations: 18.7 min each
# on average, 17.0 median, 31.6 maximum)
#   stage G   35 enumeration nodes      about 11 h
#   stage S   20 search evaluations     about 6 h
#   stage E   15 more, EQUAL=1 only     about 5 h   (search at the same budget
#                                                    as the enumeration)
#   stage C   comparison                minutes, no transport
#
# STAGES, each skipped when its marker exists, so the script can be relaunched
#   P preflight   G enumeration   S search   E equal budget   C compare   F report
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_valgrid_c9.sh --preflight
#   setsid nohup bash run_valgrid_c9.sh > valgrid_c9.log 2>&1 < /dev/null &
#   setsid nohup bash run_valgrid_c9.sh 12345 > valgrid_c9.log 2>&1 < /dev/null &  # wait for that PID first
#   bash run_valgrid_c9.sh --status
#   EQUAL=1 bash run_valgrid_c9.sh        # add the equal-budget stage
#   rm .valgrid_c9_markers/S              # force a stage to rerun
set -u -o pipefail

THREADS=${THREADS:-64}
EQUAL=${EQUAL:-0}
MARK=.valgrid_c9_markers
LIST=valgrid_c9/grid_list.json
OUT_G=out_valgrid_c9_grid
OUT_S=out_valgrid_c9_opt
WORK_G=openmc_runs_valgrid_c9_grid
WORK_S=openmc_runs_valgrid_c9_opt
FIGS=figs_valgrid_c9
REPORT=valgrid_c9_report.txt

# the Campaign 9 problem, the same values as run_c9.sh
KT=ktarget_table_c8.json
EFPD_REQ=1826
F_MAX=1.65
K_MAX=1.166
K_MIN=1.02
ENR_MAX=16
CTRL_MARGIN=1000
BORON_STEP=2000
BORON_TOP=3000
BORON_CEILING=2763
HUMP_NOISE=400
BORON_OBJ=floor

COMMON=(--ktarget-table "$KT" --k-basis core --k-max "$K_MAX" --k-min "$K_MIN"
        --f-max "$F_MAX" --enr-max "$ENR_MAX" --ctrl-margin "$CTRL_MARGIN"
        --objective-set c9 --efpd-req "$EFPD_REQ" --boron-objective "$BORON_OBJ"
        --boron-step "$BORON_STEP" --boron-top "$BORON_TOP"
        --boron-ceiling "$BORON_CEILING" --hump-noise "$HUMP_NOISE"
        --threads "$THREADS")

mkdir -p "$MARK"
die()   { echo "FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
n_arch() { python - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "optimization_checkpoint.json"
print(len(json.loads(p.read_text())["all_raw"]) if p.exists() else 0)
PY
}

if [ "${1:-}" = "--status" ]; then
  echo "time        : $(stamp)   host $(hostname)"
  p=$(pgrep -f "[r]un_valgrid_c9.sh" | head -1)
  [ -n "$p" ] && echo "runner      : ALIVE (pid $p)" || echo "runner      : not running"
  st=""; for s in P G S E C F; do done_ $s && st="$st $s"; done
  echo "stages done :${st:- none}"
  echo "enumeration : $(n_arch $OUT_G) of 35 nodes"
  echo "search      : $(n_arch $OUT_S) evaluations"
  [ -f "$REPORT" ] && echo "report      : $REPORT" || true
  exit 0
fi

# ---------------------------------------------------------------- stage P --
stage_P() {
  hr; echo " STAGE P. Preflight  ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] || die "OpenMC is not 0.15.3"
  grep -q "c9_efpd_req" openmc_evaluator.py || die "the Campaign 9 reformulation is not applied: run apply_c9_reformulation.py"
  for f in "$LIST" "$KT" run_optimization.py slice_space.py valgrid_compare.py; do
    [ -f "$f" ] || die "missing $f"
  done
  echo "  nodes       : $(python -c "import json;print(len(json.load(open('$LIST'))['designs']))")"
  echo "  frozen      : $(cat valgrid_c9/freeze.txt)"
  echo "  box         : $(cat valgrid_c9/box.txt)"
  echo "  threads     : $THREADS of $(nproc)"
  echo "  disk free   : $(df -h . | awk 'NR==2{print $4}')"
  echo "  other jobs  : $(pgrep -fc '[r]un_optimization.py' || true) run_optimization.py running"
  echo "  projection  : 35 x 18.7 min = 10.9 h enumeration, 20 x 18.7 min = 6.2 h search"
  echo "preflight OK"
  mark P
}

# ---------------------------------------------------------------- stage W --
stage_W() {
  local pid="$1"
  done_ W && return 0
  kill -0 "$pid" 2>/dev/null || die "PID $pid is not running"
  echo "waiting for PID $pid  ($(stamp))"
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  echo "PID $pid finished  ($(stamp))"
  mark W
}

# ---------------------------------------------------------------- stage G --
stage_G() {
  done_ G && { echo "[G] already done"; return 0; }
  hr; echo " STAGE G. Enumeration of the 35 nodes  ($(stamp))"; hr
  local resume=()
  [ -f "$OUT_G/optimization_checkpoint.json" ] && resume=(--resume "$OUT_G/optimization_checkpoint.json")
  python -u run_optimization.py --out "$OUT_G" --workdir "$WORK_G" "${COMMON[@]}" \
    --eval-list "$LIST" $(cat valgrid_c9/freeze.txt) $(cat valgrid_c9/box.txt) "${resume[@]}" \
    2>&1 | tee -a "$OUT_G.log" || die "enumeration failed"
  [ "$(n_arch $OUT_G)" -ge 35 ] || die "enumeration wrote only $(n_arch $OUT_G) of 35 nodes"
  mark G
}

# ---------------------------------------------------------------- stage S --
stage_S() {
  done_ S && { echo "[S] already done"; return 0; }
  hr; echo " STAGE S. Surrogate-assisted search, 8 + 2 x 6 = 20 evaluations  ($(stamp))"; hr
  python -u run_optimization.py --out "$OUT_S" --workdir "$WORK_S" "${COMMON[@]}" \
    --n-init 8 --iters 2 --n-infill 6 \
    --nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5 \
    $(cat valgrid_c9/freeze.txt) $(cat valgrid_c9/box.txt) \
    2>&1 | tee -a "$OUT_S.log" || die "search failed"
  mark S
}

# ---------------------------------------------------------------- stage E --
stage_E() {
  [ "$EQUAL" = "1" ] || return 0
  done_ E && { echo "[E] already done"; return 0; }
  hr; echo " STAGE E. Equal budget: 15 more evaluations, 35 in total  ($(stamp))"; hr
  python -u run_optimization.py --resume "$OUT_S/optimization_checkpoint.json" \
    --out "$OUT_S" --workdir "$WORK_S" "${COMMON[@]}" \
    --iters 3 --n-infill 5 \
    --nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5 \
    $(cat valgrid_c9/freeze.txt) $(cat valgrid_c9/box.txt) \
    2>&1 | tee -a "$OUT_S.log" || die "equal-budget stage failed"
  mark E
}

# ------------------------------------------------------------- stages C, F --
stage_C() {
  done_ C && { echo "[C] already done"; return 0; }
  hr; echo " STAGE C. Comparison  ($(stamp))"; hr
  python valgrid_compare.py --grid "$OUT_G/optimization_checkpoint.json" \
    --opt "$OUT_S/optimization_checkpoint.json" --out "$FIGS" \
    2>&1 | tee "$REPORT" || die "comparison failed"
  mark C
}

stage_F() {
  hr; echo " DONE  ($(stamp))"; hr
  echo "enumeration : $(n_arch $OUT_G) evaluations in $OUT_G"
  echo "search      : $(n_arch $OUT_S) evaluations in $OUT_S"
  echo "report      : $REPORT"
  echo "figures     : $FIGS/"
  echo
  echo "push the archives and the comparison:"
  echo "  git add -f $OUT_G/optimization_checkpoint.json $OUT_S/optimization_checkpoint.json"
  echo "  git add $FIGS $REPORT valgrid_c9/grid_list.json valgrid_c9/box.txt valgrid_c9/freeze.txt"
  echo "  git add -f $OUT_G.log $OUT_S.log valgrid_c9.log"
  echo "  git commit -m 'Campaign 9 verification of the search against exhaustive enumeration'"
  echo "  git push origin main"
  mark F
}

case "${1:-}" in
  --preflight) stage_P ;;
  ""|[0-9]*)   stage_P; [ -n "${1:-}" ] && stage_W "$1"; stage_G; stage_S; stage_E; stage_C; stage_F ;;
  *)           die "unknown option $1" ;;
esac
