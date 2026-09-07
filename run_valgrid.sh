#!/usr/bin/env bash
# =============================================================================
# run_valgrid.sh -- Step 0 (archive-only validations of the Campaign 8
# surrogate and front) and Step 3 (surrogate-assisted search against
# exhaustive enumeration on a two-variable slice), one detached job for
# wks720. Version 1, 7 September 2026, written against branch main after
# apply_slice_flags.py.
#
#  stage      what                                             evals   time
#  step0      c8_surrogate_cv.py + c8_front_stability.py        0     ~5 min
#  makelist   valgrid_make_list.py: 7 x 5 grid, freeze from 47   0      1 s
#  grid       run_optimization.py --eval-list (enumeration)     35   ~12 h
#  opt        run_optimization.py on the same slice, 8 + 2 x 6  20    ~7 h
#  compare    valgrid_compare.py: HV ratio, IGD, eps+, figures   0     1 min
#                                                       total  55   ~19 h
#
# Both OpenMC stages checkpoint after every evaluation (grid) or every
# infill block (opt) and are resumed by this driver, so a Ctrl+C or a
# reboot costs at most one evaluation (grid) or one block (opt).
# A stage that fails is logged and the driver moves on; the summary lists
# every failure. Delete .valgrid_markers/<stage> to force a rerun.
# Nothing is committed and nothing is deleted.
#
# USAGE
#   cd ~/master-thesis-unipi && conda activate openmc-env
#   bash run_valgrid.sh --preflight                 # checks only, runs nothing
#   setsid nohup bash run_valgrid.sh > run_valgrid.log 2>&1 < /dev/null &
#   sleep 3; pgrep -af run_valgrid; tail -f run_valgrid.log
#   bash run_valgrid.sh --no-step0                  # skip the archive-only stage
#   bash run_valgrid.sh --opt-first                 # run the 20-evaluation search before the grid
#
#   setsid        starts a new session so the job survives the terminal closing
#   nohup         ignores the hang-up signal for the same reason
#   < /dev/null   detaches stdin, so nothing ever waits for the keyboard
#   &             returns the prompt immediately
#
# Native conda job. Do NOT prefix with the Docker `lab` alias.
# =============================================================================
set -u -o pipefail

# ---- the Campaign 8 problem, verbatim (limits, screen, k_target, search) ----
C8_CKPT="out_c8/optimization_checkpoint.json"
KT_TABLE="ktarget_table_c8.json"
LIMITS="--k-basis core --k-min 1.02 --k-max 1.166 --enr-max 16 --ctrl-margin 1000 --ctrl-absorber B4C"
SEARCH="--nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5"
THREADS=64
# ---- the slice and the budgets --------------------------------------------
ANCHOR=47
N_ENR=7
N_GD=5
N_INIT=8
N_ITERS=2
N_INFILL=6
# ---- directories --------------------------------------------------------------
LIST_DIR="valgrid"
OUT_GRID="out_valgrid_grid"
OUT_OPT="out_valgrid_opt"
WD_GRID="openmc_runs_valgrid_grid"
WD_OPT="openmc_runs_valgrid_opt"
FIGS="figs_valgrid"
MARK=".valgrid_markers"
NEED_GB=20
STEP0=1
OPT_FIRST=0
FAILED=()

say()  { echo "$(date -Is)  $*"; }
die()  { echo "$(date -Is)  FAIL: $*" >&2; exit 1; }
done_stage() { [ -f "$MARK/$1" ]; }
mark() { mkdir -p "$MARK"; date -Is > "$MARK/$1"; }
run_stage() {                    # run_stage <name> <function>; failures do not abort
  local name="$1" fn="$2"
  if done_stage "$name"; then say "[$name] already done, skipping"; return 0; fi
  say "[$name] START"
  local t0; t0=$(date +%s)
  if "$fn"; then
    mark "$name"; say "[$name] DONE in $(( ($(date +%s) - t0) / 60 )) min"
  else
    FAILED+=("$name"); say "[$name] FAILED after $(( ($(date +%s) - t0) / 60 )) min (continuing)"
  fi
}
n_evals() {                      # n_evals <checkpoint> -> number of archived evaluations, or 0
  [ -f "$1" ] || { echo 0; return; }
  python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['all_raw']))" "$1" 2>/dev/null || echo 0
}

# ----------------------------------------------------------------------------- preflight
preflight() {
  say "preflight"
  [ -f preflight.sh ] || die "preflight.sh missing from the repository root"
  bash preflight.sh || die "environment gate failed"
  # the driver's own extras
  pgrep -f "run_c8_night|run_c8_post|run_c8_resume" > /dev/null && die "a Campaign 8 driver is running; wait for it"
  for f in run_optimization.py slice_space.py apply_slice_flags.py valgrid_make_list.py \
           valgrid_compare.py c8_surrogate_cv.py c8_front_stability.py "$C8_CKPT" "$KT_TABLE"; do
    [ -f "$f" ] || die "missing $f"
  done
  python apply_slice_flags.py --check | grep -q "FAIL" && die "apply_slice_flags.py anchors are inconsistent"
  python apply_slice_flags.py --check | grep -c "APPLIED" | grep -q "^5$" || die "run_optimization.py is not patched: run python apply_slice_flags.py"
  python slice_space.py --selftest > /dev/null 2>&1 || die "slice_space.py selftest failed"
  python -c "import pymoo, sklearn, matplotlib, numpy" || die "python packages missing"
  local avail; avail=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
  [ "$avail" -ge "$NEED_GB" ] || die "only ${avail} GB free, need ${NEED_GB}"
  say "commit $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD), $(nproc) cores, ${avail} GB free"
  say "preflight OK"
}

# ----------------------------------------------------------------------------- stages
stage_step0() {
  python c8_surrogate_cv.py --checkpoint "$C8_CKPT" --out figs_c8_cv 2>&1 | grep -v "loo [0-9]" || return 1
  python c8_front_stability.py --checkpoint "$C8_CKPT" --out figs_c8_stab || return 1
}

stage_makelist() {
  python valgrid_make_list.py --checkpoint "$C8_CKPT" --anchor "$ANCHOR" \
      --n-enr "$N_ENR" --n-gd "$N_GD" --out "$LIST_DIR" || return 1
  [ -f "$LIST_DIR/freeze.txt" ] && [ -f "$LIST_DIR/box.txt" ] || return 1
}

slice_flags() {                  # the flags every slice run shares
  echo "--ktarget-table $KT_TABLE $LIMITS $(cat "$LIST_DIR/box.txt") $(cat "$LIST_DIR/freeze.txt") --threads $THREADS"
}

stage_grid() {
  local ck="$OUT_GRID/optimization_checkpoint.json" resume=""
  local n_total; n_total=$(python -c "import json; print(len(json.load(open('$LIST_DIR/grid_list.json'))['designs']))")
  local n; n=$(n_evals "$ck")
  if [ "$n" -ge "$n_total" ]; then say "[grid] archive already holds $n of $n_total"; return 0; fi
  [ "$n" -gt 0 ] && { resume="--resume $ck"; say "[grid] resuming with $n of $n_total in the archive"; }
  # shellcheck disable=SC2046
  python -u run_optimization.py $(slice_flags) $SEARCH \
      --eval-list "$LIST_DIR/grid_list.json" $resume \
      --out "$OUT_GRID" --workdir "$WD_GRID" || return 1
  n=$(n_evals "$ck"); say "[grid] archive holds $n of $n_total"
  [ "$n" -ge "$n_total" ]
}

stage_opt() {
  local ck="$OUT_OPT/optimization_checkpoint.json"
  local target=$(( N_INIT + N_ITERS * N_INFILL ))
  local n; n=$(n_evals "$ck")
  if [ "$n" -ge "$target" ]; then say "[opt] archive already holds $n of $target"; return 0; fi
  if [ "$n" -gt 0 ]; then
    local left=$(( (target - n + N_INFILL - 1) / N_INFILL ))
    say "[opt] resuming with $n of $target: $left more infill iteration(s)"
    # shellcheck disable=SC2046
    python -u run_optimization.py $(slice_flags) $SEARCH \
        --resume "$ck" --iters "$left" --n-infill "$N_INFILL" \
        --out "$OUT_OPT" --workdir "$WD_OPT" || return 1
  else
    # shellcheck disable=SC2046
    python -u run_optimization.py $(slice_flags) $SEARCH \
        --n-init "$N_INIT" --iters "$N_ITERS" --n-infill "$N_INFILL" \
        --out "$OUT_OPT" --workdir "$WD_OPT" || return 1
  fi
  n=$(n_evals "$ck"); say "[opt] archive holds $n of $target"
  [ "$n" -ge "$target" ]
}

stage_compare() {
  python valgrid_compare.py --grid "$OUT_GRID/optimization_checkpoint.json" \
      --opt "$OUT_OPT/optimization_checkpoint.json" --out "$FIGS" || return 1
}

# ----------------------------------------------------------------------------- main
for arg in "$@"; do
  case "$arg" in
    --preflight) preflight; exit 0 ;;
    --no-step0)  STEP0=0 ;;
    --opt-first) OPT_FIRST=1 ;;
    *) die "unknown flag $arg" ;;
  esac
done

say "run_valgrid.sh start, pid $$"
preflight
[ "$STEP0" -eq 1 ] && run_stage step0 stage_step0
run_stage makelist stage_makelist
if [ "$OPT_FIRST" -eq 1 ]; then
  run_stage opt  stage_opt
  run_stage grid stage_grid
else
  run_stage grid stage_grid
  run_stage opt  stage_opt
fi
run_stage compare stage_compare

say "summary"
for s in step0 makelist grid opt compare; do
  if done_stage "$s"; then say "  $s: done $(cat "$MARK/$s")"; else say "  $s: NOT done"; fi
done
if [ "${#FAILED[@]}" -gt 0 ]; then say "FAILED stages: ${FAILED[*]}"; exit 1; fi
say "all stages done. Figures in figs_c8_cv/, figs_c8_stab/, $FIGS/"
