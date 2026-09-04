#!/usr/bin/env bash
# =============================================================================
# run_c8_night.sh -- Campaign 8: everything still open on the three Notion
# tasks, in one detached job for wks720. Version 1, 4 September 2026, written
# against commit f60b78a of branch campaign8. Supersedes run_c8_stageA.sh and
# shares its marker directory, so stages already finished there are skipped.
#
#  stage        what                                          solves   time
#  integrity    runs.json parse, design 13 cache count            0     10 s
#  hump         C8 k history from the chunked archive (c8_hump2)  0    < 1 min
#               c8_hump2.py must be in the repo root or under $HOME; the
#               driver copies it in and verifies its flags in preflight
#  kh_c7        C7 k history recovered from out_c7/run.log.gz     0    < 1 min
#  swing        reactivity swing + hump-adjusted margins          0    < 1 min
#  boron_sum    rebuild boron_c8/summary.json for 11 designs      0     1 min
#  d13          design 13, 1000 ppm, ARO ARI RE12, 2 seeds        12    50 min
#  refl956      closure 3a, reflector 0.956, ARO, designs 47 21    8    30 min
#  rabs4229     closure 3b, absorber 0.4229, ARI RE12, 47 21      16    65 min
#  fullb4c      closure 4,  full-B4C stack, ARI RE12, 47 21       16    65 min
#  kt13         closure 1,  burnup validation of design 13       ~14   ~2 h
#  tier2        closure 1,  design 47 depleted at 10x particles  ~14   ~3 h  (skip: --no-tier2)
#  closures     numbers of the four closures, table, draft para   0    10 s
#  rollup       stage A 2.3, roll-up fix in run_c8_post.sh        0     1 min
#  figures      stage A 2.4, c8_post_figures.py                   0     2 min
#                                                        total  ~4.5 h (+3 h tier2)
#
# A stage that fails is logged and the driver moves on; the summary at the
# end lists every failure. Delete .c8_markers_post/<stage> to force a rerun.
# Nothing is committed and nothing is deleted.
#
# USAGE
#   cd ~/master-thesis-unipi && conda activate openmc-env
#   bash run_c8_night.sh --preflight                 # checks only, runs nothing
#   setsid nohup bash run_c8_night.sh > run_c8_night.log 2>&1 < /dev/null &
#   sleep 3; pgrep -af run_c8_night; tail -f run_c8_night.log
#   bash run_c8_night.sh --no-tier2                  # same without the 3 h stage
#
# Native conda job. Do NOT prefix with the Docker `lab` alias.
# =============================================================================
set -u -o pipefail

CKPT="out_c8/optimization_checkpoint.json"
WORKDIR="openmc_runs_c8"
THREADS=32
DESIGNS="47 42 23 29 21 44 59 1 53 31 13"
SENS_DESIGNS="47 21"           # low-enrichment floor and high-enrichment knee
MARK=".c8_markers_post"
NEED_GB=15
TIER2=1
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
  if "$fn"; then mark "$name"; say "[$name] DONE in $(( ($(date +%s)-t0)/60 )) min"
  else FAILED+=("$name"); say "[$name] FAILED after $(( ($(date +%s)-t0)/60 )) min, continuing"; fi
}
need_flags() {                   # need_flags <script> <flag>... : verify the CLI before spending hours
  local script="$1"; shift; local help
  help=$(python "$script" --help 2>&1) || die "$script --help failed:
$help"
  for flag in "$@"; do grep -q -- "$flag" <<<"$help" || die "$script does not accept $flag"; done
}
locate() {                       # locate <file> : copy from $HOME if not in the repo root
  [ -f "$1" ] && return 0
  local hit; hit=$(find "$HOME" -maxdepth 4 -name "$1" -not -path "*/master-thesis-unipi/*" 2>/dev/null | head -1)
  if [ -n "$hit" ]; then cp "$hit" . && say "copied $1 from $hit"; return 0; fi
  return 1
}

# --------------------------------------------------------------- preflight --
preflight() {
  echo "=============================================================="
  echo " PREFLIGHT  $(date -Is)"
  echo "=============================================================="
  echo "  host        : $(hostname)"
  echo "  cwd         : $(pwd)"
  echo "  conda env   : ${CONDA_DEFAULT_ENV:-NONE}"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "wrong conda environment. Run: conda activate openmc-env"
  python -c "import openmc, numpy, matplotlib" 2>/dev/null || die "openmc, numpy or matplotlib not importable. You are probably in (base)."
  echo "  python      : $(python -c 'import sys; print(sys.version.split()[0])')"
  echo "  openmc      : $(python -c 'import openmc; print(openmc.__version__)')"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] || die "openmc is not 0.15.3, this is not the campaign environment"
  echo "  branch      : $(git branch --show-current)"
  [ "$(git branch --show-current)" = "campaign8" ] || die "wrong branch. Run: git checkout campaign8"
  echo "  threads     : $THREADS of $(nproc) available"
  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] && [ -f "$OPENMC_CROSS_SECTIONS" ] \
    || die "OPENMC_CROSS_SECTIONS is not set or the file is missing. Expected \$HOME/openmc_data/endfb-vii.1-hdf5/cross_sections.xml"
  echo "  XS          : $OPENMC_CROSS_SECTIONS"
  CHAIN="${OPENMC_CHAIN_FILE:-}"
  [ -n "$CHAIN" ] || CHAIN=$(find "$HOME/openmc_data" -maxdepth 2 -name "chain_endfb71_pwr.xml" 2>/dev/null | head -1)
  [ -n "$CHAIN" ] && [ -f "$CHAIN" ] || die "depletion chain chain_endfb71_pwr.xml not found under \$HOME/openmc_data; export OPENMC_CHAIN_FILE"
  echo "  chain       : $CHAIN"

  for f in "$CKPT" confirm3d.py hardware3d.py boron_worth.py validate_ktarget_burnup.py \
           ktarget_table_c8.json confirm3d_c8/summary.json confirm3d_c8/runs.json \
           boron_c8/runs.json kt_burnup/summary.json out_c7/run.log.gz out_c7/optimization_checkpoint.json; do
    [ -e "$f" ] || die "missing $f"
  done
  [ -d "$WORKDIR" ] || say "  NOTE: $WORKDIR absent, the hump stage (chunked archive) will be skipped"

  local free; free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  echo "  free disk   : ${free} GB"; [ "$free" -ge "$NEED_GB" ] || die "less than ${NEED_GB} GB free"

  local others
  others=$(pgrep -af "python.*(run_optimization|confirm3d\.py|boron_worth|validate_ktarget|c8_hump)" \
           | grep -v -E "^($$|$PPID) " | grep -v run_c8_night || true)
  if [ -n "$others" ]; then echo "  RUNNING JOBS:"; echo "$others"; die "a simulation is already running. Wait or kill it first."; fi
  echo "  no simulation currently running"

  for f in c8_reactivity_swing.py c7_khist_from_log.py c8_closures_report.py; do
    locate "$f" || die "$f not found in the repo root or under \$HOME (it is in the 4 Sep bundle)"
  done
  if locate c8_hump2.py; then HUMP=c8_hump2.py
  else HUMP=c8_khist_hump.py; say "  NOTE: c8_hump2.py not found, falling back to c8_khist_hump.py (it failed on 4 Sep; the stage may fail and will be skipped)"; fi
  need_flags "$HUMP" --checkpoint --workdir --designs --out --dry-run --selftest
  python "$HUMP" --selftest >/dev/null || die "$HUMP --selftest failed"
  echo "  hump script : $HUMP (flags verified, selftest OK)"
  [ -d "$WORKDIR" ] && python - "$WORKDIR" <<'PY'
import sys, pathlib
w = pathlib.Path(sys.argv[1]); cases = sorted(w.glob("case_*"))
h5 = sum(1 for c in cases for _ in c.glob("dep_*/depletion_results.h5"))
print(f"  {w}: {len(cases)} case directories, {h5} depletion_results.h5 files (the hump needs them)")
PY
  locate apply_rollup_fix.py  || say "  NOTE: apply_rollup_fix.py absent, rollup stage will be skipped"
  locate c8_post_figures.py   || say "  NOTE: c8_post_figures.py absent, figures stage will be skipped"

  need_flags confirm3d.py --refl-steel-vol --cr-abs-radius --rod-stack --boron-ppm --states --seeds --threads --out
  need_flags validate_ktarget_burnup.py --ktarget-table --dep-particles --seeds --threads --chain --out --designs
  need_flags boron_worth.py --checkpoint --designs --out --threads
  python c8_reactivity_swing.py --selftest >/dev/null || die "c8_reactivity_swing selftest failed"
  python c7_khist_from_log.py --selftest  >/dev/null || die "c7_khist_from_log selftest failed"
  python confirm3d.py --selftest >/dev/null 2>&1 || say "  NOTE: confirm3d --selftest returned non-zero (check by hand: python confirm3d.py --selftest)"
  echo "  scripts present and their flags verified"
  echo "  stages already done : $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"
  echo "  tier2 (3 h)         : $([ "$TIER2" = 1 ] && echo ON || echo OFF)"
  echo "preflight OK"
}

# ------------------------------------------------------------------ stages --
stage_integrity() {
  python - <<'PY' || return 1
import json, sys
r = json.load(open("confirm3d_c8/runs.json"))          # raises if the two writers of 3 Sep collided
s = json.load(open("confirm3d_c8/summary.json"))
n13 = sum(k.startswith("13|") for k in r)
print(f"  confirm3d_c8/runs.json parses: {len(r)} solves, design 13 has {n13} of 12; summary covers {sorted(s, key=int)}")
smoke = [k for k, v in r.items() if "|5000|" in k]
print(f"  smoke-tagged entries: {len(smoke)} (expected 2, design 53 ARO seed 0)")
PY
}

stage_hump() {
  [ -d "$WORKDIR" ] || { say "  $WORKDIR absent, skipping"; return 0; }
  python "$HUMP" --selftest || return 1
  python "$HUMP" --checkpoint "$CKPT" --workdir "$WORKDIR" --designs $DESIGNS --dry-run || return 1
  python "$HUMP" --checkpoint "$CKPT" --workdir "$WORKDIR" --designs $DESIGNS --out khist_c8 || return 1
  [ -s khist_c8/khist.json ]
}

stage_kh_c7() {
  python c7_khist_from_log.py --log out_c7/run.log.gz --checkpoint out_c7/optimization_checkpoint.json --out kh_c7 || return 1
  [ -s kh_c7/k_histories.json ]
}

stage_swing() {              # rerun at the end as well, once khist_c8 and design 13 exist
  python c8_reactivity_swing.py --out swing_c8 || return 1
  [ -s swing_c8/swing.json ]
}

stage_boron_sum() {          # boron_worth.py overwrites summary.json with the designs of the call;
  python - <<'PY' || { say "  cache incomplete, a rebuild would solve; skipping (run boron_worth by hand)"; return 0; }
import json, sys
r = json.load(open("boron_c8/runs.json"))
need = [f"{d}|{s}|{c}|0" for d in "47 42 23 29 21 44 59 1 53 31 13".split() for s in ("ARO","ARI","RE12") for c in ("0.0","500.0","1000.0","1500.0")]
miss = [k for k in need if k not in r]
print(f"  boron cache: {len(need)-len(miss)} of {len(need)} solves present; missing {miss or 'none'}")
sys.exit(1 if miss else 0)
PY
  cp boron_c8/summary.json boron_c8/summary.json.bak_$(date +%Y%m%d)
  python -u boron_worth.py --checkpoint "$CKPT" --designs $DESIGNS --threads "$THREADS" --out boron_c8 || return 1
  python -c "import json,sys; s=json.load(open('boron_c8/summary.json')); print('  summary designs:', sorted(s,key=int)); sys.exit(0 if len(s)==11 else 1)"
}

stage_d13() {
  if python -c "import json,sys; sys.exit(0 if '13' in json.load(open('confirm3d_c8/summary.json')) else 1)"; then
    say "  design 13 already in confirm3d_c8/summary.json"; return 0; fi
  say "  fidelity flags omitted on purpose: 150000 x 200 x 80 is in the cache key of the ten finished designs"
  python -u confirm3d.py --checkpoint "$CKPT" --designs 13 --states ARO ARI RE12 --seeds 2 --threads "$THREADS" --out confirm3d_c8 || return 1
  python -c "import json,sys; sys.exit(0 if '13' in json.load(open('confirm3d_c8/summary.json')) else 1)"
}

stage_refl956() {            # closure 3a: the benchmark reflector, unrodded state, separate --out
  python -u confirm3d.py --checkpoint "$CKPT" --designs $SENS_DESIGNS --states ARO --seeds 2 --threads "$THREADS" \
      --refl-steel-vol 0.956 --out confirm3d_c8_refl956 || return 1
  [ -s confirm3d_c8_refl956/summary.json ]
}

stage_rabs4229() {           # closure 3b: the benchmark B4C radius on both absorbers, rodded states
  python -u confirm3d.py --checkpoint "$CKPT" --designs $SENS_DESIGNS --states ARI RE12 --seeds 2 --threads "$THREADS" \
      --cr-abs-radius 0.4229 --out confirm3d_c8_rabs4229 || return 1
  [ -s confirm3d_c8_rabs4229/summary.json ]
}

stage_fullb4c() {            # closure 4: the 2D stack on the 3D core, isolates leakage from the stack
  python -u confirm3d.py --checkpoint "$CKPT" --designs $SENS_DESIGNS --states ARI RE12 --seeds 2 --threads "$THREADS" \
      --rod-stack full-b4c --out confirm3d_c8_fullb4c || return 1
  [ -s confirm3d_c8_fullb4c/summary.json ]
}

stage_kt13() {               # closure 1: validate_ktarget_burnup rewrites summary.json with the designs of
                             # the call, so all eleven are passed; the ten finished ones come from the cache
  if python -c "import json,sys; sys.exit(0 if 13 in {int(r['idx']) for r in json.load(open('kt_burnup/summary.json'))} else 1)"; then
    say "  design 13 already in kt_burnup/summary.json"; return 0; fi
  cp kt_burnup/summary.json kt_burnup/summary.json.bak_$(date +%Y%m%d)
  python -u validate_ktarget_burnup.py --checkpoint "$CKPT" --designs $DESIGNS --ktarget-table ktarget_table_c8.json \
      --seeds 3 --threads "$THREADS" --chain "$CHAIN" --out kt_burnup || return 1
  python -c "import json,sys; s=json.load(open('kt_burnup/summary.json')); print('  kt_burnup designs:', sorted(int(r['idx']) for r in s)); sys.exit(0 if len(s)==11 else 1)"
}

stage_tier2() {              # closure 1: design 47 at ten times the depletion particles, own directory
  python -u validate_ktarget_burnup.py --checkpoint "$CKPT" --designs 47 --ktarget-table ktarget_table_c8.json \
      --dep-particles 40000 --seeds 2 --threads "$THREADS" --chain "$CHAIN" --out kt_burnup_hifi47 || return 1
  [ -s kt_burnup_hifi47/summary.json ]
}

stage_closures() {
  python c8_reactivity_swing.py --out swing_c8 || return 1
  python c8_closures_report.py --out closures_c8 || return 1
  [ -s closures_c8/closures_scope.tex ]
}

stage_rollup() {
  [ -f apply_rollup_fix.py ] || { say "  apply_rollup_fix.py absent, skipping"; return 0; }
  python apply_rollup_fix.py --check || return 1
  python apply_rollup_fix.py --apply || return 1
  bash -n run_c8_post.sh
}

stage_figures() {
  [ -f c8_post_figures.py ] || { say "  c8_post_figures.py absent, skipping"; return 0; }
  python c8_post_figures.py || return 1
  ls figs_c8_post/*.pdf >/dev/null 2>&1
}

# -------------------------------------------------------------------- main --
cd "$(dirname "$0")" 2>/dev/null || true
for arg in "$@"; do case "$arg" in --no-tier2) TIER2=0;; --preflight) ;; *) die "unknown flag $arg";; esac; done
preflight
[ "${1:-}" = "--preflight" ] && exit 0

say "starting, estimated 4.5 h plus 3 h if tier2 is on"
T0=$(date +%s)
run_stage integrity stage_integrity
run_stage hump      stage_hump
run_stage kh_c7     stage_kh_c7
run_stage swing     stage_swing
run_stage boron_sum stage_boron_sum
run_stage d13       stage_d13
run_stage refl956   stage_refl956
run_stage rabs4229  stage_rabs4229
run_stage fullb4c   stage_fullb4c
run_stage kt13      stage_kt13
[ "$TIER2" = 1 ] && run_stage tier2 stage_tier2
rm -f "$MARK/closures" "$MARK/figures"          # always regenerate from the night's data
run_stage closures  stage_closures
run_stage rollup    stage_rollup
run_stage figures   stage_figures
T1=$(date +%s)
say "ALL STAGES ATTEMPTED in $(( (T1-T0)/3600 )) h $(( ((T1-T0)%3600)/60 )) min"
if [ "${#FAILED[@]}" -gt 0 ]; then say "FAILED STAGES: ${FAILED[*]}  (grep '\[<stage>\]' run_c8_night.log for the cause)"; else say "no stage failed"; fi
echo
echo "Read in the morning, in this order:"
echo "  1. swing_c8/swing_report.txt        hump and hump-adjusted 3D margins, the candidate decision"
echo "  2. closures_c8/closures_report.txt  the four closures in numbers; closures_scope.tex is the draft paragraph"
echo "  3. kh_c7/report.txt                 Campaign 7 histories, 58 of 60 validated, cases 0 and 23 excluded"
echo "  4. kt_burnup/summary_table.tex      design 13 added; kt_burnup_hifi47/ if tier2 ran"
echo "Then commit the new result files (nothing was committed):"
echo "  git add c8_reactivity_swing.py c7_khist_from_log.py c8_closures_report.py run_c8_night.sh \\"
echo "          swing_c8 kh_c7 closures_c8 confirm3d_c8 confirm3d_c8_refl956 confirm3d_c8_rabs4229 \\"
echo "          confirm3d_c8_fullb4c kt_burnup boron_c8/summary.json khist_c8 2>/dev/null"
echo "  git commit -m \"C8 night run: closures 1 to 4 measured, k histories C7 and C8, reactivity swing\""
