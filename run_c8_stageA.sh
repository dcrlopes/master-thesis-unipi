#!/usr/bin/env bash
# =============================================================================
# run_c8_stageA.sh -- Campaign 8 post-analysis, runbook steps 2.1 to 2.4.
# Version 1, 4 September 2026. Written against commit 704f190 of branch
# campaign8.
#
#   2.1  gadolinium hump on the eleven designs        < 1 min   0 solves
#   2.2  design 13, 1000 ppm, 3 states, 2 seeds      ~50 min   ~12 solves
#   2.3  roll-up verdict fix in run_c8_post.sh          1 min   0 solves
#   2.4  regenerate figures, tables, numbers          1-2 min   0 solves
#                                              total ~55 min
#
# Resumable: a finished stage writes a marker in .c8_markers_post/ and is
# skipped on the next launch. Delete a marker to force a stage to rerun.
# Nothing is committed and nothing is deleted. Stage 2.3 keeps a .bak.
#
# USAGE
#   cd ~/master-thesis-unipi && conda activate openmc-env
#   bash run_c8_stageA.sh --preflight        # checks only, runs nothing
#   setsid nohup bash run_c8_stageA.sh > run_c8_stageA.log 2>&1 < /dev/null &
#   sleep 3; pgrep -af run_c8_stageA; tail -f run_c8_stageA.log
#
# Native conda job. Do NOT prefix with the Docker `lab` alias.
# =============================================================================
set -u -o pipefail

CKPT="out_c8/optimization_checkpoint.json"
WORKDIR="openmc_runs_c8"          # meta.workdir of the checkpoint
THREADS=32                        # physical cores of wks720
DESIGNS="47 42 23 29 21 44 59 1 53 31 13"
MARK=".c8_markers_post"
NEED_GB=10

say()  { echo "$(date -Is)  $*"; }
die()  { echo "$(date -Is)  FAIL: $*" >&2; exit 1; }
done_stage() { [ -f "$MARK/$1" ]; }
mark() { mkdir -p "$MARK"; date -Is > "$MARK/$1"; }

# ------------------------------------------------------------------ locate --
# The three scripts come from c8_post_bundle.zip and may not be in the
# repository root yet. Look in the usual places and copy them in.
locate_scripts() {
  local missing=0
  for f in c8_khist_hump.py apply_rollup_fix.py c8_post_figures.py; do
    if [ -f "$f" ]; then continue; fi
    local hit
    hit=$(find "$HOME" -maxdepth 4 -name "$f" -not -path "*/master-thesis-unipi/*" 2>/dev/null | head -1)
    if [ -n "$hit" ]; then
      cp "$hit" . && say "copied $f from $hit"
    else
      echo "  MISSING: $f"; missing=1
    fi
  done
  [ "$missing" = 0 ] || die "unzip c8_post_bundle.zip and put the scripts in $(pwd), or run: find \$HOME -iname 'c8_post_bundle*'"
}

# Verify that a script really accepts the flags this driver passes, instead
# of assuming it. Stops before wasting an hour on a flag that does not exist.
need_flags() {
  local script="$1"; shift
  local help; help=$(python "$script" --help 2>&1) || die "$script --help failed:
$help"
  for flag in "$@"; do
    grep -q -- "$flag" <<<"$help" || die "$script does not accept $flag. Its help says:
$help"
  done
}

# --------------------------------------------------------------- preflight --
preflight() {
  echo "=============================================================="
  echo " PREFLIGHT  $(date -Is)"
  echo "=============================================================="
  echo "  host        : $(hostname)"
  echo "  cwd         : $(pwd)"
  echo "  conda env   : ${CONDA_DEFAULT_ENV:-NONE}"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] \
    || die "wrong conda environment. Run: conda activate openmc-env"
  python -c "import openmc, numpy, matplotlib" 2>/dev/null \
    || die "openmc, numpy or matplotlib not importable. You are probably in (base)."
  echo "  python      : $(python -c 'import sys; print(sys.version.split()[0])')"
  echo "  openmc      : $(python -c 'import openmc; print(openmc.__version__)')"
  echo "  branch      : $(git branch --show-current)"
  [ "$(git branch --show-current)" = "campaign8" ] \
    || die "wrong branch. Run: git checkout campaign8"
  echo "  threads     : $THREADS of $(nproc) available"

  [ -f "$CKPT" ]                || die "missing $CKPT"
  [ -d "$WORKDIR" ]             || die "missing $WORKDIR (the depletion files the hump needs)"
  [ -f confirm3d.py ]           || die "missing confirm3d.py"
  [ -f run_c8_post.sh ]         || die "missing run_c8_post.sh"
  [ -f confirm3d_c8/summary.json ] || die "missing confirm3d_c8/summary.json"
  [ -d boron_c8 ]               || die "missing boron_c8"

  local free; free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  echo "  free disk   : ${free} GB"
  [ "$free" -ge "$NEED_GB" ] || die "less than ${NEED_GB} GB free"

  # Look only at real python jobs, and never at this script, its parent
  # shell, or a pasted command line that merely mentions the names.
  local others
  others=$(pgrep -af "python.*(run_optimization|confirm3d\.py|boron_worth|validate_ktarget)" \
           | grep -v -E "^($$|$PPID) " | grep -v run_c8_stageA || true)
  if [ -n "$others" ]; then
    echo "  RUNNING JOBS:"; echo "$others"
    die "a simulation is already running. Wait for it or kill it before starting."
  fi
  echo "  no simulation currently running"

  locate_scripts
  need_flags c8_khist_hump.py --checkpoint --workdir --designs --out --dry-run --selftest
  need_flags apply_rollup_fix.py --check --apply
  echo "  the three bundle scripts are present and accept the flags used here"

  python - <<'PY'
import json
s = json.load(open("confirm3d_c8/summary.json"))
have = sorted(s, key=int)
print(f"  confirm3d_c8 designs : {' '.join(have)}")
print("  design 13            :", "already present, stage 2.2 will be skipped" if "13" in s else "absent, stage 2.2 will run")
PY

  echo "  stages already done  : $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"
  echo "preflight OK"
}

# ------------------------------------------------------------------ stages --
stage_hump() {                       # 2.1, under a minute
  done_stage hump && { say "2.1 hump already done, skipping"; return; }
  say "2.1 gadolinium hump, no transport, under a minute"
  python c8_khist_hump.py --selftest || die "hump selftest failed"
  say "2.1 dry run, every design must match exactly one case directory"
  python c8_khist_hump.py --checkpoint "$CKPT" --workdir "$WORKDIR" \
        --designs $DESIGNS --dry-run || die "hump dry run failed, do not proceed"
  python c8_khist_hump.py --checkpoint "$CKPT" --workdir "$WORKDIR" \
        --designs $DESIGNS --out khist_c8 || die "hump run failed"
  [ -s khist_c8/khist.json ] || die "khist_c8/khist.json not written"
  say "2.1 done. READ khist_c8 before the candidate decision:"
  say "    hump vs BOL negative  -> every BOL margin of the post-analysis stands"
  say "    hump vs BOL positive  -> subtract it from every BOL margin"
  say "    design 47 boron-free test: (0 ppm four-bank 3D margin) - hump >= 1000 pcm"
  mark hump
}

stage_d13() {                        # 2.2, about 50 minutes
  done_stage d13 && { say "2.2 design 13 already done, skipping"; return; }
  if python -c "import json,sys; sys.exit(0 if '13' in json.load(open('confirm3d_c8/summary.json')) else 1)"; then
    say "2.2 design 13 already in confirm3d_c8/summary.json, skipping"; mark d13; return
  fi
  say "2.2 design 13, ARO ARI RE12, two seeds, 1000 ppm, about 50 minutes"
  say "    fidelity flags are deliberately omitted: the defaults 150000 x 200"
  say "    with 80 inactive are part of the cache key, and changing them would"
  say "    invalidate the ten designs already in confirm3d_c8/runs.json"
  python -u confirm3d.py --checkpoint "$CKPT" \
      --designs 13 --states ARO ARI RE12 --seeds 2 --threads "$THREADS" \
      --out confirm3d_c8 || die "confirm3d on design 13 failed"
  python -c "import json,sys; s=json.load(open('confirm3d_c8/summary.json')); \
sys.exit(0 if '13' in s else 1)" || die "design 13 still absent from the summary"
  say "2.2 done, confirm3d_c8 now covers eleven designs"
  mark d13
}

stage_rollup() {                     # 2.3, one minute
  done_stage rollup && { say "2.3 roll-up fix already done, skipping"; return; }
  say "2.3 roll-up verdict fix in run_c8_post.sh"
  python apply_rollup_fix.py --check || die "apply_rollup_fix --check did not find its anchors"
  python apply_rollup_fix.py --apply || die "apply_rollup_fix --apply failed"
  bash -n run_c8_post.sh || die "run_c8_post.sh no longer parses, run: python apply_rollup_fix.py --revert"
  say "2.3 done, original kept as a .bak"
  mark rollup
}

stage_figures() {                    # 2.4, one to two minutes
  done_stage figures && { say "2.4 figures already done, skipping"; return; }
  say "2.4 regenerating figures, tables and numbers"
  python c8_post_figures.py || die "c8_post_figures.py failed"
  local out="figs_c8_post"
  [ -d "$out" ] || out=$(ls -d figs_c8* 2>/dev/null | head -1)
  [ -n "$out" ] && [ -d "$out" ] || die "no output directory found"
  say "2.4 wrote: $(ls "$out" | tr '\n' ' ')"
  echo "    pdf figures : $(ls "$out"/*.pdf 2>/dev/null | wc -l)"
  [ -f "$out/c8_post_tables.tex" ]   || say "    WARNING: c8_post_tables.tex not produced"
  [ -f "$out/c8_post_numbers.json" ] || say "    WARNING: c8_post_numbers.json not produced"
  say "2.4 done"
  mark figures
}

# The one factual defect to catch before anything reaches the thesis.
check_teval_claim() {
  say "checking the block and the numbers file for the t_eval claim"
  local hits=0
  for f in results_c8_post.tex figs_c8_post/c8_post_numbers.json figs_c8_post/c8_post_tables.tex; do
    [ -f "$f" ] || continue
    if grep -n -i -E "exclude[sd]? the (two )?control|t_eval.{0,40}exclud" "$f"; then hits=1; fi
    if grep -n -E "1[34](\\\\,)?(\\\\%|%| per cent).{0,60}(control|t_eval|evaluation)" "$f"; then hits=1; fi
  done
  if [ "$hits" = 1 ]; then
    say "  ACTION NEEDED: a file above still says t_eval excludes the control solves."
    say "  The Campaign 8 checkpoint has t_eval_s = asm + core + deplete + ctrl + ctrl12"
    say "  on all 60 records, and the two control solves are 16.2 per cent of it."
    say "  Fix that sentence before pasting into the thesis."
  else
    say "  no t_eval claim found in the regenerated files"
  fi
}

# -------------------------------------------------------------------- main --
cd "$(dirname "$0")" 2>/dev/null || true
if [ "${1:-}" = "--preflight" ]; then preflight; exit 0; fi

preflight
say "starting stages, estimated total 55 minutes"
T0=$(date +%s)
stage_hump
stage_d13
stage_rollup
stage_figures
check_teval_claim
T1=$(date +%s)
say "ALL STAGES DONE in $(( (T1-T0)/60 )) min $(( (T1-T0)%60 )) s"
echo
echo "Next, by hand:"
echo "  1. read khist_c8/ and decide whether the boron-free reading of design 47 holds"
echo "  2. git add c8_khist_hump.py apply_rollup_fix.py c8_post_figures.py run_c8_post.sh \\"
echo "             khist_c8/khist.json khist_c8/*.tex figs_c8_post/c8_post_tables.tex \\"
echo "             figs_c8_post/c8_post_numbers.json confirm3d_c8/summary.json confirm3d_c8/runs.json"
echo "  3. git commit -m \"C8 post-analysis: hump, design 13 confirmation, roll-up fix, figures\""
echo "  4. copy results_c8_post.tex, c8_post_tables.tex and figs_c8_post/*.pdf into the"
echo "     dissertation repository, then run apply_c8_post_thesis.py --check"
