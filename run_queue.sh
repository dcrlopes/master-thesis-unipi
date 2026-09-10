#!/usr/bin/env bash
# run_queue.sh -- the three open items, in one entry point.
#
# ORDER
#   Stages B and C take seconds and touch no transport, so they run first
#   and their results are on screen before the sweep occupies the machine.
#   Stage A is the only run that costs solves.
#
#   B  reflector slope basis report                 no transport
#   C  Part B patch and rerun of the retrospective  no transport
#   A  sweep_refl_at_reference.py                   8 solves, ~30 min
#
# Each stage is skipped if its marker exists, so an interruption is cheap.
#
# USAGE, from the repository root
#   bash run_queue.sh --preflight
#   bash run_queue.sh                    # foreground, B and C are quick
#   setsid nohup bash run_queue.sh > queue.log 2>&1 < /dev/null &
#
#   bash run_queue.sh --only B           # a single stage
#   rm .queue_markers/A                  # force one stage to rerun
set -u -o pipefail

THREADS=32
MARK=.queue_markers
PATCH=apply_c8_reformulation_partb.py
RETRO=c8_reformulation_retro.py
SWEEP=sweep_refl_at_reference.py
OUTDIR=kt_refl_ref

die() { echo "FAIL: $*" >&2; exit 1; }
stage_done() { [ -f "$MARK/$1" ]; }
mark() { mkdir -p "$MARK"; touch "$MARK/$1"; }
hr() { printf '=%.0s' $(seq 1 74); echo; }

# ------------------------------------------------------------- preflight --
preflight() {
  hr
  echo " PREFLIGHT"
  hr
  echo "  host        : $(hostname)"
  echo "  cwd         : $(pwd)"
  echo "  conda env   : ${CONDA_DEFAULT_ENV:-NONE}"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] \
    || die "wrong env, run: conda activate openmc-env"

  python -c "import numpy, openmc; print('  openmc      :', openmc.__version__)" \
    || die "numpy or openmc not importable"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] \
    || die "openmc is not 0.15.3, this is not the campaign environment"

  echo "  branch      : $(git rev-parse --abbrev-ref HEAD)"
  [ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] \
    || die "wrong branch, run: git checkout main"

  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] && [ -f "$OPENMC_CROSS_SECTIONS" ] \
    || die "OPENMC_CROSS_SECTIONS unset or missing"
  echo "  XS          : $OPENMC_CROSS_SECTIONS"

  for f in "$SWEEP" "$RETRO" "$PATCH"; do
    [ -f "$f" ] || die "missing $f"
  done
  echo "  scripts     : all three present"

  local busy
  busy=$(pgrep -af "python.*(run_optimization|confirm3d\.py|boron_worth|mtc_scan|sweep_ktarget|sweep_refl)" \
         | grep -v run_queue || true)
  [ -z "$busy" ] || { echo "$busy"; die "a simulation is already running"; }
  echo "  no simulation currently running"
  echo "  threads     : $THREADS of $(nproc)"
  echo "  stages done : $(ls $MARK 2>/dev/null | tr '\n' ' ')"
  echo "preflight OK"
  echo
}

# --------------------------------------- stage B, reflector slope basis --
stage_B() {
  stage_done B && { echo "[B] already done"; return 0; }
  hr; echo " STAGE B. Reflector slope basis"; hr

  echo "-- the conversion used by $SWEEP --"
  grep -En "def slope_in_pcm|1e5 \* b|assert abs\(slope_in_pcm" "$SWEEP" || true
  echo
  echo "  slope_in_pcm(b, lf_ref) = 1e5 * b / lf_ref"
  echo "  so the script normalises the stored delta-LF slope by LF_ref,"
  echo "  and a chapter that writes 1e5 * b alone does not."
  echo

  echo "-- who writes fit_slope_per_cm --"
  grep -rEln "fit_slope_per_cm" --include=*.py . || echo "  (no python writer found)"
  echo

  echo "-- the stored value and the implied LF_ref --"
  python - <<'PY'
import json, pathlib
p = pathlib.Path("ktarget_table_c8.json")
if not p.exists():
    print("  ktarget_table_c8.json not found in the repository root")
    raise SystemExit(0)
d = json.load(open(p))
print("  top type:", type(d).__name__)
keys = list(d) if isinstance(d, dict) else range(len(d))
print("  top keys:", list(keys)[:20])

def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for n, v in enumerate(o[:3]):
            yield from walk(v, path + f"[{n}]")
    else:
        yield path, o

hits = [(p_, v) for p_, v in walk(d)
        if "slope" in p_.lower() or "lf" in p_.lower().split("/")[-1]]
for p_, v in hits[:20]:
    print(f"  {p_} = {v}")

b = next((v for p_, v in walk(d) if p_.endswith("fit_slope_per_cm")), None)
if b is None:
    print("  fit_slope_per_cm not found, inspect the keys above")
else:
    print(f"\n  stored slope b            = {b:.4e} per cm")
    print(f"  un-normalised, 1e5 * b    = {1e5*b:+.1f} pcm per cm")
    print(f"  chapter currently states  = -51.6 pcm per cm")
    print(f"  script states             = -47.6 pcm per cm")
    print(f"  implied LF_ref            = {51.6/47.6:.4f}")
    print("  pick ONE basis for the whole chapter and set \\REFLSLOPE to it")
PY
  echo
  mark B
}

# ------------------------------------------- stage C, Part B correction --
stage_C() {
  stage_done C && { echo "[C] already done"; return 0; }
  hr; echo " STAGE C. Part B of the reformulation"; hr

  python "$PATCH" --selftest || die "patch selftest failed"
  echo
  python "$PATCH" --check    || die "anchors are not unique, not applying"
  echo
  python "$PATCH"            || die "patch failed to apply"
  echo
  echo "-- diff --"
  git --no-pager diff --stat -- "$RETRO" || true
  echo
  echo "-- rerun at the five-year mission --"
  python "$RETRO" --mission-efpd 1826 --f-limit 1.65 \
    2>&1 | tee reform_after_patch.log || die "retrospective failed after patch"
  echo
  echo "  if this looks wrong: python $PATCH --revert"
  mark C
}

# ----------------------------------------------- stage A, reflector sweep --
stage_A() {
  stage_done A && { echo "[A] already done"; return 0; }
  hr; echo " STAGE A. Reflector sweep at the reference composition"; hr

  python "$SWEEP" --selftest || die "sweep selftest failed"
  python "$SWEEP" --dry-run  || die "sweep dry run failed"
  echo
  echo "  launching, 8 solves, output in $OUTDIR"
  python -u "$SWEEP" --threads "$THREADS" --out "$OUTDIR" \
    2>&1 | tee "${OUTDIR}.log" || die "sweep failed"
  echo
  echo "-- verdict --"
  grep -Ei "verdict|slope|NEITHER|reproduce" "$OUTDIR/report.txt" | head -20 || true
  echo
  echo "  then substitute the matching paragraph into meth_limits_section.tex"
  mark A
}

# ------------------------------------------------------------------ main --
ONLY=""
case "${1:-}" in
  --preflight) preflight; exit 0 ;;
  --only)      ONLY="${2:-}" ;;
  "")          ;;
  *)           die "unknown argument: $1" ;;
esac

preflight
if [ -n "$ONLY" ]; then
  case "$ONLY" in
    A) stage_A ;;
    B) stage_B ;;
    C) stage_C ;;
    *) die "--only takes A, B or C" ;;
  esac
else
  stage_B
  stage_C
  stage_A
fi

hr
echo " QUEUE COMPLETE"
echo "  stages done : $(ls $MARK 2>/dev/null | tr '\n' ' ')"
hr
