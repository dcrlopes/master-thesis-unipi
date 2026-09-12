#!/usr/bin/env bash
# run_c9.sh -- Campaign 9, the reformulated campaign, in one entry point.
#
#   minimise   F_dH, c_max
#   subject to EFPD >= 1826, F_dH <= 1.65, k window, LEU cap, vessel fit,
#              four-bank margin at the operating maximum
#
# STAGES (each skipped when its marker exists, so an interruption is cheap)
#   P  preflight: host, env, OpenMC, branch, files, pure selftests
#   W  apply the patch and run the wiring test through the real optimiser
#   S  smoke run, coarse transport, ~10 min, proves the OpenMC path
#   F  full run, 24 + 6x6 = 60 evaluations, ~24 h, detached
#
# USAGE, from the repository root
#   bash run_c9.sh --preflight            # stage P only, writes nothing
#   bash run_c9.sh --smoke                # P, W, S in the foreground
#   setsid nohup bash run_c9.sh > c9.log 2>&1 < /dev/null &     # P, W, S, F
#   rm .c9_markers/S                      # force a stage to rerun
#
# Before launching F update the @reboot crontab so a power cut resumes the
# run: the resume line is printed by run_optimization.py at the end of S.
set -u -o pipefail

THREADS=64
MARK=.c9_markers
OUT=out_c9
OUT_SMOKE=out_c9_smoke
KT=ktarget_table_c8.json

# the problem, one place
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

mkdir -p "$MARK"
die()  { echo "FAIL: $*" >&2; exit 1; }
hr()   { printf '%.0s-' $(seq 1 74); echo; }
mark() { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }

COMMON=(--ktarget-table "$KT" --k-basis core --k-max "$K_MAX" --k-min "$K_MIN"
        --f-max "$F_MAX" --enr-max "$ENR_MAX" --ctrl-margin "$CTRL_MARGIN"
        --objective-set c9 --efpd-req "$EFPD_REQ" --boron-objective "$BORON_OBJ"
        --boron-step "$BORON_STEP" --boron-top "$BORON_TOP"
        --boron-ceiling "$BORON_CEILING" --hump-noise "$HUMP_NOISE"
        --threads "$THREADS")

# ---------------------------------------------------------------- stage P --
stage_P() {
  hr; echo " STAGE P. Preflight"; hr
  echo "  host        : $(hostname)"
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  echo "  conda env   : ${CONDA_DEFAULT_ENV:-none}"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('  openmc      :', openmc.__version__)" \
    || die "openmc import failed"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] \
    || die "OpenMC is not 0.15.3"
  echo "  branch      : $(git branch --show-current)"
  [ "$(git branch --show-current)" = "main" ] || die "not on main"
  echo "  head        : $(git log --oneline -1)"
  echo "  xs          : ${OPENMC_CROSS_SECTIONS:-unset}"
  echo "  chain       : ${OPENMC_CHAIN_FILE:-unset}"
  [ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
  for f in "$KT" boron_objective.py apply_c9_reformulation.py test_c9_wiring.py \
           run_optimization.py openmc_evaluator.py reactor_optimization.py; do
    [ -f "$f" ] && echo "  found       : $f" || die "missing $f"
  done
  echo "  threads     : $THREADS of $(nproc)"
  echo "  disk free   : $(df -h . | awk 'NR==2{print $4}')"
  python boron_objective.py --selftest || die "boron_objective selftest failed"
  python apply_c9_reformulation.py --selftest || die "applier selftest failed"
  echo "  other jobs  : $(pgrep -fc '[r]un_optimization.py' || true) run_optimization.py running"
  echo "preflight OK"
}

# ---------------------------------------------------------------- stage W --
stage_W() {
  done_ W && { echo "[W] already done"; return 0; }
  hr; echo " STAGE W. Patch and wiring test"; hr
  if grep -q "c9_efpd_req" openmc_evaluator.py; then
    echo "  patch already applied"
  else
    python apply_c9_reformulation.py --check || die "anchors not unique, not applying"
    python apply_c9_reformulation.py         || die "patch failed"
  fi
  git --no-pager diff --stat -- run_optimization.py openmc_evaluator.py reactor_optimization.py
  python test_c9_wiring.py 2>&1 | grep -v "Warning\|warn\|preprocessing\|scale the data\|_check_optimize\|ABNORMAL\|^$" \
    || die "wiring test failed"
  echo "  if this looks wrong: python apply_c9_reformulation.py --revert"
  mark W
}

# ---------------------------------------------------------------- stage S --
stage_S() {
  done_ S && { echo "[S] already done"; return 0; }
  hr; echo " STAGE S. Smoke run (coarse transport, 4 + 2 evaluations)"; hr
  python -c "import numpy, openmc; print('env ok')" \
    && python -u run_optimization.py --smoke --out "$OUT_SMOKE" "${COMMON[@]}" \
         --n-init 4 --n-infill 2 --iters 1 2>&1 | tee "$OUT_SMOKE.log" \
    || die "smoke run failed"
  python - <<PY || die "smoke checkpoint is not a Campaign 9 archive"
import json
d = json.load(open("$OUT_SMOKE/optimization_checkpoint.json"))
assert d["objectives"] == [["peaking", "min"], ["c_max", "min"]], d["objectives"]
assert "g_efpd" in d["constraint_names"] and "g_ctrl_peak" in d["constraint_names"]
assert d["meta"]["objective_set"] == "c9"
r = d["all_raw"]
assert all("c_max" in x and "n_boron_solves" in x for x in r)
print("  smoke archive OK:", len(r), "evaluations,",
      "boron solves", [x["n_boron_solves"] for x in r],
      "c_max", [round(x["c_max"]) for x in r])
PY
  mark S
}

# ---------------------------------------------------------------- stage F --
stage_F() {
  done_ F && { echo "[F] already done"; return 0; }
  hr; echo " STAGE F. Full campaign, 24 + 6 x 6 evaluations"; hr
  [ -d "$OUT" ] && die "$OUT exists. Resume with the line printed at the end of S, or move it away."
  python -c "import numpy, openmc; print('env ok')" \
    && python -u run_optimization.py --out "$OUT" "${COMMON[@]}" \
         --n-init 24 --n-infill 6 --iters 6 \
         --nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5 \
         2>&1 | tee "$OUT.log" \
    || die "full run failed"
  mark F
}

# ------------------------------------------------------------------- main --
case "${1:-}" in
  --preflight) stage_P; exit 0 ;;
  --smoke)     stage_P; stage_W; stage_S; exit 0 ;;
  "")          stage_P; stage_W; stage_S; stage_F ;;
  *)           die "unknown option $1" ;;
esac
hr; echo " stages done: $(ls "$MARK" | tr '\n' ' ')"; hr
