#!/usr/bin/env bash
# run_c9_axial.sh -- Campaign 9 continued under a per-design axial floor, with
# the eight-layer confirmation inside the loop as a gate.
#
# WHAT
#   The Campaign 9 problem, settings and evaluator, seeded with the 78
#   evaluations of the first continuation (60 of Campaign 9 and 18 under the
#   2270 d floor), with the cycle-length constraint read through
#   axial_ratio_model.py: every record is scored by E_req(design) - E_asm,
#   where E_req = 1826 / (ratio_hat - kappa s_loo), and every design measured
#   with eight layers takes its measured cycle instead. After each infill
#   block the front members not yet measured are depleted with eight layers,
#   the model is refitted on all measured designs, and the archive is
#   rescored, so the search and the measurement converge together. The
#   hypervolume history therefore moves between rounds for two reasons, the
#   new evaluations and the rescoring, and the report prints it per round.
#
# WHY  (25 Sep 2026)
#   The constant floor of 2270 d admitted design 72, which resolved to
#   1782 d. The gadolinia-only fit admits it too. No proxy of this accuracy
#   (38 to 89 d leave-one-out) decides the margins that decide the front
#   (+12 to +69 d), so the confirmation is a gate, not a check afterwards.
#
# TIME (wks720, 64 threads)
#   stage S   fit and seed                     seconds, no transport
#   round r   I: 6 evaluations, about 1.9 h    C: about 1 h per new front member
#             R: refit and rescore, seconds
#   three rounds: about 6 h of infill plus 3 to 9 h of confirmation
#
# STAGES, each skipped when its marker exists
#   P preflight   W wait for a PID   S fit + seed   for r in 1..ROUNDS: I_r C_r R_r   F report
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9_axial.sh --preflight
#   setsid nohup bash run_c9_axial.sh > c9_axial.log 2>&1 < /dev/null &
#   setsid nohup bash run_c9_axial.sh 618757 > c9_axial.log 2>&1 < /dev/null &   # wait for that PID first
#   bash run_c9_axial.sh --status
#   REGRESSORS="inventory" bash run_c9_axial.sh          # another regressor set (see axial_ratio_model.py --report)
#   ROUNDS=4 KAPPA=1.5 bash run_c9_axial.sh
set -u -o pipefail

THREADS=${THREADS:-64}
MISSION=${MISSION:-1826}
ROUNDS=${ROUNDS:-3}
N_INFILL=${N_INFILL:-6}
KAPPA=${KAPPA:-1.0}
REGRESSORS=${REGRESSORS:-"gd_wt hump_asm_pcm"}
SRC=${SRC:-out_c9f/optimization_checkpoint.json}
OUT=${OUT:-out_c9a}
WORK=${WORK:-openmc_runs_c9a}
DEP=${DEP:-c9a_dep_core3d}
MODEL=${MODEL:-axial_ratio_model.json}
MARK=.c9a_markers_${OUT}
REPORT=${OUT}_report.txt

# the Campaign 9 problem, as in run_c9.sh and run_c9_floor.sh, at the mission floor
KT=ktarget_table_c8.json
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
        --objective-set c9 --efpd-req "$MISSION" --boron-objective "$BORON_OBJ"
        --boron-step "$BORON_STEP" --boron-top "$BORON_TOP"
        --boron-ceiling "$BORON_CEILING" --hump-noise "$HUMP_NOISE"
        --axial-model "$MODEL" --threads "$THREADS")

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
# the front of the archive under the current reading: feasible, below the MTC
# boron limit, non-dominated in (peaking, c_max)
front_of() { python - "$1" <<'PY'
import json, sys
from pathlib import Path
ck = json.loads((Path(sys.argv[1]) / "optimization_checkpoint.json").read_text())
con = ck["constraint_names"]
pts = [(i, float(r["peaking"]), float(r["c_max"])) for i, r in enumerate(ck["all_raw"])
       if all(float(r.get(c, 0.0)) <= 1e-9 for c in con) and float(r["c_max"]) <= 2763]
print(" ".join(str(i) for i, f, c in pts if not any(
    j != i and g <= f and d <= c and (g < f or d < c) for j, g, d in pts)))
PY
}
# the front members that no eight-layer run has measured yet, by design variables
unmeasured_of() { python - "$1" "$MODEL" <<'PY'
import json, sys
from pathlib import Path
import axial_ratio_model as arm
ck = json.loads((Path(sys.argv[1]) / "optimization_checkpoint.json").read_text())
model = arm.load(sys.argv[2])
con = ck["constraint_names"]
pts = [(i, float(r["peaking"]), float(r["c_max"])) for i, r in enumerate(ck["all_raw"])
       if all(float(r.get(c, 0.0)) <= 1e-9 for c in con) and float(r["c_max"]) <= 2763]
front = [i for i, f, c in pts if not any(j != i and g <= f and d <= c and (g < f or d < c) for j, g, d in pts)]
print(" ".join(str(i) for i in front if arm.measured_lookup(model, ck["all_raw"][i]) is None))
PY
}
describe_model() { python -c "import axial_ratio_model as a, sys; print(a.describe(a.load(sys.argv[1])))" "$MODEL"; }

case "${1:-}" in
--status)
  echo "time        : $(stamp)   host $(hostname)"
  p=$(pgrep -f "[r]un_c9_axial.sh" | grep -v "^$$\$" | head -1)
  [ -n "$p" ] && echo "runner      : ALIVE (pid $p)" || echo "runner      : not running"
  st=""; for s in P W S $(for r in $(seq 1 "$ROUNDS"); do echo I$r C$r R$r; done) F; do done_ $s && st="$st $s"; done
  echo "stages done :${st:- none}"
  echo "archive     : $(n_arch $OUT) evaluations (78 seeded from the first continuation)"
  [ -f "$MODEL" ] && echo "model       : $(describe_model)"
  [ -f "$OUT/optimization_checkpoint.json" ] && echo "front       : $(front_of $OUT)   unmeasured: $(unmeasured_of $OUT)"
  [ -f "$DEP/runs.json" ] && echo "confirmed   : $(python -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1])))))" "$DEP/runs.json")"
  exit 0 ;;
esac

stage_P() {
  hr; echo " STAGE P. Preflight  ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] || die "OpenMC is not 0.15.3"
  grep -q "c9_axial_model" openmc_evaluator.py || die "the axial floor is not in the evaluator"
  grep -q "axial-model" run_optimization.py || die "the axial floor flag is not in run_optimization.py"
  for f in axial_ratio_model.py seed_c9_axial.py c9_dep_core3d.py "$SRC" "$KT"; do
    [ -f "$f" ] || die "missing $f"
  done
  python axial_ratio_model.py --selftest || die "axial_ratio_model selftest failed"
  python axial_ratio_model.py --report || die "the measured set cannot be read"
  python axial_ratio_model.py --fit $REGRESSORS --kappa "$KAPPA" --out "$MARK/preflight_model.json" || die "the fit failed"
  python seed_c9_axial.py --src "$SRC" --model "$MARK/preflight_model.json" --out "$MARK/preflight_seed" --dry-run || die "seed dry run failed"
  echo "  rounds      : $ROUNDS of $N_INFILL evaluations, about $(python -c "print(f'{$ROUNDS*$N_INFILL*18.7/60:.1f}')") h of infill plus the confirmations"
  echo "  threads     : $THREADS of $(nproc)"
  echo "  other jobs  : $(pgrep -fc '[r]un_optimization.py' || true) run_optimization.py running"
  echo "preflight OK"
  mark P
}

stage_W() {
  local pid="$1"
  done_ W && return 0
  kill -0 "$pid" 2>/dev/null || die "PID $pid is not running"
  echo "waiting for PID $pid  ($(stamp))"
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  echo "PID $pid finished  ($(stamp))"
  mark W
}

refit() {
  python axial_ratio_model.py --fit $REGRESSORS --kappa "$KAPPA" --out "$MODEL" || die "the fit failed"
}

stage_S() {
  done_ S && { echo "[S] already done"; return 0; }
  hr; echo " STAGE S. Fit the model and seed the archive  ($(stamp))"; hr
  refit
  python seed_c9_axial.py --src "$SRC" --model "$MODEL" --out "$OUT" || die "seeding failed"
  [ "$(n_arch $OUT)" -ge 78 ] || die "the seed holds $(n_arch $OUT) evaluations, expected at least 78"
  mark S
}

stage_I() {
  local r="$1"
  done_ I$r && { echo "[I$r] already done"; return 0; }
  hr; echo " STAGE I$r. Infill block $r of $ROUNDS, $N_INFILL evaluations  ($(stamp))"; hr
  python -u run_optimization.py --resume "$OUT/optimization_checkpoint.json" \
    --out "$OUT" --workdir "$WORK" "${COMMON[@]}" \
    --iters 1 --n-infill "$N_INFILL" \
    --nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5 \
    2>&1 | tee -a "$OUT.log" || die "infill $r failed"
  mark I$r
}

stage_C() {
  local r="$1"
  done_ C$r && { echo "[C$r] already done"; return 0; }
  hr; echo " STAGE C$r. Eight-layer confirmation of the unmeasured front members  ($(stamp))"; hr
  local f; f=$(unmeasured_of "$OUT")
  echo "  front: $(front_of "$OUT")   unmeasured: ${f:-none}"
  if [ -n "$f" ]; then
    python -u c9_dep_core3d.py --checkpoint "$OUT/optimization_checkpoint.json" \
      --designs $f --layers 8 --threads "$THREADS" --out "$DEP" \
      2>&1 | tee -a "$DEP.log" || die "confirmation $r failed"
  fi
  mark C$r
}

stage_R() {
  local r="$1"
  done_ R$r && { echo "[R$r] already done"; return 0; }
  hr; echo " STAGE R$r. Refit on every measured design and rescore the archive  ($(stamp))"; hr
  refit
  python seed_c9_axial.py --inplace "$OUT/optimization_checkpoint.json" --model "$MODEL" || die "rescoring failed"
  mark R$r
}

stage_F() {
  hr; echo " REPORT  ($(stamp))"; hr
  python c9_dep_core3d.py --checkpoint "$OUT/optimization_checkpoint.json" --out "$DEP" --analyse >> "$DEP.log" 2>&1 || echo "  WARNING aggregation failed"
  { echo "Campaign 9 under the axial floor, $(stamp)"
    echo "model       : $(describe_model)"
    echo "archive     : $(n_arch $OUT) evaluations, 78 seeded from the first continuation"
    echo "front       : $(front_of $OUT)   (indices below 60 are Campaign 9, 60 to 77 the 2270 d continuation)"
    echo "unmeasured  : $(unmeasured_of $OUT)"
    echo
    echo "front members against the $MISSION d mission, measured where a run exists:"
    python axial_ratio_model.py --predict "$MODEL" "$OUT/optimization_checkpoint.json" $(front_of $OUT) --mission "$MISSION"
    echo
    echo "hypervolume history (restarts at the seed; moves at each rescoring):"
    python -c "import json, sys; print('  ' + ' '.join(f'{h:.0f}' for h in json.load(open(sys.argv[1]))['hv_history']))" "$OUT/optimization_checkpoint.json"
  } | tee "$REPORT"
  echo
  echo "push:"
  echo "  git add -f $OUT/optimization_checkpoint.json $DEP/runs.json $DEP/summary.json $DEP/summary.txt"
  echo "  git add axial_ratio_model.py seed_c9_axial.py run_c9_axial.sh $MODEL $REPORT"
  echo "  git add -f $OUT.log $DEP.log c9_axial.log"
  echo "  git commit -m 'Campaign 9 under the axial floor: infill with the eight-layer gate'"
  mark F
}

case "${1:-}" in
  --preflight) stage_P ;;
  ""|[0-9]*)
    stage_P
    [ -n "${1:-}" ] && stage_W "$1"
    stage_S
    for r in $(seq 1 "$ROUNDS"); do stage_I "$r"; stage_C "$r"; stage_R "$r"; done
    stage_F ;;
  *) die "unknown option $1" ;;
esac
