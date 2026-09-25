#!/usr/bin/env bash
# run_c9_floor.sh -- Campaign 9 continued under a raised mission floor.
#
# WHAT
#   The Campaign 9 problem, settings and evaluator, unchanged, with one number
#   different: the cycle-length floor is 2270 d on the assembly-level value
#   instead of 1826 d, so that after the measured axial-burnup loss (mean
#   ratio 0.805, seven designs depleted with eight layers) the resolved cycle
#   still holds five years. The archive of Campaign 9 seeds the surrogate;
#   only the infill is new.
#
# TIME (wks720, 64 threads, 18.7 min per evaluation)
#   stage S   seed                    seconds, no transport
#   stage I   ITERS x 6 evaluations   about 1.9 h per block, 5.6 h for three
#   stage C   eight-layer depletion of each front member, about 1 h each
#
# STAGES, each skipped when its marker exists
#   P preflight   W wait for a PID   S seed   I infill   C confirm   F report
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9_floor.sh --preflight
#   setsid nohup bash run_c9_floor.sh > c9_floor.log 2>&1 < /dev/null &
#   setsid nohup bash run_c9_floor.sh 12345 > c9_floor.log 2>&1 < /dev/null &   # wait for that PID
#   bash run_c9_floor.sh --status
#   N_SEED=24 OUT=out_c9f_doe bash run_c9_floor.sh     # seed with the design of experiments only
#   ITERS=4 bash run_c9_floor.sh                        # a longer search
set -u -o pipefail

THREADS=${THREADS:-64}
FLOOR=${FLOOR:-2270}
N_SEED=${N_SEED:-60}
ITERS=${ITERS:-3}
N_INFILL=${N_INFILL:-6}
OUT=${OUT:-out_c9f}
WORK=${WORK:-openmc_runs_c9f}
DEP=${DEP:-c9f_dep_core3d}
MARK=.c9f_markers_${OUT}
REPORT=${OUT}_report.txt

# the Campaign 9 problem, as in run_c9.sh, except EFPD_REQ
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
        --objective-set c9 --efpd-req "$FLOOR" --boron-objective "$BORON_OBJ"
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

case "${1:-}" in
--status)
  echo "time        : $(stamp)   host $(hostname)"
  p=$(pgrep -f "[r]un_c9_floor.sh" | head -1)
  [ -n "$p" ] && echo "runner      : ALIVE (pid $p)" || echo "runner      : not running"
  st=""; for s in P W S I C F; do done_ $s && st="$st $s"; done
  echo "stages done :${st:- none}"
  echo "archive     : $(n_arch $OUT) evaluations ($N_SEED seeded from Campaign 9)"
  [ -f "$OUT/optimization_checkpoint.json" ] && echo "front       : $(front_of $OUT)"
  exit 0 ;;
esac

stage_P() {
  hr; echo " STAGE P. Preflight  ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] || die "OpenMC is not 0.15.3"
  grep -q "c9_efpd_req" openmc_evaluator.py || die "the Campaign 9 reformulation is not applied"
  for f in seed_c9_floor.py out_c9/optimization_checkpoint.json "$KT" c9_dep_core3d.py; do
    [ -f "$f" ] || die "missing $f"
  done
  python seed_c9_floor.py --floor "$FLOOR" --n-seed "$N_SEED" --dry-run || die "seed dry run failed"
  echo "  budget      : $ITERS blocks of $N_INFILL, about $(python -c "print(f'{$ITERS*$N_INFILL*18.7/60:.1f}')") h"
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

stage_S() {
  done_ S && { echo "[S] already done"; return 0; }
  hr; echo " STAGE S. Seed archive, floor $FLOOR d  ($(stamp))"; hr
  python seed_c9_floor.py --floor "$FLOOR" --n-seed "$N_SEED" --out "$OUT" || die "seeding failed"
  [ "$(n_arch $OUT)" -eq "$N_SEED" ] || die "the seed holds $(n_arch $OUT) evaluations, expected $N_SEED"
  mark S
}

stage_I() {
  done_ I && { echo "[I] already done"; return 0; }
  hr; echo " STAGE I. Infill, $ITERS blocks of $N_INFILL  ($(stamp))"; hr
  python -u run_optimization.py --resume "$OUT/optimization_checkpoint.json" \
    --out "$OUT" --workdir "$WORK" "${COMMON[@]}" \
    --iters "$ITERS" --n-infill "$N_INFILL" \
    --nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5 \
    2>&1 | tee -a "$OUT.log" || die "infill failed"
  mark I
}

stage_C() {
  done_ C && { echo "[C] already done"; return 0; }
  hr; echo " STAGE C. Eight-layer confirmation of the front  ($(stamp))"; hr
  local f; f=$(front_of "$OUT")
  [ -n "$f" ] || die "the front is empty"
  echo "  front: $f"
  python -u c9_dep_core3d.py --checkpoint "$OUT/optimization_checkpoint.json" \
    --designs $f --layers 8 --threads "$THREADS" --out "$DEP" \
    2>&1 | tee -a "$DEP.log" || die "confirmation failed"
  mark C
}

stage_F() {
  hr; echo " REPORT  ($(stamp))"; hr
  python c9_dep_core3d.py --checkpoint "$OUT/optimization_checkpoint.json" --out "$DEP" --analyse >> "$DEP.log" 2>&1 || echo "  WARNING aggregation failed"
  { echo "Campaign 9 under a floor of $FLOOR d, $(stamp)"
    echo "archive     : $(n_arch $OUT) evaluations, $N_SEED seeded from Campaign 9"
    echo "front       : $(front_of $OUT)   (indices below $N_SEED are Campaign 9 designs)"
    echo
    echo "eight-layer confirmation, resolved cycle against the 1826 d mission:"
    python - "$DEP" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "summary.json"
if not p.exists():
    print("  no summary yet"); raise SystemExit
for k, v in json.loads(p.read_text()).items():
    e, a = v.get("efpd_3d", 0), v.get("efpd_archive", 1)
    print(f"  design {k}: assembly {a:6.0f} d, eight layers {e:6.0f} d, ratio {e/a:.3f}, margin {e-1826:+.0f} d")
PY
  } | tee "$REPORT"
  echo
  echo "push:"
  echo "  git add -f $OUT/optimization_checkpoint.json $DEP/summary.json"
  echo "  git add seed_c9_floor.py run_c9_floor.sh $REPORT"
  echo "  git add -f $OUT.log $DEP.log c9_floor.log"
  echo "  git commit -m 'Campaign 9 under a 2270 d floor: infill and eight-layer confirmation'"
  mark F
}

case "${1:-}" in
  --preflight) stage_P ;;
  ""|[0-9]*)   stage_P; [ -n "${1:-}" ] && stage_W "$1"; stage_S; stage_I; stage_C; stage_F ;;
  *)           die "unknown option $1" ;;
esac
