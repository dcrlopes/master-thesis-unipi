#!/usr/bin/env bash
# run_c10.sh -- Campaign 10: the Campaign 9 problem with the cycle-length
# constraint read from the eight-layer three-dimensional core depletion at
# 10 000 x 100 (run_optimization.py --cycle-core3d), in place of the assembly
# depletion and the fitted axial ratio. Objectives, constraints, bounds,
# design of experiments and loop settings are those of Campaign 9, so the
# comparison with it is direct: does the loop find the three-dimensionally
# confirmed designs (C9-27, 68, 69, 70) without a post-analysis?
#
#   minimise   F_dH, c_max
#   subject to EFPD_3D >= 1826, F_dH <= 1.65, k window, LEU cap, vessel fit,
#              four-bank margin at the operating maximum
#
# STAGES (each skipped when its marker exists)
#   P  preflight: host, env, OpenMC, chain, files, pure selftests of the hook
#   S  smoke run, coarse transport and a coarse 3D depletion, about 20 min
#   F  full run, 24 + 6 x 6 = 60 evaluations, about 30 h at 30 min each.
#      Resumable: if out_c10 holds 24 or more evaluations the run continues
#      from its checkpoint with the remaining batches.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c10.sh --preflight
#   bash run_c10.sh --smoke
#   setsid nohup bash run_c10.sh > c10.log 2>&1 < /dev/null &
#   bash run_c10.sh --status
set -u -o pipefail

THREADS=${THREADS:-64}
MARK=.c10_markers
OUT=out_c10
OUT_SMOKE=out_c10_smoke
WORK=openmc_runs_c10
WORK_SMOKE=openmc_runs_c10_smoke
KT=ktarget_table_c8.json
N_INIT=24; N_INFILL=6; ITERS=6

# the problem: Campaign 9, one place (run_c9.sh)
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
HUMP_NOISE_SMOKE=2500
BORON_OBJ=floor
CYCLE3D="10000,100,50"            # the proxy that met every criterion on 7 Oct 2026
CYCLE3D_SMOKE="2000,40,20"
LAYERS=8

mkdir -p "$MARK"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
n_eval() { [ -f "$OUT/optimization_checkpoint.json" ] && python -c "import json; print(json.load(open('$OUT/optimization_checkpoint.json'))['n_real_evaluations'])" || echo 0; }

COMMON=(--ktarget-table "$KT" --k-basis core --k-max "$K_MAX" --k-min "$K_MIN"
        --f-max "$F_MAX" --enr-max "$ENR_MAX" --ctrl-margin "$CTRL_MARGIN"
        --objective-set c9 --efpd-req "$EFPD_REQ" --boron-objective "$BORON_OBJ"
        --boron-step "$BORON_STEP" --boron-top "$BORON_TOP"
        --boron-ceiling "$BORON_CEILING" --hump-noise "$HUMP_NOISE"
        --cycle-layers "$LAYERS" --threads "$THREADS")
LOOP=(--nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5)

stage_P() {
  hr; echo " STAGE P. Preflight"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('  openmc      :', openmc.__version__)" || die "openmc import failed"
  [ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
  echo "  head        : $(git log --oneline -1)"
  for f in "$KT" run_optimization.py openmc_evaluator.py reactor_optimization.py c9_dep_core3d.py test_core3d_hook.py; do
    [ -f "$f" ] || die "missing $f"
  done
  grep -q "cycle-core3d" run_optimization.py || die "run_optimization.py has no --cycle-core3d (git pull?)"
  python test_core3d_hook.py || die "hook selftest failed"
  python c9_dep_core3d.py --selftest || die "c9_dep_core3d selftest failed"
  python run_optimization.py --help | grep -q "cycle-core3d" || die "driver does not accept --cycle-core3d"
  echo "  other jobs  : $(pgrep -fc '[r]un_optimization.py|[c]9_dep_core3d.py|[c]9_dep_asm3d.py' || true) transport jobs running"
  echo "preflight OK"
}

stage_S() {
  done_ S && { echo "[S] already done"; return 0; }
  hr; echo " STAGE S. Smoke run (coarse transport, 4 + 2 evaluations, coarse 3D depletion)"; hr
  rm -rf "$WORK_SMOKE" "$OUT_SMOKE"
  python -u run_optimization.py --smoke --out "$OUT_SMOKE" --workdir "$WORK_SMOKE" "${COMMON[@]}" \
       --n-init 4 --n-infill 2 --iters 1 --hump-noise "$HUMP_NOISE_SMOKE" \
       --cycle-core3d "$CYCLE3D_SMOKE" 2>&1 | tee "$OUT_SMOKE.log" || die "smoke run failed"
  python - <<PY || die "smoke checkpoint is not a Campaign 10 archive"
import json
d = json.load(open("$OUT_SMOKE/optimization_checkpoint.json"))
r = d["all_raw"]
assert d["meta"]["campaign9"]["cycle_core3d"] is not None, "cycle_core3d missing from meta"
m = [x for x in r if x.get("axial_measured")]
print("  smoke archive OK:", len(r), "evaluations,", len(m), "with a 3D cycle length")
assert len(m) == len(r), "some evaluations kept the assembly cycle length: " + str([x.get("core3d_error") for x in r if not x.get("axial_measured")])
print("    cycle_length (3D):", [round(x["cycle_length"]) for x in r])
print("    cycle_length_asm :", [round(x["cycle_length_asm"]) for x in r])
print("    t_dep3d min      :", [round(x["t_dep3d_s"] / 60, 1) for x in r])
PY
  mark S
}

stage_F() {
  done_ F && { echo "[F] already done"; return 0; }
  hr; echo " STAGE F. Full campaign, $N_INIT + $ITERS x $N_INFILL evaluations"; hr
  n=$(n_eval)
  if [ "$n" -ge $((N_INIT + ITERS * N_INFILL)) ]; then echo "[F] $n evaluations already in $OUT"; mark F; return 0; fi
  if [ "$n" -ge "$N_INIT" ]; then
    left=$(( (N_INIT + ITERS * N_INFILL - n + N_INFILL - 1) / N_INFILL ))
    echo "  resuming $OUT at $n evaluations: $left more batch(es) of $N_INFILL"
    python -u run_optimization.py --resume "$OUT/optimization_checkpoint.json" --out "$OUT" --workdir "$WORK" \
         "${COMMON[@]}" "${LOOP[@]}" --n-init "$N_INIT" --n-infill "$N_INFILL" --iters "$left" \
         --cycle-core3d "$CYCLE3D" 2>&1 | tee -a "$OUT.log" || die "resumed run failed"
  else
    if [ -d "$OUT" ] || [ -d "$WORK" ]; then
      die "$OUT or $WORK exists with fewer than $N_INIT evaluations ($n): the design of experiments cannot be resumed. Move both away and relaunch."
    fi
    python -u run_optimization.py --out "$OUT" --workdir "$WORK" "${COMMON[@]}" "${LOOP[@]}" \
         --n-init "$N_INIT" --n-infill "$N_INFILL" --iters "$ITERS" \
         --cycle-core3d "$CYCLE3D" 2>&1 | tee "$OUT.log" || die "full run failed"
  fi
  [ "$(n_eval)" -ge $((N_INIT + ITERS * N_INFILL)) ] || die "the run ended with $(n_eval) evaluations, relaunch to resume"
  mark F
  echo "$(stamp) Campaign 10 finished. To send the results back:"
  echo "  git add -f $OUT/optimization_checkpoint.json $OUT.log && git commit -m 'Campaign 10: 3D cycle-length proxy in the loop' && git push"
}

case "${1:-}" in
  --preflight) stage_P; exit 0 ;;
  --smoke)     stage_P; stage_S; exit 0 ;;
  --status)    echo "$(stamp) markers: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"; echo "evaluations in $OUT: $(n_eval)"; tail -n 3 "$OUT.log" 2>/dev/null; exit 0 ;;
  "")          stage_P; stage_S; stage_F ;;
  *)           die "unknown option $1" ;;
esac
hr; echo " stages done: $(ls "$MARK" | tr '\n' ' ')"; hr
