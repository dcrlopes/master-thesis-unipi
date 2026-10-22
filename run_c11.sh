#!/usr/bin/env bash
# run_c11.sh -- Campaign 11: the corrected loop. The Campaign 10 problem (3D
# cycle-length constraint from the eight-layer core at 10 000 x 100) with the
# three changes the evidence of October 2026 supports (paper_finds.md 9, 11, 12):
#   1. the end-of-cycle criterion of the 3D depletion uses a fresh-assembly
#      k_bol at 20 000 x 160 (64 pcm, about 10 d) instead of the campaign solve
#      (264 pcm, about 43 d)                                 --kbol-transport
#   2. the MTC boron limit (2763 ppm) is a constraint of the loop  --constrain-boron
#   3. the constraint surrogates use the Matern 1/2 kernel, with kappa of the
#      margin rule set from their measured leave-one-out error
#                                        --con-kernel-nu 0.5 --feas-kappa-per
# Seeded with the 58 evaluations of Campaign 10, their cycle lengths recomputed
# under the corrected criterion (c11_seed_from_c10.py), then 6 batches of 6.
#
# STAGES (each skipped when its marker exists)
#   K  high-statistics k_bol of the Campaign 10 designs that lack one
#      (run_kbol_queue.sh with JOBS_FILE, about 32 min each alone on the machine)
#   S  seed file c11_seed.json and kappa (c11_seed_from_c10.py --require-all)
#   M  smoke run: coarse transport, 4 + 2 evaluations, coarse 3D and k_bol solves
#   F  the campaign, 58 + 6 x 6 = 94 evaluations, about 49 min each, with the
#      watchdog of run_c10.sh (kill and relaunch after STALL_MIN of silence)
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c11.sh --preflight
#   bash run_c11.sh --smoke             stages K, S, M only when K is done
#   setsid nohup bash run_c11.sh > c11.log 2>&1 < /dev/null &
#   bash run_c11.sh --status
set -u -o pipefail
cd "$(dirname "$(readlink -f "$0")")"

THREADS=${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
MARK=.c11_markers
OUT=out_c11
OUT_SMOKE=out_c11_smoke
WORK=openmc_runs_c11
WORK_SMOKE=openmc_runs_c11_smoke
KT=ktarget_table_c8.json
SEED=c11_seed.json
N_INFILL=6; ITERS=6
N_INIT=58                         # the seed; the driver sets n_init to the number of records it reads

# the problem: Campaign 9, one place (run_c9.sh), as in run_c10.sh
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
CYCLE3D="10000,100,50"
CYCLE3D_BU=29.5
CYCLE3D_SMOKE="2000,40,20"
KBOL="20000,160,60"               # the high-statistics fresh-assembly solve of the criterion
KBOL_SMOKE="4000,60,20"
LAYERS=8
CON_NU=0.5
STALL_MIN=${STALL_MIN:-100}
MAX_LAUNCHES=${MAX_LAUNCHES:-25}
POLL_S=60
MACHINE_CSV=c11_machine.csv

mkdir -p "$MARK"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
hr()    { printf -- '-%.0s' $(seq 1 78); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
n_eval() { [ -f "$OUT/optimization_checkpoint.json" ] && python -c "import json; print(json.load(open('$OUT/optimization_checkpoint.json'))['n_real_evaluations'])" || echo 0; }
kappa_args() { python -c "import json; k=json.load(open('c11_kappa.json')); print(' '.join(f'--feas-kappa-per {n}={v[\"kappa\"]:.2f}' for n, v in k.items()))"; }

COMMON=(--ktarget-table "$KT" --k-basis core --k-max "$K_MAX" --k-min "$K_MIN"
        --f-max "$F_MAX" --enr-max "$ENR_MAX" --ctrl-margin "$CTRL_MARGIN"
        --objective-set c9 --efpd-req "$EFPD_REQ" --boron-objective "$BORON_OBJ"
        --boron-step "$BORON_STEP" --boron-top "$BORON_TOP"
        --boron-ceiling "$BORON_CEILING" --constrain-boron --hump-noise "$HUMP_NOISE"
        --cycle-layers "$LAYERS" --threads "$THREADS" --con-kernel-nu "$CON_NU")
LOOP=(--nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5)
TARGET=$(( N_INIT + ITERS * N_INFILL ))

stage_P() {
  hr; echo " STAGE P. Preflight ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || [ -n "${ALLOW_HOST:-}" ] || die "not on wks720 (set ALLOW_HOST=1 to run elsewhere)"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  [ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
  for f in run_optimization.py openmc_evaluator.py core3d_subprocess.py c9_repro_check.py c11_seed_from_c10.py run_kbol_queue.sh \
           "$KT" out_c10/optimization_checkpoint.json openmc_runs_c10/eval_cache.json; do [ -e "$f" ] || die "missing $f"; done
  python -c "import openmc, pymoo, sklearn; print('  openmc', openmc.__version__, '| pymoo', pymoo.__version__, '| sklearn', sklearn.__version__)" || die "imports"
  python run_optimization.py --help | grep -q -- "--kbol-transport" || die "run_optimization.py lacks --kbol-transport (git pull)"
  python run_optimization.py --help | grep -q -- "--con-kernel-nu" || die "run_optimization.py lacks --con-kernel-nu (git pull)"
  python -c "import openmc_evaluator as oe; oe._core3d_selftest(); print('  core3d hook selftest OK')" || die "hook selftest"
  python test_resume_batch_seed.py >/dev/null 2>&1 || die "test_resume_batch_seed.py"
  echo "  preflight OK, $THREADS threads"
}

stage_K() {
  done_ K && { echo "[K] already done"; return 0; }
  hr; echo " STAGE K. High-statistics k_bol of the Campaign 10 designs without one ($(stamp))"; hr
  python c11_seed_from_c10.py --no-kappa || die "c11_seed_from_c10.py"
  if [ -s c11_kbol_jobs.txt ]; then
    echo "  $(wc -l < c11_kbol_jobs.txt) designs to solve, log kbol_queue.log"
    touch kbol_queue.active
    JOBS_FILE=c11_kbol_jobs.txt bash run_kbol_queue.sh run >> kbol_queue.log 2>&1 || die "run_kbol_queue.sh"
    python c11_seed_from_c10.py --no-kappa --require-all || die "designs still without a k_bol after the queue, see kbol_queue.log"
  fi
  mark K
}

stage_S() {
  done_ S && { echo "[S] already done"; return 0; }
  hr; echo " STAGE S. Seed file and kappa from the measured leave-one-out error ($(stamp))"; hr
  python c11_seed_from_c10.py --require-all || die "c11_seed_from_c10.py --require-all"
  N_INIT=$(python -c "import json; print(len(json.load(open('$SEED'))))")
  echo "  seed $SEED: $N_INIT designs; $(kappa_args)"
  mark S
}

stage_M() {
  done_ M && { echo "[M] already done"; return 0; }
  hr; echo " STAGE M. Smoke run: coarse transport, 4 + 2 evaluations, coarse 3D depletion and k_bol solve ($(stamp))"; hr
  rm -rf "$WORK_SMOKE" "$OUT_SMOKE"
  python -u run_optimization.py --smoke --out "$OUT_SMOKE" --workdir "$WORK_SMOKE" "${COMMON[@]}" \
       --n-init 4 --n-infill 2 --iters 1 --hump-noise "$HUMP_NOISE_SMOKE" \
       --cycle-core3d "$CYCLE3D_SMOKE" --kbol-transport "$KBOL_SMOKE" 2>&1 | tee "$OUT_SMOKE.log" || die "smoke run failed"
  python - <<PY || die "smoke checkpoint is not a Campaign 11 archive"
import json
d = json.load(open("$OUT_SMOKE/optimization_checkpoint.json")); r = d["all_raw"]
assert "g_boron" in d["constraint_names"], "g_boron is not a constraint"
assert d["meta"]["campaign9"]["cycle_core3d"].get("kbol_transport"), "kbol_transport missing from meta"
m = [x for x in r if x.get("axial_measured")]
assert len(m) == len(r), "some evaluations kept the assembly cycle length: " + str([x.get("core3d_error") for x in r if not x.get("axial_measured")])
assert all("k_bol_hifi" in x for x in r), "k_bol_hifi missing from an evaluation"
print("  smoke archive OK:", len(r), "evaluations, every one with a 3D cycle length and a high-statistics k_bol")
print("    k_bol campaign :", [round(x["k_bol"], 4) for x in r]); print("    k_bol hifi     :", [round(x["k_bol_hifi"], 4) for x in r])
print("    t_kbol min     :", [round(x["t_kbol_s"] / 60, 1) for x in r]); print("    t_dep3d min    :", [round(x["t_dep3d_s"] / 60, 1) for x in r])
PY
  mark M
}

newest_s() { local f; f=$(find "$WORK" "$OUT.log" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1); [ -n "$f" ] && echo $(( $(date +%s) - ${f%.*} )) || echo 0; }
descendants() { local pid=$1 child; echo "$pid"; for child in $(pgrep -P "$pid" 2>/dev/null); do descendants "$child"; done; }
kill_driver() { local pids; pids=$(descendants "$DRIVER" | tr '\n' ' '); echo "$(stamp)   killing $pids"; kill -TERM $pids 2>/dev/null; sleep 10; kill -KILL $pids 2>/dev/null; wait "$DRIVER" 2>/dev/null; }

launch_driver() {
  local n left
  n=$(n_eval)
  if [ "$n" -ge "$N_INIT" ]; then
    left=$(( (TARGET - n + N_INFILL - 1) / N_INFILL ))
    echo "$(stamp)   resuming $OUT at $n evaluations: $left more batch(es) of $N_INFILL"
    python -u run_optimization.py --resume "$OUT/optimization_checkpoint.json" --out "$OUT" --workdir "$WORK" \
         "${COMMON[@]}" "${LOOP[@]}" $(kappa_args) --n-init "$N_INIT" --n-infill "$N_INFILL" --iters "$left" \
         --cycle-core3d "$CYCLE3D" --cycle-max-burnup "$CYCLE3D_BU" --kbol-transport "$KBOL" >> "$OUT.log" 2>&1 < /dev/null &
  else
    echo "$(stamp)   fresh start, design of experiments = the $N_INIT corrected evaluations of $SEED"
    python -u run_optimization.py --out "$OUT" --workdir "$WORK" "${COMMON[@]}" "${LOOP[@]}" $(kappa_args) \
         --n-init "$N_INIT" --n-infill "$N_INFILL" --iters "$ITERS" --doe-from-cache "$SEED" \
         --cycle-core3d "$CYCLE3D" --cycle-max-burnup "$CYCLE3D_BU" --kbol-transport "$KBOL" >> "$OUT.log" 2>&1 < /dev/null &
  fi
  DRIVER=$!
  echo "$(stamp)   driver pid $DRIVER"
}

stage_F() {
  done_ F && { echo "[F] already done"; return 0; }
  N_INIT=$(python -c "import json; print(len(json.load(open('$SEED'))))"); TARGET=$(( N_INIT + ITERS * N_INFILL ))
  hr; echo " STAGE F. Campaign 11, $N_INIT + $ITERS x $N_INFILL = $TARGET evaluations, watchdog $STALL_MIN min ($(stamp))"; hr
  if command -v free >/dev/null 2>&1; then bash machine_monitor.sh "$MACHINE_CSV" 60 & else sleep 1 & fi
  MONITOR=$!; trap 'kill $MONITOR 2>/dev/null' EXIT
  local launches=0
  while [ "$(n_eval)" -lt "$TARGET" ]; do
    launches=$((launches + 1))
    [ "$launches" -le "$MAX_LAUNCHES" ] || die "$MAX_LAUNCHES launches without finishing, see $OUT.log"
    echo "$(stamp) launch $launches, $(n_eval) evaluations in the checkpoint"
    launch_driver
    while kill -0 "$DRIVER" 2>/dev/null; do
      sleep "$POLL_S"
      s=$(newest_s)
      if [ "$s" -gt $((STALL_MIN * 60)) ]; then
        echo "$(stamp)   no file written for $((s / 60)) min: killing the driver group and relaunching"; kill_driver; break
      fi
    done
    wait "$DRIVER" 2>/dev/null
    echo "$(stamp)   driver ended, $(n_eval) evaluations in the checkpoint"
  done
  kill "$MONITOR" 2>/dev/null
  mark F
  echo "$(stamp) Campaign 11 finished. To send the results back:"
  echo "  git add -f $OUT/optimization_checkpoint.json $OUT.log $MACHINE_CSV $SEED c11_kappa.json $WORK/eval_cache.json repro_c9/kbol_c10_*.json kbol_queue.log && git commit -m 'Campaign 11: the corrected loop' && git push"
}

case "${1:-}" in
  --preflight) stage_P; exit 0 ;;
  --smoke)     stage_P; stage_K; stage_S; stage_M; exit 0 ;;
  --status)    echo "$(stamp) markers: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"; echo "evaluations in $OUT: $(n_eval), newest file $(( $(newest_s) / 60 )) min ago"; [ -f c11_kbol_jobs.txt ] && echo "k_bol jobs left: $(grep -c . c11_kbol_jobs.txt)"; grep "launch\|killing\|driver ended\|FAIL" "$OUT.log" c11.log 2>/dev/null | tail -n 3; [ -f "$MACHINE_CSV" ] && { echo "machine, last 3 minutes:"; tail -n 3 "$MACHINE_CSV"; }; exit 0 ;;
  "")          stage_P; stage_K; stage_S; stage_M; stage_F ;;
  *)           die "usage: bash run_c11.sh [--preflight | --smoke | --status]" ;;
esac
