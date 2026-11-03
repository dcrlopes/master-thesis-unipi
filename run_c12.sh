#!/usr/bin/env bash
# run_c12.sh -- Campaign 12: the Campaign 11 loop with the four changes its
# results support (paper_finds.md Sections 14 and 15):
#   a. stop rule: hypervolume gain below 1 % for three batches and no censored
#      front member ends the campaign (Campaign 11 would have ended after
#      batch 4 and saved about 15 h)                 --hv-stop --max-evals
#   b. the cycle requirement is the reactivity margin at the mission burnup,
#      g_margin = k_eoc - k_3D(B_mission) <= 0, smooth where the cycle length
#      has a cliff (R^2 0.997, s_z 0.99 in the replay); the 3D depletion runs
#      exactly to B_mission = 18.23 MWd/kgHM (7 solves instead of about 10),
#      constraint kernel back to the campaign Matern 5/2
#                                          --cycle-constraint margin --con-kernel-nu 2.5
#   c. exploration: when the batch separation relaxes below 0.035, the last
#      pick comes from a space-filling sample of the predicted-feasible region
#      (Campaign 11 moved only the pin count)          --explore-sep 0.035 --n-explore 1
#   d. confirmation in the loop: every design that enters the feasible front
#      is depleted twice at 20 000 x 160 before the next batch (one of four
#      Campaign 11 front members was infeasible at full fidelity, five of six
#      rejected near misses were feasible)             --confirm-transport
# Seeded with the 99 evaluations of Campaign 11, g_margin read from their stored
# histories and the 14 confirmed designs at their confirmed values
# (c12_seed_from_c11.py). kappa of the margin rule from the leave-one-out error
# on the seed (c12_kappa.json).
#
# STAGES (each skipped when its marker exists)
#   S  seed file c12_seed.json and kappa
#   M  smoke run: coarse transport, 4 + 2 evaluations, margin constraint,
#      coarse confirmation, stop rule on
#   F  the campaign: batches of 6 until the stop rule fires or MAX_EVALS, with
#      the watchdog of run_c11.sh (kill and relaunch after STALL_MIN of silence)
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c12.sh --preflight
#   bash run_c12.sh --smoke
#   setsid nohup bash run_c12.sh > c12.log 2>&1 < /dev/null &
#   bash run_c12.sh --status
set -u -o pipefail
cd "$(dirname "$(readlink -f "$0")")"

THREADS=${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
MARK=.c12_markers
OUT=out_c12
OUT_SMOKE=out_c12_smoke
WORK=openmc_runs_c12
WORK_SMOKE=openmc_runs_c12_smoke
KT=ktarget_table_c8.json
SEED=c12_seed.json
KAPPA=c12_kappa.json
N_INFILL=6; ITERS=8               # ITERS is the upper bound; the stop rule decides
N_INIT=99                         # replaced by the number of usable seed records in stage S

# the problem: Campaign 9, one place (run_c9.sh), as in run_c10.sh and run_c11.sh
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
CYCLE3D_SMOKE="2000,40,20"
KBOL="20000,160,60"               # the high-statistics fresh-assembly solve of the criterion (Campaign 11)
KBOL_SMOKE="4000,60,20"
CONFIRM="20000,160,60"            # the two-seed confirmation of a new front member (run_c11_confirm.sh)
LAYERS=8
CON_NU=2.5
EXPLORE_SEP=0.035
HV_TOL=0.01; HV_PATIENCE=3
STALL_MIN=${STALL_MIN:-100}
MAX_LAUNCHES=${MAX_LAUNCHES:-25}
POLL_S=60
MACHINE_CSV=c12_machine.csv

mkdir -p "$MARK"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
hr()    { printf -- '-%.0s' $(seq 1 78); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
n_eval() { [ -f "$OUT/optimization_checkpoint.json" ] && python -c "import json; print(json.load(open('$OUT/optimization_checkpoint.json'))['n_real_evaluations'])" || echo 0; }
stopped() { [ -f "$OUT/optimization_checkpoint.json" ] && python -c "import json, sys; s = json.load(open('$OUT/optimization_checkpoint.json')).get('stop') or {}; sys.exit(0 if s.get('stop') else 1)"; }
n_seed() { python -c "import json; print(sum(1 for v in json.load(open('$SEED')).values() if v.get('axial_measured') is not False and 'g_margin' in v))"; }
kappa_args() { python -c "import json; k=json.load(open('$KAPPA')); print(' '.join(f'--feas-kappa-per {n}={v[\"kappa\"]:.2f}' for n, v in k.items()))"; }

COMMON=(--ktarget-table "$KT" --k-basis core --k-max "$K_MAX" --k-min "$K_MIN"
        --f-max "$F_MAX" --enr-max "$ENR_MAX" --ctrl-margin "$CTRL_MARGIN"
        --objective-set c9 --efpd-req "$EFPD_REQ" --boron-objective "$BORON_OBJ"
        --boron-step "$BORON_STEP" --boron-top "$BORON_TOP"
        --boron-ceiling "$BORON_CEILING" --constrain-boron --hump-noise "$HUMP_NOISE"
        --cycle-layers "$LAYERS" --threads "$THREADS" --con-kernel-nu "$CON_NU"
        --cycle-constraint margin --explore-sep "$EXPLORE_SEP" --n-explore 1
        --hv-stop --hv-tol "$HV_TOL" --hv-patience "$HV_PATIENCE")
LOOP=(--nsga-pop 300 --nsga-gen 400 --infill-min-sep 0.14 --feas-kappa 1.5)

stage_P() {
  hr; echo " STAGE P. Preflight ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || [ -n "${ALLOW_HOST:-}" ] || die "not on wks720 (set ALLOW_HOST=1 to run elsewhere)"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  [ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
  for f in run_optimization.py openmc_evaluator.py core3d_subprocess.py hv_stop.py c12_seed_from_c11.py "$KT" \
           out_c11/optimization_checkpoint.json openmc_runs_c11/eval_cache.json \
           c11_dep_core3d_seed1/runs.json c11_dep_core3d_seed2/runs.json c10_dep_core3d_seed1/runs.json c10_dep_core3d_seed2/runs.json; do
    [ -e "$f" ] || die "missing $f"; done
  python -c "import openmc, pymoo, sklearn; print('  openmc', openmc.__version__, '| pymoo', pymoo.__version__, '| sklearn', sklearn.__version__)" || die "imports"
  for flag in --cycle-constraint --confirm-transport --explore-sep --hv-stop; do
    python run_optimization.py --help | grep -q -- "$flag" || die "run_optimization.py lacks $flag (git pull)"; done
  python -c "import openmc_evaluator as oe; oe._core3d_selftest(); print('  core3d hook selftest OK')" || die "hook selftest"
  python -c "import dep_common; dep_common.selftest()" >/dev/null || die "dep_common selftest"
  python hv_stop.py --selftest >/dev/null || die "hv_stop selftest"
  python test_resume_batch_seed.py >/dev/null 2>&1 || die "test_resume_batch_seed.py"
  python test_c12_loop.py >/dev/null 2>&1 || die "test_c12_loop.py"
  echo "  preflight OK, $THREADS threads"
}

stage_S() {
  done_ S && { echo "[S] already done"; return 0; }
  hr; echo " STAGE S. Seed file and kappa from the leave-one-out error ($(stamp))"; hr
  python c12_seed_from_c11.py || die "c12_seed_from_c11.py"
  N_INIT=$(n_seed)
  echo "  seed $SEED: $N_INIT designs; $(kappa_args)"
  mark S
}

stage_M() {
  done_ M && { echo "[M] already done"; return 0; }
  hr; echo " STAGE M. Smoke run: coarse transport, 4 + 2 evaluations, margin constraint, coarse confirmation ($(stamp))"; hr
  rm -rf "$WORK_SMOKE" "$OUT_SMOKE"
  python -u run_optimization.py --smoke --out "$OUT_SMOKE" --workdir "$WORK_SMOKE" "${COMMON[@]}" \
       --n-init 4 --n-infill 2 --iters 1 --hump-noise "$HUMP_NOISE_SMOKE" \
       --cycle-core3d "$CYCLE3D_SMOKE" --kbol-transport "$KBOL_SMOKE" --confirm-transport "$CYCLE3D_SMOKE" \
       2>&1 | tee "$OUT_SMOKE.log" || die "smoke run failed"
  python - <<PY || die "smoke checkpoint is not a Campaign 12 archive"
import json
d = json.load(open("$OUT_SMOKE/optimization_checkpoint.json")); r = d["all_raw"]
assert "g_margin" in d["constraint_names"] and "g_efpd" not in d["constraint_names"], d["constraint_names"]
assert "g_boron" in d["constraint_names"]
m = [x for x in r if x.get("axial_measured")]
assert len(m) == len(r), "some evaluations kept the assembly cycle length: " + str([x.get("core3d_error") for x in r if not x.get("axial_measured")])
assert all("g_margin" in x and "k_bol_hifi" in x for x in r), "g_margin or k_bol_hifi missing from an evaluation"
assert d.get("stop") is not None, "no stop block in the checkpoint"
inf = [p for p in d["phase_log"] if p["stage"] == "infill"]
print("  smoke archive OK:", len(r), "evaluations, every one with g_margin and a high-statistics k_bol; stop:", d["stop"]["by"])
print("    g_margin pcm   :", [round(1e5 * x["g_margin"]) for x in r], " extrapolated:", [x.get("margin_extrapolated") for x in r])
print("    bu_hist last   :", [round(x["bu_hist_3d"][-1], 2) for x in r], "(B_mission 18.23)")
print("    confirmed      :", [x.get("confirmed_3d", False) for x in r], " n_confirmed per batch:", [p.get("n_confirmed") for p in inf])
print("    t_dep3d min    :", [round(x["t_dep3d_s"] / 60, 1) for x in r], " t_confirm min:", [round(x.get("t_confirm_s", 0) / 60, 1) for x in r])
PY
  mark M
}

newest_s() { local f; f=$(find "$WORK" "$OUT.log" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1); [ -n "$f" ] && echo $(( $(date +%s) - ${f%.*} )) || echo 0; }
descendants() { local pid=$1 child; echo "$pid"; for child in $(pgrep -P "$pid" 2>/dev/null); do descendants "$child"; done; }
kill_driver() { local pids; pids=$(descendants "$DRIVER" | tr '\n' ' '); echo "$(stamp)   killing $pids"; kill -TERM $pids 2>/dev/null; sleep 10; kill -KILL $pids 2>/dev/null; wait "$DRIVER" 2>/dev/null; }

launch_driver() {
  local n
  n=$(n_eval)
  if [ "$n" -ge "$N_INIT" ]; then
    echo "$(stamp)   resuming $OUT at $n evaluations, budget $MAX_EVALS, the stop rule decides"
    python -u run_optimization.py --resume "$OUT/optimization_checkpoint.json" --out "$OUT" --workdir "$WORK" \
         "${COMMON[@]}" "${LOOP[@]}" $(kappa_args) --n-init "$N_INIT" --n-infill "$N_INFILL" --max-evals "$MAX_EVALS" \
         --cycle-core3d "$CYCLE3D" --kbol-transport "$KBOL" --confirm-transport "$CONFIRM" >> "$OUT.log" 2>&1 < /dev/null &
  else
    echo "$(stamp)   fresh start, design of experiments = the $N_INIT records of $SEED"
    python -u run_optimization.py --out "$OUT" --workdir "$WORK" "${COMMON[@]}" "${LOOP[@]}" $(kappa_args) \
         --n-init "$N_INIT" --n-infill "$N_INFILL" --max-evals "$MAX_EVALS" --doe-from-cache "$SEED" \
         --cycle-core3d "$CYCLE3D" --kbol-transport "$KBOL" --confirm-transport "$CONFIRM" >> "$OUT.log" 2>&1 < /dev/null &
  fi
  DRIVER=$!
  echo "$(stamp)   driver pid $DRIVER"
}

stage_F() {
  done_ F && { echo "[F] already done"; return 0; }
  N_INIT=$(n_seed); MAX_EVALS=$(( N_INIT + ITERS * N_INFILL ))
  hr; echo " STAGE F. Campaign 12, $N_INIT seed + up to $ITERS x $N_INFILL = $MAX_EVALS evaluations, stop rule $HV_TOL / $HV_PATIENCE, watchdog $STALL_MIN min ($(stamp))"; hr
  if command -v free >/dev/null 2>&1; then bash machine_monitor.sh "$MACHINE_CSV" 60 & else sleep 1 & fi
  MONITOR=$!; trap 'kill $MONITOR 2>/dev/null' EXIT
  local launches=0
  while [ "$(n_eval)" -lt "$MAX_EVALS" ] && ! stopped; do
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
    echo "$(stamp)   driver ended, $(n_eval) evaluations in the checkpoint$(stopped && echo ', stop rule met')"
  done
  kill "$MONITOR" 2>/dev/null
  mark F
  echo "$(stamp) Campaign 12 finished ($(n_eval) evaluations). To send the results back:"
  echo "  git add -f $OUT/optimization_checkpoint.json $OUT.log $MACHINE_CSV $SEED $KAPPA $WORK/eval_cache.json && git commit -m 'Campaign 12: stop rule, mission margin, exploration, confirmation in the loop' && git push"
}

case "${1:-}" in
  --preflight) stage_P; exit 0 ;;
  --smoke)     stage_P; stage_S; stage_M; exit 0 ;;
  --status)    echo "$(stamp) markers: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"; echo "evaluations in $OUT: $(n_eval), newest file $(( $(newest_s) / 60 )) min ago$(stopped && echo ', stop rule met')"; grep "launch\|killing\|driver ended\|FAIL\|HV-STOP\|\[confirm\]\|exploration" "$OUT.log" c12.log 2>/dev/null | tail -n 6; [ -f "$MACHINE_CSV" ] && { echo "machine, last 3 minutes:"; tail -n 3 "$MACHINE_CSV"; }; exit 0 ;;
  "")          stage_P; stage_S; stage_M; stage_F ;;
  *)           die "usage: bash run_c12.sh [--preflight | --smoke | --status]" ;;
esac
