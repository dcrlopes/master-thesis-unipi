#!/usr/bin/env bash
# run_c10_confirm.sh -- full-fidelity eight-layer depletion of the Campaign 10
# front below the MTC boron limit (C10-42, 56, 31, 48), two seeds each, at
# the settings of the Campaign 9 reference (20 000 particles x 160 batches,
# 60 inactive). The campaign measured these designs once at 10 000 x 100 and
# three of the four have margins to 1826 d inside the 26 d seed SD of that
# proxy, so their feasibility is unresolved. Same evidence level as the
# Campaign 9 front (c9a_dep_core3d, c9a_dep_core3d_seed2).
#
# About 56 min per depletion on 64 threads, 8 depletions, about 7.5 h.
# The chain solver runs serial (dep_common.serial_depletion_pool).
#
# PROTECTIONS (18 Oct 2026, user's request)
#   watchdog  the depletion process is killed with its descendants and
#             relaunched when nothing is written under the output directory
#             or the log for STALL_MIN minutes (OpenMC writes a line per
#             batch, so a silence of minutes is already a stall); a crash is
#             relaunched the same way; at most MAX_LAUNCHES per seed
#   reboot    "install" writes an @reboot line that relaunches the job while
#             the marker c10_confirm.active exists; the job removes the
#             marker and the line when it finishes
#   resume    finished designs are cached in runs.json and skipped; the
#             directory of an interrupted design is moved aside first
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c10_confirm.sh install     the @reboot line (once)
#   bash run_c10_confirm.sh start       launch in the background, log c10_confirm.log
#   bash run_c10_confirm.sh status
#   bash run_c10_confirm.sh stop        stop the job and remove the marker
#   bash run_c10_confirm.sh uninstall   remove the @reboot line
set -u -o pipefail
cd "$(dirname "$(readlink -f "$0")")"

THREADS=${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
CKPT=${CKPT:-out_c10/optimization_checkpoint.json}
DESIGNS=${DESIGNS:-"42 56 31 48"}
PAIRS="core3d:c10_dep_core3d_seed1 core3d-seed2:c10_dep_core3d_seed2"   # seed salt:output directory
STALL_MIN=${STALL_MIN:-45}       # a full-fidelity solve is about 6 min, the serial chain solve a few
MAX_LAUNCHES=${MAX_LAUNCHES:-10}
POLL_S=60
LOG=c10_confirm.log
MARK=c10_confirm.active
LOCK=/tmp/c10_confirm.lock
TAG="# run_c10_confirm"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }

missing() {                           # designs of output directory $1 not yet in its runs.json
  python - "$1" $DESIGNS <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "runs.json")
done = json.load(open(p)) if os.path.exists(p) else {}
print(" ".join(d for d in sys.argv[2:] if f"d{d}" not in done))
PY
}

newest_s() {                          # seconds since the newest file under $1 or the log changed
  local f
  f=$(find "$1" "$LOG" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  [ -n "$f" ] && echo $(( $(date +%s) - ${f%.*} )) || echo 0
}

move_stale() {                        # an interrupted design restarts from fresh fuel in a clean directory
  local out=$1 d
  for d in $(missing "$out"); do
    if [ -d "$out/work/d$d" ]; then
      mv "$out/work/d$d" "$out/work/d${d}_stale_$(date +%Y%m%d_%H%M%S)"
      echo "$(stamp)   unfinished $out/work/d$d moved aside"
    fi
  done
}

descendants() {
  local pid=$1 child
  echo "$pid"
  for child in $(pgrep -P "$pid" 2>/dev/null); do descendants "$child"; done
}

kill_tree() {
  local pids
  pids=$(descendants "$1" | tr '\n' ' ')
  echo "$(stamp)   killing $pids"
  kill -TERM $pids 2>/dev/null; sleep 10; kill -KILL $pids 2>/dev/null
  wait "$1" 2>/dev/null
}

uninstall_line() { crontab -l 2>/dev/null | grep -v "$TAG" | crontab -; }

run_all() {
  exec 9>"$LOCK"
  flock -n 9 || die "the job is already running (lock $LOCK)"
  [ -f "$MARK" ] || die "no marker $MARK: run 'bash run_c10_confirm.sh start'"
  echo "$(stamp) job started on $(hostname), env ${CONDA_DEFAULT_ENV:-none}, head $(git log --oneline -1)"
  [ "$(hostname)" = "wks720" ] || [ -n "${ALLOW_HOST:-}" ] || die "not on wks720 (set ALLOW_HOST=1 to run elsewhere)"
  [ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
  [ -f "$CKPT" ] || die "no $CKPT"
  python c9_dep_core3d.py --selftest 9>&- || die "c9_dep_core3d selftest"
  local pair salt out launches pid s
  for pair in $PAIRS; do
    salt=${pair%%:*}; out=${pair##*:}; launches=0
    while [ -n "$(missing "$out")" ]; do
      launches=$((launches + 1))
      [ "$launches" -le "$MAX_LAUNCHES" ] || die "$MAX_LAUNCHES launches of $out without finishing, see $LOG"
      move_stale "$out"
      echo "$(stamp) launch $launches of $out (salt $salt), designs left: $(missing "$out"), watchdog $STALL_MIN min"
      # 9>&-: the children must not inherit the lock (17 Oct 2026, an orphan kept it)
      python -u c9_dep_core3d.py --checkpoint "$CKPT" --designs $(missing "$out") --layers 8 \
          --threads "$THREADS" --salt "$salt" --out "$out" 9>&- &
      pid=$!
      while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_S" 9>&-
        s=$(newest_s "$out")
        if [ "$s" -gt $((STALL_MIN * 60)) ]; then
          echo "$(stamp)   no file written for $((s / 60)) min: killing the depletion and relaunching"
          kill_tree "$pid"
          break
        fi
      done
      wait "$pid" 2>/dev/null
      echo "$(stamp)   depletion process ended (exit $?), designs left in $out: $(missing "$out")"
    done
    python c9_dep_core3d.py --checkpoint "$CKPT" --out "$out" --analyse 9>&- || echo "$(stamp) WARNING aggregation of $out failed"
  done
  rm -f "$MARK"; uninstall_line
  echo "$(stamp) job finished, marker and @reboot line removed. To send the results back:"
  echo "  git add -f c10_dep_core3d_seed1/runs.json c10_dep_core3d_seed1/summary.txt c10_dep_core3d_seed2/runs.json c10_dep_core3d_seed2/summary.txt $LOG && git commit -m 'Campaign 10 front: full-fidelity eight-layer depletions, two seeds' && git push"
}

status() {
  local pair out
  echo "$(stamp) host $(hostname), marker $([ -f "$MARK" ] && echo present || echo absent), @reboot $(crontab -l 2>/dev/null | grep -q "$TAG" && echo installed || echo absent)"
  for pair in $PAIRS; do
    out=${pair##*:}
    echo "  $out: $( [ -f "$out/runs.json" ] && python -c "import json; d=json.load(open('$out/runs.json')); print(' '.join(f\"{k} {v['efpd']:.0f} d\" for k, v in d.items()))" || echo "nothing yet"), left: $(missing "$out"), newest file $(( $(newest_s "$out") / 60 )) min ago"
  done
  pgrep -u "$USER" -af "c9_dep_core3d.py --checkpoint $CKPT" | grep -v pgrep || echo "  running: nothing"
  grep "launch\|killing\|ended\|FAIL\|finished" "$LOG" 2>/dev/null | tail -n 4
}

case "${1:-}" in
install)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "activate openmc-env first, the paths are taken from it"
  CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  [ -f "$CONDA_SH" ] || die "conda.sh not found at $CONDA_SH"
  # cron starts without the shell profile, so the OpenMC data paths of this shell go into the line
  [ -n "${OPENMC_CHAIN_FILE:-}" ]     || die "OPENMC_CHAIN_FILE unset in this shell"
  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] || die "OPENMC_CROSS_SECTIONS unset in this shell"
  ENV="export OPENMC_CHAIN_FILE=$OPENMC_CHAIN_FILE OPENMC_CROSS_SECTIONS=$OPENMC_CROSS_SECTIONS OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}"
  LINE="@reboot sleep 120 && /bin/bash -c 'cd $PWD && . $CONDA_SH && conda activate openmc-env && $ENV && [ -f $MARK ] && setsid nohup bash run_c10_confirm.sh run >> $LOG 2>&1 < /dev/null' $TAG"
  { crontab -l 2>/dev/null | grep -v "$TAG"; echo "$LINE"; } | crontab - || die "crontab failed"
  echo "$(stamp) installed in the crontab:"; crontab -l | grep "$TAG" ;;
uninstall)
  uninstall_line; echo "$(stamp) @reboot line removed" ;;
start)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  crontab -l 2>/dev/null | grep -q "$TAG" || echo "note: no @reboot line, run 'bash run_c10_confirm.sh install' for the restart after a reboot"
  b=$(pgrep -u "$USER" -af "c9_dep_core3d.py|run_optimization.py|run_c10_confirm.sh run" | grep -v pgrep || true)
  [ -z "$b" ] || die "another transport job is running: $b"
  touch "$MARK"
  setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
  echo "$(stamp) job launched, log $LOG"; sleep 3; status ;;
stop)
  rm -f "$MARK"
  for p in $(pgrep -u "$USER" -f "run_c10_confirm.sh run"); do kill_tree "$p"; done
  pkill -u "$USER" -f "c9_dep_core3d.py --checkpoint $CKPT" 2>/dev/null
  echo "$(stamp) stopped, marker removed (a reboot starts nothing)" ;;
status|--status)
  status ;;
run)
  run_all ;;
*)
  die "usage: bash run_c10_confirm.sh install | start | status | stop | uninstall" ;;
esac
