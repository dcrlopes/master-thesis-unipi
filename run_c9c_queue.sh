#!/usr/bin/env bash
# run_c9c_queue.sh -- the four suggestions of 7 October 2026 in one resumable
# queue on wks720, in order:
#
#   1  run_c9c_proxy2.sh run    seed repeats, 5 000 x 60 setting, bare single
#                               assembly, comparison                about 7 h
#   2  run_c10.sh               Campaign 10, the Campaign 9 problem with the
#                               3D cycle-length proxy in the loop  about 30 h
#
# Same mechanism as run_c9b_queue.sh: a marker file while the queue is wanted,
# a lock against a second copy, an @reboot crontab line that relaunches the
# queue after a reboot while the marker exists. Both stages skip what is
# already done (cached runs.json, markers of run_c10.sh, checkpoint resume).
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9c_queue.sh install      once: the @reboot line
#   bash run_c9c_queue.sh start        launch now, detached, log c9c_queue.log
#   bash run_c9c_queue.sh status
#   bash run_c9c_queue.sh stop         remove the marker (a running job goes on)
#   bash run_c9c_queue.sh uninstall    remove the @reboot line
set -u -o pipefail

cd "$(dirname "$(readlink -f "$0")")"
MARK=c9c_queue.active
LOG=c9c_queue.log
LOCK=/tmp/c9c_queue.lock
TAG="# run_c9c_queue"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
busy()  { pgrep -af "run_c9c_proxy2.sh|run_c10.sh|run_c9b_ctrl_mtc.sh|run_proxy_study.sh|c9_dep_core3d.py|c9_dep_asm3d.py|confirm3d.py|mtc_scan.py|run_optimization.py" | grep -v "run_c9c_queue.sh" || true; }

proxy2_done() { [ -f proxy_study2_report.txt ]; }
c10_done()    { [ -f .c10_markers/F ]; }

run_queue() {
  exec 9>"$LOCK"
  flock -n 9 || die "the queue is already running (lock $LOCK)"
  [ -f "$MARK" ] || die "no marker $MARK: run 'bash run_c9c_queue.sh start'"
  echo "$(stamp) queue started on $(hostname), env ${CONDA_DEFAULT_ENV:-none}, head $(git log --oneline -1)"
  b=$(busy); [ -z "$b" ] || die "another transport job is running: $b"
  if proxy2_done; then echo "$(stamp) 1. proxy study 2: already done"; else
    echo "$(stamp) 1. proxy study 2"
    bash run_c9c_proxy2.sh run || die "run_c9c_proxy2.sh failed, the queue stops here"
  fi
  if c10_done; then echo "$(stamp) 2. Campaign 10: already done"; else
    echo "$(stamp) 2. Campaign 10"
    bash run_c10.sh || die "run_c10.sh failed or was interrupted, the queue stops here (relaunch resumes it)"
  fi
  rm -f "$MARK"
  echo "$(stamp) queue finished. To send the results back:"
  echo "  git add -f lofi_dep_core3d_seed*/runs.json lofi5k_core3d/runs.json asm3d_bare_L8/runs.json proxy_study2_report.txt out_c10/optimization_checkpoint.json out_c10.log out_c10_smoke/optimization_checkpoint.json c9c_queue.log && git commit -m 'Proxy study 2 and Campaign 10' && git push"
}

case "${1:-}" in
install)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "activate openmc-env first, the paths are taken from it"
  CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  [ -f "$CONDA_SH" ] || die "conda.sh not found at $CONDA_SH"
  LINE="@reboot sleep 120 && /bin/bash -c 'cd $PWD && . $CONDA_SH && conda activate openmc-env && [ -f $MARK ] && setsid nohup bash run_c9c_queue.sh run >> $LOG 2>&1 < /dev/null' $TAG"
  { crontab -l 2>/dev/null | grep -v "$TAG"; echo "$LINE"; } | crontab - || die "crontab failed"
  echo "$(stamp) installed in the crontab:"; crontab -l | grep "$TAG" ;;
uninstall)
  crontab -l 2>/dev/null | grep -v "$TAG" | crontab -; echo "$(stamp) @reboot line removed" ;;
start)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  crontab -l 2>/dev/null | grep -q "$TAG" || echo "note: no @reboot line, run 'bash run_c9c_queue.sh install' for the restart after a reboot"
  b=$(busy); [ -z "$b" ] || die "another transport job is running: $b"
  touch "$MARK"
  setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
  echo "$(stamp) queue launched, log $LOG"; sleep 2; bash "$0" status ;;
run)
  run_queue ;;
stop)
  rm -f "$MARK"; echo "$(stamp) marker removed: a reboot starts nothing. A running job is not stopped." ;;
status)
  echo "$(stamp) host $(hostname), marker $([ -f "$MARK" ] && echo present || echo absent), @reboot $(crontab -l 2>/dev/null | grep -q "$TAG" && echo installed || echo not installed)"
  b=$(busy); [ -n "$b" ] && echo "running: $b" || echo "running: nothing"
  echo "1. proxy study 2: $(proxy2_done && echo done || echo not finished)"
  bash run_c9c_proxy2.sh status 2>/dev/null | sed 's/^/     /' | grep -v "host\|running"
  echo "2. Campaign 10: $(c10_done && echo done || echo not finished)"
  bash run_c10.sh --status 2>/dev/null | sed 's/^/     /'
  [ -f "$LOG" ] && { echo "last lines of $LOG:"; tail -n 5 "$LOG" | sed 's/^/     /'; } ;;
*)
  die "usage: bash run_c9c_queue.sh install | start | status | stop | uninstall" ;;
esac
