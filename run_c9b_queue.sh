#!/usr/bin/env bash
# run_c9b_queue.sh -- the two studies still to be executed on wks720, in order,
# resumable after a stop or a reboot:
#
#   1  run_c9b_ctrl_mtc.sh all     rodded 3D solves and MTC boron limit of
#                                  C9-63, C9-68, C9-94             about 3 h
#   2  run_proxy_study.sh run      proxies of the eight-layer depletion
#                                                                  6 to 10 h
#
# Both scripts skip what is already cached (finished solves in runs.json,
# report.txt of each MTC scan, runs.json of each proxy stage), so the queue
# is always started with the same command and continues from the first
# unfinished item. A stage that was interrupted inside one design repeats
# that design only.
#
# After a reboot the queue restarts by itself when "install" has been run
# once: it adds an @reboot line to the user's crontab that launches this
# script. The queue only runs while the marker file c9b_queue.active exists,
# which "start" creates and the end of the queue removes, so a reboot after
# the queue has finished starts nothing.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9b_queue.sh install      once: the @reboot line (paths of this env)
#   bash run_c9b_queue.sh start        launch now, detached, log c9b_queue.log
#   bash run_c9b_queue.sh status       what is done, what is running
#   bash run_c9b_queue.sh stop         remove the marker, so a reboot starts nothing
#   bash run_c9b_queue.sh uninstall    remove the @reboot line
set -u -o pipefail

cd "$(dirname "$(readlink -f "$0")")"
MARK=c9b_queue.active
LOG=c9b_queue.log
LOCK=/tmp/c9b_queue.lock
TAG="# run_c9b_queue"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
busy()  { pgrep -af "run_c9b_ctrl_mtc.sh|run_proxy_study.sh|run_c9b_layers.sh|c9_dep_core3d.py|c9_dep_asm3d.py|confirm3d.py|mtc_scan.py|run_optimization.py" | grep -v "run_c9b_queue.sh" || true; }

ctrl_done() {
  # the report of run_c9b_ctrl_mtc.sh has no "not finished" line once every solve exists
  [ -f c9b_ctrl_mtc_report.txt ] && ! grep -q "not finished" c9b_ctrl_mtc_report.txt
}
proxy_done() { [ -f proxy_study_report.txt ]; }

run_queue() {
  exec 9>"$LOCK"
  flock -n 9 || die "the queue is already running (lock $LOCK)"
  [ -f "$MARK" ] || die "no marker $MARK: run 'bash run_c9b_queue.sh start'"
  echo "$(stamp) queue started on $(hostname), env ${CONDA_DEFAULT_ENV:-none}"
  b=$(busy); [ -z "$b" ] || die "another transport job is running: $b"
  if ctrl_done; then echo "$(stamp) 1. control rods and MTC: already done"; else
    echo "$(stamp) 1. control rods and MTC"
    bash run_c9b_ctrl_mtc.sh all || die "run_c9b_ctrl_mtc.sh failed, the queue stops here"
    ctrl_done || die "run_c9b_ctrl_mtc.sh ended with unfinished solves, see c9b_ctrl_mtc_report.txt"
  fi
  if proxy_done; then echo "$(stamp) 2. proxy study: already done"; else
    echo "$(stamp) 2. proxy study"
    bash run_proxy_study.sh run || die "run_proxy_study.sh failed, the queue stops here"
  fi
  rm -f "$MARK"
  echo "$(stamp) queue finished. To send the results back:"
  echo "  git add -f confirm3d_c9a_rest/runs.json confirm3d_c9a_rest/summary.json mtc_c9b_d*_p128_core3d/report.txt mtc_c9b_d*_p128_core3d/summary.json c9b_ctrl_mtc_report.txt lofi_dep_core3d_all/runs.json asm3d_*/runs.json asm3d_L8/estimate_d47.json proxy_study_report.txt c9b_queue.log && git commit -m 'C9-63, 68, 94 rods and MTC; proxy study' && git push"
}

case "${1:-}" in
install)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "activate openmc-env first, the paths are taken from it"
  CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  [ -f "$CONDA_SH" ] || die "conda.sh not found at $CONDA_SH"
  # cron runs /bin/sh, so the conda activation is done inside an explicit bash
  LINE="@reboot sleep 120 && /bin/bash -c 'cd $PWD && . $CONDA_SH && conda activate openmc-env && [ -f $MARK ] && setsid nohup bash run_c9b_queue.sh run >> $LOG 2>&1 < /dev/null' $TAG"
  { crontab -l 2>/dev/null | grep -v "$TAG"; echo "$LINE"; } | crontab - || die "crontab failed"
  echo "$(stamp) installed in the crontab:"; crontab -l | grep "$TAG" ;;
uninstall)
  crontab -l 2>/dev/null | grep -v "$TAG" | crontab -; echo "$(stamp) @reboot line removed" ;;
start)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  crontab -l 2>/dev/null | grep -q "$TAG" || echo "note: no @reboot line, run 'bash run_c9b_queue.sh install' for the restart after a reboot"
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
  echo "1. control rods and MTC: $(ctrl_done && echo done || echo not finished)"
  [ -f c9b_ctrl_mtc_report.txt ] && sed 's/^/     /' c9b_ctrl_mtc_report.txt
  echo "2. proxy study: $(proxy_done && echo done || echo not finished)"
  [ -f "$LOG" ] && { echo "last lines of $LOG:"; tail -n 5 "$LOG" | sed 's/^/     /'; } ;;
*)
  die "usage: bash run_c9b_queue.sh install | start | status | stop | uninstall" ;;
esac
