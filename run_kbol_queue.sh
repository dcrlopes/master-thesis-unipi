#!/usr/bin/env bash
# run_kbol_queue.sh -- high-statistics beginning-of-life assembly k_inf of the
# front and reference designs (c9_repro_check.py, 20 000 x 160, 60 inactive),
# for the end-of-cycle criterion of the eight-layer depletions,
# k_3D(BOL) x k_target / k_bol. The campaign k_bol at 4000 x 60 carries about
# 194 pcm, which moved the cycle length of C9-27 by -99 d (21 Oct 2026).
#
#   block A  the remaining front designs: C9-69, C9-70, C10-42, C10-56, C10-48
#   block B  the other 14 reference designs of the proxy study (paper_finds.md
#            Section 1), infeasible ones included, which also tests whether the
#            front designs had an upward k_bol fluctuation (selection effect)
#
# About 32 min per design, 19 designs, about 10 h. A design is done when
# repro_c9/<tag>.json exists, which c9_repro_check.py writes at its end.
#
# PROTECTIONS, as in run_c10_confirm.sh: watchdog of STALL_MIN minutes without
# a file written, crash relaunch, at most MAX_LAUNCHES per design, @reboot line
# active while the marker exists, the work directory of an interrupted design
# moved aside, the lock not inherited by the children.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_kbol_queue.sh install     the @reboot line (once)
#   bash run_kbol_queue.sh start       launch in the background, log kbol_queue.log
#   bash run_kbol_queue.sh status
#   bash run_kbol_queue.sh stop
#   bash run_kbol_queue.sh uninstall
set -u -o pipefail
cd "$(dirname "$(readlink -f "$0")")"

C9=out_c9a/optimization_checkpoint.json
C10=out_c10/optimization_checkpoint.json
JOBS="$C9:69 $C9:70 $C10:42 $C10:56 $C10:48
      $C9:1 $C9:11 $C9:16 $C9:24 $C9:32 $C9:35 $C9:47 $C9:63 $C9:68 $C9:72 $C9:79 $C9:85 $C9:88 $C9:94"
# JOBS_FILE=<file>: one "checkpoint:index" per line replaces the list above (run_c11.sh stage K)
[ -n "${JOBS_FILE:-}" ] && JOBS=$(grep -v '^\s*$' "$JOBS_FILE" | tr '\n' ' ')
TRANSPORT="--particles 20000 --batches 160 --inactive 60"
THREADS=${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
STALL_MIN=${STALL_MIN:-30}       # a solve at 20 000 x 160 is about 8 min and OpenMC writes a line per batch
MAX_LAUNCHES=${MAX_LAUNCHES:-3}
POLL_S=60
OUT=repro_c9
LOG=kbol_queue.log
MARK=kbol_queue.active
LOCK=/tmp/kbol_queue.lock
TAG="# run_kbol_queue"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }

tag_of() {                            # out_c9a/...:69 -> kbol_c9_69_hifi, out_c10/...:42 -> kbol_c10_42_hifi
  local ck=${1%%:*} idx=${1##*:} camp
  case "$ck" in *out_c10/*) camp=c10 ;; *) camp=c9 ;; esac
  echo "kbol_${camp}_${idx}_hifi"
}

left() {                              # jobs whose result file does not exist yet
  local j
  for j in $JOBS; do [ -f "$OUT/$(tag_of "$j").json" ] || echo "$j"; done
}

newest_s() {
  local f
  f=$(find "$OUT/work" "$LOG" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  [ -n "$f" ] && echo $(( $(date +%s) - ${f%.*} )) || echo 0
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
  flock -n 9 || die "the queue is already running (lock $LOCK)"
  [ -f "$MARK" ] || die "no marker $MARK: run 'bash run_kbol_queue.sh start'"
  echo "$(stamp) queue started on $(hostname), env ${CONDA_DEFAULT_ENV:-none}, head $(git log --oneline -1)"
  [ "$(hostname)" = "wks720" ] || [ -n "${ALLOW_HOST:-}" ] || die "not on wks720 (set ALLOW_HOST=1 to run elsewhere)"
  [ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
  [ -f "$C9" ] && [ -f "$C10" ] || die "missing $C9 or $C10"
  local j tag n launches pid s
  n=$(left | wc -l); echo "$(stamp) $n of $(echo $JOBS | wc -w) designs left"
  for j in $(left); do
    tag=$(tag_of "$j"); launches=0
    while [ ! -f "$OUT/$tag.json" ]; do
      launches=$((launches + 1))
      if [ "$launches" -gt "$MAX_LAUNCHES" ]; then
        echo "$(stamp) WARNING $tag failed $MAX_LAUNCHES times, skipped"; break
      fi
      if [ -d "$OUT/work/$tag" ]; then
        mv "$OUT/work/$tag" "$OUT/work/${tag}_stale_$(date +%Y%m%d_%H%M%S)"
        echo "$(stamp)   unfinished $OUT/work/$tag moved aside"
      fi
      echo "$(stamp) launch $launches of $tag (${j%%:*}, design ${j##*:}), watchdog $STALL_MIN min"
      python -u c9_repro_check.py --checkpoint "${j%%:*}" --design "${j##*:}" --threads "$THREADS" \
          $TRANSPORT --tag "$tag" 9>&- &
      pid=$!
      while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_S" 9>&-
        s=$(newest_s)
        if [ "$s" -gt $((STALL_MIN * 60)) ]; then
          echo "$(stamp)   no file written for $((s / 60)) min: killing and relaunching"
          kill_tree "$pid"
          break
        fi
      done
      wait "$pid" 2>/dev/null
      echo "$(stamp)   $tag ended (exit $?), result $([ -f "$OUT/$tag.json" ] && echo written || echo missing)"
    done
  done
  rm -f "$MARK"; uninstall_line
  echo "$(stamp) queue finished, marker and @reboot line removed, $(left | wc -l) designs without a result. To send the results back:"
  echo "  git add -f $OUT/kbol_*.json kbol_test.log $LOG && git commit -m 'High-statistics k_bol of the front and reference designs' && git push"
}

status() {
  local j tag
  echo "$(stamp) host $(hostname), marker $([ -f "$MARK" ] && echo present || echo absent), @reboot $(crontab -l 2>/dev/null | grep -q "$TAG" && echo installed || echo absent), newest file $(( $(newest_s) / 60 )) min ago"
  for j in $JOBS; do
    tag=$(tag_of "$j")
    if [ -f "$OUT/$tag.json" ]; then
      python -c "import json; r=json.load(open('$OUT/$tag.json')); print(f\"  {'$tag':20s} k_bol {float(r['k'][0]):.5f}  {r['wall_s']/60:5.1f} min\")"
    else
      echo "  $tag  not done"
    fi
  done
  pgrep -u "$USER" -af "c9_repro_check.py" | grep -v pgrep || echo "  running: nothing"
  grep "launch\|killing\|ended\|WARNING\|FAIL\|finished" "$LOG" 2>/dev/null | tail -n 3
}

case "${1:-}" in
install)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "activate openmc-env first, the paths are taken from it"
  CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  [ -f "$CONDA_SH" ] || die "conda.sh not found at $CONDA_SH"
  [ -n "${OPENMC_CHAIN_FILE:-}" ]     || die "OPENMC_CHAIN_FILE unset in this shell"
  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] || die "OPENMC_CROSS_SECTIONS unset in this shell"
  ENV="export OPENMC_CHAIN_FILE=$OPENMC_CHAIN_FILE OPENMC_CROSS_SECTIONS=$OPENMC_CROSS_SECTIONS OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}"
  LINE="@reboot sleep 120 && /bin/bash -c 'cd $PWD && . $CONDA_SH && conda activate openmc-env && $ENV && [ -f $MARK ] && setsid nohup bash run_kbol_queue.sh run >> $LOG 2>&1 < /dev/null' $TAG"
  { crontab -l 2>/dev/null | grep -v "$TAG"; echo "$LINE"; } | crontab - || die "crontab failed"
  echo "$(stamp) installed in the crontab:"; crontab -l | grep "$TAG" ;;
uninstall)
  uninstall_line; echo "$(stamp) @reboot line removed" ;;
start)
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  crontab -l 2>/dev/null | grep -q "$TAG" || echo "note: no @reboot line, run 'bash run_kbol_queue.sh install' for the restart after a reboot"
  b=$(pgrep -u "$USER" -af "c9_repro_check.py|c9_dep_core3d.py|run_optimization.py|run_kbol_queue.sh run|run_c10_confirm.sh run" | grep -v pgrep || true)
  [ -z "$b" ] || die "another transport job is running: $b"
  touch "$MARK"
  setsid nohup bash "$0" run >> "$LOG" 2>&1 < /dev/null &
  echo "$(stamp) queue launched, log $LOG"; sleep 3; status ;;
stop)
  rm -f "$MARK"
  for p in $(pgrep -u "$USER" -f "run_kbol_queue.sh run"); do kill_tree "$p"; done
  pkill -u "$USER" -f "c9_repro_check.py" 2>/dev/null
  echo "$(stamp) stopped, marker removed (a reboot starts nothing)" ;;
status|--status)
  status ;;
run)
  run_all ;;
*)
  die "usage: bash run_kbol_queue.sh install | start | status | stop | uninstall" ;;
esac
