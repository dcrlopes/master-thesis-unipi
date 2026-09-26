#!/usr/bin/env bash
# run_c9_checks.sh -- two short checks queued after the closures queue.
#
# T  THREAD REPRODUCIBILITY (c9_repro_check.py). A rerun of a Campaign 9
#    depletion with the campaign seed matches the archive at beginning of
#    life to 1e-15 and then diverges from the first depletion step. The
#    hypothesis is multi-threaded tally accumulation. Two runs at 64 threads
#    and two at one thread, three transport solves each, on C9-47.
#    Expected if the hypothesis holds: the 64-thread pair differs from the
#    first depletion step on, the one-thread pair is bit-identical.
#
# L  PRECISION OF THE REDUCED-FIDELITY EIGHT-LAYER DEPLETION. Stage L of
#    run_c9_closures.sh measured its cost (0.42 h per design at 10 000 x 100
#    with 50 inactive batches) but not its precision. Four designs already
#    depleted at full fidelity, chosen to span the axial ratio, are depleted
#    again at the reduced fidelity: C9-24 (0.745), C9-72 (0.777), C9-27
#    (0.824), C9-69 (0.825). The comparison decides whether a further
#    campaign can measure the mission constraint in every evaluation.
#    Output folders are named lofi_* so that axial_ratio_model.py, which
#    reads c9*_dep_core3d*, never takes them as measurements.
#
# TIME (wks720, 64 threads)
#   T  two 64-thread runs of a few minutes, two one-thread runs of about
#      15 to 30 min each at 1000 x 30 / 10          about 1 h
#   L  four eight-layer depletions at about 0.42 h    about 1.7 h
#
# STAGES, each skipped when its marker exists
#   P preflight   W wait for a PID   T thread test   L reduced fidelity   F report
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9_checks.sh --preflight
#   setsid nohup bash run_c9_checks.sh 803596 > c9_checks.log 2>&1 < /dev/null &   # after the closures queue
#   bash run_c9_checks.sh --status
#   SKIP_L=1 bash run_c9_checks.sh ...      # the thread test only
set -u -o pipefail

THREADS=${THREADS:-64}
SKIP_L=${SKIP_L:-0}
MARK=.c9_checks_markers
REPORT=c9_checks_report.txt
LOFI=(--particles 10000 --batches 100 --inactive 50)

mkdir -p "$MARK"
die()   { echo "FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

case "${1:-}" in
--status)
  echo "time        : $(stamp)   host $(hostname)"
  p=$(pgrep -f "^bash run_c9_checks.sh( [0-9]+)?$" | head -1)
  [ -n "$p" ] && echo "runner      : ALIVE (pid $p)" || echo "runner      : not running"
  st=""; for s in P W T L F; do done_ $s && st="$st $s"; done
  echo "stages done :${st:- none}"
  echo "thread runs : $(ls repro_c9/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
  for d in lofi_dep_core3d_c9 lofi_dep_core3d_c9f; do
    [ -f "$d/runs.json" ] && echo "reduced     : $d $(python -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1])))))" "$d/runs.json")"
  done
  exit 0 ;;
esac

stage_P() {
  hr; echo " STAGE P. Preflight  ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] || die "OpenMC is not 0.15.3"
  for f in c9_repro_check.py c9_dep_replicas.py c9_dep_core3d.py kh_c9/k_histories.json \
           out_c9/optimization_checkpoint.json out_c9f/optimization_checkpoint.json \
           c9_dep_core3d_d24_L8/runs.json c9_dep_core3d_d27_L8/runs.json c9f_dep_core3d/runs.json; do
    [ -f "$f" ] || die "missing $f"
  done
  python c9_dep_core3d.py --selftest || die "c9_dep_core3d selftest"
  echo "  threads     : $THREADS of $(nproc)"
  echo "  other jobs  : $(pgrep -fc '[r]un_optimization.py|[c]9_dep_core3d|[c]onfirm3d|[m]tc_scan' || true) transport processes running"
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

stage_T() {
  done_ T && { echo "[T] already done"; return 0; }
  hr; echo " STAGE T. Thread reproducibility of the depletion, C9-47  ($(stamp))"; hr
  local tag
  for tag in t64a t64b; do
    [ -f "repro_c9/$tag.json" ] && { echo "  $tag already done"; continue; }
    python -u c9_repro_check.py --design 47 --threads "$THREADS" --tag "$tag" 2>&1 | tee -a repro_c9.log \
      || die "thread test $tag failed"
  done
  for tag in t1a t1b; do
    [ -f "repro_c9/$tag.json" ] && { echo "  $tag already done"; continue; }
    python -u c9_repro_check.py --design 47 --threads 1 --particles 1000 --tag "$tag" 2>&1 | tee -a repro_c9.log \
      || die "thread test $tag failed"
  done
  python c9_repro_check.py --compare 2>&1 | tee repro_c9/compare.txt
  mark T
}

stage_L() {
  [ "$SKIP_L" = "1" ] && { echo "[L] skipped (SKIP_L=1)"; return 0; }
  done_ L && { echo "[L] already done"; return 0; }
  hr; echo " STAGE L. Reduced-fidelity eight-layer depletion of four measured designs  ($(stamp))"; hr
  python -u c9_dep_core3d.py --checkpoint out_c9/optimization_checkpoint.json --designs 24 27 --layers 8 \
    "${LOFI[@]}" --threads "$THREADS" --out lofi_dep_core3d_c9 2>&1 | tee -a lofi_dep_core3d.log \
    || die "reduced fidelity, Campaign 9 designs"
  python -u c9_dep_core3d.py --checkpoint out_c9f/optimization_checkpoint.json --designs 72 69 --layers 8 \
    "${LOFI[@]}" --threads "$THREADS" --out lofi_dep_core3d_c9f 2>&1 | tee -a lofi_dep_core3d.log \
    || die "reduced fidelity, continuation designs"
  mark L
}

lofi_table() { python - <<'PY'
import json
from pathlib import Path
pairs = [("C9-24", "c9_dep_core3d_d24_L8/runs.json", "d24", "lofi_dep_core3d_c9/runs.json"),
         ("C9-27", "c9_dep_core3d_d27_L8/runs.json", "d27", "lofi_dep_core3d_c9/runs.json"),
         ("C9-72", "c9f_dep_core3d/runs.json", "d72", "lofi_dep_core3d_c9f/runs.json"),
         ("C9-69", "c9f_dep_core3d/runs.json", "d69", "lofi_dep_core3d_c9f/runs.json")]
print(f"{'design':7s} {'full [d]':>14s} {'reduced [d]':>14s} {'diff [d]':>9s} {'z':>6s} {'full h':>7s} {'reduced h':>9s}")
diffs = []
for name, full, key, lofi in pairs:
    try:
        a = json.loads(Path(full).read_text())[key]
        b = json.loads(Path(lofi).read_text())[key]
    except (FileNotFoundError, KeyError):
        print(f"{name:7s} not available"); continue
    d = b["efpd"] - a["efpd"]; s = (a["sigma_efpd"] ** 2 + b["sigma_efpd"] ** 2) ** 0.5
    diffs.append(d)
    print(f"{name:7s} {a['efpd']:7.1f} +/- {a['sigma_efpd']:4.1f} {b['efpd']:7.1f} +/- {b['sigma_efpd']:4.1f} {d:+9.1f} {d / s:+6.2f} "
          f"{a['wall_s'] / 3600:7.2f} {b['wall_s'] / 3600:9.2f}")
if diffs:
    import statistics as st
    print(f"\nmean difference {st.mean(diffs):+.1f} d, rms {st.mean([x * x for x in diffs]) ** 0.5:.1f} d over {len(diffs)} designs")
    print("the reduced fidelity is usable in the loop if the rms is well below the front margins (+12 to +69 d)")
PY
}

stage_F() {
  hr; echo " REPORT  ($(stamp))"; hr
  { echo "Campaign 9 checks, $(stamp)"
    echo
    echo "T. thread reproducibility of the depletion of C9-47 (pcm; 0 = bit-identical):"
    [ -f repro_c9/compare.txt ] && sed 's/^/  /' repro_c9/compare.txt || echo "  not run"
    echo
    echo "L. reduced-fidelity (10 000 x 100 / 50) against full-fidelity eight-layer depletion:"
    if [ "$SKIP_L" = "1" ]; then echo "  skipped"; else lofi_table | sed 's/^/  /'; fi
  } | tee "$REPORT"
  echo
  echo "push:"
  echo "  git add -f repro_c9/*.json repro_c9/compare.txt lofi_dep_core3d_c9/runs.json lofi_dep_core3d_c9f/runs.json"
  echo "  git add run_c9_checks.sh $REPORT"
  echo "  git add -f repro_c9.log lofi_dep_core3d.log c9_checks.log"
  echo "  git commit -m 'Campaign 9 checks: thread reproducibility of the depletion, precision of the reduced-fidelity eight-layer depletion'"
  mark F
}

case "${1:-}" in
  --preflight) stage_P ;;
  ""|[0-9]*)   stage_P; [ -n "${1:-}" ] && stage_W "$1"; stage_T; stage_L; stage_F ;;
  *)           die "unknown option $1" ;;
esac
