#!/usr/bin/env bash
# run_c9_closures.sh -- bring the new front members of Campaign 9 (designs 69
# and 70 of out_c9a) to the evidence level of C9-27, measure the two numbers
# that decide the next campaign, and put the valgrid run back at the end.
#
# ORDER (the user's, 25 Sep 2026)
#   K  stop the valgrid search (its enumeration is complete and kept; the
#      search restarts cleanly at the end, its partial output is set aside)
#   L  reduced-fidelity eight-layer estimate, 10 000 x 100, on design 69   ~10 min
#   R  second seed of the eight-layer depletion of 70 (+12 d) and 88 (-11 d) ~2 h
#   C  3D peaking and rodded confirmation of 69 and 70, ARO ARI RE12,
#      1000 ppm, two seeds, the settings of confirm3d_c9                  ~1.2 h
#   M  MTC boron limit of the lattices of 69 and 70, both pressures,
#      the settings of mtc_front_c9.sh                                    ~2 h
#   G  the designed gadolinia study, run_gdstudy.sh                       ~6.5 h
#   V  the valgrid run resumed with EQUAL=1                               ~11 h
#   F  report (written before V, so it exists while the valgrid run works)
#
# STAGES skip on their markers, so the script can be relaunched after an
# interruption. Every transport job runs one at a time.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9_closures.sh --preflight        # also lists what stage K would stop
#   setsid nohup bash run_c9_closures.sh > c9_closures.log 2>&1 < /dev/null &
#   bash run_c9_closures.sh --status
#   SKIP_VALGRID=1 bash run_c9_closures.sh     # end after the report
set -u -o pipefail

THREADS=${THREADS:-64}
CKPT=${CKPT:-out_c9a/optimization_checkpoint.json}
NEW=${NEW:-"69 70"}                 # the new front members
REPL=${REPL:-"70 88"}               # the designs whose margins sit inside the noise
SALT=${SALT:-core3d-seed2}
SKIP_VALGRID=${SKIP_VALGRID:-0}
MARK=.c9_closures_markers
REPORT=c9_closures_report.txt

mkdir -p "$MARK"
die()   { echo "FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
envok() { python -c "import numpy, openmc; print('env ok', openmc.__version__)"; }

valgrid_pids() { pgrep -f "near_miss_then_valgrid.sh|run_valgrid_c9.sh|out_valgrid_c9_opt" | grep -vw "$$" || true; }

case "${1:-}" in
--status)
  echo "time        : $(stamp)   host $(hostname)"
  p=$(pgrep -f "^bash run_c9_closures.sh$" | grep -vw "$$" | head -1)
  [ -n "$p" ] && echo "runner      : ALIVE (pid $p)" || echo "runner      : not running"
  st=""; for s in P K L R C M G F V; do done_ $s && st="$st $s"; done
  echo "stages done :${st:- none}"
  [ -f c9_dep_estimate_lofi/estimate_d69.json ] && echo "lofi        : $(grep -h PROJECTED_HOURS c9_closures.log 2>/dev/null | tail -1)"
  [ -f c9a_dep_core3d_seed2/runs.json ] && echo "replicas    : $(python -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1])))))" c9a_dep_core3d_seed2/runs.json)"
  [ -f confirm3d_c9a/summary.json ] && echo "confirm3d   : $(python -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1])))))" confirm3d_c9a/summary.json)"
  for d in mtc_c9a_d*_core3d; do [ -f "$d/report.txt" ] && echo "mtc         : $d  $(grep CROSSING "$d/report.txt")"; done
  [ -f .gdstudy_markers/F ] && echo "gd study    : done" || { [ -d .gdstudy_markers ] && echo "gd study    : stages $(ls .gdstudy_markers | tr '\n' ' ')"; }
  exit 0 ;;
esac

stage_P() {
  hr; echo " STAGE P. Preflight  ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  envok || die "environment"
  for f in c9_dep_core3d.py confirm3d.py mtc_scan.py run_gdstudy.sh run_valgrid_c9.sh axial_ratio_model.py "$CKPT"; do
    [ -f "$f" ] || die "missing $f"
  done
  grep -q -- "--salt" c9_dep_core3d.py || die "c9_dep_core3d.py lacks the --salt flag"
  grep -q "design_map=zn.evaluator_design_map(design)" mtc_scan.py || die "mtc_scan.py is not the zoned version"
  python c9_dep_core3d.py --selftest || die "c9_dep_core3d selftest"
  python confirm3d.py --selftest > /dev/null || die "confirm3d selftest"
  python mtc_scan.py --selftest > /dev/null || die "mtc_scan selftest"
  [ -f .valgrid_c9_markers/G ] || echo "  NOTE: the valgrid enumeration is not marked done; stage K keeps its checkpoint either way"
  echo "  designs     : new front members $NEW, replicas $REPL, from $CKPT"
  echo "  threads     : $THREADS of $(nproc)"
  echo "  stage K would stop these processes:"
  for p in $(valgrid_pids); do ps -o pid,etime,args -p "$p" | tail -1 | cut -c1-110 | sed 's/^/    /'; done
  echo "preflight OK"
  mark P
}

stage_K() {
  done_ K && { echo "[K] already done"; return 0; }
  hr; echo " STAGE K. Stop the valgrid search and set its partial output aside  ($(stamp))"; hr
  local pids; pids=$(valgrid_pids)
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null; sleep 20
    pids=$(valgrid_pids); [ -n "$pids" ] && { kill -9 $pids 2>/dev/null; sleep 5; }
  fi
  [ -z "$(valgrid_pids)" ] || die "valgrid processes survived: $(valgrid_pids)"
  local ts; ts=$(date '+%Y%m%d_%H%M')
  [ -d out_valgrid_c9_opt ] && mv out_valgrid_c9_opt "out_valgrid_c9_opt.stopped_$ts" && echo "  out_valgrid_c9_opt -> out_valgrid_c9_opt.stopped_$ts"
  [ -d openmc_runs_valgrid_c9_opt ] && mv openmc_runs_valgrid_c9_opt "openmc_runs_valgrid_c9_opt.stopped_$ts"
  rm -f .valgrid_c9_markers/S .valgrid_c9_markers/W .valgrid_c9_markers/E
  echo "  enumeration kept: $(python -c "import json; print(len(json.load(open('out_valgrid_c9_grid/optimization_checkpoint.json'))['all_raw']))" 2>/dev/null || echo '?') of 35 nodes"
  mark K
}

stage_L() {
  done_ L && { echo "[L] already done"; return 0; }
  hr; echo " STAGE L. Reduced-fidelity eight-layer estimate on design 69  ($(stamp))"; hr
  python -u c9_dep_core3d.py --estimate --checkpoint "$CKPT" --designs 69 --layers 8 \
    --particles 10000 --batches 100 --inactive 50 --threads "$THREADS" \
    --out c9_dep_estimate_lofi 2>&1 | tee -a c9_dep_estimate_lofi.log || die "estimate failed"
  mark L
}

stage_R() {
  done_ R && { echo "[R] already done"; return 0; }
  hr; echo " STAGE R. Second seed of the eight-layer depletion, designs $REPL  ($(stamp))"; hr
  python -u c9_dep_core3d.py --checkpoint "$CKPT" --designs $REPL --layers 8 \
    --threads "$THREADS" --salt "$SALT" --out c9a_dep_core3d_seed2 \
    2>&1 | tee -a c9a_dep_core3d_seed2.log || die "replicas failed"
  python c9_dep_core3d.py --checkpoint "$CKPT" --out c9a_dep_core3d_seed2 --analyse >> c9a_dep_core3d_seed2.log 2>&1 || echo "  WARNING aggregation failed"
  mark R
}

stage_C() {
  done_ C && { echo "[C] already done"; return 0; }
  hr; echo " STAGE C. 3D peaking and rodded confirmation of $NEW  ($(stamp))"; hr
  python -u confirm3d.py --checkpoint "$CKPT" --designs $NEW --states ARO ARI RE12 \
    --boron-ppm 1000 --seeds 2 --threads "$THREADS" --out confirm3d_c9a \
    2>&1 | tee -a confirm3d_c9a.log || die "confirm3d failed"
  [ -f c9_peaking_2d3d.py ] && { python c9_peaking_2d3d.py --summary confirm3d_c9a/summary.json --checkpoint "$CKPT" \
    --out confirm3d_c9a 2>&1 | tail -20 || echo "  WARNING c9_peaking_2d3d failed (not needed for the summary)"; }
  mark C
}

stage_M() {
  done_ M && { echo "[M] already done"; return 0; }
  hr; echo " STAGE M. MTC boron limit of the lattices of $NEW, both pressures  ($(stamp))"; hr
  local IDX P TLO THI TAG
  for IDX in $NEW; do
    for P in 12.8 15.5; do
      if [ "$P" = "12.8" ]; then TLO=547; THI=567; else TLO=570; THI=590; fi
      TAG="mtc_c9a_d${IDX}_p${P/./}_core3d"
      [ -f "$TAG/report.txt" ] && { echo "  $TAG already done"; continue; }
      echo "=== $TAG  ($(stamp))"
      python -u mtc_scan.py --checkpoint "$CKPT" --idx "$IDX" \
        --pressure "$P" --t-lo "$TLO" --t-hi "$THI" --level core3d \
        --boron 0,1000,2000,3000 --seeds 2 --threads "$THREADS" \
        --out "$TAG" 2>&1 | tee "$TAG.log" || die "$TAG failed"
    done
  done
  mark M
}

stage_G() {
  done_ G && { echo "[G] already done"; return 0; }
  hr; echo " STAGE G. The designed gadolinia study  ($(stamp))"; hr
  bash run_gdstudy.sh 2>&1 | tee -a gdstudy.log || die "gadolinia study failed"
  mark G
}

stage_F() {
  hr; echo " REPORT  ($(stamp))"; hr
  { echo "Campaign 9 closures on the new front members, $(stamp)"
    echo
    echo "L. reduced-fidelity estimate (10 000 x 100 x 50) on design 69:"
    grep -h "ESTIMATE\|PROJECTED_HOURS" c9_dep_estimate_lofi.log 2>/dev/null | tail -2 | sed 's/^/  /'
    echo
    echo "R. eight-layer cycle, first seed against second seed:"
    python - "$CKPT" <<'PY'
import json, sys, glob
raw = json.load(open(sys.argv[1]))["all_raw"]
first = {}
for p in ("c9f_dep_core3d/runs.json", "c9a_dep_core3d_nm/runs.json", "c9a_dep_core3d/runs.json"):
    try:
        for r in json.load(open(p)).values(): first.setdefault(int(r["idx"]), r)
    except FileNotFoundError: pass
try: second = {int(r["idx"]): r for r in json.load(open("c9a_dep_core3d_seed2/runs.json")).values()}
except FileNotFoundError: second = {}
for i in sorted(second):
    a, b = first.get(i), second[i]
    if a:
        d = b["efpd"] - a["efpd"]; s = (a["sigma_efpd"] ** 2 + b["sigma_efpd"] ** 2) ** 0.5
        print(f"  design {i}: seed 1 {a['efpd']:.0f} +/- {a['sigma_efpd']:.0f} d, seed 2 {b['efpd']:.0f} +/- {b['sigma_efpd']:.0f} d, "
              f"difference {d:+.0f} d ({d / s:+.1f} sigma of the brackets), mean {(a['efpd'] + b['efpd']) / 2:.0f} d, margin {(a['efpd'] + b['efpd']) / 2 - 1826:+.0f} d")
    else:
        print(f"  design {i}: seed 2 {b['efpd']:.0f} d, no first seed found")
PY
    echo
    echo "C. 3D confirmation of $NEW at 1000 ppm (ARO peaking, axial factor, rodded margins in pcm):"
    python - <<'PY'
import json
try: S = json.load(open("confirm3d_c9a/summary.json"))
except FileNotFoundError: S = {}
for i, d in sorted(S.items(), key=lambda kv: int(kv[0])):
    line = f"  design {i}:"
    for k in sorted(d):
        v = d[k]
        if isinstance(v, (int, float)): line += f"  {k} {v:.4g}"
        elif isinstance(v, dict) and any(s in k for s in ("2D", "3D")):
            line += "  " + k + " {" + ", ".join(f"{kk} {vv:.4g}" for kk, vv in v.items() if isinstance(vv, (int, float))) + "}"
    print(line)
PY
    echo
    echo "M. MTC boron limit of the lattices of $NEW:"
    grep -H "CROSSING\|needs" mtc_c9a_d*_core3d/report.txt 2>/dev/null | sed 's/^/  /'
    echo
    echo "G. gadolinia study:"
    [ -f gdstudy_report.txt ] && sed 's/^/  /' gdstudy_report.txt || echo "  no report"
    echo
    echo "leave-one-out table with every measurement to date:"
    python axial_ratio_model.py --report
  } | tee "$REPORT"
  echo
  echo "push:"
  echo "  git add -f c9a_dep_core3d_seed2/runs.json c9a_dep_core3d_seed2/summary.json confirm3d_c9a/summary.json confirm3d_c9a/runs.json"
  echo "  git add -f mtc_c9a_d*_core3d/summary.json mtc_c9a_d*_core3d/report.txt c9_dep_estimate_lofi/estimate_d69.json"
  echo "  git add -f out_gdstudy/optimization_checkpoint.json gdstudy_dep_core3d/runs.json gdstudy_dep_core3d/summary.json gdstudy/list.json"
  echo "  git add run_c9_closures.sh run_gdstudy.sh c9_dep_core3d.py axial_ratio_model.py $REPORT gdstudy_report.txt"
  echo "  git add -f c9_closures.log gdstudy.log c9a_dep_core3d_seed2.log confirm3d_c9a.log mtc_c9a_d*_core3d.log"
  echo "  git commit -m 'Campaign 9 closures on the new front members, replicas, MTC limits and the gadolinia study'"
  mark F
}

stage_V() {
  [ "$SKIP_VALGRID" = "1" ] && { echo "[V] skipped (SKIP_VALGRID=1)"; return 0; }
  done_ V && { echo "[V] already done"; return 0; }
  hr; echo " STAGE V. The valgrid run, resumed  ($(stamp))"; hr
  EQUAL=1 bash run_valgrid_c9.sh 2>&1 | tee -a valgrid_c9.log || die "valgrid failed"
  mark V
}

case "${1:-}" in
  --preflight) stage_P ;;
  "")          stage_P; stage_K; stage_L; stage_R; stage_C; stage_M; stage_G; stage_F; stage_V ;;
  *)           die "unknown option $1" ;;
esac
