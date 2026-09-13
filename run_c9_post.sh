#!/usr/bin/env bash
# run_c9_post.sh -- the Campaign 9 post-analysis, one stage after another.
#
# Every stage is skipped when its marker exists, so the script can be
# interrupted and relaunched. Stages A to D need no OpenMC and run in a
# minute. Stages 1 to 6 run OpenMC and are ordered so that the result you
# asked for first (the 3D peaking on the front) comes first.
#
#   A  manifest      c9_front.py         feasible set, front, two-bank, champion
#   B  figures       c9_figures.py       pareto with ceiling, Gd trade-off, hump
#   C  step 0        c9_step0.py         LOO surrogate CV, front stability
#   D  reverse retro c9_reverse_retro.py C9 archive under the C8 objectives
#   E  analysis figs  c9_analysis_figures.py two formulations, stability, mechanism
#   1  3D peaking, FRONT      confirm3d --states ARO       ~50 min  <- run first
#   2  3D confirmation, front + two-bank, ARO/ARI/RE12 at 1000 ppm    ~5 h
#   3  3D confirmation at each front design's own c_BOL, ARO/ARI/RE12  ~5 h
#   4  k-target burnup on the front           validate_ktarget_burnup  ~3 h
#   5  MTC ceiling on the champion, 2D + 3D, 12.8 and 15.5 MPa         ~1.5 h
#   6  boron worth, independent check on the front                    ~40 min
#   7  3D peaking, ALL 60 designs, ARO                                 ~5 h
#   8  2D/3D peaking analysis on front and on all (decides Campaign 10)
#
# USAGE, from the repository root
#   bash run_c9_post.sh --check                # list stages and markers, run nothing
#   bash run_c9_post.sh --quick                # A B C D only, no OpenMC
#   bash run_c9_post.sh --front3d              # A, stage 1, stage 8 only, then STOP
#                                              (not needed: the default already runs
#                                               the front first and then continues)
#   setsid nohup bash run_c9_post.sh > c9_post.log 2>&1 < /dev/null &
#       THE ONE COMMAND. Order: archive analysis (1 min), 3D peaking on the
#       front (50 min), then stages 2 to 7 (about 20 h). A failing stage is
#       reported and skipped, the run continues. Nothing else to launch.
#   bash run_c9_post.sh --only 5               # one OpenMC stage by number
#   rm .c9post_markers/2                       # force a stage to rerun
set -u -o pipefail

THREADS=64
CKPT=out_c9/optimization_checkpoint.json
CKPT8=out_c8/optimization_checkpoint.json
KT=ktarget_table_c8.json
POST=c9_post
FIGS=figs_c9
MARK=.c9post_markers
CEILING=2763          # ppm, design 47, hardware 3D, 12.8 MPa
CEILING_HI=2997       # ppm, 15.5 MPa
SEEDS3D=2

mkdir -p "$MARK" "$POST" "$FIGS"
die()  { echo "FAIL: $*" >&2; exit 1; }
hr()   { printf '%.0s-' $(seq 1 74); echo; }
mark() { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
need() { [ -f "$1" ] || die "missing $1"; }
ids()  { python -c "import json;print(' '.join(map(str,json.load(open('$POST/c9_front.json'))['$1'])))"; }

preflight() {
  hr; echo " PREFLIGHT"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('  openmc', openmc.__version__)" || die "openmc import failed"
  [ "$(git branch --show-current)" = "main" ] || die "not on main"
  for f in "$CKPT" "$CKPT8" "$KT" c9_front.py c9_figures.py c9_step0.py c9_reverse_retro.py \
           c9_analysis_figures.py \
           c9_peaking_2d3d.py confirm3d.py hardware3d.py validate_ktarget_burnup.py \
           mtc_scan.py boron_worth.py boron_objective.py; do need "$f"; done
  python -c "
import json; d=json.load(open('$CKPT'))
assert d['objectives']==[['peaking','min'],['c_max','min']], d['objectives']
assert len(d['all_raw'])==60, len(d['all_raw'])
print('  archive: 60 evaluations, objectives', d['objectives'])"
  echo "  markers: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"
}

# ------------------------------------------------------- no-OpenMC stages --
stage_A() { done_ A && { echo "[A] done"; return; }; hr; echo " A. manifest"; hr
  python c9_front.py --selftest >/dev/null || die "c9_front selftest"
  python c9_front.py --checkpoint "$CKPT" --out "$POST" --ceiling "$CEILING" || die "c9_front"
  mark A; }
stage_B() { done_ B && { echo "[B] done"; return; }; hr; echo " B. figures"; hr
  python c9_figures.py --checkpoint "$CKPT" --manifest "$POST/c9_front.json" --out "$FIGS" \
    --ceiling "$CEILING" --ceiling-hi "$CEILING_HI" || die "c9_figures"
  mark B; }
stage_C() { done_ C && { echo "[C] done"; return; }; hr; echo " C. step 0 validation"; hr
  python c9_step0.py --checkpoint "$CKPT" --manifest "$POST/c9_front.json" --out "$POST" \
    --sigma 0.010 10 --B 2000 || die "c9_step0"
  mark C; }
stage_D() { done_ D && { echo "[D] done"; return; }; hr; echo " D. reverse retrospective"; hr
  python c9_reverse_retro.py --c9 "$CKPT" --c8 "$CKPT8" --manifest "$POST/c9_front.json" \
    --out "$POST" || die "c9_reverse_retro"
  mark D; }
stage_E() { done_ E && { echo "[E] done"; return; }; hr; echo " E. analysis figures"; hr
  # needs A, C and D: reads the manifest, c9_step0.json and c9_reverse_retro.json
  python c9_analysis_figures.py --checkpoint "$CKPT" --post "$POST" --out "$FIGS" \
    --ceiling "$CEILING" || die "c9_analysis_figures"
  mark E; }

# ---------------------------------------------------------- OpenMC stages --
c3d() {   # confirm3d with the campaign-consistent hardware settings
  python -c "import numpy, openmc; print('env ok')" \
  && python -u confirm3d.py --checkpoint "$CKPT" --seeds "$SEEDS3D" --threads "$THREADS" "$@"
}

stage_1() { done_ 1 && { echo "[1] done"; return; }; hr; echo " 1. 3D peaking on the FRONT, ARO only"; hr
  local F; F=$(ids front); echo "  front ids: $F"
  c3d --designs $F --states ARO --out confirm3d_c9_front 2>&1 | tee confirm3d_c9_front.log \
    || die "stage 1"
  mark 1; }

stage_2() { done_ 2 && { echo "[2] done"; return; }; hr; echo " 2. 3D confirmation, front + two-bank, at 1000 ppm"; hr
  local F; F=$(python -c "
import json;m=json.load(open('$POST/c9_front.json'));print(' '.join(map(str,sorted(set(m['front'])|set(m['two_bank'])))))")
  echo "  ids: $F"
  c3d --designs $F --states ARO ARI RE12 --boron-ppm 1000 --out confirm3d_c9 2>&1 | tee confirm3d_c9.log \
    || die "stage 2"
  mark 2; }

stage_3() { done_ 3 && { echo "[3] done"; return; }; hr; echo " 3. 3D confirmation at each front design's own c_BOL"; hr
  # the Campaign 8 lesson: 1000 ppm is the reference, not the operating point.
  # One confirm3d call per design at its measured critical concentration.
  python - <<PY > "$POST/c3d_cbol_plan.txt" || die "plan"
import json; m=json.load(open('$POST/c9_front.json'))
for i in m['front']:
    c=m['front_c_bol'][str(i)]
    print(i, round(max(c,0.0)))
PY
  cat "$POST/c3d_cbol_plan.txt"
  while read -r i c; do
    [ -d "confirm3d_c9_cbol/d${i}" ] && [ -f "confirm3d_c9_cbol/d${i}/summary.json" ] \
      && { echo "  d$i at $c ppm done"; continue; }
    c3d --designs "$i" --states ARO ARI RE12 --boron-ppm "$c" --out "confirm3d_c9_cbol/d${i}" \
      2>&1 | tee -a confirm3d_c9_cbol.log || die "stage 3 design $i"
  done < "$POST/c3d_cbol_plan.txt"
  mark 3; }

stage_4() { done_ 4 && { echo "[4] done"; return; }; hr; echo " 4. k-target burnup validation on the front"; hr
  # --front is NOT used: its front detection assumes the C8 objectives
  local F; F=$(ids front)
  python -c "import numpy, openmc; print('env ok')" \
  && python -u validate_ktarget_burnup.py --checkpoint "$CKPT" --designs $F \
       --ktarget-table "$KT" --seeds 2 --threads "$THREADS" --out kt_burnup_c9 \
       2>&1 | tee kt_burnup_c9.log || die "stage 4"
  mark 4; }

stage_5() { done_ 5 && { echo "[5] done"; return; }; hr; echo " 5. MTC ceiling on the Campaign 9 champion"; hr
  local CH; CH=$(ids champion); echo "  champion id: $CH"
  for P in 12.8 15.5; do
    for LV in core2d core3d; do
      tag="mtc_c9_p${P/./}_${LV}"
      [ -f "$tag/report.txt" ] && { echo "  $tag done"; continue; }
      python -c "import numpy, openmc; print('env ok')" \
      && python -u mtc_scan.py --checkpoint "$CKPT" --idx "$CH" --pressure "$P" --level "$LV" \
           --boron 0,1000,2000,3000 --seeds 2 --threads "$THREADS" --out "$tag" \
           2>&1 | tee "$tag.log" || die "stage 5 $tag"
    done
  done
  mark 5; }

stage_6() { done_ 6 && { echo "[6] done"; return; }; hr; echo " 6. boron worth, independent check on the front"; hr
  local F; F=$(ids front)
  python -c "import numpy, openmc; print('env ok')" \
  && python -u boron_worth.py --checkpoint "$CKPT" --designs $F --ppm 0 1000 1500 2500 \
       --states ARO --seeds 1 --threads "$THREADS" --out boron_c9 2>&1 | tee boron_c9.log \
    || die "stage 6"
  mark 6; }

stage_7() { done_ 7 && { echo "[7] done"; return; }; hr; echo " 7. 3D peaking on ALL 60 designs, ARO only"; hr
  c3d --designs $(seq 0 59) --states ARO --out confirm3d_c9_all 2>&1 | tee confirm3d_c9_all.log \
    || die "stage 7"
  mark 7; }

stage_8() { hr; echo " 8. 2D/3D peaking analysis"; hr
  [ -f confirm3d_c9_front/summary.json ] \
    && python c9_peaking_2d3d.py --summary confirm3d_c9_front/summary.json --checkpoint "$CKPT" \
         --manifest "$POST/c9_front.json" --out "$POST" --tag front
  [ -f confirm3d_c9_all/summary.json ] \
    && python c9_peaking_2d3d.py --summary confirm3d_c9_all/summary.json --checkpoint "$CKPT" \
         --manifest "$POST/c9_front.json" --out "$POST" --tag all
  return 0; }

# ------------------------------------------------------------------- main --
FAILED=()
# Each stage runs in a subshell, so its "die" ends the stage and not the run.
# A failed stage leaves no marker, so relaunching retries only that stage.
run() {   # isolate the stage: its failure must not end the run
  if ( "stage_$1" ); then return 0; fi
  FAILED+=("$1"); echo; echo "!! stage $1 FAILED, continuing with the next stage"; echo
}
banner() {
  echo; printf '=%.0s' $(seq 1 74); echo
  echo " $*"
  echo " $(date '+%F %H:%M:%S')"
  printf '=%.0s' $(seq 1 74); echo; echo
}
case "${1:-}" in
  --check)   preflight; echo; grep -E '^#   [A-D0-9] ' "$0"; exit 0 ;;
  --quick)   preflight; run A; run B; run C; run D; run E
             echo; echo "failed: ${FAILED[*]:-none}"; exit 0 ;;
  --front3d) preflight; stage_A; stage_1; stage_8; exit 0 ;;
  --only)    preflight; stage_A; "stage_$2"; exit 0 ;;
  "")        preflight
             banner "PART 1 of 3: archive analysis, no OpenMC, about one minute"
             run A; run B; run C; run D; run E
             banner "PART 2 of 3: 3D peaking on the front, about 50 minutes"
             run 1; run 8
             banner "FRONT RESULTS ARE READY. Read them now, the run continues on its own:
   c9_post/c9_status.txt              the front, the champion, the two-bank set
   c9_post/c9_step0.txt               surrogate cross-validation, front stability
   c9_post/c9_reverse_retro.txt       what the Campaign 8 objectives would have picked
   c9_post/c9_peaking_2d3d_front.txt  2D vs 3D peaking and the Campaign 10 decision
   figs_c9/                           the three figures
 Nothing further to launch. Stages 2 to 7 are running, about 20 hours."
             run 2; run 3; run 4; run 5; run 6; run 7
             banner "PART 3 of 3: 2D/3D peaking over all 60 designs"
             run 8 ;;
  *)         die "unknown option $1" ;;
esac
banner "POST-ANALYSIS COMPLETE"
echo " stages done  : $(ls "$MARK" 2>/dev/null | sort | tr '\n' ' ')"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo " stages FAILED: ${FAILED[*]}"
  echo " retry just those with:  bash run_c9_post.sh --only <n>"
  exit 1
fi
echo " no failures"
