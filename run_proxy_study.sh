#!/usr/bin/env bash
# run_proxy_study.sh -- proxies of the eight-layer core depletion, for the
# cycle length inside the optimisation loop. One job at a time, 64 threads.
# Every stage is cached per design, so the same command resumes after a stop.
#
#   1  low-fidelity eight-layer CORE depletion, 10 000 x 100, of the 13
#      measured designs that do not have one                      about 4.5 h
#   2  single ASSEMBLY, 8 layers, C9-47: --estimate (two short steps). Checks
#      that the new model builds and projects its cost             minutes
#   3  single assembly, C9-47, layer sensitivity: 1, 4, 8 and 12 layers
#   4  single assembly, C9-47, 8 layers, bare fuel (no axial structure)
#   5  single assembly, 8 layers, the other 16 measured designs
#   6  comparison of every proxy with the reference (proxy_compare.py)
#
# The cost of stages 3 to 5 is not known before stage 2. If the single
# assembly costs like the campaign assembly per solve, they take 2 to 4 h.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_proxy_study.sh preflight
#   setsid nohup bash run_proxy_study.sh run > proxy_study.log 2>&1 < /dev/null &
#   bash run_proxy_study.sh status
set -u -o pipefail

THREADS=${THREADS:-64}
CKPT=out_c9a/optimization_checkpoint.json
MEASURED="1 11 16 24 27 32 35 47 63 68 69 70 72 79 85 88 94"   # eight-layer reference exists
LOFI_NEW="1 11 16 32 35 47 63 68 70 79 85 88 94"               # 24, 27, 69, 72 already at low fidelity
PILOT=47
REST=$(for d in $MEASURED; do [ "$d" = "$PILOT" ] || printf '%s ' "$d"; done)
LOFI=(--particles 10000 --batches 100 --inactive 50)
REPORT=proxy_study_report.txt

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; echo " $*  ($(stamp))"; }
busy()  { pgrep -af "run_c9b_ctrl_mtc.sh|run_c9b_layers.sh|c9_dep_core3d.py|c9_dep_asm3d.py|confirm3d.py|mtc_scan.py|run_optimization.py" || true; }
asm()   { python -u c9_dep_asm3d.py --checkpoint "$CKPT" "${LOFI[@]}" --threads "$THREADS" "$@"; }

preflight() {
  [ "$(hostname)" = "wks720" ]                || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"
  for f in "$CKPT" c9_dep_core3d.py c9_dep_asm3d.py proxy_compare.py; do [ -e "$f" ] || die "missing $f (git pull?)"; done
  python c9_dep_asm3d.py --selftest || die "selftest of c9_dep_asm3d.py failed"
  b=$(busy); [ -z "$b" ] || die "another transport job is running, one 64-thread job at a time: $b"
}

case "${1:-}" in
status)
  echo "$(stamp) host $(hostname)"; b=$(busy); [ -n "$b" ] && echo "running: $b" || echo "running: nothing"
  for d in lofi_dep_core3d_all asm3d_L1 asm3d_L4 asm3d_L8 asm3d_L12 asm3d_bare_L8; do
    [ -f "$d/runs.json" ] && echo "$d: $(python -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1])), key=lambda k: int(k[1:]))))" "$d/runs.json")" || echo "$d: nothing yet"
  done
  python proxy_compare.py --checkpoint "$CKPT" ;;
preflight)
  preflight; echo "$(stamp) preflight ok" ;;
run)
  preflight
  hr "1. low-fidelity eight-layer core depletion: $LOFI_NEW"
  python -u c9_dep_core3d.py --checkpoint "$CKPT" --designs $LOFI_NEW --layers 8 "${LOFI[@]}" \
    --threads "$THREADS" --out lofi_dep_core3d_all || die "stage 1"
  hr "2. single assembly, estimate on C9-$PILOT"
  asm --designs "$PILOT" --layers 8 --estimate --out asm3d_L8 || die "stage 2: the single-assembly model did not run"
  grep -q '"ok": true' "asm3d_L8/estimate_d$PILOT.json" || die "stage 2: the estimate found no statepoint"
  hr "3. single assembly, layer sensitivity on C9-$PILOT"
  for L in 1 4 8 12; do asm --designs "$PILOT" --layers "$L" --out "asm3d_L$L" || die "stage 3, $L layers"; done
  hr "4. single assembly, bare fuel, C9-$PILOT"
  asm --designs "$PILOT" --layers 8 --axial bare --out asm3d_bare_L8 || die "stage 4"
  hr "5. single assembly, 8 layers: $REST"
  asm --designs $REST --layers 8 --out asm3d_L8 || die "stage 5"
  hr "6. comparison"
  python proxy_compare.py --checkpoint "$CKPT" | tee "$REPORT"
  echo "$(stamp) finished. To send the results back:"
  echo "  git add -f lofi_dep_core3d_all/runs.json asm3d_*/runs.json asm3d_L8/estimate_d$PILOT.json $REPORT && git commit -m 'Proxy study: low-fidelity core and single-assembly depletions' && git push" ;;
*)
  die "usage: bash run_proxy_study.sh preflight | run | status" ;;
esac
