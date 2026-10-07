#!/usr/bin/env bash
# run_c9c_proxy2.sh -- second round of the proxy study (suggestions 1 to 3 of
# 7 October 2026), after run_proxy_study.sh showed that the eight-layer core
# depletion at 10 000 x 100 meets every criterion. One job at a time, 64
# threads. Every stage is cached per design, so the same command resumes.
#
#   A  seed repeats of the low-fidelity core on C9-27, 69, 70, 47: two more
#      seeds each (salts core3d-seed2, core3d-seed3), to measure the noise of
#      the proxy itself instead of inferring it           about 2.7 h
#   B  the same model at 5 000 x 60 (30 inactive) on the 17 reference
#      designs, to see whether the cost can halve        about 3 h
#   C  single assembly, 8 layers, bare fuel (no grids, plenum or nozzles)
#      on C9-27, 69, 70, to read the +84 d bias of the structured
#      single-assembly model                             about 1 h
#   D  comparison (proxy_compare.py), with the seed SD of each proxy
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_c9c_proxy2.sh preflight
#   setsid nohup bash run_c9c_proxy2.sh run > proxy2.log 2>&1 < /dev/null &
#   bash run_c9c_proxy2.sh status
set -u -o pipefail

THREADS=${THREADS:-64}
CKPT=out_c9a/optimization_checkpoint.json
SEED_DESIGNS="27 69 70 47"
ALL17="1 11 16 24 27 32 35 47 63 68 69 70 72 79 85 88 94"
BARE_DESIGNS="27 69 70"
LOFI=(--particles 10000 --batches 100 --inactive 50)
LOFI5K=(--particles 5000 --batches 60 --inactive 30)
REPORT=proxy_study2_report.txt

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; echo " $*  ($(stamp))"; }
busy()  { pgrep -af "run_c10.sh|run_c9b_ctrl_mtc.sh|run_c9b_layers.sh|run_proxy_study.sh|c9_dep_core3d.py|c9_dep_asm3d.py|confirm3d.py|mtc_scan.py|run_optimization.py" || true; }
core()  { python -u c9_dep_core3d.py --checkpoint "$CKPT" --layers 8 --threads "$THREADS" "$@"; }
asm()   { python -u c9_dep_asm3d.py --checkpoint "$CKPT" --layers 8 --threads "$THREADS" "$@"; }

preflight() {
  [ "$(hostname)" = "wks720" ]                || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"
  [ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
  for f in "$CKPT" c9_dep_core3d.py c9_dep_asm3d.py proxy_compare.py; do [ -e "$f" ] || die "missing $f (git pull?)"; done
  python c9_dep_core3d.py --selftest || die "selftest of c9_dep_core3d.py failed"
  b=$(busy); [ -z "$b" ] || die "another transport job is running, one 64-thread job at a time: $b"
}

case "${1:-}" in
status)
  echo "$(stamp) host $(hostname)"; b=$(busy); [ -n "$b" ] && echo "running: $b" || echo "running: nothing"
  for d in lofi_dep_core3d_seed2 lofi_dep_core3d_seed3 lofi5k_core3d asm3d_bare_L8; do
    [ -f "$d/runs.json" ] && echo "$d: $(python -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1])), key=lambda k: int(k[1:]))))" "$d/runs.json")" || echo "$d: nothing yet"
  done
  [ -f "$REPORT" ] && { echo "report:"; tail -n 8 "$REPORT"; } ;;
preflight)
  preflight; echo "$(stamp) preflight ok" ;;
run)
  preflight
  hr "A. seed repeats of the low-fidelity core: $SEED_DESIGNS"
  for s in 2 3; do
    core --designs $SEED_DESIGNS "${LOFI[@]}" --salt "core3d-seed$s" --out "lofi_dep_core3d_seed$s" || die "stage A, seed $s"
  done
  hr "B. low-fidelity core at 5 000 x 60: $ALL17"
  core --designs $ALL17 "${LOFI5K[@]}" --out lofi5k_core3d || die "stage B"
  hr "C. single assembly, bare fuel, 8 layers: $BARE_DESIGNS"
  asm --designs $BARE_DESIGNS "${LOFI[@]}" --axial bare --out asm3d_bare_L8 || die "stage C"
  hr "D. comparison"
  python proxy_compare.py --checkpoint "$CKPT" | tee "$REPORT"
  echo "$(stamp) finished. To send the results back:"
  echo "  git add -f lofi_dep_core3d_seed*/runs.json lofi5k_core3d/runs.json asm3d_bare_L8/runs.json $REPORT && git commit -m 'Proxy study 2: seed repeats, 5000 x 60 setting, bare single assembly' && git push" ;;
*)
  die "usage: bash run_c9c_proxy2.sh preflight | run | status" ;;
esac
