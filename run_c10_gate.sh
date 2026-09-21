#!/usr/bin/env bash
# run_c10_gate.sh -- the gate that decides whether a Campaign 10 on the axial
# peaking factor F_z is worth running. One entry point, resumable.
#
#   question   does F_z respond to the four design variables by more than its
#              own Monte Carlo noise, over the whole Campaign 9 archive?
#   method     one ARO solve per archived design on the hardware model, at
#              1000 ppm and beginning of life, with the axial mesh of
#              axial_shape_c9.py. Then c10_fz_gate.py, no transport.
#   cost       53 new solves at about 247 s, about 3.6 h. The seven designs of
#              axial_c9 are reused from its cache.
#
# STAGES (each skipped when its marker exists)
#   P  preflight: host, env, OpenMC, branch, files, selftests
#   D  dry run: the 60 designs are listed, nothing is solved
#   R  the solves, resumable through axial_shape_c9.py's own cache
#   G  the gate analysis and its verdict
#
# USAGE, from the repository root
#   bash run_c10_gate.sh --preflight          # stage P only, writes nothing
#   setsid nohup bash run_c10_gate.sh > c10_gate.log 2>&1 < /dev/null &
#   tail -f c10_gate.log
#   rm .c10_gate_markers/R                    # force a stage to rerun
set -u -o pipefail

THREADS=64
MARK=.c10_gate_markers
CKPT=out_c9/optimization_checkpoint.json
SRC=axial_c9                 # the seven-design study, left untouched
OUT=axial_c9_all             # its copy, extended to the whole archive
GATE=c10_gate
STATE=ARO
SEEDS=1                      # one solve per design, as a campaign would see it
BORON=1000

mkdir -p "$MARK"
die()   { echo "FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }

N_DESIGNS=$(python3 -c "import json;print(len(json.load(open('$CKPT'))['all_raw']))" 2>/dev/null) \
  || die "cannot read $CKPT"
DESIGNS=$(seq 0 $((N_DESIGNS - 1)))

# ---------------------------------------------------------------- stage P --
stage_P() {
  hr; echo " STAGE P. Preflight"; hr
  echo "  host        : $(hostname)"
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  echo "  conda env   : ${CONDA_DEFAULT_ENV:-none}"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  python -c "import numpy, openmc; print('  openmc      :', openmc.__version__)" \
    || die "openmc import failed"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] \
    || die "OpenMC is not 0.15.3"
  echo "  branch      : $(git branch --show-current)"
  [ "$(git branch --show-current)" = "main" ] || die "not on main"
  echo "  head        : $(git log --oneline -1)"
  echo "  xs          : ${OPENMC_CROSS_SECTIONS:-unset}"
  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] || die "OPENMC_CROSS_SECTIONS unset"
  for f in "$CKPT" axial_shape_c9.py c10_fz_gate.py c9_step0.py hardware3d.py zoning.py; do
    [ -f "$f" ] && echo "  found       : $f" || die "missing $f"
  done
  echo "  archive     : $N_DESIGNS designs in $CKPT"
  echo "  threads     : $THREADS of $(nproc)"
  echo "  disk free   : $(df -h . | awk 'NR==2{print $4}')"
  python axial_shape_c9.py --selftest || die "axial_shape_c9 selftest failed"
  python c10_fz_gate.py --selftest   || die "c10_fz_gate selftest failed"
  echo "  other jobs  : $(pgrep -xc openmc || true) openmc solver processes running"
  echo "preflight OK"
}

# ---------------------------------------------------------------- stage D --
stage_D() {
  done_ D && { echo "[D] already done"; return 0; }
  hr; echo " STAGE D. Dry run"; hr
  if [ -d "$OUT" ]; then
    echo "  $OUT exists, resuming from its cache"
  elif [ -d "$SRC" ]; then
    cp -r "$SRC" "$OUT" || die "could not copy $SRC to $OUT"
    echo "  copied $SRC to $OUT, its cached solves are reused"
  else
    echo "  $SRC not found, starting $OUT from empty"
  fi
  # shellcheck disable=SC2086
  python axial_shape_c9.py --checkpoint "$CKPT" --designs $DESIGNS --states "$STATE" \
      --seeds "$SEEDS" --threads "$THREADS" --boron-ppm "$BORON" --out "$OUT" --dry-run \
      | tee "$OUT.dryrun.txt" | tail -4 || die "dry run failed"
  n=$(grep -c "^=== design C9-" "$OUT.dryrun.txt")
  [ "$n" = "$N_DESIGNS" ] || die "dry run listed $n designs, expected $N_DESIGNS"
  echo "  $n designs listed"
  mark D
}

# ---------------------------------------------------------------- stage R --
stage_R() {
  done_ R && { echo "[R] already done"; return 0; }
  hr; echo " STAGE R. $N_DESIGNS designs x $STATE x $SEEDS seed, resumable"; hr
  # shellcheck disable=SC2086
  python -c "import numpy, openmc; print('env ok')" \
    && python -u axial_shape_c9.py --checkpoint "$CKPT" --designs $DESIGNS --states "$STATE" \
         --seeds "$SEEDS" --threads "$THREADS" --boron-ppm "$BORON" --out "$OUT" \
    || die "solves failed. Relaunch the same command: finished solves are cached."
  mark R
}

# ---------------------------------------------------------------- stage G --
stage_G() {
  hr; echo " STAGE G. Gate"; hr
  python c10_fz_gate.py --axial-dir "$OUT" --checkpoint "$CKPT" --state "$STATE" --out "$GATE" \
    || die "gate analysis failed"
  hr; echo " GATE RESULT IS READY: $GATE/fz_gate.txt"; hr
  mark G
}

# ------------------------------------------------------------------- main --
case "${1:-}" in
  --preflight) stage_P; exit 0 ;;
  "")          stage_P; stage_D; stage_R; stage_G ;;
  *)           die "unknown option $1" ;;
esac
