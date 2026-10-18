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
# Resumable: c9_dep_core3d.py skips the designs already in runs.json.
# The chain solver runs serial (dep_common.serial_depletion_pool), which is
# also the first test of that change: no forked pool, so no silent hang.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   setsid nohup bash run_c10_confirm.sh > c10_confirm.log 2>&1 < /dev/null &
#   bash run_c10_confirm.sh --status
set -u -o pipefail

THREADS=${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
CKPT=${CKPT:-out_c10/optimization_checkpoint.json}
DESIGNS=${DESIGNS:-"42 56 31 48"}
OUT1=c10_dep_core3d_seed1
OUT2=c10_dep_core3d_seed2

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die() { echo "$(stamp) FAIL: $*" >&2; exit 1; }

if [ "${1:-}" = "--status" ]; then
  for o in $OUT1 $OUT2; do
    echo "$o: $( [ -f $o/runs.json ] && python -c "import json; d=json.load(open('$o/runs.json')); print(' '.join(f\"{k} {v['efpd']:.0f}d\" for k,v in d.items()))" || echo "nothing yet")"
  done
  pgrep -u "$USER" -af "c9_dep_core3d.py --checkpoint $CKPT" | grep -v pgrep || echo "running: nothing"
  exit 0
fi

[ "$(hostname)" = "wks720" ] || [ -n "${ALLOW_HOST:-}" ] || die "not on wks720 (set ALLOW_HOST=1 to run elsewhere)"
[ -n "${OPENMC_CHAIN_FILE:-}" ] || die "OPENMC_CHAIN_FILE unset"
[ -f "$CKPT" ] || die "no $CKPT"
python c9_dep_core3d.py --selftest || die "c9_dep_core3d selftest"
python c9_dep_core3d.py --checkpoint "$CKPT" --designs $DESIGNS --layers 8 --dry-run || die "dry run"

for pair in "core3d:$OUT1" "core3d-seed2:$OUT2"; do
  salt=${pair%%:*}; out=${pair##*:}
  echo "$(stamp) seed salt $salt -> $out, designs $DESIGNS, $THREADS threads"
  python -u c9_dep_core3d.py --checkpoint "$CKPT" --designs $DESIGNS --layers 8 \
      --threads "$THREADS" --salt "$salt" --out "$out" || die "depletions of $out failed"
  python c9_dep_core3d.py --checkpoint "$CKPT" --out "$out" --analyse || echo "$(stamp) WARNING aggregation of $out failed"
done
echo "$(stamp) done. To send the results back:"
echo "  git add -f $OUT1/runs.json $OUT1/summary.txt $OUT2/runs.json $OUT2/summary.txt c10_confirm.log && git commit -m 'Campaign 10 front: full-fidelity eight-layer depletions, two seeds' && git push"
