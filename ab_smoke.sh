#!/usr/bin/env bash
# ab_smoke.sh -- A/B test of the OpenMC aborts of 3 Nov 2026 (heap corruption
# in libopenmc/HDF5, five aborts in four smoke attempts of Campaign 12, none in
# Campaigns 10 and 11). Runs the Campaign 11 smoke, which passed on 24 Oct, with
# the code of commit a6f8668 (before any Campaign 12 change), in a separate
# worktree so the main checkout and its outputs are untouched.
#   passes  -> the aborts come from the Campaign 12 code
#   aborts  -> the machine or the OpenMC installation, the code is not the cause
#
#   setsid nohup bash ab_smoke.sh > ab.out 2>&1 < /dev/null &      about 1.5 h
#   cat ab.out                                                      the verdict line
set -u -o pipefail
REPO=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
WT=$HOME/c11code
[ -d "$WT" ] || git -C "$REPO" worktree add --detach "$WT" a6f8668 || exit 1
cd "$WT"
# the problem settings exactly as run_c11.sh defines them
source <(sed -n '/^THREADS=/,/^LOOP=/p' run_c11.sh)
rm -rf out_ab runs_ab
echo "$(date '+%F %T') A/B smoke started in $WT ($(git log -1 --oneline))"
python -u run_optimization.py --smoke --out out_ab --workdir runs_ab "${COMMON[@]}" \
     --n-init 4 --n-infill 2 --iters 1 --hump-noise "$HUMP_NOISE_SMOKE" \
     --cycle-core3d "$CYCLE3D_SMOKE" --kbol-transport "$KBOL_SMOKE" > ab.log 2>&1 < /dev/null
rc=$?
echo "$(date '+%F %T') exit $rc | aborts $(grep -c 'Signal: Aborted' ab.log) | evaluations finished $(grep -c '^  \[case [0-9]*\] e=' ab.log) | log $WT/ab.log"
