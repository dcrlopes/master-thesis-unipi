#!/usr/bin/env bash
# run_c9_axial_post.sh -- Campaign 9 axial and ceiling post-processing, steps 1 to 5.
#
# No transport. About one minute. Every step reads existing results.
#
#   1. environment, branch, git pull, apply_axial_c9_fixes.py (selftest, check, apply)
#   2. axial figures redrawn, must report 3 grid bands
#   3. corrected design headers, dry run of axial_shape_c9.py into /tmp
#   4. c9_axial_checks.py, seed agreement, F_dH estimators, entropy, boron
#   5. MTC ceiling tables and figures at 12.8 and 15.5 MPa, relabelled 15.5 MPa
#      table, two-pressure figure
#
# Step 0 is done on the laptop before this script:
#   scp apply_axial_c9_fixes.py c9_axial_checks.py c9_ceiling_two_pressures.py \
#       run_c9_axial_post.sh diogo@wks720:~/master-thesis-unipi/
#
# LAUNCH (foreground, it is short)
#   cd ~/master-thesis-unipi
#   python -c "import numpy, openmc; print('env ok')" && \
#     bash run_c9_axial_post.sh 2>&1 | tee run_c9_axial_post.log
#
# The script stops at the first failure and says which step failed.
# Re-running is safe: the applier skips patches already applied, and every
# other step overwrites its own outputs.

set -u
set -o pipefail
cd ~/master-thesis-unipi || { echo "FAIL: repository not found"; exit 1; }

CKPT=out_c9/optimization_checkpoint.json
DESIGNS="47 44 34 40 35 58 12"
STEP=0

fail () { echo; echo "!!! FAILED at step $STEP: $1"; exit 1; }
step () { STEP="$1"; echo; echo "=================== step $1, $2  $(date '+%F %H:%M:%S')"; }

# ------------------------------------------------------------------ step 1 --
step 1 "environment, update, fixes"
python -c "import numpy, openmc; print('env ok', openmc.__version__)" \
  || fail "openmc-env is not active"
echo "host   : $(hostname)"
BR=$(git branch --show-current)
echo "branch : $BR"
[ "$BR" = "main" ] || fail "not on main"

for f in "$CKPT" apply_axial_c9_fixes.py c9_axial_checks.py c9_ceiling_two_pressures.py \
         axial_shape_c9.py axial_figures_c9.py mtc_front_table.py \
         axial_c9/runs.json confirm3d_c9_all/runs.json; do
  [ -e "$f" ] || fail "$f not found"
done
RUNNING='python[0-9.]* .*(axial_shape_c9|mtc_scan|confirm3d)\.py'
if pgrep -f "$RUNNING" >/dev/null; then
  pgrep -af "$RUNNING" | head -3
  fail "a transport job is still running, wait for it"
fi

git pull --ff-only || fail "git pull did not fast-forward, check git status"
echo "commit : $(git log --oneline -1)"

python apply_axial_c9_fixes.py --selftest || fail "applier selftest"
echo "(the 'stored edges differ' warning above is the deliberate shifted-edges test)"
python apply_axial_c9_fixes.py --check    || fail "applier anchors do not match"
python apply_axial_c9_fixes.py            || fail "applier"
python -m py_compile axial_shape_c9.py axial_figures_c9.py || fail "patched files do not compile"

# ------------------------------------------------------------------ step 2 --
step 2 "axial figures with the real grid bands"
python axial_figures_c9.py --out axial_c9 --champion 47 --png > axial_c9_figures.log 2>&1 \
  || { tail -20 axial_c9_figures.log; fail "axial_figures_c9.py"; }
head -3 axial_c9_figures.log
grep -q " 3 grid bands" axial_c9_figures.log \
  || fail "figures did not recover 3 grid bands, see axial_c9_figures.log"
echo "grid bands OK, $(ls axial_c9/figs/*.pdf | wc -l) PDFs in axial_c9/figs"

# ------------------------------------------------------------------ step 3 --
step 3 "design headers with the as-built pin count, no transport"
rm -rf /tmp/axial_dry
python axial_shape_c9.py --checkpoint "$CKPT" --designs $DESIGNS \
  --states ARO RE12 --seeds 2 --out /tmp/axial_dry --dry-run > /tmp/axial_dry.log 2>&1 \
  || { tail -20 /tmp/axial_dry.log; fail "axial_shape_c9.py --dry-run"; }
grep "=== design" /tmp/axial_dry.log | tee c9_post/axial_design_headers.txt
[ "$(grep -c 'as built' c9_post/axial_design_headers.txt)" -eq 7 ] \
  || fail "expected 7 headers with the as-built pin count"
rm -rf /tmp/axial_dry

# ------------------------------------------------------------------ step 4 --
step 4 "seed agreement, peaking estimators, entropy, boron"
python c9_axial_checks.py --axial axial_c9 --confirm confirm3d_c9_all --out c9_post \
  || fail "c9_axial_checks.py"
grep -q "same conditions, the solves can be pooled" c9_post/c9_axial_checks.txt \
  || echo "WARNING: fidelity or boron differ, do not pool the F_dH solves"

# ------------------------------------------------------------------ step 5 --
step 5 "MTC ceiling tables and figures at both pressures"
python mtc_front_table.py --checkpoint "$CKPT" \
  --glob 'mtc_c9_*_core3d*' --pressure 12.8 --out c9_post --figure \
  > c9_post/mtc_front_table_p128.out 2>&1 \
  || { tail -20 c9_post/mtc_front_table_p128.out; fail "mtc_front_table.py at 12.8 MPa"; }
sed -n '/  id  Gd/,/fail their own/p' c9_post/mtc_front_table_p128.out

python mtc_front_table.py --checkpoint "$CKPT" \
  --glob 'mtc_c9_*p155_core3d' --pressure 15.5 --out c9_post/p155 --figure \
  > c9_post/mtc_front_table_p155.out 2>&1 \
  || { tail -20 c9_post/mtc_front_table_p155.out; fail "mtc_front_table.py at 15.5 MPa"; }
sed -n '/  id  Gd/,/fail their own/p' c9_post/mtc_front_table_p155.out

N128=$(python -c "import json; print(len(json.load(open('c9_post/mtc_ceiling_table.json'))))")
N155=$(python -c "import json; print(len(json.load(open('c9_post/p155/mtc_ceiling_table.json'))))")
echo "rows: $N128 at 12.8 MPa (expected 8), $N155 at 15.5 MPa (expected 5)"
[ "$N128" -eq 8 ] && [ "$N155" -eq 5 ] || fail "unexpected number of lattices, check the globs"

sed 's/tab:c9-ceilings}/tab:c9-ceilings-p155}/' c9_post/p155/mtc_ceiling_table.tex \
  > c9_post/p155/mtc_ceiling_table_p155.tex
grep -q "tab:c9-ceilings-p155" c9_post/p155/mtc_ceiling_table_p155.tex \
  || fail "relabelling the 15.5 MPa table"

python c9_ceiling_two_pressures.py --out c9_post || fail "c9_ceiling_two_pressures.py"

# ------------------------------------------------------------------ summary --
STEP="done"
echo
echo "=================== all steps done  $(date '+%F %H:%M:%S')"
echo "--- outputs"
ls -l c9_post/c9_axial_checks.txt c9_post/axial_design_headers.txt \
      c9_post/mtc_ceiling_table.tex c9_post/c9_ceiling_inventory.pdf \
      c9_post/p155/mtc_ceiling_table_p155.tex c9_post/p155/c9_ceiling_inventory.pdf \
      c9_post/c9_ceiling_two_pressures.pdf
echo "--- seed agreement"
grep -E "^  (axial|confirm3d|both) +pairs" c9_post/c9_axial_checks.txt
echo "--- files changed or new, review before step 6 (commit)"
git status --short | head -40
