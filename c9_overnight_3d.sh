#!/usr/bin/env bash
# c9_overnight_3d.sh -- overnight 3D depletions after the C9-35 axial test.
#
# QUESTIONS FOR THE MORNING
#   A  Which C9 designs meet the 1826 d floor once axial burnup is resolved?
#      8-layer runs of the candidates that survive both measured losses
#      (13.1 % and 16.9 %) and sit below the 2763 ppm boron ceiling.
#   B  Does the size of the axial loss follow the gadolinia content?
#      1-layer runs of C9-1 (2.76 wt% Gd) and C9-27 (5.70 wt%), paired with
#      their 8-layer runs, next to C9-35 (2.63 wt%, -13.1 %) and C9-47
#      (4.42 wt%, -16.9 %).
#   C9-10 is left out: its c_max of 4371 ppm is 1608 ppm above the ceiling.
#
# ORDER AND TIME (about 7.3 min per solve at 64 threads)
#   1 C9-27 8 layers   ~1.0 h   corrected-front candidate, lowest boron
#   2 C9-16 8 layers   ~1.0 h   tightest candidate, cycle and ceiling
#   3 C9-1  8 layers   ~1.2 h   candidate, low gadolinia
#   4 C9-27 1 layer    ~1.2 h   loss of a high-gadolinia design
#   5 C9-1  1 layer    ~1.5 h   loss of a low-gadolinia design
#   6 C9-24 8 layers   ~1.2 h   candidate, dominated by C9-27
#   7 C9-11 8 layers   ~1.0 h   candidate, dominated by C9-27
#   total about 8 to 9 h. The most important runs go first, so a morning
#   report is useful even if the queue has not finished.
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   python -c "import numpy, openmc; print('env ok')" && \
#   setsid nohup bash c9_overnight_3d.sh > c9_overnight.log 2>&1 < /dev/null &
#   bash c9_overnight_3d.sh --status
#   cat c9_overnight_report.txt            # in the morning
# Relaunching skips every run whose summary.txt already exists.
set -u -o pipefail

THREADS=${THREADS:-64}
QUEUE="27:8 16:8 1:8 27:1 1:1 24:8 11:8"
REPORT=c9_overnight_report.txt

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
dir_of() { echo "c9_dep_core3d_d$1_L$2"; }

status() {
  echo "time    : $(stamp)"
  for job in $QUEUE; do
    d=${job%:*}; L=${job#*:}; o=$(dir_of "$d" "$L")
    if [ -f "$o/summary.txt" ]; then s="done"
    elif pgrep -f -- "--designs $d --layers $L " > /dev/null; then s="RUNNING"
    elif grep -qs "FAILED C9-$d $L layer" c9_overnight.log; then s="FAILED, see c9_overnight.log"
    else s="waiting"; fi
    printf "  C9-%-3s %2s layer(s)  %s\n" "$d" "$L" "$s"
  done
  [ -f "$REPORT" ] && echo "report  : $REPORT"
}
[ "${1:-}" = "--status" ] && { status; exit 0; }

# ------------------------------------------------------------- preflight --
echo "$(stamp) preflight"
[ "$(hostname)" = "wks720" ]                || { echo "FAIL: not on wks720"; exit 1; }
[ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || { echo "FAIL: conda env is not openmc-env"; exit 1; }
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || { echo "FAIL: env check"; exit 1; }
[ -n "${OPENMC_CHAIN_FILE:-}" ] || { echo "FAIL: OPENMC_CHAIN_FILE unset"; exit 1; }
for f in c9_dep_core3d.py dep_common.py out_c9/optimization_checkpoint.json; do
  [ -f "$f" ] || { echo "FAIL: missing $f"; exit 1; }
done
python c9_dep_core3d.py --selftest || { echo "FAIL: selftest"; exit 1; }

# never share the machine with another 3D depletion
while pgrep -f "python -u c9_dep_core3d.py" > /dev/null; do
  echo "$(stamp) another c9_dep_core3d.py is running, waiting"; sleep 120
done

# ------------------------------------------------------------------ runs --
for job in $QUEUE; do
  d=${job%:*}; L=${job#*:}; o=$(dir_of "$d" "$L")
  if [ -f "$o/summary.txt" ]; then echo "$(stamp) C9-$d $L layer(s): already done"; continue; fi
  echo "$(stamp) START C9-$d with $L layer(s) -> $o"
  python -c "import numpy, openmc; print('env ok')" || { echo "FAIL: env check"; exit 1; }
  if python -u c9_dep_core3d.py --designs "$d" --layers "$L" --threads "$THREADS" --out "$o"; then
    python c9_dep_core3d.py --analyse --out "$o" > /dev/null && echo "$(stamp) DONE  C9-$d $L layer(s)"
  else
    echo "$(stamp) FAILED C9-$d $L layer(s), continuing with the next run"
  fi
done

# ---------------------------------------------------------------- report --
python - > "$REPORT" <<'EOF'
import json, glob, re
from pathlib import Path
FLOOR, CEIL = 1826.0, 2763.0
ck = json.load(open("out_c9/optimization_checkpoint.json")); raw = ck["all_raw"]
runs = {}
for p in glob.glob("c9_dep_core3d*/runs.json"):
    for key, r in json.load(open(p)).items():
        runs[(int(r["idx"]), int(r["layers"]))] = r
print("=== Overnight 3D depletions, relative end-of-cycle criterion, 1000 ppm, all rods out")
print(f"{'design':>7} {'enr':>5} {'Gd':>5} {'c_max':>6} {'F_dH':>6} {'archive':>7} {'1 layer':>8} {'8 layers':>8} {'loss':>7} {'margin 8L':>9}  verdict")
for d in sorted({k[0] for k in runs}):
    r = raw[d]; a = r["cycle_length"]
    e1 = runs.get((d, 1), {}).get("efpd"); e8 = runs.get((d, 8), {}).get("efpd")
    loss = f"{100 * (e8 / e1 - 1):+6.1f}%" if e1 and e8 else "     --"
    m8 = e8 - FLOOR if e8 else None
    if m8 is None: v = "no 8-layer run"
    elif r["c_max"] > CEIL: v = "above the boron ceiling"
    elif m8 > 30: v = "MEETS the floor in 3D"
    elif m8 > -30: v = "at the floor, within noise"
    else: v = "misses the floor in 3D"
    f = lambda x: f"{x:8.0f}" if x else "      --"
    print(f"  C9-{d:<3d} {r['enrich']:5.2f} {r['gd_wt']:5.2f} {r['c_max']:6.0f} {r['peaking']:6.3f} {a:7.0f} "
          f"{f(e1)} {f(e8)} {loss} {('%+9.0f' % m8) if m8 is not None else '       --'}  {v}")
print("\nloss = 8 layers against 1 layer. Noise about 12 d per run (propagated sigma x 1.33).")
print("Margins within 30 d of the floor are not resolved by one run.")
EOF
echo "$(stamp) REPORT written: $REPORT"; cat "$REPORT"
