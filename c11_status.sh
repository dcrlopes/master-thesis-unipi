#!/usr/bin/env bash
# c11_status.sh -- where Campaign 11 is, read-only. Separate from run_c11.sh so
# that it can be updated while the campaign executes.
#   bash c11_status.sh
cd "$(dirname "$(readlink -f "$0")")"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
age_min() { local f; f=$(find "$@" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1); [ -n "$f" ] && echo "$(( ($(date +%s) - ${f%.*}) / 60 )) min ago" || echo "no file yet"; }

echo "$(stamp) Campaign 11 on $(hostname)"
echo "stages done: $(ls .c11_markers 2>/dev/null | tr '\n' ' ' || true)"
pgrep -u "$USER" -af "run_c11.sh|run_kbol_queue.sh run|c9_repro_check.py|run_optimization.py --.*out_c11" | grep -v pgrep | sed 's/^/  running: /' || echo "  running: nothing"

n=$(ls repro_c9/kbol_c10_*_hifi.json 2>/dev/null | wc -l)
echo "stage K, high-statistics k_bol of Campaign 10: $n of 58 designs"
if [ -f .c11_markers/K ]; then
  echo "  done"
else
  first=$(grep -m1 "launch 1 of kbol_c10_" kbol_queue.log 2>/dev/null | cut -c1-19)
  if [ -n "$first" ] && [ "$n" -gt 4 ]; then
    el=$(( $(date +%s) - $(date -d "$first" +%s) )); per=$(( el / (n - 4) ))
    echo "  first solve $first, $(( per / 60 )) min per design, about $(( (58 - n) * per / 3600 )) h left"
  fi
  grep "^2026" kbol_queue.log 2>/dev/null | grep "kbol_c10_" | tail -n 2 | sed 's/^/  /'
  echo "  relaunches or skips: $(grep -c "kbol_c10_.*\(killing\|WARNING\)\|WARNING kbol_c10_" kbol_queue.log 2>/dev/null)"
fi

if [ -f out_c11/optimization_checkpoint.json ]; then
  echo "stage F: $(python -c "import json; print(json.load(open('out_c11/optimization_checkpoint.json'))['n_real_evaluations'])") evaluations in the checkpoint (target 94), newest file $(age_min openmc_runs_c11 out_c11.log)"
  grep "launch\|killing\|driver ended" out_c11.log c11.log 2>/dev/null | tail -n 2 | sed 's/^/  /'
elif [ -f .c11_markers/M ] || [ -d openmc_runs_c11_smoke ]; then
  echo "stage M or F starting, newest file $(age_min openmc_runs_c11_smoke openmc_runs_c11 out_c11_smoke.log out_c11.log)"
fi
tail -n 2 c11.log 2>/dev/null | sed 's/^/  c11.log: /'
