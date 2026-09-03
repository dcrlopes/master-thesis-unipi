#!/bin/bash
# monitor.sh — watch OpenMC optimization progress from another terminal
# Usage:  bash monitor.sh    (from ~/master-thesis-unipi)
#         runs in a loop, refreshes every 10 seconds
# Kill with Ctrl+C

cd ~/master-thesis-unipi || exit 1

while true; do
  clear
  echo "=== LABGENE MOO Optimization Progress ===" 
  echo "Updated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  
  # case count
  n=$(ls -d openmc_runs/case_* 2>/dev/null | wc -l)
  echo "✓ Completed cases: $n"
  
  # latest case number
  if [ $n -gt 0 ]; then
    latest=$(ls -d openmc_runs/case_* 2>/dev/null | tail -1 | sed 's/.*case_//')
    echo "  Latest: case_$latest"
  fi
  echo ""
  
  # disk usage
  echo "Disk usage:"
  du -sh openmc_runs/ 2>/dev/null | awk '{print "  OpenMC scratch: " $1}'
  du -sh out/ 2>/dev/null | awk '{print "  Results: " $1}'
  echo ""
  
  # current CPU load
  echo "System load:"
  uptime | sed 's/^/  /'
  echo ""
  
  # container status if running
  echo "Docker container status:"
  docker ps --filter "status=running" --format "table {{.Status}}" | tail -1 | sed 's/^/  /'
  echo ""
  
  # most recent case log snippet
  if [ $n -gt 0 ]; then
    latest_dir=$(ls -d openmc_runs/case_* 2>/dev/null | tail -1)
    echo "Latest case output (tail -3):"
    if [ -f "$latest_dir/bol/run_log.txt" ]; then
      tail -3 "$latest_dir/bol/run_log.txt" | sed 's/^/  /'
    elif [ -f smoke.log ]; then
      tail -3 smoke.log | sed 's/^/  /'
    else
      echo "  (no log yet)"
    fi
  fi
  echo ""
  echo "Press Ctrl+C to stop. Refreshing in 10 seconds..."
  sleep 10
done
