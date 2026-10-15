#!/usr/bin/env bash
# machine_monitor.sh -- one line per minute on the load of the machine, so
# that a stall of a campaign can be read against what the other users were
# doing at the time. Started by run_c10.sh with the watchdog, usable alone.
#
#   bash machine_monitor.sh OUT.csv [interval_s]      (default 60 s)
#
# Columns, separated by ";":
#   time              local time
#   load1             1-minute load average (64 CPUs: 64 is full)
#   cpu_idle_pct      idle percentage over a 1 s sample
#   mem_free_gb       free memory, GB
#   mem_avail_gb      available memory, GB (free plus reclaimable cache)
#   swap_used_gb
#   n_proc            processes on the machine
#   users             per user with any CPU use or more than 20 processes:
#                     user:processes:cpu_pct:rss_gb  (cpu_pct over the 1 s
#                     sample, 6400 is every CPU busy; rss_gb is the sum of the
#                     resident set sizes of the user's processes, which counts
#                     shared pages once per process, so it exceeds the memory
#                     really used and can exceed the machine's memory: compare
#                     users and follow growth with it, read the absolute from
#                     mem_avail_gb)
set -u
OUT=${1:?usage: machine_monitor.sh OUT.csv [interval_s]}
INTERVAL=${2:-60}
[ -f "$OUT" ] || echo "time;load1;cpu_idle_pct;mem_free_gb;mem_avail_gb;swap_used_gb;n_proc;users(user:processes:cpu_pct:rss_gb)" > "$OUT"
while true; do
  snap=$(top -bn2 -d 1 -w 512 2>/dev/null)
  idle=$(echo "$snap" | grep "^%Cpu" | tail -1 | sed 's/.*, *\([0-9.]*\) id.*/\1/')
  load1=$(cut -d' ' -f1 /proc/loadavg)
  read -r mfree mavail sused <<< "$(free -m | awk 'NR==2{f=$4; a=$7} NR==3{s=$3} END{printf "%.1f %.1f %.1f", f/1024, a/1024, s/1024}')"
  nproc=$(ps -e --no-headers | wc -l)
  # per-user summary from the second top iteration (instantaneous CPU)
  users=$(echo "$snap" | awk '
    /^ *PID +USER/ {hdr++; next}
    hdr==2 && NF>=12 {u=$2; n[u]++; cpu[u]+=$9; mem[u]+=$10}
    END {for (u in n) if (cpu[u] > 0.5 || n[u] > 20) printf "%s:%d:%.0f:%.1f ", u, n[u], cpu[u], mem[u]}' )
  # resident memory per user in GB, from ps (top gives %MEM, this is absolute)
  users=$(ps -eo user:24,rss --no-headers | awk -v U="$users" '
    {r[$1]+=$2}
    END {n=split(U, a, " "); for (i=1;i<=n;i++) {split(a[i], f, ":"); printf "%s:%s:%s:%.1f ", f[1], f[2], f[3], r[f[1]]/1048576}}')
  echo "$(date '+%Y-%m-%d %H:%M:%S');$load1;$idle;$mfree;$mavail;$sused;$nproc;$users" >> "$OUT"
  sleep "$INTERVAL"
done
