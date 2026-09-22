#!/usr/bin/env bash
# c10_seed2_queue.sh -- second seed on the 15 feasible designs, then the gate
# and the feasible-region check. Waits for the job already running.
#
# TIME ESTIMATE
#   wait        until your current job ends (you said about 5 h)
#   solves      15 designs x about 250 s = about 65 min (seed 1 is cached)
#   analysis    under 1 min
#   total       about 1 h 10 min after the current job ends
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   1. find the PID of the job you are running now:
#        pgrep -af python
#   2. queue this script behind it (replace 12345 with that PID):
#        setsid nohup bash c10_seed2_queue.sh 12345 > c10_seed2.log 2>&1 < /dev/null &
#      Without a PID it waits until the machine is idle for 15 minutes:
#        setsid nohup bash c10_seed2_queue.sh > c10_seed2.log 2>&1 < /dev/null &
#   3. check it any time:   tail -3 c10_seed2.log
#   4. the result:          cat c10_seed2_report.txt
set -u -o pipefail

WAIT_PID="${1:-}"
DESIGNS="1 10 11 16 24 25 26 27 29 30 32 42 51 54 57"
N_SEED2_EXPECTED=22           # 7 already had seed 2, plus these 15
THREADS=64
SEC_PER_SOLVE=250
CKPT=out_c9/optimization_checkpoint.json
OUT=axial_c9_all
REPORT=c10_seed2_report.txt
IDLE_LOAD=8                   # 1-minute load average counted as idle
IDLE_MINUTES=15               # consecutive idle minutes before starting

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
die()   { echo "$(stamp) FAIL: $*"; exit 1; }

# ------------------------------------------------------------ preflight ----
echo "$(stamp) preflight"
[ "$(hostname)" = "wks720" ]                || die "not on wks720"
[ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
python -c "import numpy, openmc; print('env ok', openmc.__version__)" || die "env check failed"
for f in "$CKPT" axial_shape_c9.py c10_fz_gate.py "$OUT"; do
  [ -e "$f" ] || die "missing $f (run from ~/master-thesis-unipi)"
done
if [ -n "$WAIT_PID" ]; then
  kill -0 "$WAIT_PID" 2>/dev/null || die "PID $WAIT_PID is not running. Check it with: pgrep -af python"
  echo "$(stamp) will wait for PID $WAIT_PID: $(ps -o args= -p "$WAIT_PID" | cut -c1-100)"
else
  echo "$(stamp) no PID given, will wait for $IDLE_MINUTES idle minutes (load below $IDLE_LOAD)"
fi

# ----------------------------------------------------------------- wait ----
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "$(stamp) PID $WAIT_PID has ended"
  sleep 60                    # let the finished job release memory and files
else
  idle=0
  while [ "$idle" -lt "$IDLE_MINUTES" ]; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" -v t="$IDLE_LOAD" 'BEGIN{exit !(l < t)}'; then
      idle=$((idle + 1))
    else
      idle=0
    fi
    sleep 60
  done
  echo "$(stamp) machine idle for $IDLE_MINUTES minutes"
fi

# --------------------------------------------------------------- solves ----
n=$(echo $DESIGNS | wc -w)
eta=$(date -d "+$((n * SEC_PER_SOLVE)) seconds" '+%H:%M')
echo "$(stamp) START $n seed-2 solves, expected end about $eta"
python -c "import numpy, openmc; print('env ok')" || die "env check failed at start"
# shellcheck disable=SC2086
python -u axial_shape_c9.py --checkpoint "$CKPT" --designs $DESIGNS --states ARO \
    --seeds 2 --threads "$THREADS" --boron-ppm 1000 --out "$OUT" \
  || die "solves failed. Relaunch the same command: finished solves are cached."

got=$(ls "$OUT"/d*/ARO_s2.npz 2>/dev/null | wc -l)
echo "$(stamp) designs with seed 2: $got (expected $N_SEED2_EXPECTED)"
[ "$got" -ge "$N_SEED2_EXPECTED" ] || die "only $got designs have seed 2"

# ------------------------------------------------------------- analysis ----
{
  echo "=== C10 gate after the second seed, $(stamp)"
  echo
  echo "=== 1. gate over all 60 designs"
  python c10_fz_gate.py --axial-dir "$OUT" --checkpoint "$CKPT" --out c10_gate \
    | grep -v "^wrote"
  echo
  echo "=== 2. feasible region only"
  python - <<'EOF'
import json, numpy as np
from scipy.stats import spearmanr
r = json.load(open('c10_gate/fz_gate.json'))
t = [x for x in r['table'] if x['feasible']]
fz1 = np.array([x['fz'][0] for x in t])
fz2 = np.array([np.mean(x['fz'][:2]) for x in t])
e = np.array([x['enrich'] for x in t])
sd = r['sd_seed']
def snr(v, noise):
    obs = v.std(ddof=1); true = np.sqrt(max(obs**2 - noise**2, 0.0))
    return obs, true, true / noise
o1, t1, s1 = snr(fz1, sd)
o2, t2, s2 = snr(fz2, sd / np.sqrt(2))
print(f"feasible designs : {len(t)}   seed s.d. {sd:.4f} ({r['sd_seed_dof']} degrees of freedom)")
print(f"enrichment       : {e.min():.2f} to {e.max():.2f} wt%")
print(f"F_z, one seed    : span {100*np.ptp(fz1)/fz1.mean():.2f} %   true s.d. {t1:.4f}   SNR {s1:.2f}")
print(f"F_z, two seeds   : span {100*np.ptp(fz2)/fz2.mean():.2f} %   true s.d. {t2:.4f}   SNR {s2:.2f}")
print(f"Spearman F_z vs enrichment: {spearmanr(e, fz2)[0]:+.2f}")
# the same without the highest-enrichment design, to see if one outlier drives it
k = np.argsort(e)[:-1]
_, _, s3 = snr(fz2[k], sd / np.sqrt(2))
print(f"two seeds, without C9-{t[int(np.argmax(e))]['idx']} ({e.max():.2f} wt%): "
      f"span {100*np.ptp(fz2[k])/fz2[k].mean():.2f} %   SNR {s3:.2f}")
print()
print(f"{'design':>7} {'enrich':>7} {'F_z s1':>8} {'F_z s2':>8}")
for x in sorted(t, key=lambda x: x['enrich']):
    f = x['fz'] + [float('nan')]
    print(f"  C9-{x['idx']:<3d} {x['enrich']:7.2f} {f[0]:8.4f} {f[1]:8.4f}")
EOF
} 2>&1 | tee "$REPORT"

echo "$(stamp) DONE. Report: $REPORT"
