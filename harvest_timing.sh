#!/usr/bin/env bash
# harvest_timing.sh -- collect the SMALL metadata needed to reconstruct the
# wall-clock cost of past campaigns, on the machine where they ran.
#
# It copies NO statepoint payloads. It collects:
#   - an index of every .h5 file (path, modification time, size)
#   - every optimization checkpoint / results JSON it can find
#   - run logs that mention the optimiser
#   - host facts (CPU model, core count, reboot history, versions, crontab)
#   - git branch and last commits of each repository that holds a work tree
#
# usage:
#   bash harvest_timing.sh LABEL [SEARCH_ROOT]
#     LABEL        short machine tag for the bundle name, e.g. wks720, aws-c1
#     SEARCH_ROOT  tree to scan (default: your home directory)
# result:
#   $HOME/timing_bundle_LABEL.tar.gz   (text only, usually a few MB)

set -u
LABEL="${1:?usage: bash harvest_timing.sh LABEL [SEARCH_ROOT]}"
ROOT="${2:-$HOME}"
OUT="$HOME/timing_harvest_$LABEL"
rm -rf "$OUT"
mkdir -p "$OUT/checkpoints" "$OUT/logs"

echo "== [1/7] host facts"
{
  echo "label: $LABEL"
  echo "hostname: $(hostname)"
  echo "harvested_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "search_root: $ROOT"
  uname -a
  echo "nproc: $(nproc 2>/dev/null || echo '?')"
} > "$OUT/hostinfo.txt"
lscpu >> "$OUT/hostinfo.txt" 2>/dev/null || true
last -x reboot 2>/dev/null | head -25 > "$OUT/reboots.txt" || true
crontab -l > "$OUT/crontab.txt" 2>/dev/null || true
python3 - <<'PY' > "$OUT/versions.txt" 2>&1 || true
import sys
print("python", sys.version.split()[0])
try:
    import openmc
    print("openmc", openmc.__version__)
except Exception as exc:
    print("openmc not importable from this shell:", exc)
PY

echo "== [2/7] locating campaign work trees under $ROOT (can take a minute)"
find "$ROOT" -maxdepth 6 -type d -name 'case_0000' 2>/dev/null \
  | sed 's|/case_0000$||' | sort -u > "$OUT/workdirs.txt"
cat "$OUT/workdirs.txt"
if [ ! -s "$OUT/workdirs.txt" ]; then
  echo "   (none found. If the runs live inside a Docker container, run this"
  echo "    script INSIDE the container, or against the bind-mounted path.)"
fi

echo "== [3/7] indexing every .h5 file (path, mtime, size). NO payload copied"
: > "$OUT/statepoint_index.txt"
while IFS= read -r d; do
  find "$d" -type f -name '*.h5' -printf '%T@ %s %p\n' 2>/dev/null
done < "$OUT/workdirs.txt" | sort -n >> "$OUT/statepoint_index.txt"
echo "   $(wc -l < "$OUT/statepoint_index.txt") files indexed"

echo "== [4/7] work tree sizes (for the git decision, nothing is copied)"
while IFS= read -r d; do du -sh "$d" 2>/dev/null; done \
  < "$OUT/workdirs.txt" > "$OUT/workdir_sizes.txt"
cat "$OUT/workdir_sizes.txt"

echo "== [5/7] checkpoints and result JSONs"
find "$ROOT" -maxdepth 7 -type f \
     \( -name 'optimization_checkpoint*.json' \
        -o -name 'optimization_results*.json' \
        -o -name 'pareto*.json' -o -name 'pareto*.csv' \) \
     -size -80M 2>/dev/null | sort > "$OUT/checkpoint_paths.txt"
i=0
while IFS= read -r f; do
  i=$((i + 1))
  cp "$f" "$OUT/checkpoints/$(printf '%02d' "$i")__$(basename "$(dirname "$f")")__$(basename "$f")"
done < "$OUT/checkpoint_paths.txt"
echo "   $i files copied"

echo "== [6/7] run logs that mention the optimiser"
{ while IFS= read -r d; do dirname "$d"; done < "$OUT/workdirs.txt"
  echo "$ROOT"; } | sort -u > "$OUT/roots.txt"
: > "$OUT/log_paths.txt"
while IFS= read -r r; do
  find "$r" -maxdepth 2 -type f \
       \( -name '*.log' -o -name 'nohup.out' -o -name '*.out' \) \
       -size -20M 2>/dev/null
done < "$OUT/roots.txt" | sort -u >> "$OUT/log_paths.txt"
i=0
while IFS= read -r f; do
  if grep -qa -e 'Done in' -e 'real eval' -e 'Stage' -e 'RESUME' "$f" 2>/dev/null; then
    i=$((i + 1))
    cp "$f" "$OUT/logs/$(printf '%02d' "$i")__$(basename "$f")"
  fi
done < "$OUT/log_paths.txt"
echo "   $i logs copied"

echo "== [7/7] git provenance of each repository holding a work tree"
: > "$OUT/gitinfo.txt"
while IFS= read -r r; do
  if [ -d "$r/.git" ]; then
    {
      echo "repo: $r"
      echo "branch: $(git -C "$r" branch --show-current 2>/dev/null)"
      git -C "$r" log -10 --format='%ci | %h | %s' 2>/dev/null
      echo
    } >> "$OUT/gitinfo.txt"
  fi
done < "$OUT/roots.txt"

tar -czf "$HOME/timing_bundle_$LABEL.tar.gz" -C "$HOME" "timing_harvest_$LABEL"
echo
echo "DONE -> $HOME/timing_bundle_$LABEL.tar.gz " \
     "($(du -h "$HOME/timing_bundle_$LABEL.tar.gz" | cut -f1))"
echo "Copy this one file back and upload it to the chat."
