#!/bin/bash
# watchdog.sh -- shuts down this EC2 instance once the Docker optimization run finishes
# Usage:  bash watchdog.sh
# Run in a SEPARATE tmux window from the one running "docker run ...".
# Cancel any time with Ctrl+C, up until the final shutdown line executes.

IMAGE="labgene-openmc"     # must match the image name in your docker run command
CHECK_EVERY=30             # seconds between checks
GRACE_PERIOD=60            # seconds you have to Ctrl+C before it powers off
LOG="$HOME/master-thesis-unipi/watchdog.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

is_running() {
  [ -n "$(docker ps --filter "ancestor=$IMAGE" --format '{{.ID}}' 2>/dev/null)" ]
}

log "===== watchdog started (watching image: $IMAGE) ====="

# Phase 1: wait for a container to START. Without this, launching the
# watchdog BEFORE "docker run" would see 0 containers immediately and
# shut the instance down before your simulation ever begins.
if is_running; then
  log "container already running -- skipping straight to phase 2"
else
  log "no container yet -- waiting for '$IMAGE' to start..."
  while ! is_running; do
    sleep "$CHECK_EVERY"
  done
  log "container detected running."
fi

# Phase 2: wait for it to FINISH
while is_running; do
  log "still running..."
  sleep "$CHECK_EVERY"
done
log "container finished."

# Phase 3: grace period (cancel window), then shutdown
log "shutting down in $GRACE_PERIOD seconds -- press Ctrl+C now to cancel"
sleep "$GRACE_PERIOD"

if is_running; then
  log "a new container started during the grace period -- aborting shutdown"
  exit 0
fi

log "powering off now."
sudo shutdown -h now
