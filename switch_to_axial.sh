#!/usr/bin/env bash
# Wait for the valgrid enumeration (stage G) to finish, stop the valgrid run
# before its search does any work, start the axial continuation, and queue
# the valgrid run behind it with EQUAL=1.
cd ~/master-thesis-unipi
echo "$(date '+%F %T') waiting for .valgrid_c9_markers/G"
while [ ! -f .valgrid_c9_markers/G ]; do sleep 20; done
echo "$(date '+%F %T') enumeration done, stopping the valgrid group 618757"
kill -TERM -- -618757; sleep 30; kill -KILL -- -618757 2>/dev/null
pgrep -af "run_valgrid_c9|run_optimization" && echo "WARNING: something survived"
rm -f .valgrid_c9_markers/W .valgrid_c9_markers/S
bash run_c9_axial.sh > c9_axial.log 2>&1 < /dev/null &
AX=$!
echo "$(date '+%F %T') axial runner started, pid $AX"
sleep 5
EQUAL=1 bash run_valgrid_c9.sh "$AX" >> valgrid_c9.log 2>&1 < /dev/null &
echo "$(date '+%F %T') valgrid queued behind $AX, pid $!"
