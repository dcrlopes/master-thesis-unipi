#!/usr/bin/env bash
# run_gdstudy.sh -- the designed gadolinia study: which quantity governs the
# axial burnup loss, the gadolinia weight fraction, the poisoned-pin count or
# the gadolinia inventory (their product).
#
# WHAT
#   Five designs at the gadolinia inventory of design 70 of the first
#   continuation, gd_wt x gd_pins = 4.407 x 20 = 88.1 wt%.rods, on the pin
#   ladder 12, 16, 24, 32 and 40, with the enrichment (4.868 wt%) and the
#   reflector thickness (5.660 cm, the upper bound) of design 70. Design 70
#   itself, whose 19.37 pins snap to 20, is the sixth point and is already
#   measured (ratio 0.806). Each design is evaluated by the Campaign 9
#   evaluator, which gives the assembly cycle length and the reactivity hump,
#   and then depleted with eight axial layers, which gives the 3D cycle.
#
# WHY
#   The three single-regressor fits on the eleven measured designs predict
#   opposite slopes of the ratio E_3D / E_asm along this ladder:
#       pins            12     16     20     24     32     40
#       gd_wt fit     0.754  0.789  0.810  0.824  0.842  0.853   rises
#       gd_pins fit   0.843  0.826  0.810  0.793  0.760  0.726   decreases
#       inventory fit 0.812  0.812  0.812  0.812  0.812  0.812   constant
#   One eight-layer depletion resolves the ratio to about 0.004 (7 to 10 d),
#   so the sign of the slope decides between the three.
#
# TIME (wks720, 64 threads)
#   stage E   5 evaluations, about 19 min each            about 1.6 h
#   stage D   5 eight-layer depletions, about 1 h each    about 5 h
#
# STAGES, each skipped when its marker exists
#   P preflight and list   W wait for a PID   E evaluate   D deplete   F report
#
# USAGE, from ~/master-thesis-unipi in the activated openmc-env
#   bash run_gdstudy.sh --preflight
#   setsid nohup bash run_gdstudy.sh > gdstudy.log 2>&1 < /dev/null &
#   setsid nohup bash run_gdstudy.sh 12345 > gdstudy.log 2>&1 < /dev/null &   # wait for that PID first
#   bash run_gdstudy.sh --status
set -u -o pipefail

THREADS=${THREADS:-64}
PINS=${PINS:-"12 16 24 32 40"}
ANCHOR_CKPT=${ANCHOR_CKPT:-out_c9f/optimization_checkpoint.json}
ANCHOR=${ANCHOR:-70}
OUT=${OUT:-out_gdstudy}
WORK=${WORK:-openmc_runs_gdstudy}
DEP=${DEP:-gdstudy_dep_core3d}
LIST=gdstudy/list.json
MARK=.gdstudy_markers
REPORT=gdstudy_report.txt

# the Campaign 9 problem, as in run_c9.sh
COMMON=(--ktarget-table ktarget_table_c8.json --k-basis core --k-max 1.166 --k-min 1.02
        --f-max 1.65 --enr-max 16 --ctrl-margin 1000
        --objective-set c9 --efpd-req 1826 --boron-objective floor
        --boron-step 2000 --boron-top 3000 --boron-ceiling 2763 --hump-noise 400
        --threads "$THREADS")

mkdir -p "$MARK" gdstudy
die()   { echo "FAIL: $*" >&2; exit 1; }
hr()    { printf '%.0s-' $(seq 1 74); echo; }
mark()  { touch "$MARK/$1"; }
done_() { [ -f "$MARK/$1" ]; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
n_arch() { python -c "import json, sys; print(len(json.load(open(sys.argv[1]))['all_raw']))" "$OUT/optimization_checkpoint.json" 2>/dev/null || echo 0; }
refl_of_anchor() { python -c "import json, sys; print(json.load(open(sys.argv[1]))['all_raw'][int(sys.argv[2])]['refl_thick'])" "$ANCHOR_CKPT" "$ANCHOR"; }

case "${1:-}" in
--status)
  echo "time        : $(stamp)   host $(hostname)"
  p=$(pgrep -f "^bash run_gdstudy.sh( [0-9]+)?$" | head -1)
  [ -n "$p" ] && echo "runner      : ALIVE (pid $p)" || echo "runner      : not running"
  st=""; for s in P W E D F; do done_ $s && st="$st $s"; done
  echo "stages done :${st:- none}"
  echo "evaluated   : $(n_arch) of $(echo $PINS | wc -w)"
  [ -f "$DEP/runs.json" ] && echo "depleted    : $(python -c "import json, sys; print(' '.join(sorted(json.load(open(sys.argv[1])))))" "$DEP/runs.json")"
  exit 0 ;;
esac

stage_P() {
  hr; echo " STAGE P. Preflight and design list  ($(stamp))"; hr
  [ "$(hostname)" = "wks720" ] || die "not on wks720"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "conda env is not openmc-env"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] || die "OpenMC is not 0.15.3"
  grep -q "spec.design_space.names" c9_dep_core3d.py || die "c9_dep_core3d.py lacks the frozen-checkpoint fix"
  python c9_dep_core3d.py --selftest || die "c9_dep_core3d selftest failed"
  python - "$ANCHOR_CKPT" "$ANCHOR" "$LIST" $PINS <<'PY' || die "the design list could not be written"
import json, sys
ck, idx, out, pins = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(p) for p in sys.argv[4:]]
import reactor_model as rm
r = json.load(open(ck))["all_raw"][idx]
n0 = rm.snap_gd_pins(float(r["gd_pins"]))
inv = float(r["gd_wt"]) * n0
designs = [dict(enrich=float(r["enrich"]), gd_wt=round(inv / n, 6), gd_pins=float(n)) for n in pins]
for d in designs:
    assert 0.0 <= d["gd_wt"] <= 8.0, f"gd_wt {d['gd_wt']} outside the search box"
    assert rm.snap_gd_pins(d["gd_pins"]) == int(d["gd_pins"]), "a listed pin count is not on the ladder"
json.dump({"designs": designs, "anchor": {"checkpoint": ck, "index": idx, "gd_pins_snapped": n0,
                                          "inventory": inv, "refl_thick": float(r["refl_thick"])}},
          open(out, "w"), indent=1)
print(f"anchor {ck}[{idx}]: enrich {float(r['enrich']):.3f}, gd_wt {float(r['gd_wt']):.3f}, "
      f"pins {float(r['gd_pins']):.2f} -> {n0}, inventory {inv:.1f}, refl {float(r['refl_thick']):.4f}")
for d in designs:
    print(f"  pins {int(d['gd_pins']):2d}  gd_wt {d['gd_wt']:.3f}")
PY
  echo "  refl_thick frozen at $(refl_of_anchor) cm"
  echo "  threads     : $THREADS of $(nproc)"
  echo "  other jobs  : $(pgrep -fc '[r]un_optimization.py' || true) run_optimization.py processes (depletion workers included)"
  echo "preflight OK"
  mark P
}

stage_W() {
  local pid="$1"
  done_ W && return 0
  kill -0 "$pid" 2>/dev/null || die "PID $pid is not running"
  echo "waiting for PID $pid  ($(stamp))"
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  echo "PID $pid finished  ($(stamp))"
  mark W
}

stage_E() {
  done_ E && { echo "[E] already done"; return 0; }
  hr; echo " STAGE E. Evaluate the listed designs  ($(stamp))"; hr
  local resume=()
  [ -f "$OUT/optimization_checkpoint.json" ] && resume=(--resume "$OUT/optimization_checkpoint.json")
  python -u run_optimization.py --out "$OUT" --workdir "$WORK" "${COMMON[@]}" \
    --eval-list "$LIST" --freeze "refl_thick=$(refl_of_anchor)" "${resume[@]}" \
    2>&1 | tee -a "$OUT.log" || die "evaluation failed"
  mark E
}

stage_D() {
  done_ D && { echo "[D] already done"; return 0; }
  hr; echo " STAGE D. Eight-layer depletion of every evaluated design  ($(stamp))"; hr
  local idx; idx=$(python -c "import json, sys; print(' '.join(str(i) for i in range(len(json.load(open(sys.argv[1]))['all_raw']))))" "$OUT/optimization_checkpoint.json")
  [ -n "$idx" ] || die "nothing was evaluated"
  python -u c9_dep_core3d.py --checkpoint "$OUT/optimization_checkpoint.json" \
    --designs $idx --layers 8 --threads "$THREADS" --out "$DEP" \
    2>&1 | tee -a "$DEP.log" || die "depletion failed"
  mark D
}

stage_F() {
  hr; echo " REPORT  ($(stamp))"; hr
  python c9_dep_core3d.py --checkpoint "$OUT/optimization_checkpoint.json" --out "$DEP" --analyse >> "$DEP.log" 2>&1 || echo "  WARNING aggregation failed"
  python - "$OUT/optimization_checkpoint.json" "$DEP/runs.json" "$ANCHOR_CKPT" "$ANCHOR" <<'PY' | tee "$REPORT"
import json, sys
import numpy as np
import reactor_model as rm
import axial_ratio_model as arm
raw = json.load(open(sys.argv[1]))["all_raw"]
runs = {int(r["idx"]): r for r in json.load(open(sys.argv[2])).values()}
anc = json.load(open(sys.argv[3]))["all_raw"][int(sys.argv[4])]
rows = []
for i, r in enumerate(raw):
    if i in runs:
        rows.append((rm.snap_gd_pins(float(r["gd_pins"])), float(r["gd_wt"]), float(r["cycle_length"]),
                     float(r["hump_asm_pcm"]), runs[i]["efpd"], runs[i]["sigma_efpd"], f"gdstudy/{i}"))
m70 = arm.measured_lookup(arm.fit(arm.measurements(), []), anc)
if m70:
    rows.append((rm.snap_gd_pins(float(anc["gd_pins"])), float(anc["gd_wt"]), float(anc["cycle_length"]),
                 float(anc["hump_asm_pcm"]), m70["efpd_3d"], m70["sigma_3d"], "design 70"))
rows.sort()
print("Designed gadolinia study: fixed inventory, pin ladder\n")
print(f"{'pins':>4s} {'gd_wt':>6s} {'E_asm':>6s} {'hump':>6s} {'E_3D':>6s} {'sigma':>5s} {'ratio':>6s}  source")
for n, g, E, h, e3, s, src in rows:
    print(f"{n:4d} {g:6.3f} {E:6.0f} {h:6.0f} {e3:6.0f} {s:5.1f} {e3 / E:6.3f}  {src}")
x = np.array([r[0] for r in rows], float)
y = np.array([r[4] / r[2] for r in rows])
w = np.array([r[2] / max(r[5], 1.0) for r in rows]) ** 2          # 1 / var(ratio)
A = np.column_stack([np.ones_like(x), x])
b, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
res = y - A @ b
se = np.sqrt(max(np.sum(w * res ** 2) / max(len(x) - 2, 1), 1.0) * np.linalg.inv((A * w[:, None]).T @ A)[1, 1])
print(f"\nslope of the ratio against the pin count: {b[1]:+.5f} +/- {se:.5f} per pin")
print("predictions: gd_wt fit +0.0036 per pin (rises), gd_pins fit -0.0042 (decreases), inventory fit 0 (constant)")
t = b[1] / se if se > 0 else float('inf')
verdict = ("the ratio rises with the pin count: the weight fraction governs" if t > 2 else
           "the ratio decreases with the pin count: the pin count governs" if t < -2 else
           "no slope within two standard errors: the inventory governs")
print(f"verdict at two standard errors: {verdict}  (t = {t:+.1f})")
PY
  echo
  echo "the leave-one-out table with the new designs included:"
  python axial_ratio_model.py --report
  echo
  echo "push:"
  echo "  git add -f $OUT/optimization_checkpoint.json $DEP/runs.json $DEP/summary.json $DEP/summary.txt"
  echo "  git add $LIST $REPORT"
  echo "  git add -f $OUT.log $DEP.log gdstudy.log"
  echo "  git commit -m 'Designed gadolinia study: fixed inventory on the pin ladder, eight-layer depletions'"
  mark F
}

case "${1:-}" in
  --preflight) stage_P ;;
  ""|[0-9]*)   stage_P; [ -n "${1:-}" ] && stage_W "$1"; stage_E; stage_D; stage_F ;;
  *)           die "unknown option $1" ;;
esac
