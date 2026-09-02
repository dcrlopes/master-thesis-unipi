#!/usr/bin/env bash
# =============================================================================
# run_c8_post.sh -- Campaign 8 post-analysis, all remaining runs, in order.
# Revision 2, 2 September 2026. Adds the open items of
# corrections_and_next_steps_2026-09-02.md that the current code can reach.
#
# Working set, 11 designs:
#   FRONT (8) 47 42 23 29 21 44 59 1   the feasible Pareto front of Campaign 8
#   KEEP  (3) 53 31 13                 feasible, dominated, and the only designs
#                                      whose regulating banks hold a PWR-like
#                                      2544 to 4001 pcm at power. The front
#                                      needs 8918 to 11651 pcm (item C3.5).
#
# Stages
#   0  entropy audit of the three flagged front solves      ~1 min   0 solves
#   A  repair design 53, smoke-contaminated cache entries   ~8 min   2 solves
#   D  zero-boron ALL-RE margin, 2D and 3D, all 11          ~3.1 h  44 solves
#   B  boron sweep on the three unswept front members       ~1.4 h  36 solves
#   C  1000 ppm 3D confirmation, five members lacking one   ~4.2 h  60 solves
#   E  parked-rod sensitivity, design 47 ARO                ~0.3 h   2 solves
#   F  g_ctrl re-score of designs 54 and 11, 3 seeds        ~0.3 h   6 solves
#   G  downcomer 3.7 cm sensitivity, ARO (OPTIONAL)         ~0.9 h  12 solves
#                                       required total     ~9.5 h 150 solves
#                                       with stage G      ~10.4 h 162 solves
#
# NOT covered, deliberately: the cold-state all-CRAs-minus-one shutdown check
# of item C2. It needs a shutdown-bank position set (absent from zoning.py),
# stuck-rod logic (absent), and a temperature-dependent water density
# (make_water ignores its T argument and hardcodes 0.72 g/cm3 at
# reactor_model.py line 145). That is new code plus a density source, not a run.
#
# Resumable at two levels: finished stages are skipped via .c8_markers/,
# interrupted stages resume from the runs.json of their own output directory.
#
# Native conda job. Do NOT prefix with the Docker `lab` alias.
#
# USAGE
#   cd ~/master-thesis-unipi && conda activate openmc-env
#   bash run_c8_post.sh --preflight              # checks only, runs nothing
#   setsid nohup bash run_c8_post.sh > run_c8_post.log 2>&1 < /dev/null &
#   RUN_OPTIONAL=1 setsid nohup bash run_c8_post.sh > ... to include stage G
# =============================================================================
set -u -o pipefail

CKPT="out_c8/optimization_checkpoint.json"
THREADS=32
FRONT="47 42 23 29 21 44 59 1"
KEEP="53 31 13"
ALL="$FRONT $KEEP"
NEW3="42 29 59"           # front members with no boron sweep and no 3D run
NO3D="42 29 59 44 1"      # front members with no 3D confirmation at 1000 ppm
MARGINAL="54 11"          # fail g_ctrl by 17 and 63 pcm, within solve resolution
DOWNC="47 21 59"          # thick-reflector front members, for the downcomer test
REFL_37="3.969"           # reflector budget giving a 3.7 cm downcomer at pitch
                          # 1.26. VERIFY against the Campaign 7 note before
                          # trusting stage G. Derivation from the d47 log line:
                          # downcomer = 90 - (R_env + refl + barrel)
                          #           = 90 - (77.231 + refl + 5.08)
                          # so refl = 3.989 for 3.7 cm, less the geometry pad.
RUN_OPTIONAL="${RUN_OPTIONAL:-0}"
MARK=".c8_markers"
NEED_GB=28

say() { echo "$(date -Is)  $*"; }
die() { echo "FAIL: $*" >&2; exit 1; }

# --------------------------------------------------------------- preflight --
preflight() {
  echo "=============================================================="
  echo " PREFLIGHT  $(date -Is)"
  echo "=============================================================="
  echo "  host            : $(hostname)"
  echo "  cwd             : $(pwd)"
  echo "  conda env       : ${CONDA_DEFAULT_ENV:-NONE}"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] \
    || die "wrong conda environment. Run: conda activate openmc-env"

  python -c "import openmc, numpy" 2>/dev/null \
    || die "openmc or numpy not importable. You are probably in (base)."
  echo "  python          : $(python -c 'import sys;print(sys.version.split()[0])')"
  echo "  openmc          : $(python -c 'import openmc;print(openmc.__version__)')"

  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] || die "OPENMC_CROSS_SECTIONS is not exported"
  [ -f "$OPENMC_CROSS_SECTIONS" ]     || die "cross sections not found: $OPENMC_CROSS_SECTIONS"
  echo "  cross sections  : $OPENMC_CROSS_SECTIONS"
  echo "  chain           : ${OPENMC_CHAIN_FILE:-not set} (no depletion in these stages)"

  local br; br=$(git branch --show-current 2>/dev/null)
  echo "  git branch      : ${br:-NOT A REPO}"
  [ "$br" = "campaign8" ] || die "expected branch campaign8, got '${br:-none}'"

  [ -f "$CKPT" ] || die "checkpoint not found: $CKPT"
  echo "  checkpoint      : $CKPT ($(du -h "$CKPT" | cut -f1))"

  for f in confirm3d.py boron_worth.py zoning.py hardware3d.py reactor_model.py; do
    [ -f "$f" ] || die "missing script: $f"
  done
  echo "  scripts         : present"
  [ -f check_entropy.py ] || echo "  NOTE: check_entropy.py absent, stage 0 will be skipped"

  python confirm3d.py --help 2>/dev/null | grep -q -- "--boron-ppm" \
    || die "confirm3d.py has no --boron-ppm flag. Run: python fix_confirm3d_cachekey.py --apply"
  echo "  cache-key patch : applied (--boron-ppm present)"

  if d53_clean; then
    echo "  d53 state       : clean, stage A will be skipped"
  else
    echo "  d53 state       : needs repair, stage A will rerun it (about 8 min)"
  fi

  local nc; nc=$(nproc)
  echo "  cores available : $nc  (requesting $THREADS)"
  [ "$nc" -ge "$THREADS" ] || die "only $nc cores, THREADS=$THREADS"

  local free_gb; free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  echo "  free disk       : ${free_gb} GB  (need >= ${NEED_GB} GB)"
  [ "$free_gb" -ge "$NEED_GB" ] \
    || die "only ${free_gb} GB free. Delete old d*/ directories under boron_c8 or confirm3d_c8."

  echo "  FRONT           : $FRONT"
  echo "  KEEP            : $KEEP"
  echo "  optional stage G: $([ "$RUN_OPTIONAL" = "1" ] && echo enabled || echo disabled)"
  echo "=============================================================="
}

stage_done() { [ -f "$MARK/$1" ]; }
mark()       { mkdir -p "$MARK"; date -Is > "$MARK/$1"; }

# Design 53 ARO seed 0 was contaminated by a --smoke run that shared its cache
# key (5000 x 40 reused by a 150000 x 200 run). Clean means: four production
# ARO entries in runs.json, a peaking below 1.8, and two seeds averaged in
# both ARO modes. Returns 0 when clean, 1 when the repair is needed. Quiet.
d53_clean() {
  python3 - << 'PYEOF' > /dev/null 2>&1
import json, os, sys
dr, ds = "confirm3d_c8/runs.json", "confirm3d_c8/summary.json"
if not (os.path.exists(dr) and os.path.exists(ds)): sys.exit(1)
runs = json.load(open(dr))
n = sum(1 for k in runs
        if k.startswith("53|ARO|") and k.endswith("|150000|200|80|1000.0"))
try:
    r = json.load(open(ds))["53"]
except (KeyError, ValueError):
    sys.exit(1)
sys.exit(0 if (n == 4 and r["ARO_2D"]["F"] < 1.8
               and r["ARO_2D"]["n"] == 2 and r["ARO_3Dhw"]["n"] == 2) else 1)
PYEOF
}

# ------------------------------------------------------------------- main --
if [ "${1:-}" = "--preflight" ]; then preflight; echo "preflight OK, nothing was run."; exit 0; fi
preflight || exit 1
mkdir -p "$MARK"
say "START run_c8_post.sh revision 2"
T0=$SECONDS

# --- stage 0: entropy audit, item C3.1 -------------------------------------
if stage_done Z_entropy || [ ! -f check_entropy.py ]; then
  say "STAGE 0 skipped"
else
  say "STAGE 0  entropy audit of the three flagged front solves"
  python check_entropy.py confirm3d_c8/d23/ARO_2D_s1 confirm3d_c8/d23/ARO_3Dhw_s1 \
      confirm3d_c8/d21/ARO_2D_s0 confirm3d_c8/d47/ARO_2D_s0 \
    || say "STAGE 0 returned non-zero, continuing (diagnostic only)"
  mark Z_entropy
fi

# --- stage A: repair design 53, item from the cache-key collision ----------
# Must run before stage C, which writes into the same output directory.
if stage_done A_fix53; then
  say "STAGE A already done, skipping"
else
  if d53_clean; then
    say "STAGE A  design 53 already clean, nothing to rerun"
  else
    say "STAGE A  repair design 53, 2 new solves plus 10 from cache, about 8 min"
    python -u confirm3d.py --checkpoint "$CKPT" \
        --designs 53 --states ARO ARI RE12 --seeds 2 --threads $THREADS \
        --out confirm3d_c8 \
      || die "stage A failed. Relaunch this script to resume."
  fi
  python3 - << 'PYEOF' || die "design 53 is still not clean after stage A. Stop and inspect before stage C."
import json, sys
runs = json.load(open("confirm3d_c8/runs.json"))
r = json.load(open("confirm3d_c8/summary.json"))["53"]
n = sum(1 for k in runs
        if k.startswith("53|ARO|") and k.endswith("|150000|200|80|1000.0"))
print(f"  d53 verify: {len(runs)} cache entries, {n} of 4 production ARO solves, "
      f"F_2D {r['ARO_2D']['F']:.3f} (want ~1.500), seeds {r['ARO_2D']['n']}/{r['ARO_3Dhw']['n']}, "
      f"L_ax {r['ARO_Lax_hw']:.4f} (want ~1.0312)")
sys.exit(0 if (n == 4 and r["ARO_2D"]["F"] < 1.8
               and r["ARO_2D"]["n"] == 2 and r["ARO_3Dhw"]["n"] == 2) else 1)
PYEOF
  mark A_fix53; say "STAGE A complete"
fi

# --- stage D: zero-boron ALL-RE margin, all 11 -----------------------------
if stage_done D_zero_boron; then say "STAGE D already done, skipping"; else
  say "STAGE D  zero-boron ALL-RE margin, 11 designs, 44 solves, about 3.1 h"
  python -u confirm3d.py --checkpoint "$CKPT" \
      --designs $ALL --states ARI --seeds 2 --threads $THREADS \
      --boron-ppm 0 --out confirm3d_c8_0ppm \
    || die "stage D failed. Relaunch this script to resume."
  mark D_zero_boron; say "STAGE D complete"
fi

# --- stage B: boron sweep on the three unswept front members ---------------
if stage_done B_boron_new3; then say "STAGE B already done, skipping"; else
  say "STAGE B  boron sweep on $NEW3, 36 solves, about 1.4 h"
  python -u boron_worth.py --checkpoint "$CKPT" \
      --designs $NEW3 --threads $THREADS --out boron_c8 \
    || die "stage B failed. Relaunch this script to resume."
  mark B_boron_new3; say "STAGE B complete"
fi

# --- stage C: 1000 ppm 3D confirmation, item N3 ----------------------------
if stage_done C_confirm3d_1000; then say "STAGE C already done, skipping"; else
  say "STAGE C  1000 ppm 3D confirmation on $NO3D, 60 solves, about 4.2 h"
  python -u confirm3d.py --checkpoint "$CKPT" \
      --designs $NO3D --states ARO ARI RE12 --seeds 2 --threads $THREADS \
      --out confirm3d_c8 \
    || die "stage C failed. Relaunch this script to resume."
  mark C_confirm3d_1000; say "STAGE C complete"
fi

# --- stage E: parked-rod sensitivity ---------------------------------------
if stage_done E_no_parked; then say "STAGE E already done, skipping"; else
  say "STAGE E  parked-rod sensitivity, design 47 ARO, 2 solves, about 20 min"
  python -u confirm3d.py --checkpoint "$CKPT" \
      --designs 47 --states ARO --seeds 2 --threads $THREADS \
      --no-parked-rods --out confirm3d_c8_noparked \
    || die "stage E failed. Relaunch this script to resume."
  mark E_no_parked; say "STAGE E complete"
fi

# --- stage F: g_ctrl re-score of the marginal designs, item C3.2 -----------
# Both are dominated on the objectives even if they turn feasible (54 by 21
# and 44, 11 by 44), so this settles the feasibility count only.
if stage_done F_marginal; then say "STAGE F already done, skipping"; else
  say "STAGE F  g_ctrl re-score of $MARGINAL, 3 seeds, 6 solves, about 15 min"
  python -u boron_worth.py --checkpoint "$CKPT" \
      --designs $MARGINAL --ppm 1000 --states ARI --seeds 3 \
      --threads $THREADS --out boron_c8_marginal \
    || die "stage F failed. Relaunch this script to resume."
  mark F_marginal; say "STAGE F complete"
fi

# --- stage G: downcomer 3.7 cm, item N4, OPTIONAL --------------------------
if [ "$RUN_OPTIONAL" != "1" ]; then
  say "STAGE G skipped (set RUN_OPTIONAL=1 to enable)"
elif stage_done G_downcomer; then say "STAGE G already done, skipping"; else
  say "STAGE G  downcomer 3.7 cm on $DOWNC, refl override $REFL_37, 12 solves, about 50 min"
  python -u confirm3d.py --checkpoint "$CKPT" \
      --designs $DOWNC --states ARO --seeds 2 --threads $THREADS \
      --refl-override "$REFL_37" --out confirm3d_c8_dc37 \
    || die "stage G failed. Relaunch this script to resume."
  mark G_downcomer; say "STAGE G complete"
fi

# ------------------------------------------------------------- roll-up ----
say "ALL STAGES COMPLETE in $(( (SECONDS-T0)/3600 ))h $(( ((SECONDS-T0)%3600)/60 ))m"
echo
echo "=============================================================="
echo " ZERO-BORON ALL-RE MARGIN, MEASURED (stage D)"
echo "=============================================================="
python3 - << 'PYEOF'
import json, os
p = "confirm3d_c8_0ppm/summary.json"
if not os.path.exists(p): print("  stage D summary not found"); raise SystemExit
s = json.load(open(p)); FRONT = {"47","42","23","29","21","44","59","1"}
rows = [(i, r["design"]["enrich_inner"], r["ARI_margin2D_pcm"], r["ARI_margin3D_pcm"],
         r["ARI_Lax_hw"], i in FRONT) for i, r in s.items() if "ARI_margin2D_pcm" in r]
rows.sort(key=lambda t: t[1])
print(f"{'idx':>4} {'enr':>6} {'M16(0) 2D':>10} {'M16(0) 3D':>10} {'Lax':>7}  {'set':>9}  verdict")
for i, e, m2, m3, lax, onf in rows:
    print(f"{i:>4} {e:6.2f} {m2:10.0f} {m3:10.0f} {lax:7.4f}  {'front' if onf else 'retained':>9}  "
          f"{'subcritical' if m3 > 0 else 'SUPERCRITICAL under ALL-RE'}")
print(f"\n  {sum(1 for r in rows if r[3] > 0)} of {len(rows)} subcritical under the four "
      f"regulating banks with no soluble boron, 3D, BOL.")
PYEOF
echo
echo "=============================================================="
echo " PARKED RODS (E) AND MARGINAL RE-SCORE (F)"
echo "=============================================================="
python3 - << 'PYEOF'
import json, os, statistics as st
a, b = "confirm3d_c8/summary.json", "confirm3d_c8_noparked/summary.json"
if os.path.exists(a) and os.path.exists(b):
    x = json.load(open(a)).get("47", {}).get("ARO_Lax_hw")
    y = json.load(open(b)).get("47", {}).get("ARO_Lax_hw")
    if x and y:
        print(f"  d47 L_ax(ARO): parked {x:.4f}, unparked {y:.4f}, difference {1e5*(x-y):.0f} pcm")
        print(f"  campaign water-slab value 1.0289. Parked rods explain "
              f"{'most of' if abs(y-1.0289) < abs(x-1.0289) else 'none of'} the gap.")
p = "boron_c8_marginal/runs.json"
if os.path.exists(p):
    d = json.load(open(p)); by = {}
    for k, r in d.items():
        i = k.split("|")[0]; by.setdefault(i, []).append(r)
    print()
    for i, recs in sorted(by.items()):
        ks = [r["keff"] for r in recs]
        m = st.mean(ks); sdk = 1e5 * st.stdev(ks) if len(ks) > 1 else 0.0
        print(f"  d{i} ALL-RE k = {m:.5f} +/- {sdk:.0f} pcm over {len(ks)} seeds, "
              f"{'subcritical, g_ctrl SATISFIED' if m < 1 else 'supercritical, g_ctrl violated'}")
    print("  Both are dominated on the objectives either way (54 by 21 and 44, 11 by 44),")
    print("  so the front is unchanged whichever way the re-score falls.")
PYEOF
echo
echo "=============================================================="
echo " DOWNCOMER 3.7 cm TRADE (stage G)"
echo "=============================================================="
python3 - << 'PYEOF'
import json, os
a, b = "confirm3d_c8/summary.json", "confirm3d_c8_dc37/summary.json"
if not (os.path.exists(a) and os.path.exists(b)):
    print("  stage G not run (set RUN_OPTIONAL=1)"); raise SystemExit
A, B = json.load(open(a)), json.load(open(b))
# t_dc = 7.669 - t_refl at pitch 1.26, verified against the runs.json downcomers
print(f"{'idx':>4} {'refl':>6} {'t_dc':>6} | {'k3D nat':>8} {'k3D 3.7':>8} {'dk':>7} | "
      f"{'F3D nat':>8} {'F3D 3.7':>8} {'dF':>7}")
def rho(k): return 1e5*(k-1)/k
for i in sorted(set(A) & set(B), key=lambda z: int(z)):
    if "ARO_3Dhw" not in A[i] or "ARO_3Dhw" not in B[i]: continue
    tr = A[i]["design"]["refl_thick"]
    k1, k2 = A[i]["ARO_3Dhw"]["k"], B[i]["ARO_3Dhw"]["k"]
    f1, f2 = A[i]["ARO_3Dhw"]["F"], B[i]["ARO_3Dhw"]["F"]
    print(f"{i:>4} {tr:6.2f} {7.669-tr:6.3f} | {k1:8.5f} {k2:8.5f} {rho(k2)-rho(k1):7.0f} | "
          f"{f1:8.3f} {f2:8.3f} {f2-f1:+7.3f}")
print("\n  dk is the reactivity cost in pcm of trading reflector for downcomer at fixed")
print("  vessel radius, at BOL with all rods out. The cycle-length effect is NOT")
print("  measured here, since no depletion is run. Designs 42, 1 and 13 already")
print("  exceed 3.7 cm natively and need no override.")
PYEOF
echo
say "Next: python3 c8_boron_3d_analysis.py, then commit the json outputs."
