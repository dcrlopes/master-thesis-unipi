#!/usr/bin/env bash
# run_mtc_zoned.sh -- one entry point for the whole moderator-coefficient study,
# rerun with the zoned core so every number matches the campaign model.
#
# WHY THIS EXISTS
#   The first MTC scans solved an UNZONED core, because mtc_scan.py called
#   make_core_model without design_map. The campaign evaluator passes
#   design_map=zn.evaluator_design_map(design) at openmc_evaluator.py:406.
#   The two models differ by about 2470 pcm, which made the measured ceiling
#   and the campaign critical boron incomparable. This driver reruns
#   everything zoned so both quantities come from one state.
#
# STAGES, each skipped if its marker exists, so an interruption is cheap
#   1  wide 2D zoned scan, both pressures, locates the crossing
#   2  refined 2D zoned scan, bracket chosen automatically from stage 1
#   3  3D zoned scan, gives the axial leakage contribution
#   4  critical boron recomputed from a measured worth curve
#
# COST on wks720 at 32 threads, roughly
#   stage 1   40 solves   ~50 min
#   stage 2   60 solves   ~70 min
#   stage 3   32 solves   ~110 min
#   stage 4    0 solves   seconds
#
# USAGE, from the repository root
#   bash run_mtc_zoned.sh --preflight
#   setsid nohup bash run_mtc_zoned.sh > mtc_zoned.log 2>&1 < /dev/null &
set -u -o pipefail

IDX=47
THREADS=32
MARK=.mtc_markers
PA=15.5 ; TA_LO=570 ; TA_HI=590
PB=12.8 ; TB_LO=547 ; TB_HI=567
die() { echo "FAIL: $*" >&2; exit 1; }

preflight() {
  echo "=========================================================="
  echo " PREFLIGHT"
  echo "=========================================================="
  echo "  host        : $(hostname)"
  echo "  cwd         : $(pwd)"
  echo "  conda env   : ${CONDA_DEFAULT_ENV:-NONE}"
  [ "${CONDA_DEFAULT_ENV:-}" = "openmc-env" ] || die "wrong env, run: conda activate openmc-env"
  python -c "import numpy, openmc, iapws; print('  openmc      :', openmc.__version__)" \
    || die "numpy, openmc or iapws not importable"
  [ "$(python -c 'import openmc; print(openmc.__version__)')" = "0.15.3" ] \
    || die "openmc is not 0.15.3, this is not the campaign environment"
  echo "  branch      : $(git branch --show-current)"
  [ "$(git branch --show-current)" = "main" ] || die "wrong branch, run: git checkout main"
  [ -n "${OPENMC_CROSS_SECTIONS:-}" ] && [ -f "$OPENMC_CROSS_SECTIONS" ] \
    || die "OPENMC_CROSS_SECTIONS unset or missing"
  echo "  XS          : $OPENMC_CROSS_SECTIONS"

  # the whole point of this rerun: the 2D path must pass design_map
  grep -q "design_map=zn.evaluator_design_map(design)" mtc_scan.py \
    || die "mtc_scan.py does not pass design_map. Patch it before running."
  echo "  zoning      : mtc_scan.py passes design_map, matching the evaluator"
  grep -q '"core3d"' mtc_scan.py \
    || die "mtc_scan.py has no core3d level. Apply the core3d patch first."
  echo "  core3d      : level present"
  python mtc_scan.py --selftest >/dev/null || die "mtc_scan.py selftest failed"
  echo "  selftest    : OK"

  echo "  threads     : $THREADS of $(nproc)"
  local busy
  busy=$(pgrep -af "python.*(run_optimization|confirm3d\.py|boron_worth|validate_ktarget|sweep_ktarget|sweep_refl)" | grep -v run_mtc_zoned || true)
  [ -z "$busy" ] || { echo "$busy"; die "a simulation is already running"; }
  echo "  no simulation currently running"
  echo "  stages done : $(ls $MARK 2>/dev/null | tr '\n' ' ')"
  echo "preflight OK"
  echo
}

stage_done() { [ -f "$MARK/$1" ]; }
mark()       { mkdir -p "$MARK"; touch "$MARK/$1"; }

# ---------------------------------------------------------------- archive --
archive_unzoned() {
  stage_done archive && { echo "[archive] already done"; return 0; }
  echo "[archive] moving the unzoned results aside"
  for d in mtc_47_p155 mtc_47_p128 mtc_47_p155_fine mtc_47_p128_fine mtc_47_zoned_check; do
    [ -d "$d" ] && [ ! -d "${d}_unzoned" ] && { mv "$d" "${d}_unzoned"; echo "  $d -> ${d}_unzoned"; }
  done
  mkdir -p mtc_unzoned
  for d in *_unzoned; do [ -d "$d" ] && mv "$d" mtc_unzoned/; done
  cat > mtc_unzoned/README.md <<'EOF'
# Unzoned moderator-coefficient scans, superseded

These runs solved a UNIFORM core, because mtc_scan.py called make_core_model
without design_map. The campaign evaluator passes
design_map=zn.evaluator_design_map(design) at openmc_evaluator.py:406.

Measured consequence on design 47 at 1000 ppm and 570 K:

  unzoned  k = 1.13065   rho = 11555 pcm
  zoned    k = 1.09995   rho =  9087 pcm
  campaign k = 1.09805   rho =  8930 pcm   (580 K, density 0.72)

So zoning is worth about 2470 pcm and the remaining 157 pcm is the density
difference between the IAPWS value at 570 K and the 0.72 hardcoded in
make_water. The zoned scan agrees with the campaign model.

RETAINED, NOT DELETED. The coefficient itself barely moved, -24.87 +/- 2.15
zoned against -26.39 +/- 1.42 unzoned, so these runs measure the sensitivity
of the coefficient to radial zoning, which is a result in its own right.

Do NOT quote the crossings in these directories. Use the zoned reruns.
EOF
  mark archive
}

# ------------------------------------------------------------- stage 1, 2D --
stage_wide() {
  stage_done wide && { echo "[wide] already done"; return 0; }
  echo "[wide] zoned 2D scan, 1000 to 5000 ppm, both pressures"
  for spec in "$PA $TA_LO $TA_HI mtc_z_p155_wide" "$PB $TB_LO $TB_HI mtc_z_p128_wide"; do
    set -- $spec
    python -u mtc_scan.py --idx $IDX --level core2d --pressure "$1" \
      --t-lo "$2" --t-hi "$3" --boron 1000,2000,3000,4000,5000 \
      --seeds 2 --threads $THREADS --doppler --out "$4" || return 1
  done
  mark wide
}

# ------------------------------------------------------------- stage 2, 2D --
stage_fine() {
  stage_done fine && { echo "[fine] already done"; return 0; }
  echo "[fine] refined 2D scan, bracket chosen from stage 1"
  for spec in "$PA $TA_LO $TA_HI mtc_z_p155_wide mtc_z_p155_fine" \
              "$PB $TB_LO $TB_HI mtc_z_p128_wide mtc_z_p128_fine"; do
    set -- $spec
    local br
    br=$(python - "$4" <<'PY'
import json, sys
s = json.load(open(f"{sys.argv[1]}/summary.json"))
c = s.get("crossing_ppm")
if not c:
    sys.exit("no crossing found in the wide scan, widen --boron")
c = round(c / 100.0) * 100
print(",".join(str(int(c + d)) for d in (-400, -200, 0, 200, 400)))
PY
    ) || return 1
    echo "  $1 MPa: bracket $br"
    python -u mtc_scan.py --idx $IDX --level core2d --pressure "$1" \
      --t-lo "$2" --t-hi "$3" --boron "$br" \
      --seeds 3 --threads $THREADS --out "$5" || return 1
  done
  mark fine
}

# --------------------------------------------------------------- stage 3, 3D --
stage_3d() {
  stage_done threed && { echo "[3d] already done"; return 0; }
  echo "[3d] zoned 3D scan, gives the axial leakage contribution"
  for spec in "$PA $TA_LO $TA_HI mtc_z_p155_3d" "$PB $TB_LO $TB_HI mtc_z_p128_3d"; do
    set -- $spec
    python -u mtc_scan.py --idx $IDX --level core3d --pressure "$1" \
      --t-lo "$2" --t-hi "$3" --boron 0,1000,2000,3000 \
      --seeds 2 --threads $THREADS \
      --particles 100000 --batches 170 --inactive 60 --out "$4" || return 1
  done
  mark threed
}

# --------------------------------------------------------- stage 4, analysis --
stage_cbol() {
  echo "[cbol] recomputing the critical boron from a measured worth curve"
  python - <<'PY' | tee mtc_cbol_report.txt
import json, math
from pathlib import Path
import numpy as np

def load(d):
    p = Path(d) / "runs.json"
    if not p.is_file():
        return None
    out = {}
    for k, v in json.loads(p.read_text()).items():
        parts = k.split("_")
        ppm = float(next(x[1:] for x in parts if x.startswith("b") and x[1:].replace('.','',1).isdigit()))
        T   = float(next(x[1:] for x in parts if x.startswith("T") and x[1:].replace('.','',1).isdigit()))
        out.setdefault((ppm, T), []).append(v["k"])
    return {kk: float(np.mean(vv)) for kk, vv in out.items()}

def rho(k): return 1e5 * (1 - 1 / k)

print("=" * 70)
print("CRITICAL BORON FROM THE MEASURED k CURVE, design 47, zoned")
print("=" * 70)
print("The campaign c_BOL extrapolates linearly above 1000 ppm at the")
print("1000-1500 ppm worth of 6.63 pcm/ppm. Boron self-shields, so the worth")
print("falls with concentration and that extrapolation understates c_BOL.")
print()

runs = {}
for d in ("mtc_z_p155_wide", "mtc_z_p155_fine"):
    r = load(d)
    if r: runs.update(r)
if not runs:
    raise SystemExit("no zoned 2D results found, run stages 1 and 2 first")

TLO = min(T for _, T in runs)
pts = sorted((c, k) for (c, T), k in runs.items() if T == TLO)
print(f"  cold-state points at {TLO:.0f} K:")
for c, k in pts:
    print(f"    {c:>6.0f} ppm   k = {k:.5f}   rho = {rho(k):+8.0f} pcm")
print()

cs = np.array([c for c, _ in pts]); ks = np.array([k for _, k in pts])
w = -np.gradient(rho(ks), cs)
print("  differential worth from the curve:")
for c, wi in zip(cs, w):
    print(f"    {c:>6.0f} ppm   {wi:5.2f} pcm/ppm")
a, b = np.polyfit(cs, w, 1)
print(f"  linear worth model: w(c) = {b:.3f} {a:+.3e} * c   pcm/ppm")
print()

# the campaign state is 157 pcm less reactive than this scan's cold state,
# density 0.72 against the IAPWS value, so campaign criticality sits where
# this model reads +157 pcm.
OFF = 157.0
r = rho(ks)
i = int(np.argmin(np.abs(r - OFF)))
lo = max(i - 1, 0); hi = min(i + 1, len(cs) - 1)
c_crit = float(np.interp(-OFF, -r[lo:hi + 1][::-1], cs[lo:hi + 1][::-1])) \
    if r[lo] > OFF > r[hi] else float(np.interp(OFF, r[::-1], cs[::-1]))
print(f"  campaign-state offset assumed {OFF:.0f} pcm (density 0.73291 -> 0.72)")
print(f"  CRITICAL BORON, design 47 = {c_crit:.0f} ppm")
print(f"  campaign linear estimate  = 2348 ppm")
print(f"  correction                = {c_crit - 2348:+.0f} ppm")
print()
print("  Compare this against the crossing in mtc_z_p155_fine/summary.json.")
print("  Both now come from the same zoned model, so the comparison is valid.")
PY
}

case "${1:-}" in
  --preflight) preflight; exit 0 ;;
esac

preflight
archive_unzoned
stage_wide || die "stage 1 failed"
stage_fine || die "stage 2 failed"
stage_3d   || die "stage 3 failed"
stage_cbol

echo
echo "=========================================================="
echo " ALL STAGES FINISHED"
echo "=========================================================="
for f in mtc_z_p155_fine mtc_z_p128_fine mtc_z_p155_3d mtc_z_p128_3d; do
  [ -f "$f/report.txt" ] && { echo; echo "--- $f ---"; cat "$f/report.txt"; }
done
