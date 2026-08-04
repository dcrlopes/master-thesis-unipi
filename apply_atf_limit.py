#!/usr/bin/env python3
"""
apply_atf_limit.py -- impose an ATF (Accident Tolerant Fuel) discharge burnup
limit on an EXISTING optimization checkpoint, without re-running any OpenMC
(Open source Monte Carlo particle transport code) evaluation.

WHY THIS IS EXACT
-----------------
1. Every evaluation stores `bu_eoc_mwd_kg`, the end-of-cycle burnup, and
   `cycle_length` in EFPD (Effective Full Power Days). The evaluator relates
   them by a single global constant (openmc_evaluator.py):

       EFPD = bu_eoc * 1000 / spec_power        [spec_power in W/gHM]

   `spec_power` is computed once per evaluator, not per design, so the map is
   linear and design-independent. It is recovered here from the data itself.

2. A discharge burnup limit does not change the reactor -- it changes WHEN you
   unload it. A design whose reactivity would allow 90 MWd/kgHM but whose fuel
   is licensed to 75 simply runs a shorter cycle:

       EFPD_ATF = min(EFPD_reactivity, B_limit * 1000 / spec_power)

3. F_dh (the enthalpy-rise hot channel factor) is computed by
   `_bol_peaking()` from a FRESH-assembly mesh tally at BOL (Beginning of
   Life). It is independent of cycle length, so truncating the cycle leaves
   the second objective untouched. No re-evaluation is needed.

4. CENSORED designs (those that reached the 100 MWd/kgHM computational
   ceiling still above k_target, whose EFPD was only a LOWER BOUND) become
   EXACT under any limit below that ceiling: their true burnup is >= the cap,
   so the capped EFPD is the cap. The script refuses to run if the limit
   exceeds the ceiling, where this reasoning would break.

USAGE
-----
  # sensitivity table only, writes nothing:
  python apply_atf_limit.py --checkpoint out/optimization_checkpoint.json \
      --burnup-limit 75 --report-only

  # write a rebased checkpoint you can --resume from:
  python apply_atf_limit.py --checkpoint out/optimization_checkpoint.json \
      --burnup-limit 75 --out out_atf75/optimization_checkpoint.json
"""
import argparse
import json
import shutil
from pathlib import Path

# --------------------------------------------------------------------------- #
# command line                                                                 #
# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True,
                help="input checkpoint written by save_checkpoint()")
ap.add_argument("--burnup-limit", type=float, required=True,
                help="ATF discharge burnup limit [MWd/kgHM]. Reference points: "
                     "62 = current LWR licensing limit, 75 = typical ATF "
                     "programme target, 100 = the computational ceiling "
                     "(i.e. no licensing limit at all)")
ap.add_argument("--out", default=None,
                help="path of the rebased checkpoint to write "
                     "(omit together with --report-only)")
ap.add_argument("--report-only", action="store_true",
                help="print the analysis and write nothing")
ap.add_argument("--ceiling", type=float, default=100.0,
                help="the max_burnup the campaign was run with [MWd/kgHM]; "
                     "the limit must not exceed it (default 100)")
ap.add_argument("--csv", default=None,
                help="optional path for a CSV dump of the capped archive")
args = ap.parse_args()

if args.burnup_limit > args.ceiling:
    raise SystemExit(
        f"ERROR: limit {args.burnup_limit} exceeds the campaign ceiling "
        f"{args.ceiling} MWd/kgHM. Censored designs have unknown true burnup "
        f"above the ceiling, so a limit above it cannot be applied exactly. "
        f"Run a fresh campaign with a higher --max-burnup instead.")
if not args.report_only and not args.out:
    raise SystemExit("ERROR: give --out PATH, or use --report-only.")

ckpt = json.loads(Path(args.checkpoint).read_text())
raw = ckpt["all_raw"]
con_names = ckpt.get("constraint_names", [])

# --------------------------------------------------------------------------- #
# recover spec_power from the data (median over uncensored, non-zero designs)  #
# --------------------------------------------------------------------------- #
ratios = [r["bu_eoc_mwd_kg"] * 1000.0 / r["cycle_length"]
          for r in raw
          if r.get("cycle_length", 0) > 0 and r.get("bu_eoc_mwd_kg", 0) > 0]
if not ratios:
    raise SystemExit("ERROR: no usable (burnup, EFPD) pair found in the checkpoint.")
ratios.sort()
spec_power = ratios[len(ratios) // 2]
spread = (max(ratios) - min(ratios)) / spec_power
if spread > 1e-3:
    print(f"!! WARNING: spec_power is not constant across designs "
          f"(relative spread {spread:.2%}). Using the median, "
          f"{spec_power:.4f} W/gHM.")

efpd_cap = args.burnup_limit * 1000.0 / spec_power

# --------------------------------------------------------------------------- #
# apply the cap                                                                #
# --------------------------------------------------------------------------- #
capped = json.loads(json.dumps(raw))     # deep copy, so `raw` stays pristine
n_limited = n_was_censored = 0
for r in capped:
    r["burnup_limited"] = False
    if r.get("bu_eoc_mwd_kg", 0.0) > args.burnup_limit:
        n_limited += 1
        if r.get("censored"):
            n_was_censored += 1
        r["cycle_length"] = efpd_cap
        r["bu_eoc_mwd_kg"] = args.burnup_limit
        r["censored"] = False            # exact now: true burnup >= the cap
        r["burnup_limited"] = True


# --------------------------------------------------------------------------- #
# feasibility + Pareto front (maximise cycle_length, minimise peaking)         #
# --------------------------------------------------------------------------- #
def feasible(rec, tol=1e-9):
    """All constraints are written in the <= 0 convention."""
    return all(float(rec.get(c, 0.0)) <= tol for c in con_names)


def pareto(recs):
    """Non-dominated subset. A dominates B if A is at least as good on both
    objectives and strictly better on one."""
    front = []
    for a in recs:
        dominated = any(
            (b["cycle_length"] >= a["cycle_length"]
             and b["peaking"] <= a["peaking"]
             and (b["cycle_length"] > a["cycle_length"]
                  or b["peaking"] < a["peaking"]))
            for b in recs)
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda r: -r["cycle_length"])


feas = [r for r in capped if feasible(r)]
front = pareto(feas) if feas else []

# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #
print("=" * 72)
print(f"ATF (Accident Tolerant Fuel) discharge burnup limit: "
      f"{args.burnup_limit:g} MWd/kgHM")
print(f"specific power recovered from data : {spec_power:.4f} W/gHM")
print(f"equivalent cycle-length ceiling    : {efpd_cap:.0f} EFPD")
print("-" * 72)
print(f"evaluations in checkpoint          : {len(raw)}")
print(f"feasible (all constraints <= 0)    : {len(feas)}")
print(f"truncated by the ATF limit         : {n_limited}")
print(f"  of which previously CENSORED     : {n_was_censored}  "
      f"(their EFPD is now EXACT, not a lower bound)")
print(f"Pareto-optimal designs after cap   : {len(front)}")
print("-" * 72)
if front:
    hdr = ["EFPD", "F_dH", "bu_eoc", "lim?"] + ckpt["design_variables"]
    print("  ".join(f"{h:>9s}" for h in hdr))
    for r in front:
        row = ([f"{r['cycle_length']:9.0f}", f"{r['peaking']:9.3f}",
                f"{r['bu_eoc_mwd_kg']:9.2f}",
                f"{'YES' if r['burnup_limited'] else '-':>9s}"]
               + [f"{float(r[v]):9.3f}" for v in ckpt["design_variables"]])
        print("  ".join(row))
print("=" * 72)

if args.csv:
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    cols = (["cycle_length", "peaking", "bu_eoc_mwd_kg", "burnup_limited"]
            + list(ckpt["design_variables"]) + list(con_names))
    on_front = {id(r) for r in front}
    with open(args.csv, "w") as fh:
        fh.write(",".join(cols + ["feasible", "pareto"]) + "\n")
        for r in capped:
            fh.write(",".join(str(r.get(c, "")) for c in cols)
                     + f",{int(feasible(r))},{int(id(r) in on_front)}\n")
    print(f"CSV written: {args.csv}")

# --------------------------------------------------------------------------- #
# write the rebased checkpoint                                                 #
# --------------------------------------------------------------------------- #
if args.report_only:
    print("--report-only: nothing written.")
else:
    out = dict(ckpt)
    out["all_raw"] = capped
    # The hypervolume history and its frozen reference point were computed on
    # the UNCAPPED objective and are not comparable. Clearing them makes the
    # optimizer re-freeze a reference and restart HV tracking on the capped
    # objective at the next run (see ActiveLearningMOO.run).
    out["hv_history"] = []
    out["hv_ref"] = None
    meta = dict(out.get("meta") or {})
    meta.update({"atf_burnup_limit_mwd_kg": args.burnup_limit,
                 "atf_efpd_cap": efpd_cap,
                 "atf_spec_power_w_per_g": spec_power,
                 "rebased_from": str(args.checkpoint)})
    out["meta"] = meta

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        backup = dest.with_suffix(dest.suffix + ".bak")
        shutil.copy2(dest, backup)
        print(f"existing file backed up to {backup}")
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"rebased checkpoint written: {dest}")
    print(f"resume with:  --resume {dest} --max-burnup {args.burnup_limit:g}")
