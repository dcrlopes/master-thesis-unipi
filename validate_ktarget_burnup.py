#!/usr/bin/env python3
"""
validate_ktarget_burnup.py
==========================
Measures the BURNUP dependence of the Route-B leakage factor

        LF(B) = k_inf(assembly, B) / k_eff(core, B)

for selected designs of a campaign checkpoint, and states what it does to
the cycle length. This closes the one assumption of Route B that the
existing closure check (validate_core_proxy.py, BOL only) does not test:
the calibration table k_target(pitch, refl_thick) is measured at BOL on a
reference composition and held CONSTANT over the cycle.

WHAT THE SCRIPT DOES, PER DESIGN
--------------------------------
  1. Depletes the reflective single assembly ONCE, in a single shot (no
     restart chunks), with the campaign schedule (BOL block + 4 MWd/kgHM
     steps) up to a planned burnup that covers the design's end of cycle.
     Fidelity defaults to the checkpoint's transport settings.
  2. Reads the k_inf(B) history, locates the end of cycle exactly as the
     optimiser does (last downward crossing of the table value, or the
     burnup cap when the design is capped), and picks the depletion steps
     that bracket it.
  3. Exports the depleted fuel compositions at BOL and at the bracketing
     steps (openmc.deplete.Results.export_to_materials), loads them
     UNIFORMLY into all 32 assemblies of the same make_core_model used for
     the calibration table, and runs one core solve (vacuum boundary) and
     one reflective-assembly solve per step and per seed.
  4. Reports LF at BOL and at EOC, the table value, the residuals in pcm,
     and the implied correction of the cycle length in EFPD.

Every OpenMC run is checkpointed in <out>/runs.json, so an interrupted job
resumes where it stopped.

ROUTE B RULE (unchanged): the depletion runs at infinite medium, and ALL
leakage enters through the target. Nothing here wraps the depleting
assembly in a reflector.

TARGET FORMS SUPPORTED (choose one)
  --ktarget-table T.json      2-D schema (pitch_cm, refl_thick_cm, k_target)
                              or 1-D schema (refl_thick_cm, k_target), as
                              openmc_evaluator loads them
  --ktarget-fit A B           straight line k = A + B * refl_thick, e.g. the
                              Campaign 8 refit  --ktarget-fit 1.054528 -0.000516
  --lax L                     axial factor multiplying the 2-D target (Campaign
                              8: 1.0289). The 2-D leakage factor measured here
                              is multiplied by the same L before comparison.

USAGE (wks720, conda env openmc-env, see RUNBOOK.md)
  python validate_ktarget_burnup.py --smoke --threads 32 --out kt_smoke
  python validate_ktarget_burnup.py \
      --checkpoint <campaign 8 checkpoint> --designs 53 31 47 \
      --ktarget-fit 1.054528 -0.000516 --lax 1.0289 \
      --seeds 2 --threads 32 --out kt_burnup
  python validate_ktarget_burnup.py --checkpoint ... --front --dry-run

Run it inside the working copy that ran the campaign, so that
reactor_model.py, core_geometry.py and openmc_evaluator.py are the ones
the campaign used. The script only calls build_materials(design, op),
make_assembly_model(design, op, geo, bc, particles, batches, inactive)
and make_core_model(design, op, geo, refl_thick, particles, batches,
inactive), and it tags every material role that build_materials returns,
so three enrichment rings or a gadolinia ladder need no change here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Pure-python helpers (no OpenMC needed): schedule, bracketing, implications.
# They are unit-tested by  python validate_ktarget_burnup.py --selftest
# --------------------------------------------------------------------------

def build_schedule(bol_steps, dep_step, b_plan):
    """Burnup steps [MWd/kgHM] reproducing the evaluator's marching rule:
    the BOL block, then dep_step increments, with the LAST step clipped so
    that the cumulative burnup lands exactly on b_plan."""
    steps = [float(s) for s in bol_steps]
    cum = sum(steps)
    if cum >= b_plan - 1e-9:
        return steps
    while cum < b_plan - 1e-9:
        s = min(float(dep_step), b_plan - cum)
        steps.append(s)
        cum += s
    return steps


def cumulative_burnup(steps):
    bu = [0.0]
    for s in steps:
        bu.append(bu[-1] + s)
    return bu


def last_downward_crossing(bu, k, k_target):
    """Index i such that k[i] > k_target >= k[i+1] (LAST such i), and the
    interpolated burnup of the crossing. Returns (None, None) if absent.
    Same rule as core_geometry.eoc_crossing_burnup."""
    idx = None
    for i in range(len(k) - 1):
        if k[i] > k_target >= k[i + 1]:
            idx = i
    if idx is None:
        return None, None
    f = (k[idx] - k_target) / (k[idx] - k[idx + 1])
    return idx, float(bu[idx] + f * (bu[idx + 1] - bu[idx]))


def pick_eoc(bu, k, k_target, b_cap):
    """Decide how the optimiser ends this cycle and which steps bracket it.

    Returns dict(kind, b_eoc, i_lo, i_hi) where
      kind  = 'reactivity' (k crossed the target before the cap) or
              'capped'     (still above target at the cap, or crossing at
                            the cap step itself)
      i_lo, i_hi = indices of the depletion steps bracketing b_eoc
    """
    i, b_cross = last_downward_crossing(bu, k, k_target)
    if i is not None and b_cross < b_cap - 1e-9:
        return dict(kind="reactivity", b_eoc=b_cross, i_lo=i, i_hi=i + 1)
    n = len(bu) - 1
    return dict(kind="capped", b_eoc=float(bu[n]), i_lo=n - 1, i_hi=n)


def interp_at(b, b_lo, b_hi, v_lo, v_hi):
    if abs(b_hi - b_lo) < 1e-12:
        return v_hi
    f = (b - b_lo) / (b_hi - b_lo)
    return v_lo + f * (v_hi - v_lo)


def cycle_implication(kind, bu, k, i_lo, i_hi, b_eoc, kt_table, lf_eoc,
                      spec_power):
    """What the measured LF at EOC does to the cycle length.

    reactivity-defined EOC: the optimiser stopped where k_inf = kt_table.
        The true requirement is k_inf = lf_eoc. With the local slope
        s = dk/dB between the bracketing steps (negative), the burnup shift
        is  dB = (lf_eoc - kt_table) / s.
    capped EOC: the cycle is set by the burnup cap. It changes only if the
        true requirement is no longer met at the cap, i.e. k(cap) < lf_eoc,
        in which case the corrected crossing is found on the history with
        the target replaced by lf_eoc.
    Returns dict(db_mwd_kg, defpd_days, note).
    """
    s = (k[i_hi] - k[i_lo]) / (bu[i_hi] - bu[i_lo])      # dk/dB, per MWd/kg
    if kind == "reactivity":
        if s >= 0.0:
            return dict(db_mwd_kg=float("nan"), defpd_days=float("nan"),
                        slope=s, note="non-negative local slope, cannot invert")
        db = (lf_eoc - kt_table) / s
        return dict(db_mwd_kg=db, defpd_days=db * 1000.0 / spec_power, slope=s,
                    note="reactivity-defined EOC, shift from the local slope")
    # capped
    margin_true = k[i_hi] - lf_eoc
    if margin_true >= 0.0:
        return dict(db_mwd_kg=0.0, defpd_days=0.0, slope=s,
                    note=f"capped EOC, still above the true requirement at the "
                         f"cap by {1e5*margin_true:+.0f} pcm, cycle unchanged")
    i2, b2 = last_downward_crossing(bu, k, lf_eoc)
    if i2 is None:
        return dict(db_mwd_kg=float("nan"), defpd_days=float("nan"), slope=s,
                    note="capped EOC, below the true requirement at the cap "
                         "but no crossing found on the history")
    db = b2 - b_eoc
    return dict(db_mwd_kg=db, defpd_days=db * 1000.0 / spec_power, slope=s,
                note="capped EOC, true requirement crossed before the cap")


def bilinear_clamped(x, y, xs, ys, Z):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    Z = np.asarray(Z, float)
    x = float(np.clip(x, xs[0], xs[-1]))
    y = float(np.clip(y, ys[0], ys[-1]))
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
    tx = 0.0 if xs[i + 1] == xs[i] else (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = 0.0 if ys[j + 1] == ys[j] else (y - ys[j]) / (ys[j + 1] - ys[j])
    return float((1 - tx) * (1 - ty) * Z[i, j] + tx * (1 - ty) * Z[i + 1, j]
                 + (1 - tx) * ty * Z[i, j + 1] + tx * ty * Z[i + 1, j + 1])


def selftest():
    # schedule: BOL block then 4-step marching, last step clipped to the plan
    st = build_schedule((0.5, 1, 2, 4, 6), 4.0, 75.0)
    assert abs(sum(st) - 75.0) < 1e-9 and abs(st[-1] - 1.5) < 1e-9, st
    bu = cumulative_burnup(st)
    assert abs(bu[-1] - 75.0) < 1e-9 and abs(bu[5] - 13.5) < 1e-9
    # crossing on a Gd-hump history
    b = [0, 1, 2, 3, 4]
    k = [1.04, 1.08, 1.10, 1.06, 1.02]
    i, bc = last_downward_crossing(b, k, 1.05)
    assert i == 3 and abs(bc - 3.25) < 1e-12
    e = pick_eoc(b, k, 1.05, b_cap=4.0)
    assert e["kind"] == "reactivity" and e["i_lo"] == 3 and e["i_hi"] == 4
    e = pick_eoc(b, k, 1.01, b_cap=4.0)
    assert e["kind"] == "capped" and e["i_lo"] == 3 and e["i_hi"] == 4
    # implication, reactivity case: LF_eoc 100 pcm above the table with
    # slope -0.04 per unit burnup -> dB = -0.025
    imp = cycle_implication("reactivity", b, k, 3, 4, 3.25, 1.05, 1.051, 10.0)
    assert abs(imp["db_mwd_kg"] + 0.025) < 1e-9, imp
    # implication, capped case with margin -> unchanged
    imp = cycle_implication("capped", b, k, 3, 4, 4.0, 1.01, 1.015, 10.0)
    assert imp["db_mwd_kg"] == 0.0
    # capped case where the true requirement is crossed before the cap
    imp = cycle_implication("capped", b, k, 3, 4, 4.0, 1.01, 1.03, 10.0)
    assert imp["db_mwd_kg"] < 0.0
    assert abs(bilinear_clamped(1.29, 11.0, [1.15, 1.29, 1.43], [2, 11, 19.5],
                                [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) - 5.0) < 1e-12
    print("selftest OK")


# --------------------------------------------------------------------------
# OpenMC part
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="optimization_checkpoint.json of the campaign")
    ap.add_argument("--designs", type=int, nargs="*", default=None,
                    help="indices into all_raw (as in validate_core_proxy.py)")
    ap.add_argument("--front", action="store_true",
                    help="every design on the feasible Pareto front")
    ap.add_argument("--design-json", default=None,
                    help="explicit design as JSON, e.g. "
                         "'{\"enrich_inner\":9.5,\"enrich_outer\":8.0,"
                         "\"gd_wt\":6.0,\"pitch\":1.26,\"refl_thick\":12.0}'")
    ap.add_argument("--ktarget-table", default=None,
                    help="k_target JSON table, 2-D or 1-D schema")
    ap.add_argument("--ktarget-fit", type=float, nargs=2, default=None,
                    metavar=("A", "B"),
                    help="2-D target as a straight line A + B*refl_thick "
                         "(Campaign 8: 1.054528 -0.000516)")
    ap.add_argument("--lax", type=float, default=1.0,
                    help="axial leakage factor multiplying the 2-D target "
                         "(Campaign 8: 1.0289, default 1.0)")
    # depletion (defaults: checkpoint meta, else campaign values)
    ap.add_argument("--dep-particles", type=int, default=None)
    ap.add_argument("--dep-batches", type=int, default=None)
    ap.add_argument("--dep-inactive", type=int, default=None)
    ap.add_argument("--bol-steps", default=None,
                    help="comma list, default from checkpoint or 0.5,1,2,4,6")
    ap.add_argument("--dep-step", type=float, default=None)
    ap.add_argument("--max-burnup", type=float, default=None,
                    help="burnup cap [MWd/kgHM], default from checkpoint meta "
                         "(75 for the ATF campaigns, 100 before)")
    ap.add_argument("--extra-steps", type=int, default=2,
                    help="marching steps planned beyond the checkpoint EOC "
                         "(margin for statistical differences), default 2")
    # eigenvalue solves on the depleted compositions (defaults: the sweep)
    ap.add_argument("--asm-particles", type=int, default=10000)
    ap.add_argument("--asm-batches", type=int, default=120)
    ap.add_argument("--asm-inactive", type=int, default=30)
    ap.add_argument("--core-particles", type=int, default=20000)
    ap.add_argument("--core-batches", type=int, default=150)
    ap.add_argument("--core-inactive", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2,
                    help="seed replicates of the core and assembly solves")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chain", default=None, help="depletion chain path "
                    "(default $OPENMC_CHAIN_FILE)")
    ap.add_argument("--out", default="kt_burnup")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fidelity and a 3-step schedule to validate the "
                         "whole chain in minutes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (designs, schedule, burnups) and exit")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def load_checkpoint(path):
    ck = json.loads(Path(path).read_text())
    return ck


def pareto_front(raw, cn):
    def feasible(r, tol=1e-9):
        return all(float(r.get(c, 0.0)) <= tol for c in cn)
    feas = [(i, r) for i, r in enumerate(raw) if feasible(r)]
    picks = [(i, a) for i, a in feas
             if not any((b["cycle_length"] >= a["cycle_length"]
                         and b["peaking"] <= a["peaking"]
                         and (b["cycle_length"] > a["cycle_length"]
                              or b["peaking"] < a["peaking"]))
                        for _, b in feas)]
    picks.sort(key=lambda t: -t[1]["cycle_length"])
    return picks


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    # ---- designs -------------------------------------------------------- #
    dv = ["enrich_inner", "enrich_outer", "gd_wt", "pitch", "refl_thick"]
    meta = {}
    picks = []
    if args.checkpoint:
        ck = load_checkpoint(args.checkpoint)
        raw, cn = ck["all_raw"], ck.get("constraint_names", [])
        dv = ck.get("design_variables", dv)
        meta = ck.get("meta", {}) or {}
        if args.designs:
            picks = [(i, raw[i]) for i in args.designs]
        elif args.front:
            picks = pareto_front(raw, cn)
    if args.design_json:
        d = json.loads(args.design_json)
        picks.append((f"json{len(picks)}", dict(d)))
    if args.smoke and not picks:
        picks = [("smoke", {"enrich_inner": 4.55, "enrich_outer": 4.05,
                            "gd_wt": 0.0, "pitch": 1.26, "refl_thick": 12.0})]
    if not picks:
        sys.exit("no designs selected: give --checkpoint with --designs or "
                 "--front, or --design-json, or --smoke")

    # ---- schedule and fidelity ------------------------------------------ #
    sched = meta.get("schedule", {}) if isinstance(meta, dict) else {}
    tr_ck = meta.get("transport", {}) if isinstance(meta, dict) else {}
    bol_steps = ([float(s) for s in args.bol_steps.split(",")] if args.bol_steps
                 else [float(s) for s in sched.get("bol_steps", [0.5, 1, 2, 4, 6])])
    dep_step = args.dep_step or float(sched.get("dep_step", 4.0))
    max_burnup = args.max_burnup or float(sched.get("max_burnup", 100.0))
    dep_tr = dict(particles=args.dep_particles or int(tr_ck.get("particles", 4000)),
                  batches=args.dep_batches or int(tr_ck.get("batches", 60)),
                  inactive=args.dep_inactive or int(tr_ck.get("inactive", 20)))
    asm_tr = dict(particles=args.asm_particles, batches=args.asm_batches,
                  inactive=args.asm_inactive)
    core_tr = dict(particles=args.core_particles, batches=args.core_batches,
                   inactive=args.core_inactive)
    if args.smoke:
        bol_steps, dep_step, max_burnup = [0.5, 1.0, 2.0], 2.0, 3.5
        dep_tr = dict(particles=500, batches=20, inactive=5)
        asm_tr = dict(particles=1000, batches=20, inactive=5)
        core_tr = dict(particles=2000, batches=20, inactive=5)
        args.seeds = 1

    # ---- Route-B 2-D target: table (2-D or 1-D) or straight-line fit ----- #
    table_lax = None
    if args.ktarget_fit is not None:
        A, B = args.ktarget_fit

        def kt2d_at(design):
            return A + B * float(design["refl_thick"])
        kt_desc = f"fit {A:.6f} + {B:.6f} * refl_thick"
    else:
        tpath = args.ktarget_table or "ktarget_table.json"
        tab = json.loads(Path(tpath).read_text())
        kr = np.asarray(tab["refl_thick_cm"], float)
        # Schema 3 (Campaign 8) stores the axial-corrected target in
        # "k_target" and the pre-axial 2-D target in "k_target_2d_fit". Read
        # the pre-axial values here and take L_ax from the table, so the 2-D
        # leakage factor measured below is scaled by L_ax exactly once.
        table_lax = None
        if "k_target_2d_fit" in tab and "axial_leakage_factor" in tab:
            kv = np.asarray(tab["k_target_2d_fit"], float)
            table_lax = float(tab["axial_leakage_factor"])
            schema = "schema-3 (pre-axial fit, L_ax from table)"
        else:
            kv = np.asarray(tab["k_target"], float)
            schema = "k_target"
        if "pitch_cm" in tab:
            kp = np.asarray(tab["pitch_cm"], float)

            def kt2d_at(design):
                return bilinear_clamped(design.get("pitch", 1.26),
                                        design["refl_thick"], kp, kr, kv)
            kt_desc = f"2-D table {tpath} ({schema})"
        else:
            def kt2d_at(design):
                return float(np.interp(design["refl_thick"], kr, kv))
            kt_desc = f"1-D table {tpath} ({schema})"
    # --lax on the command line wins; otherwise a schema-3 table supplies it;
    # otherwise 1.0 (two-dimensional campaigns).
    if args.lax != 1.0:
        lax = float(args.lax)          # command line wins
    elif table_lax:
        lax = table_lax                # schema-3 table supplies L_ax
    else:
        lax = float(args.lax)          # 1.0, two-dimensional campaigns

    def kt_at(design):
        """Target the optimiser used: 2-D target times the axial factor."""
        return kt2d_at(design) * lax

    # ---- plan ------------------------------------------------------------ #
    # Build keys the model factories actually read. The C8 checkpoint stores
    # design_variables = [enrich, gd_wt, refl_thick, gd_pins], but
    # build_materials needs enrich_inner/enrich_outer and make_assembly_model
    # needs pitch and gd_pins. The raw record carries the derived keys, so we
    # read them the way confirm3d.py does, falling back to design_variables.
    BUILD_KEYS = ("enrich_inner", "enrich_outer", "gd_wt", "pitch",
                  "refl_thick", "gd_pins")

    def design_from(rec):
        keys = [k for k in BUILD_KEYS if k in rec] or dv
        return {k: (int(rec[k]) if isinstance(rec[k], bool) is False
                    and isinstance(rec[k], int) else float(rec[k]))
                for k in keys}

    plan = []
    for idx, rec in picks:
        design = design_from(rec)
        kt = kt_at(design)
        bu_ck = rec.get("bu_eoc_mwd_kg")
        censored = bool(rec.get("censored", False))
        if bu_ck is None or censored or float(bu_ck) <= 0.0:
            b_plan = max_burnup
        else:
            b_plan = min(max_burnup, float(bu_ck) + args.extra_steps * dep_step)
        steps = build_schedule(bol_steps, dep_step, b_plan)
        plan.append(dict(idx=idx, design=design, kt_table=kt, b_plan=b_plan,
                         steps=steps, bu=cumulative_burnup(steps),
                         efpd_ck=rec.get("cycle_length"), bu_ck=bu_ck,
                         censored=censored))

    print("=" * 84)
    print("Route-B burnup-dependence check")
    print(f"depletion {dep_tr}  assembly {asm_tr}  core {core_tr}  "
          f"seeds {args.seeds}")
    print(f"schedule: BOL {bol_steps} then {dep_step} MWd/kg steps, "
          f"cap {max_burnup} MWd/kg")
    print(f"target: {kt_desc}, axial factor {lax}")
    for p in plan:
        d = p["design"]
        print(f"[design {p['idx']}] " + "  ".join(f"{k}={d[k]:.3f}" for k in d)
              + f"  kt_table={p['kt_table']:.4f}  plan to {p['b_plan']:.1f} "
              f"MWd/kg in {len(p['steps'])} steps"
              + (f"  (checkpoint EFPD {p['efpd_ck']:.0f}, "
                 f"B_EOC {float(p['bu_ck']):.2f}{' CEN' if p['censored'] else ''})"
                 if p["efpd_ck"] is not None and p["bu_ck"] is not None else ""))
    print("=" * 84)
    if args.dry_run:
        return

    # ---- OpenMC imports (deferred so --dry-run/--selftest need no OpenMC) - #
    import openmc
    import openmc.deplete
    import reactor_model as rm
    try:
        from openmc_evaluator import _design_seed
    except Exception:                       # identical fallback
        import zlib

        def _design_seed(design):
            key = json.dumps({k: round(float(v), 10)
                              for k, v in sorted(design.items())})
            return 1 + zlib.crc32(key.encode()) % 2_000_000_000

    chain = args.chain or os.environ.get("OPENMC_CHAIN_FILE")
    if not chain:
        sys.exit("set OPENMC_CHAIN_FILE or pass --chain")
    openmc.config["chain_file"] = chain

    op, geo = rm.Operating(), rm.Geometry17x17()
    spec_power = rm.core_specific_power_w_per_g(op, geo)     # W/gHM
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    store = outdir / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}

    def save():
        store.write_text(json.dumps(done, indent=1))

    @contextmanager
    def materials_override(mats):
        """Make reactor_model build its geometry on OUR materials dict."""
        orig = rm.build_materials
        rm.build_materials = lambda design, op_: dict(mats)
        try:
            yield
        finally:
            rm.build_materials = orig

    # ---------------------------------------------------------------- #
    def deplete_once(p, case):
        """Single-shot depletion of the reflective assembly. Returns
        (bu list, k list, k_sd list, dep_dir, Results, set of roles)."""
        dep_dir = case / "dep"
        dep_dir.mkdir(parents=True, exist_ok=True)
        design = p["design"]
        captured = {}
        orig = rm.build_materials

        def capturing(design_, op_):
            d = orig(design_, op_)
            captured.update(d)
            return d

        rm.build_materials = capturing
        try:
            model, fuel_cells, _lat = rm.make_assembly_model(
                design, op, geo, bc="reflective", **dep_tr)
        finally:
            rm.build_materials = orig
        model.settings.seed = _design_seed(design)
        # Tag every material with its ROLE. The tag is written into the
        # depletion materials.xml, so a resumed process (new object ids)
        # still maps the exported materials to the right roles.
        for role, m in captured.items():
            m.name = f"role:{role}"
        roles = set(captured)

        # volumes + depletable, exactly as openmc_evaluator._cycle_length
        pin_vol = math.pi * geo.fuel_or ** 2 * geo.active_height
        counts = Counter(c.fill.id for c in fuel_cells)
        id2mat = {m.id: m for m in model.materials}
        for mat_id, npins in counts.items():
            id2mat[mat_id].volume = npins * pin_vol
            id2mat[mat_id].depletable = True

        h5 = dep_dir / "depletion_results.h5"
        if not h5.exists():
            cwd = Path.cwd()
            try:
                os.chdir(dep_dir)
                op_dep = openmc.deplete.CoupledOperator(
                    model, diff_burnable_mats=False)
                power_w = spec_power * op_dep.heavy_metal
                days = [s * 1000.0 / spec_power for s in p["steps"]]
                integ = openmc.deplete.PredictorIntegrator(
                    op_dep, days, power=power_w, timestep_units="d")
                t0 = time.time()
                integ.integrate()
                print(f"   depletion done in {(time.time()-t0)/60:.1f} min")
            finally:
                os.chdir(cwd)
        else:
            print("   depletion_results.h5 found, reusing")
        res = openmc.deplete.Results(str(h5))
        _t, karr = res.get_keff()
        k = [float(v) for v in karr[:, 0]]
        ksd = [float(v) for v in karr[:, 1]]
        bu = p["bu"]
        if len(k) != len(bu):
            raise RuntimeError(f"{len(k)} k values for {len(bu)} burnups")
        return bu, k, ksd, dep_dir, res, roles

    # ---------------------------------------------------------------- #
    def mats_at_step(res, dep_dir, step, roles):
        """Materials dict (roles -> depleted openmc.Material) at a step."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # duplicate-id notices
            exported = res.export_to_materials(
                step, path=str(dep_dir / "materials.xml"))
        mats = {}
        for m in exported:
            if m.name and m.name.startswith("role:"):
                mats[m.name[len("role:"):]] = m
        missing = set(roles) - set(mats)
        if missing:
            raise RuntimeError(f"roles not found in exported materials: {missing}")
        return mats

    # ---------------------------------------------------------------- #
    def solve_pair(p, mats, step, seed, case):
        """k_inf (reflective assembly) and k_eff (32-assembly core) on the
        SAME depleted materials. Returns dict."""
        design = p["design"]
        key = f"d{p['idx']}_s{step}_seed{seed}"
        if key in done:
            return done[key]
        with materials_override(mats):
            asm, _fc, _lat = rm.make_assembly_model(
                design, op, geo, bc="reflective", **asm_tr)
            core, _fc2 = rm.make_core_model(
                design, op, geo, refl_thick=design.get("refl_thick"), **core_tr)
        asm.settings.seed = seed
        core.settings.seed = seed
        t0 = time.time()
        sp = asm.run(cwd=str(case / f"asm_s{step}_seed{seed}"), output=False,
                     threads=args.threads)
        with openmc.StatePoint(sp) as s:
            kinf, kinf_sd = float(s.keff.nominal_value), float(s.keff.std_dev)
        sp = core.run(cwd=str(case / f"core_s{step}_seed{seed}"), output=False,
                      threads=args.threads)
        with openmc.StatePoint(sp) as s:
            keff, keff_sd = float(s.keff.nominal_value), float(s.keff.std_dev)
        lf = kinf / keff * lax          # 2-D leakage factor times the axial factor
        lf_sd = lf * math.sqrt((kinf_sd / kinf) ** 2 + (keff_sd / keff) ** 2)
        done[key] = dict(kinf=kinf, kinf_sd=kinf_sd, keff=keff, keff_sd=keff_sd,
                         lf=lf, lf_sd=lf_sd, wall_s=time.time() - t0)
        save()
        return done[key]

    # ---------------------------------------------------------------- #
    def combine(vals):
        """Seed average with the larger of (spread, propagated) as sd."""
        lf = np.array([v["lf"] for v in vals])
        prop = np.sqrt(np.mean([v["lf_sd"] ** 2 for v in vals]) / len(vals))
        spread = lf.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return float(lf.mean()), float(max(prop, spread))

    rows = []
    for p in plan:
        idx = p["idx"]
        case = outdir / f"design{idx}"
        case.mkdir(parents=True, exist_ok=True)
        print(f"\n[design {idx}] depleting the reflective assembly "
              f"({len(p['steps'])} steps to {p['b_plan']:.1f} MWd/kg)")
        bu, k, ksd, dep_dir, res, roles = deplete_once(p, case)
        kt = p["kt_table"]
        eoc = pick_eoc(bu, k, kt, b_cap=max_burnup)
        i_lo, i_hi = eoc["i_lo"], eoc["i_hi"]
        print(f"   k_inf(BOL)={k[0]:.4f}  EOC kind={eoc['kind']}  "
              f"B_EOC={eoc['b_eoc']:.2f} MWd/kg  bracket steps {i_lo},{i_hi} "
              f"(B={bu[i_lo]:.1f},{bu[i_hi]:.1f})  table k_target={kt:.4f}")

        lf_by_step = {}
        for step in sorted({0, i_lo, i_hi}):
            mats = mats_at_step(res, dep_dir, step, roles)
            vals = [solve_pair(p, mats, step, seed, case)
                    for seed in range(1, args.seeds + 1)]
            lf, lf_sd = combine(vals)
            lf_by_step[step] = (lf, lf_sd)
            print(f"   step {step:2d} (B={bu[step]:5.1f}): k_inf={vals[0]['kinf']:.4f} "
                  f"k_eff={vals[0]['keff']:.4f}  LF={lf:.4f} +/- {lf_sd:.4f}")

        lf_bol, lf_bol_sd = lf_by_step[0]
        lf_eoc = interp_at(eoc["b_eoc"], bu[i_lo], bu[i_hi],
                           lf_by_step[i_lo][0], lf_by_step[i_hi][0])
        lf_eoc_sd = max(lf_by_step[i_lo][1], lf_by_step[i_hi][1])
        r_bol = 1e5 * (lf_bol - kt) / kt
        r_eoc = 1e5 * (lf_eoc - kt) / kt
        drift = 1e5 * (lf_eoc - lf_bol) / lf_bol
        drift_sd = 1e5 * math.sqrt(lf_bol_sd ** 2 + lf_eoc_sd ** 2) / lf_bol
        imp = cycle_implication(eoc["kind"], bu, k, i_lo, i_hi, eoc["b_eoc"],
                                kt, lf_eoc, spec_power)
        efpd_here = eoc["b_eoc"] * 1000.0 / spec_power
        row = dict(idx=idx, **p["design"], kt_table=kt, lax=lax, kind=eoc["kind"],
                   b_eoc=eoc["b_eoc"], efpd_here=efpd_here,
                   efpd_checkpoint=p["efpd_ck"],
                   lf_bol=lf_bol, lf_bol_sd=lf_bol_sd,
                   lf_eoc=lf_eoc, lf_eoc_sd=lf_eoc_sd,
                   resid_bol_pcm=r_bol, resid_eoc_pcm=r_eoc,
                   drift_pcm=drift, drift_sd_pcm=drift_sd,
                   slope_dk_dB=imp["slope"], db_mwd_kg=imp["db_mwd_kg"],
                   defpd_days=imp["defpd_days"],
                   defpd_pct=(100.0 * imp["defpd_days"] / efpd_here
                              if efpd_here > 0 else float("nan")),
                   note=imp["note"], k_hist=k, k_sd_hist=ksd, bu_hist=bu)
        rows.append(row)
        (outdir / "summary.json").write_text(json.dumps(rows, indent=1))

    # ---- report ----------------------------------------------------------- #
    print("\n" + "=" * 96)
    print("SUMMARY: burnup dependence of the Route-B leakage factor")
    print("=" * 96)
    hdr = (f"{'idx':>5} {'kind':>10} {'B_EOC':>6} {'kt_tab':>7} {'LF_BOL':>7} "
           f"{'LF_EOC':>7} {'r_BOL':>7} {'r_EOC':>7} {'drift':>7} {'+/-':>5} "
           f"{'dEFPD':>7} {'%':>6}")
    print(hdr)
    for r in rows:
        print(f"{str(r['idx']):>5} {r['kind']:>10} {r['b_eoc']:6.1f} "
              f"{r['kt_table']:7.4f} {r['lf_bol']:7.4f} {r['lf_eoc']:7.4f} "
              f"{r['resid_bol_pcm']:+7.0f} {r['resid_eoc_pcm']:+7.0f} "
              f"{r['drift_pcm']:+7.0f} {r['drift_sd_pcm']:5.0f} "
              f"{r['defpd_days']:+7.0f} {r['defpd_pct']:+6.1f}")
    print("\nLF includes the axial factor when --lax is given, so it compares "
          "directly with the optimiser's target")
    print("r_BOL, r_EOC: LF minus table value, pcm (negative = table demands "
          "more reactivity than needed, conservative)")
    print("drift: LF_EOC minus LF_BOL, pcm, with its 1-sigma uncertainty")
    print("dEFPD: cycle-length correction implied by LF_EOC (days and percent)")
    for r in rows:
        print(f"  design {r['idx']}: {r['note']}")

    # ---- LaTeX table ------------------------------------------------------ #
    tex = [r"\begin{table}[htbp]", r"  \centering",
           r"  \caption{Burnup dependence of the Route B leakage factor on the "
           r"finalists. The residuals are the leakage factor minus the table "
           r"value, in pcm. The drift is the change of the leakage factor from "
           r"beginning of life to end of cycle. The last column is the "
           r"cycle-length correction implied by the end-of-cycle value.}",
           r"  \label{tab:kt-burnup}",
           r"  \begin{tabular}{lcccccccc}", r"    \toprule",
           r"    Design & $t_\mathrm{refl}$ [cm] & $B_\mathrm{EOC}$ [MWd/kgHM] & "
           r"$k_\mathrm{target}$ & $\mathrm{LF}_\mathrm{BOL}$ & "
           r"$\mathrm{LF}_\mathrm{EOC}$ & Residual EOC [pcm] & Drift [pcm] & "
           r"$\Delta\mathrm{EFPD}$ [d] \\", r"    \midrule"]
    for r in rows:
        tex.append(f"    {r['idx']} & {float(r.get('refl_thick', float('nan'))):.2f} & {r['b_eoc']:.1f} & "
                   f"{r['kt_table']:.4f} & {r['lf_bol']:.4f} & {r['lf_eoc']:.4f} & "
                   f"{r['resid_eoc_pcm']:+.0f} & {r['drift_pcm']:+.0f} $\\pm$ "
                   f"{r['drift_sd_pcm']:.0f} & {r['defpd_days']:+.0f} \\\\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (outdir / "summary_table.tex").write_text("\n".join(tex) + "\n")
    print(f"\nwrote {outdir/'summary.json'} and {outdir/'summary_table.tex'}")


if __name__ == "__main__":
    main()
