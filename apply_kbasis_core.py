#!/usr/bin/env python3
"""
apply_kbasis_core.py -- wire the reactivity screen to the CORE eigenvalue and
close the limits provenance gap. No-op by default.

WHAT THIS FIXES
---------------
Campaign 5 screened reactivity on k_inf of a single infinite-lattice
assembly. The archive proves it: g_kmax reproduces k_bol - 1.35 with a
residual of exactly 0.0, while the core-basis residual is 8.06e-2.

The leakage gap between the two eigenvalues is not a constant offset:

    k_bol - keff_core : mean +0.0656, min +0.0430, max +0.0806

so the effective strictness of the screen varied by 3760 pcm across the
design space. A constraint whose strictness depends on the design is not
screening designs on a consistent criterion. That is the defect, and it is a
methodological one rather than a matter of the limit being too tight.

openmc_evaluator.py ALREADY supports the correction. k_basis is validated at
line 222, guarded at line 227 so that k_basis='core' requires an explicit
k_max, and used at line 291 to pick k_ref. Nothing in run_optimization.py
ever passes it, so the machinery is built and unwired. This applier wires it.

THE SECOND FIX: PROVENANCE
--------------------------
Five numbers define the constrained problem and NONE of them is recorded in
the Campaign 5 checkpoint metadata:

    k_basis   which eigenvalue the reactivity screen acts on
    k_max     upper reactivity bound
    k_min     lower reactivity bound
    f_max     peaking bound
    enr_max   LEU (Low Enriched Uranium) enrichment cap

A campaign whose checkpoint cannot state its own constraint set is not
reproducible from its archive. This applier records all five and warns on
resume when any of them differs, matching the guards that already exist for
k_target and the core transport settings.

THE SAFETY PROPERTY
-------------------
Every new flag defaults to the Campaign 5 value:

    --k-basis assembly   --k-max (unset, evaluator uses 1.35)
    --k-min 1.02         --f-max 2.0        --enr-max 19.75

Applying this patch and running with no new flags reproduces Campaign 5's
constraint definitions exactly. Only the metadata block grows. The change of
basis is a separate, explicit decision made at launch time, not by patching.

WHAT THE CORE BASIS BUYS, ON YOUR OWN ARCHIVE
---------------------------------------------
    basis      limit    feasible designs in the C5 archive
    assembly   1.350    0
    assembly   1.400    2   idx17 (F 1.9534), idx49 (F 1.9828)
    core       1.284    1   idx49
    core       1.300    1   idx49
    core       1.350    2   idx17, idx49

Core basis at 1.35 recovers the same two designs as assembly basis at 1.40,
and it survives the tightened m_P = 1.15 enrichment box unchanged.

WHAT IT DOES NOT BUY
--------------------
No primary source supports 1.35 on the core basis either. The value remains a
declared screening bound. The number that would justify one is the reactivity
the control system can hold down with no soluble boron, which is what a
control-rod worth study would measure. Until that exists, say so plainly in
the thesis rather than implying a provenance the number does not have.

BLAST RADIUS
------------
Four edits, all in run_optimization.py. openmc_evaluator.py and
reactor_optimization.py are NOT touched, so this applier is independent of
apply_leu_box.py and either can be reverted without disturbing the other.

    argparse block      five new flags
    evaluator call      pass the five limits through
    resume guard        warn when a limit differs from the checkpoint
    checkpoint meta     record all five limits

USAGE
    cd ~/master-thesis-unipi
    python3 apply_kbasis_core.py --check         # verify anchors, no writes
    python3 apply_kbasis_core.py                 # apply
    python3 run_optimization.py --smoke --workdir smoke_kb --out out_smoke_kb
    python3 apply_kbasis_core.py --revert        # restore from .bak

Then Campaign 6 launches with the basis stated explicitly:

    python3 run_optimization.py --k-basis core --k-max 1.35 \\
        --workdir openmc_runs_c6 --out out_c6 ...

FLAGS
    --check    verify every anchor and print planned edits, write nothing
    --revert   restore run_optimization.py from its .bak backup
    --root D   repository root (default: the current directory)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FNAME = "run_optimization.py"

# --------------------------------------------------------------------------
# anchors, each verified to occur exactly once
# --------------------------------------------------------------------------
A_ARGS = '    args = ap.parse_args()\n'
R_ARGS = (
    '    # ---------------- constraint definition (Campaign 6) --------------- #\n'
    '    # Every default below reproduces the Campaign 5 behaviour, so adding\n'
    '    # these flags changes nothing until one is passed explicitly.\n'
    '    ap.add_argument("--k-basis", choices=["assembly", "core"],\n'
    '                    default="assembly",\n'
    '                    help="which eigenvalue the reactivity screen acts on. "\n'
    '                         "\'assembly\' uses k_inf of the single infinite-"\n'
    '                         "lattice assembly, the Campaign 5 behaviour. "\n'
    '                         "\'core\' uses k_eff of the 32-assembly core at "\n'
    '                         "Beginning of Life, which is the quantity the "\n'
    '                         "reactor actually has. The two differ by the "\n'
    '                         "leakage gap, measured at 4300 to 8060 pcm across "\n'
    '                         "the Campaign 5 archive, so the limit is NOT "\n'
    '                         "transferable between bases without thought. "\n'
    '                         "\'core\' requires an explicit --k-max.")\n'
    '    ap.add_argument("--k-max", type=float, default=None,\n'
    '                    help="upper reactivity bound applied to the eigenvalue "\n'
    '                         "selected by --k-basis. Left unset on the assembly "\n'
    '                         "basis the evaluator uses its historical 1.35.")\n'
    '    ap.add_argument("--k-min", type=float, default=1.02,\n'
    '                    help="lower reactivity bound, the criticality floor")\n'
    '    ap.add_argument("--f-max", type=float, default=2.0,\n'
    '                    help="upper bound on the core radial enthalpy-rise hot "\n'
    '                         "channel factor F_dH")\n'
    '    ap.add_argument("--enr-max", type=float, default=19.75,\n'
    '                    help="LEU (Low Enriched Uranium) enrichment cap in "\n'
    '                         "wt%% U-235")\n'
    '    args = ap.parse_args()\n'
)

A_CALL = '                         workdir=args.workdir, **schedule)\n'
R_CALL = (
    '                         k_basis=args.k_basis,\n'
    '                         k_max=args.k_max,\n'
    '                         k_min=args.k_min,\n'
    '                         f_max=args.f_max,\n'
    '                         enr_max=args.enr_max,\n'
    '                         workdir=args.workdir, **schedule)\n'
)

A_GUARD = '        prev_geom = prev_meta.get("geometry")\n'
R_GUARD = (
    '        # Constraint-definition guard. Mixing constraint sets across a\n'
    '        # resumed session makes the accumulated archive describe two\n'
    '        # different optimization problems, exactly as mixing k_target or\n'
    '        # transport fidelity would. The stored g values are NOT recomputed\n'
    '        # on load, so a changed limit silently applies only to the NEW\n'
    '        # evaluations. Raising here rather than warning, because unlike a\n'
    '        # noise-level mismatch this one cannot be reasoned about after\n'
    '        # the fact from the archive alone.\n'
    '        prev_lim = prev_meta.get("limits")\n'
    '        cur_lim = {"k_basis": args.k_basis, "k_max": args.k_max,\n'
    '                   "k_min": args.k_min, "f_max": args.f_max,\n'
    '                   "enr_max": args.enr_max}\n'
    '        if prev_lim is not None:\n'
    '            diffs = [k for k, v in cur_lim.items()\n'
    '                     if k in prev_lim and prev_lim[k] != v]\n'
    '            if diffs:\n'
    '                detail = ", ".join(f"{k}: {prev_lim[k]!r} -> {v!r}"\n'
    '                                   for k, v in cur_lim.items()\n'
    '                                   if k in diffs)\n'
    '                raise SystemExit(\n'
    '                    "constraint definition differs from the checkpoint "\n'
    '                    f"({detail}). Every evaluation sharing a checkpoint "\n'
    '                    "must use the same constraint set. Start a fresh "\n'
    '                    "campaign instead of resuming this one.")\n'
    '        elif args.k_basis != "assembly" or args.k_max is not None:\n'
    '            print("!! WARNING: this checkpoint predates constraint-set "\n'
    '                  "recording and its limits cannot be verified. It was "\n'
    '                  "almost certainly written on the assembly basis at "\n'
    '                  "k_max = 1.35. Resuming it under different limits mixes "\n'
    '                  "two problems in one archive.")\n'
    '        prev_geom = prev_meta.get("geometry")\n'
)

A_META = '                           "geometry": "v2-envelope",\n'
R_META = (
    '                           "geometry": "v2-envelope",\n'
    '                           # the five numbers that define the constrained\n'
    '                           # problem, so the archive can state its own\n'
    '                           # constraint set without reading the source\n'
    '                           "limits": {"k_basis": args.k_basis,\n'
    '                                      "k_max": args.k_max,\n'
    '                                      "k_min": args.k_min,\n'
    '                                      "f_max": args.f_max,\n'
    '                                      "enr_max": args.enr_max},\n'
)

EDITS = [
    (A_ARGS, R_ARGS, "add the five constraint-definition flags"),
    (A_CALL, R_CALL, "pass the five limits into OpenMCEvaluator"),
    (A_GUARD, R_GUARD, "refuse a resume whose constraint set differs"),
    (A_META, R_META, "record the five limits in the checkpoint metadata"),
]


def verify(root: Path):
    errors = []
    path = root / FNAME
    if not path.is_file():
        return [f"missing file: {path}"]
    text = path.read_text()
    for anchor, _, label in EDITS:
        n = text.count(anchor)
        if n != 1:
            head = anchor.strip().splitlines()[0][:64]
            errors.append(f"{FNAME}: anchor for '{label}' found {n} times, "
                          f"expected 1  ({head}...)")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify anchors and print planned edits, no writes")
    ap.add_argument("--revert", action="store_true",
                    help="restore run_optimization.py from its .bak backup")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    path = root / FNAME
    print(f"repository: {root}")

    if args.revert:
        bak = root / (FNAME + ".bak")
        if bak.is_file():
            shutil.copy2(bak, path)
            print(f"restored {FNAME} from {bak.name}")
        else:
            print(f"no backup for {FNAME}, nothing to restore")
        return

    errors = verify(root)
    if errors:
        print("\nANCHOR CHECK FAILED, nothing was written:")
        for e in errors:
            print("  " + e)
        print("\nEither the tree is already patched, or a local change moved "
              "an anchor. Run --revert if a previous attempt left a backup, "
              "then send me the line the anchor expected.")
        sys.exit(1)
    print(f"anchor check: all {len(EDITS)} anchors found exactly once")

    if args.check:
        print("\nplanned edits:")
        for _, _, label in EDITS:
            print(f"  {FNAME}: {label}")
        print("\n--check given, nothing written.")
        return

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    text = path.read_text()
    for anchor, repl, label in EDITS:
        text = text.replace(anchor, repl, 1)
        print(f"patched  {FNAME}: {label}")
    path.write_text(text)
    print(f"\nbackup written as {FNAME}.bak")

    print("\nNEXT STEPS")
    print("  1. python3 -m py_compile run_optimization.py")
    print("  2. python3 run_optimization.py --help | grep -A2 'k-basis'")
    print("  3. python3 run_optimization.py --smoke --workdir smoke_kb "
          "--out out_smoke_kb")
    print("     (no new flags, so this must reproduce Campaign 5 behaviour)")
    print("  4. Only then launch Campaign 6 with the basis stated:")
    print("     --k-basis core --k-max 1.35 --workdir openmc_runs_c6 "
          "--out out_c6")
    print("\n  The resume guard now RAISES on a constraint mismatch, so "
          "out_c5 cannot be resumed under the core basis. That is deliberate.")


if __name__ == "__main__":
    main()
