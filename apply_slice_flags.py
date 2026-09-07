#!/usr/bin/env python3
"""
apply_slice_flags.py -- three flags for run_optimization.py, applied as
exact string edits. Written against branch main, 7 September 2026, for the
Step 3 verification of the surrogate-assisted search against an
exhaustive grid (valgrid_*.py, run_valgrid.sh).

WHAT IT CHANGES (and nothing else), all in run_optimization.py
---------------------------------------------------------------
E1  argparse: --enr-box-high, --freeze (repeatable), --eval-list.
E2  after the ENR-BOX loop: apply --enr-box-high to the enrichment box,
    refuse an empty box, then freeze the listed variables through
    slice_space.freeze_variables (the search box shrinks, every design
    dict stays complete).
E3  resume guard: a checkpoint written under different frozen variables
    is refused, the same way a changed constraint set is refused.
E4  meta record: "frozen" and "eval_list" are written into the checkpoint
    so the archive states its own slice.
E5  branch: with --eval-list the run enumerates the listed designs
    (slice_space.run_eval_list, checkpoint after every evaluation)
    instead of DOE + infill. Everything after (results, checkpoint,
    figure) is unchanged.

Behaviour without the new flags is bit-for-bit the previous behaviour:
every new code path sits behind an `if args.<flag>`.

USAGE
    python3 apply_slice_flags.py --check     report, change nothing
    python3 apply_slice_flags.py            apply, writing .bak
    python3 apply_slice_flags.py --revert   restore from .bak
    python3 apply_slice_flags.py --selftest compile + AST checks

The script refuses to write unless every anchor occurs exactly once.
"""
from __future__ import annotations

import argparse
import ast
import py_compile
import shutil
import sys
from pathlib import Path

TARGET = "run_optimization.py"
EDITS = []

# --------------------------------------------------------------------- E1
EDITS.append(dict(
    path=TARGET,
    name="E1 argparse: --enr-box-high, --freeze, --eval-list",
    old='''    ap.add_argument("--enr-box-low", type=float, default=None,
                    help="ENR-BOX: optional lower bound of the enrichment "
                         "search box in wt%% (design variable), to stop the "
                         "design of experiments sampling designs that fail "
                         "k_min. Unset keeps the 2.0 wt%% floor.")
''',
    new='''    ap.add_argument("--enr-box-low", type=float, default=None,
                    help="ENR-BOX: optional lower bound of the enrichment "
                         "search box in wt%% (design variable), to stop the "
                         "design of experiments sampling designs that fail "
                         "k_min. Unset keeps the 2.0 wt%% floor.")
    ap.add_argument("--enr-box-high", type=float, default=None,
                    help="ENR-BOX: optional upper bound of the enrichment "
                         "search box in wt%% (design variable), applied "
                         "AFTER the LEU cap, so it can only narrow the box. "
                         "Leaves --enr-max and the g_enr limit untouched.")
    ap.add_argument("--freeze", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="SLICE: remove NAME from the search box and hold "
                         "it at VALUE in every design (repeatable, e.g. "
                         "--freeze refl_thick=4.2889 --freeze gd_pins=12). "
                         "The enrichment cannot be frozen. See "
                         "slice_space.py.")
    ap.add_argument("--eval-list", default=None, metavar="PATH.json",
                    help="ENUMERATION MODE: evaluate exactly the designs "
                         "listed in PATH.json (keys = live design "
                         "variables), one at a time, checkpoint after "
                         "each, skipping designs already in the archive. "
                         "Replaces the DOE and the infill loop. Combine "
                         "with --resume to finish an interrupted list.")
''',
))

# --------------------------------------------------------------------- E2
EDITS.append(dict(
    path=TARGET,
    name="E2 enr-box-high and freeze after the ENR-BOX loop",
    old='''            if args.enr_box_low is not None:
                _v.low = max(_v.low, float(args.enr_box_low))
''',
    new='''            if args.enr_box_low is not None:
                _v.low = max(_v.low, float(args.enr_box_low))
            if args.enr_box_high is not None:
                _v.high = min(_v.high, float(args.enr_box_high))
            if not _v.low < _v.high:
                raise SystemExit(
                    f"enrichment search box is empty: low {_v.low:g} >= "
                    f"high {_v.high:g} (--enr-max, --enr-box-low, "
                    f"--enr-box-high)")
    # SLICE: freeze design variables. The spec is rebuilt with a shorter
    # vector; every design dict is still complete (slice_space.py).
    if args.freeze:
        from slice_space import freeze_variables, parse_freeze
        _frozen = parse_freeze(args.freeze)
        spec = freeze_variables(spec, _frozen)
        print(f"SLICE: frozen {_frozen} | live variables "
              f"{spec.design_space.names}")
    else:
        _frozen = {}
''',
))

# --------------------------------------------------------------------- E3
EDITS.append(dict(
    path=TARGET,
    name="E3 resume guard on the frozen variables",
    old='''        prev_geom = prev_meta.get("geometry")
''',
    new='''        # SLICE: an archive is one slice of one problem. Mixing slices in
        # a resumed session is the same error as mixing constraint sets.
        prev_frozen = prev_meta.get("frozen")
        if prev_frozen is not None and \\
                {k: float(v) for k, v in dict(prev_frozen).items()} != _frozen:
            raise SystemExit(
                f"frozen variables differ from the checkpoint: "
                f"{prev_frozen} vs {_frozen}. Pass the same --freeze "
                f"flags, or start a fresh run.")
        prev_geom = prev_meta.get("geometry")
''',
))

# --------------------------------------------------------------------- E4
EDITS.append(dict(
    path=TARGET,
    name="E4 meta records frozen variables and the eval list",
    old='''                           "started_utc": datetime.now(timezone.utc)
                               .isoformat(timespec="seconds")}
''',
    new='''                           "started_utc": datetime.now(timezone.utc)
                               .isoformat(timespec="seconds")}
    # SLICE: the archive states its own slice and its own enumeration.
    opt.checkpoint_meta["frozen"] = dict(_frozen)
    opt.checkpoint_meta["eval_list"] = (str(args.eval_list)
                                        if args.eval_list else None)
    opt.checkpoint_meta["enr_box_high"] = args.enr_box_high
''',
))

# --------------------------------------------------------------------- E5
EDITS.append(dict(
    path=TARGET,
    name="E5 enumeration branch instead of DOE + infill",
    old='''    res = opt.run(verbose=True)
''',
    new='''    if args.eval_list:
        # ENUMERATION MODE (slice_space.run_eval_list): listed designs,
        # one at a time, checkpoint after each, resume-safe.
        from slice_space import run_eval_list
        res = run_eval_list(opt, args.eval_list, ckpt_out,
                            opt.checkpoint_meta, verbose=True)
    else:
        res = opt.run(verbose=True)
''',
))


def files():
    return sorted({e["path"] for e in EDITS})


def check(apply: bool) -> int:
    print("anchors:")
    ok = True
    for e in EDITS:
        text = Path(e["path"]).read_text()
        n_old, n_new = text.count(e["old"]), text.count(e["new"])
        inside = n_new * e["new"].count(e["old"])
        n_old_outside = n_old - inside
        state = ("APPLIED" if (n_new == 1 and n_old_outside == 0) else
                 "ready" if (n_new == 0 and n_old == 1) else
                 f"FAIL (old x{n_old_outside} outside, new x{n_new})")
        print(f"  {e['name']:<55s} {state}")
        if state.startswith("FAIL"):
            ok = False
    if not ok:
        print("Anchors are not unique. The file changed since this patch "
              "was written. Apply the intent by hand.")
        return 2
    if not apply:
        return 0
    if not Path("slice_space.py").exists():
        print("FAIL: slice_space.py is not in the repository root; the "
              "edited driver imports it.")
        return 2
    for p in files():
        shutil.copy2(p, p + ".bak")
        print(f"  backup -> {p}.bak")
    for e in EDITS:
        p = Path(e["path"])
        t = p.read_text()
        if e["old"] in t and e["new"] not in t:
            p.write_text(t.replace(e["old"], e["new"], 1))
            print(f"  applied {e['name']}")
    return 0


def revert() -> int:
    rc = 0
    for p in files():
        b = Path(p + ".bak")
        if b.exists():
            shutil.copy2(b, p)
            print(f"  reverted {p}")
        else:
            print(f"  MISSING {b}, cannot revert {p}")
            rc = 2
    return rc


def selftest() -> int:
    print("selftest:")
    for p in files() + ["slice_space.py"]:
        py_compile.compile(p, doraise=True)
        print(f"  compiles: {p}")
    tree = ast.parse(Path(TARGET).read_text())
    flags = {n.args[0].value for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "add_argument"
             and n.args and isinstance(n.args[0], ast.Constant)}
    need = {"--enr-box-high", "--freeze", "--eval-list"}
    if not need <= flags:
        print(f"  FAIL: flags missing {sorted(need - flags)}")
        return 2
    print(f"  flags present: {sorted(need)}")
    src = Path(TARGET).read_text()
    for token in ("freeze_variables(spec, _frozen)",
                  "run_eval_list(opt, args.eval_list, ckpt_out",
                  'opt.checkpoint_meta["frozen"]',
                  'prev_meta.get("frozen")'):
        if src.count(token) != 1:
            print(f"  FAIL: expected exactly one occurrence of {token!r}")
            return 2
    # the enumeration branch must come AFTER the meta is assigned
    if src.index('opt.checkpoint_meta["frozen"]') > src.index(
            "run_eval_list(opt, args.eval_list"):
        print("  FAIL: meta must be recorded before the enumeration branch")
        return 2
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    print(f"python  : {sys.version.split()[0]}")
    print(f"cwd     : {Path.cwd()}")
    print("OpenMC required: no (string edits, compile and AST checks only)")
    missing = [p for p in files() if not Path(p).exists()]
    if missing:
        print(f"FAIL: not in the repository root, missing {missing}")
        return 2
    if a.revert:
        return revert()
    if a.selftest:
        return selftest()
    rc = check(apply=not a.check)
    if rc == 0 and not a.check:
        rc = selftest()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
