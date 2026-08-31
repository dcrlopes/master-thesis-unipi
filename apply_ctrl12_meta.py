#!/usr/bin/env python3
"""
apply_ctrl12_meta.py -- record the provenance of g_ctrl12 in the checkpoint.

THE GAP
-------
apply_ctrl12.py made the evaluator measure and store k_re12, F_re12,
g_ctrl12 and t_ctrl12_s on every design. But the checkpoint meta block
"ctrl_screen" still records only the sixteen ALL-RE positions:

    "ctrl_screen": {"enabled": ..., "margin_pcm": ..., "absorber": ...,
                    "re_positions": [ ...sixteen... ]}

So a reader of out_c8/optimization_checkpoint.json can see g_ctrl12 in
every record but cannot tell WHICH eight assemblies produced it, nor that
it was deliberately excluded from the constraint set rather than lost. For
a campaign whose Pareto front is presented as candidate designs, that
provenance belongs in the file, not only in the code that wrote it.

THE FIX
-------
Extend the "ctrl_screen" meta block with four keys:

    re12_positions        the eight RE1 + RE2 positions, sorted, read from
                          zoning.RE12_POSITIONS so the file can never drift
                          from the code that solved them
    re12_constrained      False, stating explicitly that g_ctrl12 is
                          recorded and not enforced
    re12_note             one sentence on what the quantity means
    bank_definitions      which named banks make up each set

Metadata only. No design, no constraint, no eigenvalue and no cost changes.
Campaign 7 checkpoints are unaffected because the block is only written when
--ctrl-margin is passed.

REQUIRES
    apply_ctrl12.py applied first (it creates zoning.RE12_POSITIONS).

USAGE
    python3 apply_ctrl12_meta.py --check     report, change nothing
    python3 apply_ctrl12_meta.py            apply, writing the .bak
    python3 apply_ctrl12_meta.py --revert   restore from the .bak
    python3 apply_ctrl12_meta.py --selftest compile + AST + render checks
"""

from __future__ import annotations

import argparse
import ast
import py_compile
import shutil
import sys
from pathlib import Path

EDITS = []

EDITS.append(dict(
    path="run_optimization.py",
    name="M1 ctrl_screen records the RE12 provenance",
    old='''                           "ctrl_screen": {
                               "enabled": args.ctrl_margin is not None,
                               "margin_pcm": args.ctrl_margin,
                               "absorber": args.ctrl_absorber,
                               "re_positions": sorted(
                                   _zn.RE_BANK_POSITIONS)},
''',
    new='''                           "ctrl_screen": {
                               "enabled": args.ctrl_margin is not None,
                               "margin_pcm": args.ctrl_margin,
                               "absorber": args.ctrl_absorber,
                               "re_positions": sorted(
                                   _zn.RE_BANK_POSITIONS),
                               # CAMPAIGN 8: provenance of g_ctrl12. Read
                               # from zoning so the recorded positions can
                               # never drift from the ones actually solved.
                               "re12_positions": sorted(
                                   _zn.RE12_POSITIONS),
                               "re12_constrained": False,
                               "bank_definitions": {
                                   "ALLRE": "RE1 inner ring, RE2 M "
                                            "diagonals, RE3 and RE4 M edge "
                                            "orbits (sixteen assemblies)",
                                   "RE12": "RE1 inner ring and RE2 M "
                                           "diagonals (eight assemblies)",
                                   "SH": "outer sixteen assemblies, "
                                         "reserved for scram, never "
                                         "inserted in either screen"},
                               "re12_note": (
                                   "k_re12 is the beginning-of-life core "
                                   "eigenvalue with only the first two "
                                   "regulating banks inserted. g_ctrl12 = "
                                   "k_re12 - (1 - margin) is RECORDED and "
                                   "NOT constrained: it never enters "
                                   "constraint_names, so feasibility is set "
                                   "by the ALL-RE screen alone and the "
                                   "front can afterwards be split by "
                                   "whether two banks suffice.")},
''',
))


def files():
    return sorted({e["path"] for e in EDITS})


def check(apply: bool) -> int:
    ok = True
    for e in EDITS:
        text = Path(e["path"]).read_text()
        n_old, n_new = text.count(e["old"]), text.count(e["new"])
        inside = n_new * e["new"].count(e["old"])
        n_old_outside = n_old - inside
        state = ("APPLIED" if (n_new == 1 and n_old_outside == 0) else
                 "ready" if (n_new == 0 and n_old == 1) else
                 f"FAIL (old x{n_old_outside} outside, new x{n_new})")
        print(f"  {e['name']:<45s} {state}")
        if state.startswith("FAIL"):
            ok = False
    if not ok:
        print("Anchors are not unique. The file changed since this patch was "
              "written. Apply the intent by hand.")
        return 2
    if not apply:
        return 0
    for p in files():
        shutil.copy2(p, p + ".bak.meta")
        print(f"  backup -> {p}.bak.meta")
    for e in EDITS:
        p = Path(e["path"])
        t = p.read_text()
        if e["old"] in t:
            p.write_text(t.replace(e["old"], e["new"], 1))
            print(f"  applied {e['name']}")
    return 0


def revert() -> int:
    rc = 0
    for p in files():
        b = Path(p + ".bak.meta")
        if b.exists():
            shutil.copy2(b, p)
            print(f"  reverted {p}")
        else:
            print(f"  MISSING {b}, cannot revert {p}")
            rc = 2
    return rc


def _frozenset_literal(tree, name):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", "") == name):
            return frozenset(ast.literal_eval(node.value.args[0]))
    return None


def selftest() -> int:
    print("selftest:")
    py_compile.compile("run_optimization.py", doraise=True)
    print("  compiles: run_optimization.py")

    # the prerequisite must be in place, or the meta would raise at runtime
    ztree = ast.parse(Path("zoning.py").read_text())
    allre = _frozenset_literal(ztree, "RE_BANK_POSITIONS")
    re12 = _frozenset_literal(ztree, "RE12_POSITIONS")
    if re12 is None:
        print("  FAIL: zoning.RE12_POSITIONS not found. Run apply_ctrl12.py "
              "first, otherwise this meta block raises AttributeError at "
              "launch.")
        return 2
    if not (len(re12) == 8 and len(allre) == 16 and re12 <= allre):
        print(f"  FAIL: |RE12| = {len(re12)}, |ALLRE| = {len(allre)}, "
              f"subset = {re12 <= allre}")
        return 2
    print(f"  zoning: RE12 is an {len(re12)}-position subset of "
          f"{len(allre)}")

    src = Path("run_optimization.py").read_text()
    for key in ('"re12_positions"', '"re12_constrained": False',
                '"bank_definitions"', '"re12_note"'):
        if key not in src:
            print(f"  FAIL: {key} missing from the meta block")
            return 2
    print("  meta block carries all four new keys")

    # the promise the note makes must still be true in the code
    if '"g_ctrl12"' in src:
        print("  FAIL: g_ctrl12 appears in run_optimization.py, so it may be "
              "constrained. The note would be false.")
        return 2
    print("  g_ctrl12 still absent from the constraint machinery")

    # render the block in isolation, so a syntax or key error cannot reach
    # the launch, and confirm it is JSON-serialisable
    import json
    i0 = src.index('"ctrl_screen": {')
    depth, i = 0, src.index("{", i0)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block = src[src.index("{", i0):i + 1]

    class _A:
        ctrl_margin, ctrl_absorber = 1000.0, "B4C"

    class _Z:
        RE_BANK_POSITIONS, RE12_POSITIONS = allre, re12

    rendered = eval(block, {"args": _A(), "_zn": _Z(), "sorted": sorted})
    json.dumps(rendered)
    if len(rendered["re12_positions"]) != 8:
        print("  FAIL: the rendered block does not carry eight positions")
        return 2
    print(f"  rendered ok, re12_positions = "
          f"{[list(p) for p in rendered['re12_positions']]}")
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
    print("OpenMC required: no (metadata only, string edits and checks)")
    missing = [p for p in files() + ["zoning.py"] if not Path(p).exists()]
    if missing:
        print(f"FAIL: not in the repository root, missing {missing}")
        return 2
    if a.revert:
        return revert()
    if a.selftest:
        return selftest()
    # PRE-FLIGHT: the meta block reads _zn.RE12_POSITIONS at launch, so the
    # prerequisite must be verified BEFORE anything is written. Otherwise a
    # failed selftest would leave a tree that raises AttributeError on the
    # real run.
    ztree = ast.parse(Path("zoning.py").read_text())
    if _frozenset_literal(ztree, "RE12_POSITIONS") is None:
        print("FAIL: zoning.RE12_POSITIONS not found. Run apply_ctrl12.py "
              "first. Nothing was written.")
        return 2
    rc = check(apply=not a.check)
    if rc == 0 and not a.check:
        rc = selftest()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
