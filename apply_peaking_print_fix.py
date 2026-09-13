#!/usr/bin/env python3
r"""
apply_peaking_print_fix.py
==========================
The per-case console line of OpenMCEvaluator.evaluate_one prints the label
F_dh next to the ASSEMBLY peaking, while the objective and the g_peak
constraint both use the CORE peaking. Both quantities are archived
correctly ("peaking" is core["fdh_core"], "peaking_asm" is the assembly
value), so no result is wrong. Only the log is misleading, and it has been
since the objective moved to the core basis in Campaign 4.

The two differ by about a factor 1.31 on the Campaign 8 archive, because
the assembly solve has reflective boundaries and therefore sees only
pin-to-pin structure, while the core solve also sees the radial tilt
across the three-ring loading map. Reading the log as if it showed the
objective understates it by roughly 0.39.

    Campaign 8, 60 designs, full fidelity
      assembly F_dH   1.166 to 1.358
      core     F_dH   1.510 to 1.782
      ratio           1.313, sd 0.029  (2.2 % relative)
      difference     +0.386, sd 0.033  (8.4 % relative)

This patch prints both, core first under the F_dh label, assembly in
parentheses. Nothing else changes: no archived key, no objective, no
constraint, no return value. Campaigns 1 to 9 are unaffected, and the
archives already on disk stay valid.

USAGE, from the repository root on main
    python apply_peaking_print_fix.py --selftest
    python apply_peaking_print_fix.py --check
    python apply_peaking_print_fix.py
    python apply_peaking_print_fix.py --revert
"""

import argparse
import datetime
import pathlib
import sys

FILE = "openmc_evaluator.py"
MARKER = "(asm {peaking:.3f})"

OLD = '                  f"F_dh={peaking:.3f} k_bol={k_bol:.4f} "\n'
NEW = ('                  # F_dh is the CORE value, the objective and the g_peak\n'
       '                  # constraint. The assembly value follows in brackets as\n'
       '                  # a diagnostic: it is archived as "peaking_asm" and runs\n'
       '                  # about a factor 1.31 lower, since a reflective-boundary\n'
       '                  # assembly cannot see the radial tilt of the core.\n'
       '                  f"F_dh={core[\'fdh_core\']:.3f} (asm {peaking:.3f}) "\n'
       '                  f"k_bol={k_bol:.4f} "\n')


def selftest():
    assert OLD != NEW
    assert MARKER in NEW and MARKER not in OLD
    assert "fdh_core" in NEW
    # the replacement must be a syntactically valid run of f-string pieces
    import ast
    ast.parse("print(\n" + NEW.replace("                  ", "    ") + ")")
    print("selftest OK")
    return 0


def run(mode):
    p = pathlib.Path(FILE)
    if not p.is_file():
        sys.exit(f"FAIL: {FILE} not found. Run from the repository root.")
    text = p.read_text(encoding="utf-8")
    if MARKER in text:
        sys.exit("Already applied (marker found). Nothing to do.")
    n = text.count(OLD)
    print(f"  [{'OK  ' if n == 1 else 'FAIL'}] {FILE}  {n} match  case-line peaking label")
    if n != 1:
        sys.exit("\nAnchor did not match exactly once. Nothing written.")
    if mode == "check":
        print("\nanchor unique, nothing written")
        return 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = pathlib.Path(f"{FILE}.bak_{stamp}")
    bak.write_text(text, encoding="utf-8")
    p.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    import py_compile
    py_compile.compile(FILE, doraise=True)
    print(f"  wrote {FILE}  (backup {bak.name}), compiles")
    return 0


def revert():
    baks = sorted(pathlib.Path(".").glob(f"{FILE}.bak_*"))
    if not baks:
        print(f"  no backup for {FILE}")
        return 1
    pathlib.Path(FILE).write_text(baks[-1].read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  restored {FILE} from {baks[-1].name}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.revert:
        sys.exit(revert())
    sys.exit(run("check" if a.check else "apply"))
