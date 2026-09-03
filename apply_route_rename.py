#!/usr/bin/env python3
"""
apply_route_rename.py
=====================
Removes the naming clash in the simulation repository.

  * reactor_model.py, sweep_ktarget.py and the thesis call "Route A" the
    reflected-assembly SLEEVE model (explicit reflector around the
    depleting assembly) and "Route B" the pre-computed k_target table.
  * run_optimization.py and openmc_evaluator.py call "ROUTE A" the FROZEN
    CONSTANT k_target (one number for every design), which is not a
    leakage construction at all but a legacy calibration option.

This script rewrites the second usage as "FROZEN TARGET (legacy)" so that
"Route A" has one meaning everywhere. Only comments, docstrings and help
strings change. No executable statement is touched, so results are
bit-for-bit identical.

Run from the repository root, on a clean working tree:

    python -c "import numpy, openmc; print('env ok')" && \
    git status --short && python apply_route_rename.py && git diff --stat

Every replacement is exact. If a string is not found (the file changed
since this patch was written) the script stops and nothing is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

EDITS = {
    "run_optimization.py": [
        ("# ROUTE A default: a single FROZEN leakage-corrected EOC (End Of Cycle)\n"
         "# target, the SAME for every design regardless of refl_thick.",
         "# FROZEN TARGET (legacy, formerly labelled 'ROUTE A' in this file): a\n"
         "# single FROZEN leakage-corrected EOC (End Of Cycle) target, the SAME for\n"
         "# every design regardless of refl_thick. NOTE: in reactor_model.py,\n"
         "# sweep_ktarget.py and the thesis, 'Route A' means the reflected-assembly\n"
         "# sleeve model, NOT this constant."),
        ("# ROUTE B (reflector thickness IS a real design variable): this constant is",
         "# ROUTE B (reflector thickness IS a real design variable): this constant is"),
        ('help="ROUTE A: frozen leakage-corrected EOC target "',
         'help="FROZEN TARGET (legacy): frozen leakage-corrected EOC target "'),
        ("# ROUTE A (float) vs ROUTE B (per-design table) -- computed once, used by",
         "# FROZEN TARGET (float) vs ROUTE B (per-design table) -- computed once, used by"),
        ("# Numeric vs numeric (both Route A): tolerate float noise. Any",
         "# Numeric vs numeric (both frozen targets): tolerate float noise. Any"),
        ("# other combination (a table PATH string, or Route A meeting",
         "# other combination (a table PATH string, or a frozen target meeting"),
    ],
    "openmc_evaluator.py": [
        ("(End Of Cycle) target -- Route A frozen value or Route B per-design",
         "(End Of Cycle) target -- a frozen value (legacy) or the Route B per-design"),
        ("        ROUTE A -- a single float: the FROZEN leakage-corrected EOC target,",
         "        FROZEN TARGET (legacy) -- a single float: the FROZEN leakage-corrected EOC target,"),
        ("        # ROUTE A (float) vs ROUTE B (1-D or 2-D table) -- detected once here;",
         "        # FROZEN TARGET (float) vs ROUTE B (1-D or 2-D table) -- detected once here;"),
        ("    # EOC target for THIS design -- frozen (Route A), refl-interpolated   #",
         "    # EOC target for THIS design -- frozen (legacy), refl-interpolated    #"),
    ],
    "sweep_ktarget.py": [
        ("ROUTE A vs ROUTE B (unchanged rule)\n"
         "-----------------------------------\n",
         "ROUTE A vs ROUTE B (unchanged rule)\n"
         "-----------------------------------\n"
         "ROUTE A (thesis nomenclature): the reflected-assembly SLEEVE model,\n"
         "make_assembly_model(reflector=True), where the leakage is computed by\n"
         "the transport solver inside the depletion. Implemented, not used in\n"
         "any campaign. (The frozen constant K_TARGET of run_optimization.py is\n"
         "a legacy calibration option, not a route.)\n"),
    ],
}


def main():
    root = Path.cwd()
    staged = {}
    for fname, edits in EDITS.items():
        path = root / fname
        if not path.exists():
            sys.exit(f"{fname} not found. Run from the repository root.")
        text = path.read_text()
        for old, new in edits:
            n = text.count(old)
            if n != 1:
                sys.exit(f"{fname}: expected exactly one occurrence of\n"
                         f"    {old[:70]!r}\n  found {n}. Nothing written.")
            text = text.replace(old, new)
        staged[path] = text
    for path, text in staged.items():
        path.write_text(text)
        print(f"rewrote {path.name}")
    print("done. Review with: git diff")


if __name__ == "__main__":
    main()
