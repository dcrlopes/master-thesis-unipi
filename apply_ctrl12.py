#!/usr/bin/env python3
"""
apply_ctrl12.py -- Campaign 8 two-level control screen, plus the timing fix.

WHAT IT CHANGES (and nothing else)
----------------------------------
1. zoning.py: adds RE12_POSITIONS, the first two regulating banks
   (RE1 inner ring + RE2 M diagonals, eight assemblies), as a named
   subset of RE_BANK_POSITIONS.
2. openmc_evaluator.py, _ctrl_solve: takes optional `positions` and
   `salt` so the same solver serves the ALL-RE screen and the RE1+RE2
   reading, each with its own deterministic seed and case directory.
   Defaults reproduce the Campaign 7 behaviour bit for bit.
3. openmc_evaluator.py, the result dictionary: when --ctrl-margin is
   active, one additional zoned core solve with only RE1+RE2 inserted,
   recorded as k_re12, F_re12, t_ctrl12_s and g_ctrl12.
   g_ctrl12 is RECORDED, NOT CONSTRAINED: it never enters
   constraint_names, so feasibility is unchanged, and the Pareto front
   can afterwards be split into designs controllable with two banks and
   designs needing all four.
4. openmc_evaluator.py, timing: t_eval_s now includes every control
   solve, closing the Campaign 7 under-report of 12.6 per cent, and the
   per-case log line prints the full cost.

USAGE
    python3 apply_ctrl12.py --check     report, change nothing
    python3 apply_ctrl12.py            apply, writing .bak files
    python3 apply_ctrl12.py --revert   restore from the .bak files
    python3 apply_ctrl12.py --selftest compile + AST subset check

The script refuses to write unless every anchor occurs exactly once.
"""

from __future__ import annotations

import argparse
import ast
import py_compile
import shutil
import sys
from pathlib import Path

EDITS = []

# --------------------------------------------------------------------- Z1
EDITS.append(dict(
    path="zoning.py",
    name="Z1 RE12_POSITIONS named subset",
    old='''    (1, 3), (3, 4), (4, 2), (2, 1),          # RE4  M edges, orbit B
])
''',
    new='''    (1, 3), (3, 4), (4, 2), (2, 1),          # RE4  M edges, orbit B
])

# CAMPAIGN 8: the first two regulating banks alone (RE1 + RE2, eight
# assemblies). Recorded next to the ALL-RE screen so the front can be
# split into designs controllable with two banks and designs needing all
# four. Must stay a subset of RE_BANK_POSITIONS (checked by the applier).
RE12_POSITIONS = frozenset([
    (2, 2), (2, 3), (3, 2), (3, 3),          # RE1  inner ring
    (1, 1), (1, 4), (4, 1), (4, 4),          # RE2  M diagonals
])
''',
))

# --------------------------------------------------------------------- V1
EDITS.append(dict(
    path="openmc_evaluator.py",
    name="V1 _ctrl_solve takes positions and salt",
    old='''def _ctrl_solve(ev, design):
    """One zoned core solve with the sixteen regulating-bank CRAs inserted
    (zn.RE_BANK_POSITIONS), SH banks out. Same fidelity, zoning path and
    deterministic seeding as every other core solve; the case directory is
    keyed by the design hash so re-evaluations reuse it."""
    tag = _design_seed(design, salt="ctrl") & 0xFFFFFFFF
    return zn.core_bol_solve(
        design, zn.evaluator_design_map(design), ev.op, ev.geo,
        particles=ev.core_particles, batches=ev.core_batches,
        inactive=ev.core_inactive,
        seed=_design_seed(design, salt="ctrl"),
        case=ev.workdir / f"ctrl_{tag:08x}",
        rodded_map=(set(zn.RE_BANK_POSITIONS),
                    getattr(ev, "ctrl_absorber", "B4C")))
''',
    new='''def _ctrl_solve(ev, design, positions=None, salt="ctrl"):
    """One zoned core solve with a set of regulating-bank CRAs inserted,
    SH banks out. Same fidelity, zoning path and deterministic seeding as
    every other core solve; the case directory is keyed by the design hash
    AND the salt, so the ALL-RE and the RE1+RE2 readings each reuse their
    own cache. Defaults (positions=None, salt="ctrl") reproduce the
    Campaign 7 ALL-RE behaviour bit for bit."""
    pos = zn.RE_BANK_POSITIONS if positions is None else positions
    tag = _design_seed(design, salt=salt) & 0xFFFFFFFF
    return zn.core_bol_solve(
        design, zn.evaluator_design_map(design), ev.op, ev.geo,
        particles=ev.core_particles, batches=ev.core_batches,
        inactive=ev.core_inactive,
        seed=_design_seed(design, salt=salt),
        case=ev.workdir / f"{salt}_{tag:08x}",
        rodded_map=(set(pos),
                    getattr(ev, "ctrl_absorber", "B4C")))
''',
))

# --------------------------------------------------------------------- V2
EDITS.append(dict(
    path="openmc_evaluator.py",
    name="V2 RE1+RE2 reading recorded next to ALL-RE",
    old='''            **({"k_allre": (_c := _ctrl_solve(self, design))["keff"],
                "F_allre": _c["fdh_core"],
                "t_ctrl_s": _c.get("wall_s"),
                "g_ctrl":  _c["keff"] - (1.0 - self.ctrl_margin_dk)}
               if getattr(self, "ctrl_margin_dk", None) is not None
               else {}),
''',
    new='''            **({"k_allre": (_c := _ctrl_solve(self, design))["keff"],
                "F_allre": _c["fdh_core"],
                "t_ctrl_s": _c.get("wall_s"),
                "g_ctrl":  _c["keff"] - (1.0 - self.ctrl_margin_dk),
                # CAMPAIGN 8: first-two-banks reading (RE1 + RE2, eight
                # CRAs). RECORDED ONLY: g_ctrl12 is not appended to
                # constraint_names, so it never enters feasibility. It
                # lets the front be split by whether the first two banks
                # alone hold the core subcritical by the same margin.
                "k_re12": (_c2 := _ctrl_solve(self, design,
                                              positions=zn.RE12_POSITIONS,
                                              salt="ctrl12"))["keff"],
                "F_re12": _c2["fdh_core"],
                "t_ctrl12_s": _c2.get("wall_s"),
                "g_ctrl12": _c2["keff"] - (1.0 - self.ctrl_margin_dk)}
               if getattr(self, "ctrl_margin_dk", None) is not None
               else {}),
''',
))

# --------------------------------------------------------------------- V3
EDITS.append(dict(
    path="openmc_evaluator.py",
    name="V3 control solves folded into t_eval_s",
    old='''        }
        if self.verbose:
''',
    new='''        }
        # CAMPAIGN 8: the control solves are real evaluation cost. Fold
        # them into t_eval_s so the cost tables and the case line report
        # the whole evaluation. Campaign 7 under-reported t_eval_s by the
        # ctrl share, 12.6 per cent over that archive.
        for _tk in ("t_ctrl_s", "t_ctrl12_s"):
            if res.get(_tk):
                res["t_eval_s"] += float(res[_tk])
        if self.verbose:
''',
))

# --------------------------------------------------------------------- V4
EDITS.append(dict(
    path="openmc_evaluator.py",
    name="V4 case line prints the full cost",
    old='''                  f"[{n_solves} solves, "
                  f"{(t_asm + t_core + t_dep) / 60.0:.1f} min]")
''',
    new='''                  f"[{n_solves} solves, "
                  f"{res['t_eval_s'] / 60.0:.1f} min]")
''',
))


def files():
    return sorted({e["path"] for e in EDITS})


def check(apply: bool) -> int:
    ok = True
    for e in EDITS:
        text = Path(e["path"]).read_text()
        n_old, n_new = text.count(e["old"]), text.count(e["new"])
        # An anchor may legitimately survive INSIDE its own replacement when
        # the edit only appends (Z1 does: the new block starts with the old
        # text). Discount those occurrences before judging the state, or a
        # correctly applied edit reads as FAIL.
        inside = n_new * e["new"].count(e["old"])
        n_old_outside = n_old - inside
        state = ("APPLIED" if (n_new == 1 and n_old_outside == 0) else
                 "ready" if (n_new == 0 and n_old == 1) else
                 f"FAIL (old x{n_old_outside} outside, new x{n_new})")
        print(f"  {e['name']:<45s} {state}")
        if state.startswith("FAIL"):
            ok = False
    if not ok:
        print("Anchors are not unique. The files changed since this patch "
              "was written. Apply the intent by hand.")
        return 2
    if not apply:
        return 0
    for p in files():
        shutil.copy2(p, p + ".bak")
        print(f"  backup -> {p}.bak")
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
        b = Path(p + ".bak")
        if b.exists():
            shutil.copy2(b, p)
            print(f"  reverted {p}")
        else:
            print(f"  MISSING {b}, cannot revert {p}")
            rc = 2
    return rc


def _frozenset_literal(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", "") == name):
            return frozenset(ast.literal_eval(node.value.args[0]))
    return None


def selftest() -> int:
    print("selftest:")
    for p in files():
        py_compile.compile(p, doraise=True)
        print(f"  compiles: {p}")
    tree = ast.parse(Path("zoning.py").read_text())
    allre = _frozenset_literal(tree, "RE_BANK_POSITIONS")
    re12 = _frozenset_literal(tree, "RE12_POSITIONS")
    if allre is None or re12 is None:
        print("  FAIL: could not read the two position sets from zoning.py")
        return 2
    if len(re12) != 8 or not re12 <= allre or len(allre) != 16:
        print(f"  FAIL: |RE12| = {len(re12)}, |ALLRE| = {len(allre)}, "
              f"subset = {re12 <= allre}")
        return 2
    print(f"  RE12 is a {len(re12)}-position subset of the "
          f"{len(allre)}-position ALL-RE set")
    src = Path("openmc_evaluator.py").read_text()
    for key in ("k_re12", "F_re12", "g_ctrl12", "t_ctrl12_s"):
        if key not in src:
            print(f"  FAIL: {key} not present in openmc_evaluator.py")
            return 2
    if '"g_ctrl12"' in Path("run_optimization.py").read_text():
        print("  FAIL: g_ctrl12 must NOT appear in run_optimization.py "
              "(recorded, not constrained)")
        return 2
    print("  g_ctrl12 is recorded only, not a constraint")
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
