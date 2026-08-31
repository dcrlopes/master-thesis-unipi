#!/usr/bin/env python3
"""
apply_c8_space.py -- Campaign 8 design space, applied as exact string edits.

WHAT IT CHANGES (and nothing else)
----------------------------------
1. reactor_optimization.py, DesignSpace.as_dict:
   the single vector-to-design choke point now DERIVES
   enrich_inner = enrich_outer = enrich and pitch = 1.26 cm, so every
   downstream consumer (evaluator, zoning, k_target interpolation, logs,
   archive) keeps its keys and its behaviour.
2. reactor_optimization.py, the DesignSpace list: four variables
   (enrich, gd_wt, refl_thick, gd_pins), reflector re-bounded to
   [2.0, 5.66] cm for the fixed pitch and the new vessel clearance.
3. run_optimization.py: the --enr-max / --enr-box-low box logic and the
   meta record accept the variable name "enrich".
4. core_geometry.py: VESSEL_CLEARANCE_CM = 7.08 cm, the 5.08 cm SS304L
   barrel of the NuScale-like benchmark plus the 2.0 cm downcomer
   decided for Campaign 8. This tightens g_geom uniformly.

GEOMETRY CHECK BEHIND THE BOUNDS (pitch fixed at 1.26 cm)
    R_env = 77.231 cm, pad 0.02 cm, vessel 90.0 cm, clearance 7.08 cm
    -> max reflector = 90.0 - 7.08 - 77.231 - 0.02 = 5.669 cm
    The design-variable bound is 5.66 cm so the box edge itself is
    strictly feasible and g_geom stays the analytic guard.

USAGE
    python3 apply_c8_space.py --check     report, change nothing
    python3 apply_c8_space.py            apply, writing .bak files
    python3 apply_c8_space.py --revert   restore every file from .bak
    python3 apply_c8_space.py --selftest compile + AST checks after apply

The script refuses to write unless every anchor occurs exactly once in its
file. Reverting restores all files or reports which .bak is missing.
"""

from __future__ import annotations

import argparse
import ast
import py_compile
import shutil
import sys
from pathlib import Path

C8_PITCH = 1.26
C8_REFL_HI = 5.66
C8_CLEARANCE = 7.08

EDITS = []

# --------------------------------------------------------------------- E1
EDITS.append(dict(
    path="reactor_optimization.py",
    name="E1 as_dict derives the collapsed keys",
    old='''    def as_dict(self, x: Sequence[float]) -> dict:
        return {v.name: float(xi) for v, xi in zip(self.variables, x)}
''',
    new='''    def as_dict(self, x: Sequence[float]) -> dict:
        d = {v.name: float(xi) for v, xi in zip(self.variables, x)}
        # CAMPAIGN 8: single-enrichment design space with a fixed pitch.
        # The optimiser sees "enrich" only. Every downstream consumer
        # (evaluator, zoning path, k_target interpolation, log line,
        # archive) keeps reading enrich_inner / enrich_outer / pitch, so
        # the keys are DERIVED here, at the single vector-to-design choke
        # point, and nothing else in the pipeline changes.
        if "enrich" in d:
            d.setdefault("enrich_inner", d["enrich"])
            d.setdefault("enrich_outer", d["enrich"])
            if "pitch" not in d:
                d["pitch"] = 1.26   # cm, fixed Westinghouse 17x17 pitch (C8)
        return d
''',
))

# --------------------------------------------------------------------- E2
EDITS.append(dict(
    path="reactor_optimization.py",
    name="E2 four-variable design space",
    old='''    ds = DesignSpace([
        # Upper bound is LEU_CAP_WTPC / M_P_DESIGN, so the highest
        # enrichment anywhere in the ZONED core stays at or below the
        # LEU (Low Enriched Uranium) cap by construction. At
        # M_P_DESIGN = 1.0 this is exactly 19.75, the previous bound.
        # See leu_policy.py.
        DesignVariable("enrich_inner", 2.0, _leu.E_SEARCH_MAX, "%"),
        DesignVariable("enrich_outer", 2.0, _leu.E_SEARCH_MAX, "%"),
        DesignVariable("gd_wt",        0.0,  8.0,  "wt% Gd2O3"),
        DesignVariable("pitch",        1.15, 1.43, "cm"),
        DesignVariable("refl_thick",   2.0,  19.5, "cm"),
        # CAMPAIGN 5: gadolinia-bearing rod count. Continuous for the
        # surrogate and NSGA-II; the model snaps to the nested symmetric
        # ladder {12,16,...,40} (reactor_model.snap_gd_pins) and the snapped
        # value is recorded per evaluation as "gd_pins_used".
        DesignVariable("gd_pins",      12.0, 40.0, "rods"),
    ])
''',
    new='''    ds = DesignSpace([
        # CAMPAIGN 8: four design variables. ONE assembly enrichment
        # ("enrich"): the core-level variation comes from the C/M/P ring
        # multipliers of zoning.py (0.720 / 0.8933 / 1.150), so the
        # as-built peripheral enrichment is enrich * M_P_DESIGN and the
        # search-box upper bound stays LEU-capped by construction
        # (leu_policy.py). Pitch is FIXED at 1.26 cm, the Westinghouse
        # 17x17 value, derived in DesignSpace.as_dict. The reflector upper
        # bound is the vessel budget at that pitch with the Campaign 8
        # clearance (5.08 cm barrel + 2.0 cm downcomer):
        #   90.0 - 7.08 - 77.231 - 0.02 = 5.669 cm  ->  bound 5.66 cm.
        DesignVariable("enrich",       2.0, _leu.E_SEARCH_MAX, "%"),
        DesignVariable("gd_wt",        0.0,  8.0,  "wt% Gd2O3"),
        DesignVariable("refl_thick",   2.0,  5.66, "cm"),
        # CAMPAIGN 5: gadolinia-bearing rod count. Continuous for the
        # surrogate and NSGA-II; the model snaps to the nested symmetric
        # ladder {12,16,...,40} (reactor_model.snap_gd_pins) and the snapped
        # value is recorded per evaluation as "gd_pins_used".
        DesignVariable("gd_pins",      12.0, 40.0, "rods"),
    ])
''',
))

# --------------------------------------------------------------------- E3
EDITS.append(dict(
    path="run_optimization.py",
    name="E3 --enr-max box accepts the single variable",
    old='''    for _v in spec.design_space.variables:
        if _v.name in ("enrich_inner", "enrich_outer"):
''',
    new='''    for _v in spec.design_space.variables:
        if _v.name in ("enrich", "enrich_inner", "enrich_outer"):
''',
))

# --------------------------------------------------------------------- E4
EDITS.append(dict(
    path="run_optimization.py",
    name="E4 meta record accepts the single variable",
    old='''                               "e_box_used_wtpc": [
                                   next(v.low for v in spec.design_space.variables
                                        if v.name == "enrich_inner"),
                                   next(v.high for v in spec.design_space.variables
                                        if v.name == "enrich_inner")]},
''',
    new='''                               "e_box_used_wtpc": [
                                   next(v.low for v in spec.design_space.variables
                                        if v.name in ("enrich", "enrich_inner")),
                                   next(v.high for v in spec.design_space.variables
                                        if v.name in ("enrich", "enrich_inner"))]},
''',
))

# --------------------------------------------------------------------- E5
EDITS.append(dict(
    path="core_geometry.py",
    name="E5 vessel clearance = barrel + downcomer",
    old='''VESSEL_CLEARANCE_CM = 0.0    # cm  (barrel + downcomer allowance; edit knowingly)
''',
    new='''VESSEL_CLEARANCE_CM = 7.08   # cm  (CAMPAIGN 8: 5.08 cm SS304L core barrel,
                             #  the NuScale-like benchmark value, plus a
                             #  2.0 cm downcomer. Sensitivity to 3.7 cm
                             #  downcomer is run on the candidates only.)
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


def selftest() -> int:
    print("selftest:")
    for p in files():
        py_compile.compile(p, doraise=True)
        print(f"  compiles: {p}")
    # AST check: the design space holds exactly the four Campaign 8 names.
    tree = ast.parse(Path("reactor_optimization.py").read_text())
    names = [n.args[0].value for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "DesignVariable"
             and n.args and isinstance(n.args[0], ast.Constant)]
    expect = ["enrich", "gd_wt", "refl_thick", "gd_pins"]
    if names != expect:
        print(f"  FAIL: DesignVariable names {names}, expected {expect}")
        return 2
    print(f"  design space: {names}")
    # geometry closure: bound vs analytic budget, without importing numpy.
    import math
    r_env = math.sqrt(13.0) * 17.0 * C8_PITCH
    max_refl = 90.0 - C8_CLEARANCE - r_env - 0.02
    print(f"  budget check: R_env {r_env:.3f} cm, max reflector "
          f"{max_refl:.3f} cm, bound {C8_REFL_HI} cm")
    if not (0 < C8_REFL_HI < max_refl):
        print("  FAIL: the reflector bound does not fit the budget")
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
