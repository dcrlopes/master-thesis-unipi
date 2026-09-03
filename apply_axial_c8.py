#!/usr/bin/env python3
"""
apply_axial_c8.py -- make axial_leakage_study.py read a Campaign 8 checkpoint.

THE DEFECT
  The study builds each design as {k: raw[k] for k in design_variables}.
  In Campaign 8 the design variables are enrich, gd_wt, refl_thick, gd_pins,
  so the dictionary lacks enrich_inner, enrich_outer and pitch, and
  zoning.zone_designs raises KeyError. The archive records DO carry the
  derived keys (DesignSpace.as_dict stores them), so the fix reads the six
  physical keys directly, which also works for every earlier checkpoint.

USAGE   python3 apply_axial_c8.py --check | (apply) | --revert
"""
import argparse, py_compile, shutil, sys
from pathlib import Path
P = Path("axial_leakage_study.py")
OLD = '''    d = {k: float(ck["all_raw"][idx][k]) for k in dv}
'''
NEW = '''    # CAMPAIGN 8: read the six PHYSICAL keys, not the design-variable names.
    # The archive stores enrich_inner / enrich_outer / pitch even when the
    # optimiser searched a collapsed space, so this works for every campaign.
    _raw = ck["all_raw"][idx]
    d = {k: float(_raw[k]) for k in
         ("enrich_inner", "enrich_outer", "gd_wt", "pitch", "refl_thick", "gd_pins")}
'''
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    print(f"python  : {sys.version.split()[0]}   cwd: {Path.cwd()}   OpenMC required: no")
    if not P.exists(): print("FAIL: run from the repository root"); return 2
    if a.revert:
        b = Path(str(P) + ".bak.c8")
        if not b.exists(): print("FAIL: no backup"); return 2
        shutil.copy2(b, P); print("reverted"); return 0
    t = P.read_text(); n_old, n_new = t.count(OLD), t.count(NEW)
    state = "APPLIED" if (n_new == 1 and n_old == 0) else "ready" if (n_old == 1 and n_new == 0) else f"FAIL (old x{n_old}, new x{n_new})"
    print(f"  A1 read the six physical keys                  {state}")
    if state.startswith("FAIL"): return 2
    if a.check or state == "APPLIED": return 0
    shutil.copy2(P, str(P) + ".bak.c8"); P.write_text(t.replace(OLD, NEW, 1))
    py_compile.compile(str(P), doraise=True); print("  applied, compiles, backup axial_leakage_study.py.bak.c8"); return 0
if __name__ == "__main__": raise SystemExit(main())
