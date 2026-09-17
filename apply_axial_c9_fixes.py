#!/usr/bin/env python3
"""apply_axial_c9_fixes.py -- two fixes to the Campaign 9 axial study.

FIX 1, axial_figures_c9.py, grid bands.
  grid_bands() guessed the spacer grids from the bin lengths, keeping bins
  shorter than 0.8 of the median. With the default HardwareSpec the edges are
      3.555 cm (fuel bottom to grid 1), 4.445 cm x 3 (grids),
      5.279 cm x 14, 4.867 cm x 6
  so the median is 5.279 cm, the threshold 4.223 cm, and the only "band" found
  is the 3.555 cm bin below the first grid. The three real grids were never
  shaded and a bin that is not a grid was. The replacement takes the bands
  from axial_shape_c9.axial_edges(), the function the transport used, and
  draws them only if its edges reproduce the stored ones.

FIX 2, axial_shape_c9.py, pin count in the design header.
  The header printed the continuous optimiser variable gd_pins. The model
  snaps it to reactor_model.GD_PIN_COUNTS and the archive records the result
  as gd_pins_used. The header now prints the snapped count, which is the one
  that was built, with the continuous value beside it.

Neither fix changes a transport result. Fix 1 changes figures only, fix 2
changes one printed line.

USAGE (repository root)
  python apply_axial_c9_fixes.py --selftest   # checks the band logic, no files touched
  python apply_axial_c9_fixes.py --check      # anchors present? nothing written
  python apply_axial_c9_fixes.py              # apply, keeps *.pre-axfix copies
  python apply_axial_c9_fixes.py --revert     # restore the copies
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SUFFIX = ".pre-axfix"

OLD_BANDS = '''def grid_bands(edges):
    """The grid bands are the short bins. Recover them from the edge spacing."""
    dz = np.diff(edges); short = dz < 0.8 * np.median(dz)
    return [(edges[i], edges[i + 1]) for i in np.where(short)[0]]
'''

NEW_BANDS = '''def grid_bands(edges):
    """Spacer-grid bands inside the active fuel, from the same function the
    transport used (axial_shape_c9.axial_edges on the default HardwareSpec).
    Drawn only if that function reproduces the stored edges, since a band
    guessed from bin lengths can mark a bin that is not a grid."""
    try:
        import hardware3d as hw
        from axial_shape_c9 import axial_edges
        e_ref, bands, _ = axial_edges(hw.HardwareSpec(), hw)
    except Exception as exc:                      # noqa: BLE001
        print(f"  WARNING: grid bands not recovered ({exc}), none drawn")
        return []
    e_ref = np.asarray(e_ref, dtype=float); edges = np.asarray(edges, dtype=float)
    if e_ref.shape != edges.shape or not np.allclose(e_ref, edges, atol=1e-6):
        print("  WARNING: stored edges differ from the default HardwareSpec, "
              "no grid bands drawn")
        return []
    return [(float(g0), float(g1)) for g0, g1 in bands]
'''

OLD_PINS = '''f"pins {d['gd_pins']:.0f}, refl {d['refl_thick']:.2f} cm")'''
NEW_PINS = ('''f"pins {rm.snap_gd_pins(d['gd_pins'])} as built "\n'''
            '''              f"(optimiser value {d['gd_pins']:.2f}), refl {d['refl_thick']:.2f} cm")''')

PATCHES = [
    (Path("axial_figures_c9.py"), OLD_BANDS, NEW_BANDS),
    (Path("axial_shape_c9.py"), OLD_PINS, NEW_PINS),
]


def check():
    ok = True
    for path, old, new in PATCHES:
        if not path.exists():
            print(f"  MISSING {path}"); ok = False; continue
        txt = path.read_text(encoding="utf-8")
        n_old, n_new = txt.count(old), txt.count(new)
        state = ("already applied" if n_new == 1 and n_old == 0 else
                 "ready" if n_old == 1 else f"ANCHOR NOT UNIQUE (old {n_old}, new {n_new})")
        print(f"  {path}: {state}")
        ok &= state in ("ready", "already applied")
    return ok


def apply():
    if not check():
        print("ABORT: anchors do not match, nothing written"); return 1
    for path, old, new in PATCHES:
        txt = path.read_text(encoding="utf-8")
        if txt.count(new) == 1 and txt.count(old) == 0:
            continue
        bak = path.with_name(path.name + SUFFIX)
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(txt.replace(old, new, 1), encoding="utf-8")
        print(f"  patched {path}  (copy in {bak.name})")
    return 0


def revert():
    for path, _, _ in PATCHES:
        bak = path.with_name(path.name + SUFFIX)
        if bak.exists():
            shutil.move(bak, path); print(f"  restored {path}")
        else:
            print(f"  no copy for {path}")
    return 0


def selftest():
    import numpy as np
    import hardware3d as hw
    from axial_shape_c9 import axial_edges
    edges, bands, fuel = axial_edges(hw.HardwareSpec(), hw)
    dz = np.diff(edges)
    old = [(edges[i], edges[i + 1]) for i in np.where(dz < 0.8 * np.median(dz))[0]]
    ns = {"np": np}
    exec(NEW_BANDS, ns)
    new = ns["grid_bands"](edges)
    print(f"  {len(edges) - 1} bins, fuel {fuel[0]:.3f} to {fuel[1]:.3f} cm")
    print(f"  old heuristic : {len(old)} band(s) {[(round(a, 3), round(b, 3)) for a, b in old]}")
    print(f"  new function  : {len(new)} band(s) {[(round(a, 3), round(b, 3)) for a, b in new]}")
    assert len(new) == 3, "expected three grids inside the active fuel"
    assert all(abs((b - a) - hw.HardwareSpec().grid_height) < 1e-9 for a, b in new)
    assert ns["grid_bands"](edges + 0.1) == [], "shifted edges must give no bands"
    import reactor_model as rm
    for raw, want in ((20.98, 20), (18.0, 16), (30.4, 32), (15.2, 16)):
        assert rm.snap_gd_pins(raw) == want, (raw, rm.snap_gd_pins(raw))
    print("  snap_gd_pins agrees with the counts in mtc_ceiling_table")
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not Path("axial_shape_c9.py").exists():
        print("ABORT: run from the repository root"); return 2
    if a.selftest:
        return selftest()
    if a.check:
        return 0 if check() else 1
    if a.revert:
        return revert()
    return apply()


if __name__ == "__main__":
    sys.exit(main())
