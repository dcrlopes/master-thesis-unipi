#!/usr/bin/env python3
"""
fix_confirm3d_print.py -- make the per-design summary line state-aware.

THE BUG
confirm3d.py builds its per-design print from hardcoded ARO keys:

    print(f"design {idx}: L_ax_hw(ARO) {res['ARO_Lax_hw']:.4f} ...")

The ARI half of the same line is guarded by `if "ARI" in a.states`, but the
ARO half is not. Running with `--states ARI` alone therefore raises
KeyError: 'ARO_Lax_hw' after the solves are finished. Every earlier run used
`--states ARO ARI RE12`, so the path was never exercised.

Nothing numerical is affected. `summary[str(idx)] = res` and the summary.json
write both execute BEFORE the print, and runs.json is written per solve, so a
crash here loses no results. Relaunching reads every finished solve from cache.

THE FIX
Build the line from whichever states actually ran, and report the L_ax and the
margins of each rodded state rather than ARI only.

USAGE (wks720, branch campaign8)
  python fix_confirm3d_print.py --check
  python fix_confirm3d_print.py --apply     # writes confirm3d.py.bak2
  python fix_confirm3d_print.py --revert
"""
import argparse, shutil, sys
from pathlib import Path

TARGET = "confirm3d.py"
BAK = "confirm3d.py.bak2"          # .bak is used by fix_confirm3d_cachekey.py

OLD = '''        print(f"design {idx}: L_ax_hw(ARO) {res['ARO_Lax_hw']:.4f}  F2D {res['ARO_2D']['F']:.3f}  F3D {res['ARO_3Dhw']['F']:.3f}"
              + (f"  ARI margin 2D/3D {res['ARI_margin2D_pcm']:.0f}/{res['ARI_margin3D_pcm']:.0f} pcm" if "ARI" in a.states else ""))'''

NEW = '''        parts = [f"design {idx}:"]
        if "ARO" in a.states:
            parts.append(f"L_ax_hw(ARO) {res['ARO_Lax_hw']:.4f}  "
                         f"F2D {res['ARO_2D']['F']:.3f}  F3D {res['ARO_3Dhw']['F']:.3f}")
        for _st in ("ARI", "RE12"):
            if _st in a.states:
                _m2 = res[f"{_st}_margin2D_pcm"]
                _m3 = res[f"{_st}_margin3D_pcm"]
                _lx = res[f"{_st}_Lax_hw"]
                parts.append(f"{_st} margin 2D/3D {_m2:.0f}/{_m3:.0f} pcm, L_ax {_lx:.4f}")
        print("  ".join(parts), flush=True)'''


def state(t):
    n_old, n_new = t.count(OLD), t.count(NEW)
    if n_new == 1 and n_old == 0:
        return "applied"
    if n_old == 1 and n_new == 0:
        return "pending"
    return f"UNEXPECTED (old x{n_old}, new x{n_new})"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    p = Path(TARGET)
    if not p.exists():
        print(f"FAIL: {TARGET} not found. Run from ~/master-thesis-unipi."); return 1

    if a.revert:
        b = Path(BAK)
        if not b.exists():
            print(f"FAIL: {BAK} not found, nothing to revert."); return 1
        shutil.copy2(b, p); print(f"{TARGET}: restored from {BAK}"); return 0

    t = p.read_text()
    st = state(t)
    print(f"{TARGET}: {st}")
    if a.check:
        return 0 if st in ("applied", "pending") else 1

    if st == "applied":
        print("already applied, nothing to do"); return 0
    if st != "pending":
        print("ABORT: the anchor is not present exactly once. The file may have")
        print("       changed. Apply the intent by hand: guard the ARO half of the")
        print("       per-design print with `if \"ARO\" in a.states`.")
        return 1

    if not Path(BAK).exists():
        shutil.copy2(p, BAK)
    p.write_text(t.replace(OLD, NEW, 1))
    print(f"{TARGET}: patched ({BAK} written)")
    print(f"{TARGET}: {state(p.read_text())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
