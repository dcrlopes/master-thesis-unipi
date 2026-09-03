#!/usr/bin/env python3
"""make_leu_policy.py -- regenerate leu_policy.py, and nothing else.

WHY THIS EXISTS
---------------
leu_policy.py is not hand-written. It is produced by apply_leu_box.py from
the POLICY_MODULE template held inside that file. But apply_leu_box.py
cannot be used to regenerate it on an already-patched tree: its main()
runs the anchor check first, and on a patched tree two anchors report 0
occurrences, so it aborts before writing anything.

That is correct behaviour for the applier and useless when all you need is
the generated module, for example on a second machine that has the
repository but not the untracked file.

This script extracts the SAME template from apply_leu_box.py and writes
only leu_policy.py. It never touches reactor_optimization.py,
openmc_evaluator.py, or run_optimization.py, so it cannot undo the LEU box
patch or the constraint normalization patch.

The values must match the machine where the campaign ran, otherwise the
enrichment policy silently changes. Defaults are the Campaign 5 settings.

USAGE
    python make_leu_policy.py --check      # show what would be written
    python make_leu_policy.py              # write leu_policy.py
    python make_leu_policy.py --force      # overwrite an existing file
    python make_leu_policy.py --m-p 1.15   # a zoned peripheral multiplier

VERIFY AFTERWARDS
    python verify_leu_box.py               # replays the archived evaluations
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

SOURCE = "apply_leu_box.py"
TARGET = "leu_policy.py"


def extract_template(root: Path) -> str:
    """Pull POLICY_MODULE out of apply_leu_box.py WITHOUT executing it."""
    src_path = root / SOURCE
    if not src_path.is_file():
        sys.exit(f"ABORT: {src_path} not found. Run from the repository root.")
    src = src_path.read_text()
    m = re.search(r"POLICY_MODULE\s*=\s*'''(.*?)'''", src, re.S)
    if m is None:
        m = re.search(r'POLICY_MODULE\s*=\s*"""(.*?)"""', src, re.S)
    if m is None:
        sys.exit(f"ABORT: no POLICY_MODULE template found in {SOURCE}. "
                 f"Has the applier changed?")
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leu", type=float, default=19.75,
                    help="LEU cap in wt%% U-235 (default 19.75)")
    ap.add_argument("--m-p", type=float, default=1.0, dest="mp",
                    help="peripheral zoning multiplier, dimensionless "
                         "(default 1.0, the Campaign 5 value)")
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--check", action="store_true",
                    help="print the values and the target path, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite leu_policy.py if it already exists")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    tpl = extract_template(root)
    text = tpl.format(leu=args.leu, mp=args.mp)
    target = root / TARGET

    print(f"repository   : {root}")
    print(f"template from: {SOURCE}")
    print(f"LEU cap      : {args.leu} wt% U-235")
    print(f"m_P          : {args.mp}"
          + ("   (no-op: search box and LEU cap coincide)"
             if args.mp == 1.0 else ""))
    print(f"E_SEARCH_MAX : {args.leu / args.mp:.4f} wt%")
    print(f"target       : {target}")

    if args.check:
        print("\n--check given, nothing written.")
        return
    if target.exists() and not args.force:
        sys.exit(f"\nREFUSED: {target} already exists. Compare it first, and "
                 f"pass --force only if you mean to overwrite it.")

    target.write_text(text)
    print(f"\nwrote {target}")

    # self-check: import it and confirm the three constants
    sys.path.insert(0, str(root))
    import importlib
    mod = importlib.import_module("leu_policy")
    assert mod.LEU_CAP_WTPC == args.leu, "LEU_CAP_WTPC mismatch"
    assert mod.M_P_DESIGN == args.mp, "M_P_DESIGN mismatch"
    assert abs(mod.E_SEARCH_MAX - args.leu / args.mp) < 1e-12, \
        "E_SEARCH_MAX mismatch"
    print(f"self-check ok: LEU_CAP_WTPC={mod.LEU_CAP_WTPC}, "
          f"M_P_DESIGN={mod.M_P_DESIGN}, "
          f"E_SEARCH_MAX={mod.E_SEARCH_MAX}")


if __name__ == "__main__":
    main()
