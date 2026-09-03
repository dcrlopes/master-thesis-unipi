#!/usr/bin/env python3
"""
fix_runlog_regex.py -- repair the per-case regular expression of
parse_runlog.py, which has never matched a single line of any campaign log.

THE DEFECT
----------
openmc_evaluator.py prints, at lines 350 to 358,

    -> EFPD={cycle_efpd:7.0f}{'(CEN)' if censored else '     '} F_dh=...

so between the cycle length and the F_dh field there are either the five
characters "(CEN)" plus one space, or six spaces. RE_CASE in
parse_runlog.py expects exactly one space,

    -> EFPD=\\s*([\\d.]+) F_dh=([\\d.]+)

and therefore matches nothing. The consequence is silent: the parser still
prints the optimisation progress, the depletion schedule, the transport
statistics and the wall time, but the per-case table is empty and
--csv writes no file at all, with no error and no warning.

THE FIX
-------
Accept the optional censoring marker and one or more spaces, in a
NON-CAPTURING group so that the number and the order of the capture groups,
and hence CASE_FIELDS, are unchanged.

    -> EFPD=\\s*([\\d.]+)(?:\\(CEN\\))?\\s+F_dh=([\\d.]+)

Nothing else in the file is touched. The censoring flag is not recovered
here because it is already stored per design in the optimisation
checkpoint, which is the authoritative record.

USAGE
-----
    python3 fix_runlog_regex.py                 apply, writing parse_runlog.py.bak
    python3 fix_runlog_regex.py --check         report only, change nothing
    python3 fix_runlog_regex.py --revert        restore from the .bak

The script refuses to write unless the anchor string occurs exactly once.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("parse_runlog.py")
BACKUP = Path("parse_runlog.py.bak")

# The anchor is the middle line of the RE_CASE definition, verbatim as it
# appears in the source file, including the leading indentation.
ANCHOR = ('    r"refl=\\s*([\\d.]+) k_target=([\\d.]+) -> EFPD=\\s*([\\d.]+) '
          'F_dh=([\\d.]+) "\n')

REPLACEMENT = ('    r"refl=\\s*([\\d.]+) k_target=([\\d.]+) -> EFPD=\\s*([\\d.]+)"\n'
               '    r"(?:\\(CEN\\))?\\s+F_dh=([\\d.]+) "\n')


def selftest() -> bool:
    """Prove the new expression matches both line forms and still yields the
    ten capture groups that CASE_FIELDS expects."""
    new = re.compile(
        r"\[case (\d+)\] e=\(\s*([\d.]+)/\s*([\d.]+)\) Gd=([\d.]+) p=([\d.]+) "
        r"refl=\s*([\d.]+) k_target=([\d.]+) -> EFPD=\s*([\d.]+)"
        r"(?:\(CEN\))?\s+F_dh=([\d.]+) "
        r"k_bol=([\d.]+)")

    def line(censored: bool) -> str:
        return (f"  [case 0048] e=({8.71:5.2f}/{7.80:5.2f}) Gd={1.93:4.2f} "
                f"p={1.237:.3f} refl={14.06:5.1f} k_target={1.0553:.4f} "
                f"-> EFPD={4745:7.0f}{'(CEN)' if censored else '     '} "
                f"F_dh={1.565:.3f} k_bol={1.1192:.4f} "
                f"[12 solves, 15.8 min]")

    ok = True
    for cen in (False, True):
        m = new.search(line(cen))
        if m is None or len(m.groups()) != 10:
            ok = False
            print(f"  selftest FAILED for censored={cen}")
        else:
            print(f"  selftest ok, censored={cen}, "
                  f"EFPD={m.group(8)} F_dh={m.group(9)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report the state of the file and change nothing")
    ap.add_argument("--revert", action="store_true",
                    help="restore parse_runlog.py from parse_runlog.py.bak")
    args = ap.parse_args()

    # ---- environment guard -------------------------------------------- #
    print(f"python  : {sys.version.split()[0]}")
    print(f"cwd     : {Path.cwd()}")
    print("OpenMC required: no, this edits one regular expression")
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found. Run this from the root of "
              f"master-thesis-unipi.")
        return 2

    if args.revert:
        if not BACKUP.exists():
            print(f"FAIL: {BACKUP} not found, nothing to revert to.")
            return 2
        shutil.copy2(BACKUP, TARGET)
        print(f"reverted {TARGET} from {BACKUP}")
        return 0

    print("\nselftest of the replacement expression:")
    if not selftest():
        print("FAIL: the replacement does not behave as specified, "
              "nothing written.")
        return 2

    text = TARGET.read_text()
    n_old = text.count(ANCHOR)
    n_new = text.count(REPLACEMENT)

    print(f"\nanchor occurrences   : {n_old}")
    print(f"replacement present  : {n_new}")

    if n_new == 1 and n_old == 0:
        print("already patched, nothing to do.")
        return 0
    if n_old != 1:
        print("FAIL: expected exactly one occurrence of the anchor. "
              "The file has changed since this patch was written. "
              "Apply the intent by hand.")
        return 2
    if args.check:
        print("check only, nothing written.")
        return 0

    shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(text.replace(ANCHOR, REPLACEMENT))
    print(f"\nbackup  -> {BACKUP}")
    print(f"patched -> {TARGET}")
    print("\nNow rerun:  python3 parse_runlog.py out_c7/run.log "
          "--csv out_c7/c7_cases.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
