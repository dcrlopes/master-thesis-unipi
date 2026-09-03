#!/usr/bin/env python3
"""
apply_rollup_fix.py -- correct the stage F verdict in the roll-up of
run_c8_post.sh.

The bug. The roll-up prints "subcritical, g_ctrl SATISFIED" when the mean
ALL-RE eigenvalue is below 1. The campaign screen is
    g_ctrl = k_ALLRE - (1 - margin) = k_ALLRE - 0.99 <= 0
so a design with 0.99 < k < 1 is subcritical and still infeasible. With
three seeds, design 54 gives k = 0.99038 and design 11 gives 0.99053. Both
would be reported as satisfying g_ctrl by the current text. The fix prints
the g_ctrl value in pcm and the verdict against 0.99, and quotes the
one-sigma seed spread next to it.

Also patched, same block: the stage D roll-up prints "subcritical" for any
positive 3D margin. It now prints three classes, against the campaign
margin of 1000 pcm.

USAGE
  python apply_rollup_fix.py --check     verify anchors, change nothing
  python apply_rollup_fix.py --apply     patch in place, keep run_c8_post.sh.bak
  python apply_rollup_fix.py --revert    restore from run_c8_post.sh.bak
"""
import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path("run_c8_post.sh")
BAK = Path("run_c8_post.sh.bak")

EDITS = [
    # 1. stage F verdict
    (
        """        print(f"  d{i} ALL-RE k = {m:.5f} +/- {sdk:.0f} pcm over {len(ks)} seeds, "
              f"{'subcritical, g_ctrl SATISFIED' if m < 1 else 'supercritical, g_ctrl violated'}")""",
        """        g = 1e5 * (m - 0.99)      # g_ctrl in pcm, the screen is k_ALLRE <= 0.99
        verdict = ("g_ctrl SATISFIED" if m <= 0.99
                   else "subcritical but g_ctrl VIOLATED (0.99 < k < 1)" if m < 1.0
                   else "supercritical, g_ctrl VIOLATED")
        print(f"  d{i} ALL-RE k = {m:.5f} +/- {sdk:.0f} pcm over {len(ks)} seeds, "
              f"g_ctrl {g:+.0f} pcm ({abs(g)/max(sdk,1):.1f} sigma), {verdict}")""",
    ),
    # 2. stage D verdict against the 1000 pcm margin
    (
        """          f"{'subcritical' if m3 > 0 else 'SUPERCRITICAL under ALL-RE'}")""",
        """          f"{'holds with margin' if m3 >= 1000 else 'subcritical, margin below 1000 pcm' if m3 > 0 else 'SUPERCRITICAL under ALL-RE'}")""",
    ),
    (
        """print(f"\\n  {sum(1 for r in rows if r[3] > 0)} of {len(rows)} subcritical under the four "
      f"regulating banks with no soluble boron, 3D, BOL.")""",
        """print(f"\\n  {sum(1 for r in rows if r[3] >= 1000)} of {len(rows)} hold with the 1000 pcm margin, "
      f"{sum(1 for r in rows if 0 < r[3] < 1000)} subcritical below the margin, "
      f"{sum(1 for r in rows if r[3] <= 0)} supercritical, under the four regulating banks "
      f"with no soluble boron, 3D, BOL.")""",
    ),
]


def check(text: str) -> bool:
    ok = True
    for old, _new in EDITS:
        n = text.count(old)
        print(f"  anchor {'ok ' if n == 1 else 'FAIL'} ({n} occurrence{'s' if n != 1 else ''}): {old.splitlines()[0].strip()[:70]}")
        ok &= (n == 1)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if a.revert:
        if not BAK.exists():
            sys.exit("FAIL: no run_c8_post.sh.bak to restore")
        shutil.copy2(BAK, TARGET)
        print("restored run_c8_post.sh from run_c8_post.sh.bak")
        return 0
    if not TARGET.exists():
        sys.exit("FAIL: run_c8_post.sh not found in the current directory")
    text = TARGET.read_text()
    print("checking anchors in run_c8_post.sh:")
    if not check(text):
        sys.exit("FAIL: expected exactly one occurrence of every anchor. The file differs from revision 2. Nothing changed.")
    if a.check:
        print("check OK, nothing changed")
        return 0
    shutil.copy2(TARGET, BAK)
    for old, new in EDITS:
        text = text.replace(old, new)
    TARGET.write_text(text)
    print("applied 3 edits, backup in run_c8_post.sh.bak")
    print("verify with:  bash -n run_c8_post.sh && python apply_rollup_fix.py --check   (the check must now FAIL on all anchors, which proves they were replaced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
