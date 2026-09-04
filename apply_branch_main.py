#!/usr/bin/env python3
"""
apply_branch_main.py -- correct the branch check of run_c8_night.sh.

The driver was written against the old branch name and refuses to start on
main, which is the only branch in use. This applier replaces the check and
the stale header line. Two anchored edits, nothing else touched.

USAGE
  python apply_branch_main.py --selftest    # patch a copy in a temp dir
  python apply_branch_main.py --check       # report, change nothing
  python apply_branch_main.py --apply       # patch in place, .bak written
  python apply_branch_main.py --revert      # restore from the .bak

Flags: --check reports whether each anchor is found and whether the file is
already patched; --apply writes run_c8_night.sh.bak then edits; --revert
copies the .bak back; --selftest runs the whole cycle on a throwaway copy
and never touches the real file.
"""
from __future__ import annotations
import argparse, shutil, sys, tempfile
from pathlib import Path

TARGET = "run_c8_night.sh"

EDITS = [
    ('  [ "$(git branch --show-current)" = "campaign8" ] || die "wrong branch. Run: git checkout campaign8"',
     '  BRANCH=$(git branch --show-current)\n'
     '  [ "$BRANCH" = "main" ] || die "wrong branch ($BRANCH). This work lives on main. Run: git checkout main"\n'
     '  DIRTY=$(git status --porcelain | head -5)\n'
     '  [ -z "$DIRTY" ] || { echo "  uncommitted changes:"; echo "$DIRTY" | sed "s/^/    /"; }'),
    ('# against commit f60b78a of branch campaign8. Supersedes run_c8_stageA.sh and',
     '# against commit af3332a of branch main. Supersedes run_c8_stageA.sh and'),
]

MARKER = 'This work lives on main'


def apply_to(text, revert=False):
    for old, new in EDITS:
        a, b = (new, old) if revert else (old, new)
        if text.count(a) != 1:
            raise ValueError(f"anchor found {text.count(a)} times, expected 1: {a[:60]!r}")
        text = text.replace(a, b)
    return text


def selftest():
    d = Path(tempfile.mkdtemp())
    f = d / TARGET
    f.write_text("# against commit f60b78a of branch campaign8. Supersedes run_c8_stageA.sh and\n"
                 "  echo x\n"
                 '  [ "$(git branch --show-current)" = "campaign8" ] || die "wrong branch. Run: git checkout campaign8"\n'
                 "  echo y\n")
    orig = f.read_text()
    patched = apply_to(orig)
    assert MARKER in patched and "campaign8" not in patched, patched
    assert apply_to(patched, revert=True) == orig, "revert is not exact"
    assert "echo x" in patched and "echo y" in patched, "collateral damage"
    shutil.rmtree(d)
    print("selftest OK: both anchors match, the patch is reversible, nothing else changes")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=TARGET)
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("check", "apply", "revert", "selftest"):
        g.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    p = Path(a.path)
    if not p.exists():
        sys.exit(f"{p} not found. Run from the repository root.")
    text = p.read_text()
    if a.revert:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            sys.exit(f"{bak} not found, nothing to revert to")
        shutil.copy2(bak, p)
        print(f"restored {p} from {bak}")
        return 0
    if MARKER in text:
        print(f"{p} is already patched, nothing to do")
        return 0
    for old, _ in EDITS:
        n = text.count(old)
        print(f"  anchor {'OK ' if n == 1 else 'MISS'} ({n} match): {old.strip()[:70]}")
    if a.check:
        print("check only, nothing written")
        return 0
    new = apply_to(text)
    shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
    p.write_text(new)
    print(f"patched {p}, backup at {p}.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
