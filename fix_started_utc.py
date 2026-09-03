#!/usr/bin/env python
"""
fix_started_utc.py -- keep the true campaign start time across a --resume.

THE BUG
-------
run_optimization.py builds a fresh `meta` dict on every launch, with
`started_utc` set to the instant THAT PROCESS started. ActiveLearningMOO.
save_checkpoint() then writes it verbatim:

    if meta:
        out["meta"] = dict(meta)

So after a resume, `meta["started_utc"]` records when the LAST BLOCK started,
not when the campaign started. The earlier value is overwritten and lost.

Campaign 5 is the worked example. The checkpoint claims 2026-08-24T14:22 UTC.
The first statepoint in openmc_runs_c5/case_0000 was written 2026-08-21 13:23
local, three days earlier. Anyone using `started_utc` as a lower bound (a
timing reconstruction with --after, an archive audit, a reproducibility check)
silently loses the whole DOE phase.

THE FIX
-------
1. load_checkpoint() remembers the campaign start and the list of block starts
   already recorded in the file being resumed.
2. save_checkpoint() writes the ORIGINAL start back into `started_utc` and
   appends the current block's start to a new `block_started_utc` list.

Nothing else changes. Fresh runs are unaffected, because there is no prior
checkpoint to remember, so `started_utc` is the campaign start by definition.

USAGE
-----
    python fix_started_utc.py --dry-run       # show what would change
    python fix_started_utc.py                 # patch, keeping a .bak
    python fix_started_utc.py --verify        # re-check an already patched file

The script is idempotent: running it twice is harmless.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET = "reactor_optimization.py"

# --------------------------------------------------------------------------
# Edit 1: save_checkpoint() preserves the campaign start
# --------------------------------------------------------------------------
OLD_SAVE = '''        if meta:
            out["meta"] = dict(meta)
        Path(path).write_text(json.dumps(out, indent=2, default=float))
        return path'''

NEW_SAVE = '''        if meta:
            out["meta"] = dict(meta)
            # A resume rebuilds `meta` from scratch, so meta["started_utc"] is
            # THIS block's start. Keep the campaign start from the checkpoint
            # being resumed and record every block start separately, so a later
            # timing reconstruction or archive audit can bound the whole run.
            this_block = out["meta"].get("started_utc")
            blocks = list(getattr(self, "_ckpt_block_starts", []))
            if this_block and this_block not in blocks:
                blocks.append(this_block)
            if blocks:
                out["meta"]["block_started_utc"] = blocks
            campaign_start = getattr(self, "_ckpt_started_utc", None)
            if campaign_start:
                out["meta"]["started_utc"] = campaign_start
        Path(path).write_text(json.dumps(out, indent=2, default=float))
        return path'''

# --------------------------------------------------------------------------
# Edit 2: load_checkpoint() remembers what it read
# --------------------------------------------------------------------------
OLD_LOAD = '''        # continue case numbering so OpenMC scratch dirs never collide
        self.evaluator.n_calls = len(ckpt["all_raw"])
        return len(ckpt["all_raw"])'''

NEW_LOAD = '''        # remember when the CAMPAIGN started, not when this block started, so
        # save_checkpoint() can write it back instead of overwriting it
        _m = ckpt.get("meta") or {}
        self._ckpt_started_utc = _m.get("started_utc")
        _blocks = _m.get("block_started_utc")
        if not _blocks:
            _blocks = [_m["started_utc"]] if _m.get("started_utc") else []
        self._ckpt_block_starts = list(_blocks)
        # continue case numbering so OpenMC scratch dirs never collide
        self.evaluator.n_calls = len(ckpt["all_raw"])
        return len(ckpt["all_raw"])'''

EDITS = [("save_checkpoint", OLD_SAVE, NEW_SAVE),
         ("load_checkpoint", OLD_LOAD, NEW_LOAD)]

MARKER = "_ckpt_block_starts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=TARGET,
                    help="file to patch, default reactor_optimization.py")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and exit without writing")
    ap.add_argument("--verify", action="store_true",
                    help="only check whether the patch is already applied")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.is_file():
        sys.exit(f"not found: {p}. Run this from the repository root.")
    src = p.read_text()

    already = MARKER in src
    if args.verify:
        print(f"{p}: patch {'IS' if already else 'is NOT'} applied")
        sys.exit(0 if already else 1)
    if already:
        print(f"{p}: already patched, nothing to do")
        return

    out = src
    for name, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            sys.exit(f"anchor for {name} matched {n} times, expected 1. "
                     f"The file has drifted from the version this patch was "
                     f"written against. Do not force it, patch by hand.")
        print(f"  {name:18s} anchor found, {len(old.splitlines())} lines -> "
              f"{len(new.splitlines())} lines")
        out = out.replace(old, new)

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    bak = p.with_suffix(p.suffix + ".started_utc.bak")
    shutil.copy2(p, bak)
    p.write_text(out)
    print(f"\npatched {p}")
    print(f"backup  {bak}")
    print("\nverify with:  python -c \"import reactor_optimization\"")


if __name__ == "__main__":
    main()
