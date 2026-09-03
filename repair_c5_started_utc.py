#!/usr/bin/env python
"""
repair_c5_started_utc.py -- restore the campaign start time in an existing
checkpoint whose meta["started_utc"] was overwritten by a --resume.

The code fix (fix_started_utc.py) prevents this happening again. It cannot
recover what was already lost, because the original value is not in the file.
The statepoints can: every evaluation wrote one, and the earliest write is a
tight upper bound on the campaign start.

This script reads the earliest statepoint modification time under --workdir,
converts it to UTC, and writes it back as meta["started_utc"], moving the
value that is currently there into meta["block_started_utc"]. It records
meta["started_utc_source"] so the substitution is never mistaken for a value
the run itself recorded.

USAGE
-----
    python repair_c5_started_utc.py --dry-run
    python repair_c5_started_utc.py

The original file is copied to <name>.pre_repair.bak before anything is
written. Run this from the repository root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path


def earliest_statepoint(workdir: Path):
    """Earliest modification time of any .h5 under workdir, as an aware UTC
    datetime. Returns None if the tree holds no .h5 file."""
    best = None
    for p in workdir.rglob("*.h5"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if best is None or m < best[0]:
            best = (m, p)
    if best is None:
        return None, None
    return dt.datetime.fromtimestamp(best[0], dt.timezone.utc), best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="out_c5/optimization_checkpoint.json")
    ap.add_argument("--workdir", default="openmc_runs_c5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    wd = Path(args.workdir)
    if not ck_path.is_file():
        sys.exit(f"not found: {ck_path}")
    if not wd.is_dir():
        sys.exit(f"not found: {wd}")

    ck = json.loads(ck_path.read_text())
    meta = ck.get("meta") or {}
    current = meta.get("started_utc")

    true_start, source_file = earliest_statepoint(wd)
    if true_start is None:
        sys.exit(f"no .h5 files under {wd}, cannot recover the start time")

    iso = true_start.isoformat()
    print(f"checkpoint            : {ck_path}")
    print(f"evaluations           : {len(ck.get('all_raw', []))}")
    print(f"meta.started_utc now  : {current}")
    print(f"earliest statepoint   : {iso}")
    print(f"  from                : {source_file}")

    if current and current <= iso:
        print("\nThe recorded start already precedes the earliest statepoint. "
              "Nothing to repair.")
        return

    blocks = list(meta.get("block_started_utc") or [])
    for v in (iso, current):
        if v and v not in blocks:
            blocks.append(v)
    blocks.sort()

    meta["started_utc"] = iso
    meta["block_started_utc"] = blocks
    meta["started_utc_source"] = (
        f"earliest statepoint mtime under {wd} ({source_file.name}), "
        "recovered after a resume overwrote the recorded value")
    ck["meta"] = meta

    print(f"\nwould set started_utc : {iso}")
    print(f"       block starts   : {blocks}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    bak = ck_path.with_suffix(ck_path.suffix + ".pre_repair.bak")
    shutil.copy2(ck_path, bak)
    ck_path.write_text(json.dumps(ck, indent=2, default=float))
    print(f"\nwrote  {ck_path}")
    print(f"backup {bak}")


if __name__ == "__main__":
    main()
