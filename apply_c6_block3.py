#!/usr/bin/env python3
"""apply_c6_block3.py -- batch-diversity acquisition for Campaign 6 block 3.

WHY
---
Block 2 exposed two defects in the infill acquisition.

1. Top-k by Gaussian Process uncertainty on a continuous front returns k
   NEIGHBOURS, because uncertainty is a smooth field and its k highest
   values are adjacent. Iteration 2 of block 2 spent six real evaluations
   on designs spanning 0.03 wt% in inner enrichment (cases 48 to 53).

2. The duplicate test was < 1e-6 Euclidean in RAW units (wt%, cm, pins),
   so it only caught exact numerical copies. Designs 0.01 wt% apart both
   passed. Eighteen evaluations bought three distinct design points.

FIX
---
Rank by uncertainty as before (exploration is kept, nothing is discarded,
and no candidate is filtered by its predicted objectives), but enforce a
MINIMUM SEPARATION between the picks, and between each pick and the
archive, measured in the unit design box:

    || (a - b) / (xu - xl) || / sqrt(n_var)  >=  min_sep

min_sep is therefore a mean per-variable fraction of range. Default 0.05,
CLI flag --infill-min-sep, recorded in meta["surrogate_policy"]. If the
candidate set cannot supply n_infill picks at min_sep, the threshold is
halved and the scan repeats, so a small front still fills the batch with
the best diversity it can offer. Only if the candidates are exhausted
entirely is the batch topped up with space-filling random designs.

The selection CRITERION is unchanged (pure uncertainty). Expected
hypervolume improvement remains future work and is deliberately NOT
introduced here, so block 3 differs from block 2 by exactly one thing.

MECHANICS
---------
Prerequisite: the EFPD-CLIP marker of apply_c6_block2.py must be present.
Refuses to run twice (BATCH-DIVERSITY marker). Anchor-verified: every
target string must occur exactly once or nothing is written. Originals
are backed up as <file>.bak.c6b3.
    python apply_c6_block3.py --check    verify anchors, write nothing
    python apply_c6_block3.py            apply, back up, py_compile
    python apply_c6_block3.py --revert   restore the .bak.c6b3 backups
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path

SUFFIX = ".bak.c6b3"
MARKER = "BATCH-DIVERSITY"
PREREQ = "EFPD-CLIP"

RO = "reactor_optimization.py"
RU = "run_optimization.py"

EDITS = [

# ---- reactor_optimization.py ---------------------------------------------
(RO,
"""    efpd_cap: float | None = None  # EFPD-CLIP: ceiling on the SURROGATE's
                                   # predicted cycle length [EFPD]; None = off
""",
"""    efpd_cap: float | None = None  # EFPD-CLIP: ceiling on the SURROGATE's
                                   # predicted cycle length [EFPD]; None = off
    infill_min_sep: float = 0.05   # BATCH-DIVERSITY: minimum separation of
                                   # the infill picks (and of each pick from
                                   # the archive) in the unit design box,
                                   # as ||dx/span|| / sqrt(n_var)
""",
),

(RO,
"""            # ---- infill / acquisition: pick the most UNCERTAIN candidates ----
            # (exploration). You can blend in predicted hypervolume gain later.
            _, std = obj_sur.predict(cand)
            score = (std / (std.max(axis=0) + 1e-12)).sum(axis=1)
            # de-duplicate against already-evaluated points
            order = np.argsort(-score)
            chosen, picked = [], 0
            for idx in order:
                x = cand[idx]
                if self.X.size and np.min(np.linalg.norm(self.X - x, axis=1)) < 1e-6:
                    continue
                chosen.append(x); picked += 1
                if picked >= self.cfg.n_infill:
                    break
            if not chosen:                      # fallback: random explore
                chosen = list(self.spec.design_space.lhs(self.cfg.n_infill,
                                                         seed=self.cfg.seed + 99 + it))
            Xinf = np.array(chosen)
""",
"""            # ---- infill / acquisition: most UNCERTAIN candidates, spread ----
            # BATCH-DIVERSITY: rank by GP uncertainty (exploration, unchanged)
            # but force a minimum separation between the picks, and between
            # each pick and the archive, in the unit design box. Block 2
            # showed why: top-k by uncertainty on a continuous front returns
            # k neighbours, and the old 1e-6 raw-unit duplicate test let six
            # copies of one design through (cases 48-53 span 0.03 wt%).
            # Separation:  ||(a - b) / (xu - xl)|| / sqrt(n_var) >= min_sep,
            # a mean per-variable fraction of range. If the front cannot
            # supply n_infill picks at min_sep, halve it and rescan, so a
            # small front still fills the batch as diversely as it can. No
            # candidate is discarded for its predicted objectives.
            _, std = obj_sur.predict(cand)
            score = (std / (std.max(axis=0) + 1e-12)).sum(axis=1)
            order = np.argsort(-score)
            _xl = np.asarray(self.spec.design_space.xl, dtype=float)
            _xu = np.asarray(self.spec.design_space.xu, dtype=float)
            span = np.where(_xu > _xl, _xu - _xl, 1.0)
            rootn = np.sqrt(float(self.spec.design_space.n))

            def _sep(a, B):
                if B is None or len(B) == 0:
                    return np.inf
                d = (np.atleast_2d(np.asarray(B, dtype=float)) - a) / span
                return float(np.linalg.norm(d, axis=1).min()) / rootn

            chosen = []
            min_sep = float(getattr(self.cfg, "infill_min_sep", 0.05))
            while len(chosen) < self.cfg.n_infill and min_sep > 1e-4:
                for idx in order:
                    if len(chosen) >= self.cfg.n_infill:
                        break
                    x = cand[idx]
                    if _sep(x, self.X) < min_sep:
                        continue
                    if chosen and _sep(x, np.array(chosen)) < min_sep:
                        continue
                    chosen.append(x)
                if len(chosen) < self.cfg.n_infill:
                    min_sep *= 0.5          # relax and rescan the ranking
                    if verbose:
                        print(f"           [acquisition] diversity relaxed "
                              f"to min_sep={min_sep:.4f} "
                              f"({len(chosen)}/{self.cfg.n_infill} picked)")
            if len(chosen) < self.cfg.n_infill:
                # candidates exhausted even after relaxation: top up with
                # space-filling randoms rather than duplicating a pick
                extra = np.atleast_2d(self.spec.design_space.lhs(
                    self.cfg.n_infill - len(chosen),
                    seed=self.cfg.seed + 99 + it))
                chosen.extend(list(extra))
            Xinf = np.array(chosen[: self.cfg.n_infill])
""",
),

# ---- run_optimization.py -------------------------------------------------
(RU,
"""    ap.add_argument("--no-efpd-clip", action="store_true",
""",
"""    ap.add_argument("--infill-min-sep", type=float, default=0.05,
                    help="BATCH-DIVERSITY: minimum separation of the infill "
                         "picks (and of each pick from the archive) in the "
                         "unit design box, ||dx/span||/sqrt(n_var). 0.05 "
                         "means a mean per-variable spacing of 5%% of range. "
                         "Halved automatically when the surrogate front is "
                         "too small to supply n_infill picks at this value.")
    ap.add_argument("--no-efpd-clip", action="store_true",
""",
),

(RU,
"""    if args.nsga_gen is not None:
        cfg.nsga_gen = int(args.nsga_gen)
""",
"""    if args.nsga_gen is not None:
        cfg.nsga_gen = int(args.nsga_gen)
    cfg.infill_min_sep = float(args.infill_min_sep)  # BATCH-DIVERSITY
""",
),

(RU,
"""                               "nsga_gen": cfg.nsga_gen,
""",
"""                               "nsga_gen": cfg.nsga_gen,
                               "infill_min_sep": cfg.infill_min_sep,
""",
),
]

FILES = sorted({f for f, _, _ in EDITS})


def fail(msg: str) -> None:
    sys.exit(f"apply_c6_block3: REFUSED. {msg}")


def check(root: Path) -> None:
    for f in FILES:
        p = root / f
        if not p.is_file():
            fail(f"{f} not found. Run from the repository root.")
        text = p.read_text()
        if PREREQ not in text:
            fail(f"{f} lacks the {PREREQ} marker. Apply apply_c6_block2.py "
                 f"first, block 3 builds on the block 2 state.")
        if MARKER in text:
            fail(f"{f} already contains the {MARKER} marker. Already "
                 f"applied. Use --revert first if you want to re-apply.")
    for f, old, _ in EDITS:
        n = (root / f).read_text().count(old)
        if n != 1:
            fail(f"{f}: anchor occurs {n} times, expected exactly once. "
                 f"The file has changed since this patch was written. "
                 f"First anchor line: {old.splitlines()[0]!r}")
    print("check OK: all anchors unique, prerequisites present.")


def apply(root: Path) -> None:
    check(root)
    for f in FILES:
        src = root / f
        bak = root / (f + SUFFIX)
        if bak.exists():
            fail(f"{bak.name} already exists. Revert or remove it first.")
        shutil.copy2(src, bak)
        print(f"backup: {bak.name}")
    for f, old, new in EDITS:
        p = root / f
        p.write_text(p.read_text().replace(old, new, 1))
    for f in FILES:
        py_compile.compile(str(root / f), doraise=True)
        print(f"applied + compiles: {f}")
    print("done. Revert with: python apply_c6_block3.py --revert")


def revert(root: Path) -> None:
    missing = [f for f in FILES if not (root / (f + SUFFIX)).exists()]
    if missing:
        fail(f"no backup for: {', '.join(missing)}")
    for f in FILES:
        shutil.copy2(root / (f + SUFFIX), root / f)
        (root / (f + SUFFIX)).unlink()
        print(f"reverted: {f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="verify anchors and prerequisites, write nothing")
    g.add_argument("--revert", action="store_true",
                   help="restore the .bak.c6b3 backups")
    args = ap.parse_args()
    root = Path(".").resolve()
    if args.revert:
        revert(root)
    elif args.check:
        check(root)
    else:
        apply(root)


if __name__ == "__main__":
    main()
