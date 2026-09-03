#!/usr/bin/env python3
"""apply_c6_block4.py -- feasibility-margin acquisition for Campaign 6 block 4.

WHY
---
Block 3 fixed the batch collapse (six distinct designs, full spread in
pitch and reflector) and exposed the next defect: the acquisition ranks
by uncertainty with no regard to predicted feasibility. Five of the six
picks came back infeasible on the reactivity limit, with core k_eff
exceeding 1.35 by 1530 to 7750 pcm, because the surrogate drove
gadolinia to zero and enrichment to the ceiling. The constraint
Gaussian Process predicted all five feasible even though the archive
holds design 25 at k_core = 1.40210 in that very corner: the GP mean is
optimistic where the data is thin, and nothing in the selection asked
for margin.

FIX
---
Before the uncertainty ranking, every candidate is scored on the
constraint surrogate with an uncertainty margin:

    s = max over GP-predicted constraints of ( g_mean + kappa * g_std )

A candidate is margin-feasible when s <= 0, in the normalised constraint
units of CONSTRAINT-NORM. Where the GP is confident the margin is small,
where it extrapolates the margin is large, which is exactly where block 3
failed. Constraints computed exactly (geometry, enrichment cap) are
excluded from the test because the NSGA population already satisfies
them exactly.

NOTHING IS DISCARDED. Margin-feasible candidates are ranked first, by
uncertainty as before. Candidates that fail the margin follow, ordered
by how close they come to passing it, so a front with too few
margin-feasible members still fills the batch with the least risky
picks. The BATCH-DIVERSITY separation then applies to the combined
ranking unchanged.

kappa defaults to 1.0 (one standard deviation of margin), CLI flag
--feas-kappa, recorded in meta["surrogate_policy"]. kappa = 0 reproduces
the block 3 behaviour exactly.

MECHANICS
---------
Prerequisite: the BATCH-DIVERSITY marker of apply_c6_block3.py must be
present. Refuses to run twice (FEAS-MARGIN marker). Anchor-verified:
every target string must occur exactly once or nothing is written.
Originals are backed up as <file>.bak.c6b4.
    python apply_c6_block4.py --check    verify anchors, write nothing
    python apply_c6_block4.py            apply, back up, py_compile
    python apply_c6_block4.py --revert   restore the .bak.c6b4 backups
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path

SUFFIX = ".bak.c6b4"
MARKER = "FEAS-MARGIN"
PREREQ = "BATCH-DIVERSITY"

RO = "reactor_optimization.py"
RU = "run_optimization.py"

EDITS = [

# ---- reactor_optimization.py ---------------------------------------------
(RO,
"""    infill_min_sep: float = 0.05   # BATCH-DIVERSITY: minimum separation of
                                   # the infill picks (and of each pick from
                                   # the archive) in the unit design box,
                                   # as ||dx/span|| / sqrt(n_var)
""",
"""    infill_min_sep: float = 0.05   # BATCH-DIVERSITY: minimum separation of
                                   # the infill picks (and of each pick from
                                   # the archive) in the unit design box,
                                   # as ||dx/span|| / sqrt(n_var)
    feas_kappa: float = 1.0        # FEAS-MARGIN: candidates must satisfy
                                   # g_mean + kappa*g_std <= 0 on the GP
                                   # constraints to rank first; 0 restores
                                   # the pure-uncertainty ordering
""",
),

(RO,
"""            _, std = obj_sur.predict(cand)
            score = (std / (std.max(axis=0) + 1e-12)).sum(axis=1)
            order = np.argsort(-score)
""",
"""            _, std = obj_sur.predict(cand)
            score = (std / (std.max(axis=0) + 1e-12)).sum(axis=1)
            # FEAS-MARGIN: block 3 selected five of six infill designs past
            # the reactivity limit (k_core over 1.35 by 1530 to 7750 pcm)
            # because the constraint GP is optimistic where the data is
            # thin and the ranking never asked for margin. Score every
            # candidate as
            #     s = max_j ( g_mean_j + kappa * g_std_j )
            # over the GP-predicted constraints, in the normalised units of
            # CONSTRAINT-NORM. Exact constraints (geometry, enrichment) are
            # excluded: the NSGA population satisfies them exactly.
            # Margin-feasible candidates (s <= 0) rank first, by
            # uncertainty as before. The rest follow, ordered by s, so
            # nothing is discarded and the batch always fills. kappa = 0
            # reproduces the block 3 ranking exactly.
            g_mean, g_std = con_sur.predict(cand)
            g_mean = np.atleast_2d(np.asarray(g_mean, dtype=float))
            g_std = np.atleast_2d(np.asarray(g_std, dtype=float))
            kappa = float(getattr(self.cfg, "feas_kappa", 1.0))
            _exact_idx = {self.spec.constraint_names.index(n)
                          for n in self.spec.exact_constraints}
            _gp_cols = [j for j in range(g_mean.shape[1])
                        if j not in _exact_idx]
            if _gp_cols:
                s_marg = (g_mean[:, _gp_cols]
                          + kappa * g_std[:, _gp_cols]).max(axis=1)
            else:
                s_marg = np.zeros(len(cand), dtype=float)
            eligible = s_marg <= 0.0
            if verbose:
                print(f"           [acquisition] margin-feasible: "
                      f"{int(eligible.sum())}/{len(cand)} candidates "
                      f"at kappa={kappa:g}")
            order = np.concatenate([
                np.flatnonzero(eligible)[np.argsort(-score[eligible])],
                np.flatnonzero(~eligible)[np.argsort(s_marg[~eligible])],
            ]).astype(int)
""",
),

# ---- run_optimization.py -------------------------------------------------
(RU,
"""    ap.add_argument("--no-efpd-clip", action="store_true",
""",
"""    ap.add_argument("--feas-kappa", type=float, default=1.0,
                    help="FEAS-MARGIN: infill candidates must satisfy "
                         "g_mean + kappa*g_std <= 0 on the surrogate "
                         "constraints to rank first in the acquisition. "
                         "Larger kappa is more conservative. 0 restores "
                         "the pure-uncertainty ranking of block 3.")
    ap.add_argument("--no-efpd-clip", action="store_true",
""",
),

(RU,
"""    cfg.infill_min_sep = float(args.infill_min_sep)  # BATCH-DIVERSITY
""",
"""    cfg.infill_min_sep = float(args.infill_min_sep)  # BATCH-DIVERSITY
    cfg.feas_kappa = float(args.feas_kappa)          # FEAS-MARGIN
""",
),

(RU,
"""                               "infill_min_sep": cfg.infill_min_sep,
""",
"""                               "infill_min_sep": cfg.infill_min_sep,
                               "feas_kappa": cfg.feas_kappa,
""",
),
]

FILES = sorted({f for f, _, _ in EDITS})


def fail(msg: str) -> None:
    sys.exit(f"apply_c6_block4: REFUSED. {msg}")


def check(root: Path) -> None:
    for f in FILES:
        p = root / f
        if not p.is_file():
            fail(f"{f} not found. Run from the repository root.")
        text = p.read_text()
        if PREREQ not in text:
            fail(f"{f} lacks the {PREREQ} marker. Apply apply_c6_block3.py "
                 f"first, block 4 builds on the block 3 state.")
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
    print("done. Revert with: python apply_c6_block4.py --revert")


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
                   help="restore the .bak.c6b4 backups")
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
