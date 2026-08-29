#!/usr/bin/env python3
"""apply_c6_block2.py -- the two code changes for Campaign 6 block 2.

WHAT IT CHANGES
---------------
1. EFPD-CLIP. The truth evaluator censors every cycle length at the depletion
   ceiling (max_burnup), but the objective GP (Gaussian Process) extrapolates
   past it: on the C6 DOE archive, all 288 infill picks of the NSGA
   sensitivity study predicted 10448 to 10810 EFPD against a 10016.7 EFPD
   ceiling. Block 2 would spend real evaluations confirming the ceiling.
   Fix: _SurrogateProblem floors the minimise-space first objective at
   -efpd_cap, so beyond the cap the surrogate front is ranked by peaking
   alone, exactly as constrained dominance ranks censored truth points.
   The cap is DERIVED at launch from the same numbers the banner prints:

       efpd_cap = max_burnup [MWd/kgHM] * 1000 / spec_power [W/gHM]

   which is 10016.7 EFPD at the C6 cap of 100 MWd/kgHM (and 7512.5 at the
   C3-C5 cap of 75). Nothing is hardcoded and TRUTH values are untouched.
   Disable with --no-efpd-clip.

2. NSGA-SET. Adds --nsga-pop / --nsga-gen so the surrogate-search setting is
   a recorded launch decision instead of a hardcoded profile value. The C6
   sensitivity study (8 seeds, 6 settings) measured seed-to-seed hypervolume
   std 12.7 at the profile's 60x80 against 0.59 at 300x400, for a search
   cost of 8 s against a multi-hour infill iteration.

3. Both are recorded in checkpoint meta as meta["surrogate_policy"], and a
   resume of a checkpoint that predates the record prints a note.

WHAT IT DOES NOT CHANGE
-----------------------
Constraint normalisation: already applied on this branch (CONSTRAINT-NORM
markers in reactor_optimization.py and run_optimization.py). This script
verifies the markers are present and refuses to run if they are not, so the
three block-2 changes cannot be applied out of order.

MECHANICS
---------
Anchor-verified: every target string must occur EXACTLY once in the current
file or nothing is written. Originals are backed up as <file>.bak.c6b2.
    python apply_c6_block2.py --check    verify anchors, write nothing
    python apply_c6_block2.py            apply, back up, py_compile
    python apply_c6_block2.py --revert   restore the .bak.c6b2 backups
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path

SUFFIX = ".bak.c6b2"
MARKER = "EFPD-CLIP"          # presence in target files means already applied
PREREQ = "CONSTRAINT-NORM"    # must already be present (block-1 state)

RO = "reactor_optimization.py"
RU = "run_optimization.py"

# --------------------------------------------------------------------------
# (file, old, new) -- old must occur exactly once
# --------------------------------------------------------------------------
EDITS = [

# ---- reactor_optimization.py ---------------------------------------------
(RO,
"""    hv_ref: tuple | None = None # reference point for hypervolume (in MIN space)
""",
"""    hv_ref: tuple | None = None # reference point for hypervolume (in MIN space)
    efpd_cap: float | None = None  # EFPD-CLIP: ceiling on the SURROGATE's
                                   # predicted cycle length [EFPD]; None = off
""",
),

(RO,
"""    def __init__(self, spec, obj_surrogate, con_surrogate):
""",
"""    def __init__(self, spec, obj_surrogate, con_surrogate, efpd_cap=None):
""",
),

(RO,
"""        self.con_surrogate = con_surrogate
        self._exact_cols = [(spec.constraint_names.index(name), fn,
""",
"""        self.con_surrogate = con_surrogate
        self.efpd_cap = efpd_cap        # EFPD-CLIP (None disables the clip)
        self._exact_cols = [(spec.constraint_names.index(name), fn,
""",
),

(RO,
"""    def _evaluate(self, X, out, *a, **k):
        f_mean, _ = self.obj_surrogate.predict(X)
        out["F"] = f_mean
""",
"""    def _evaluate(self, X, out, *a, **k):
        f_mean, _ = self.obj_surrogate.predict(X)
        f_mean = np.atleast_2d(np.asarray(f_mean, dtype=float))
        if self.efpd_cap is not None:
            # EFPD-CLIP: the truth evaluator censors every cycle length at
            # the depletion ceiling, but a GP trained on the archive
            # extrapolates past it (C6 DOE: all 288 sensitivity-study picks
            # predicted 10448-10810 EFPD against a 10016.7 EFPD cap).
            # Column 0 is MINUS the cycle length (minimise space), so the
            # ceiling is a floor at -efpd_cap. On the resulting plateau the
            # first objective is flat and NSGA-II ranks by peaking alone,
            # exactly as dominance ranks censored truth points.
            f_mean[:, 0] = np.maximum(f_mean[:, 0], -float(self.efpd_cap))
        out["F"] = f_mean
""",
),

(RO,
"""            prob = _SurrogateProblem(self.spec, obj_sur, con_sur)
""",
"""            prob = _SurrogateProblem(self.spec, obj_sur, con_sur,
                                     efpd_cap=self.cfg.efpd_cap)  # EFPD-CLIP
""",
),

# ---- run_optimization.py -------------------------------------------------
(RU,
"""    ap.add_argument("--enr-max", type=float, default=19.75,
                    help="LEU (Low Enriched Uranium) enrichment cap in "
                         "wt%% U-235")
    args = ap.parse_args()
""",
"""    ap.add_argument("--enr-max", type=float, default=19.75,
                    help="LEU (Low Enriched Uranium) enrichment cap in "
                         "wt%% U-235")
    ap.add_argument("--nsga-pop", type=int, default=None,
                    help="NSGA-SET: NSGA-II population on the surrogate, "
                         "overriding the profile (full run: 60). The C6 "
                         "sensitivity study measured seed-to-seed HV std "
                         "12.7 at 60x80 vs 0.59 at 300x400, at <9 s of "
                         "search per iteration.")
    ap.add_argument("--nsga-gen", type=int, default=None,
                    help="NSGA-SET: NSGA-II generations on the surrogate, "
                         "overriding the profile (full run: 80).")
    ap.add_argument("--no-efpd-clip", action="store_true",
                    help="EFPD-CLIP: disable capping the surrogate's "
                         "predicted cycle length at the depletion ceiling "
                         "(max_burnup converted to EFPD). The clip is ON by "
                         "default because the truth evaluator censors "
                         "there, so predictions beyond it are fiction.")
    args = ap.parse_args()
""",
),

(RU,
"""    if args.n_infill is not None:
        cfg.n_infill = args.n_infill
""",
"""    if args.n_infill is not None:
        cfg.n_infill = args.n_infill
    # NSGA-SET: surrogate-search setting as a recorded launch decision
    if args.nsga_pop is not None:
        cfg.nsga_pop = int(args.nsga_pop)
    if args.nsga_gen is not None:
        cfg.nsga_gen = int(args.nsga_gen)
""",
),

(RU,
"""        "g_geom": _cg.R_VESSEL_INNER - _cg.VESSEL_CLEARANCE_CM,
    })
    opt = ActiveLearningMOO(spec, ev, cfg)
""",
"""        "g_geom": _cg.R_VESSEL_INNER - _cg.VESSEL_CLEARANCE_CM,
    })
    # EFPD-CLIP: cap the surrogate's predicted cycle length at the depletion
    # ceiling, converted with the same specific power the banner prints
    # (cap [MWd/kgHM] * 1000 / spec_power [W/gHM] = cap [EFPD]; 100 MWd/kgHM
    # is 10016.7 EFPD on this geometry). Truth values are untouched.
    if not args.no_efpd_clip:
        cfg.efpd_cap = float(schedule["max_burnup"]) * 1000.0 / ev.spec_power
    _cap_txt = ("off" if cfg.efpd_cap is None else f"{cfg.efpd_cap:.1f} EFPD")
    print(f"surrogate policy: cycle-length clip {_cap_txt} | "
          f"NSGA-II {cfg.nsga_pop}x{cfg.nsga_gen} | constraint norm active")
    opt = ActiveLearningMOO(spec, ev, cfg)
""",
),

(RU,
"""        n_loaded = opt.load_checkpoint(args.resume)
""",
"""        if prev_meta.get("surrogate_policy") is None:
            print("NOTE: this checkpoint predates the surrogate-policy "
                  "record. Earlier blocks searched without the EFPD clip "
                  "and at the profile NSGA setting; archived truth values "
                  "are unaffected. This block's policy is recorded in "
                  "meta['surrogate_policy'].")
        n_loaded = opt.load_checkpoint(args.resume)
""",
),

(RU,
"""                           "omp_threads": n_threads,
""",
"""                           "surrogate_policy": {
                               "efpd_cap_efpd": cfg.efpd_cap,
                               "nsga_pop": cfg.nsga_pop,
                               "nsga_gen": cfg.nsga_gen,
                               "constraint_norm": "g / own limit "
                                                  "(CONSTRAINT-NORM)"},
                           "omp_threads": n_threads,
""",
),
]

FILES = sorted({f for f, _, _ in EDITS})


def fail(msg: str) -> None:
    sys.exit(f"apply_c6_block2: REFUSED. {msg}")


def check(root: Path) -> None:
    for f in FILES:
        p = root / f
        if not p.is_file():
            fail(f"{f} not found. Run from the repository root.")
        text = p.read_text()
        if PREREQ not in text:
            fail(f"{f} lacks the {PREREQ} marker. This branch state does "
                 f"not match Campaign 6 block 1. Apply the constraint "
                 f"normalisation first (apply_constraint_norm.py).")
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
    print("done. Revert with: python apply_c6_block2.py --revert")


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
                   help="restore the .bak.c6b2 backups")
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
