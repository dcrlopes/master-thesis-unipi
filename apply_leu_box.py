#!/usr/bin/env python3
"""
apply_leu_box.py -- tie the enrichment SEARCH BOX to the LEU cap after zoning.

THE PROBLEM THIS FIXES
----------------------
Two enrichment quantities exist in this pipeline and only one of them was
bounded.

  1. The DESIGN enrichment, the pair (enrich_inner, enrich_outer) that the
     optimizer proposes. Its box was [2.0, 19.75] wt% U-235 and the
     constraint g_enr = max(e_in, e_out) - 19.75 audited that same number.
  2. The AS-BUILT enrichment of the zoned core. zoning.max_zoned_enrichment
     shows the peripheral ring carries max(e_in, e_out) * m_P, because the
     zoning map scales both intra-assembly enrichments of a ring by one
     multiplier.

Nothing bounded the second quantity. Campaign 5's transfer study
(campaign5/transfer_summary.json) recorded n_leu_screen_violations = 10, that
is, ten of sixty designs exceeded the LEU (Low Enriched Uranium) cap of
19.75 wt% U-235 once the peripheral multiplier was applied, even though every
one of them satisfied g_enr as written.

THE FIX
-------
The design box becomes the enforcement and the constraint becomes the audit.

    E_SEARCH_MAX = LEU_CAP / M_P_DESIGN

With that upper bound on both enrichment variables, the highest enrichment
anywhere in the zoned core cannot exceed LEU_CAP by construction. The
optimizer never spends an evaluation on a core it is not allowed to build,
and NSGA-II (Non-dominated Sorting Genetic Algorithm II) never proposes one,
because a box bound is a hard bound rather than a penalty.

g_enr is redefined on the physically meaningful quantity:

    g_enr = max(e_in, e_out) * M_P_DESIGN - LEU_CAP

It is now satisfied by construction and will read as a small negative number
for every design. That is intentional. It is kept in constraint_names for two
reasons:

  1. removing it would change the constraint list and load_checkpoint would
     then refuse every existing Campaign 5 checkpoint,
  2. a constraint that is always satisfied because the search box makes it so
     is exactly the kind of thing an examiner asks about, and a live audit
     column in the archive answers the question with data.

WHAT IT COSTS ON THE EXISTING ARCHIVE
-------------------------------------
At M_P_DESIGN = 1.150, the Stage 2 optimum, the box cap is 17.174 wt% and 17
of the 60 Campaign 5 designs fall outside it. Fourteen already violated
g_kmax. The other three are idx6, idx8 and idx33, at core F_dH (radial
enthalpy-rise hot channel factor) of 2.63, 2.65 and 3.78, far from any front.
The near-feasible designs all survive: idx49 at 7.39 wt%, idx54 at 15.00 wt%,
idx57 at 15.13 wt%, idx17 at 16.83 wt%, and every zoning champion.

THE HAZARD THIS ALSO CLOSES
---------------------------
ActiveLearningMOO.load_checkpoint compares design-variable NAMES and
constraint names, but never the BOUNDS. Resuming a Campaign 5 checkpoint under
a tightened box therefore succeeded silently, with part of the archive sitting
outside the box the search was using. This applier adds a loud warning that
counts those points.

The warning is deliberately not an error. Archive points outside the new box
remain valid training data for the Gaussian Process (GP) surrogate, and
throwing them away would waste real OpenMC (Open source Monte Carlo particle
transport code) evaluations. What changes is only that NSGA-II may no longer
propose designs in that region. Keeping them as training data while forbidding
them as proposals is the correct treatment.

CHOOSING M_P_DESIGN
-------------------
Pass --m-p. The value must match the zoning map the candidate cores will
actually use.

    m_P     box cap [wt%]   Campaign 5 designs outside the box
    1.000       19.750          0 of 60   (no zoning, current behaviour)
    1.075       18.372         10 of 60   (transfer_summary map)
    1.150       17.174         17 of 60   (Stage 2 base-grid optimum)
    1.250       15.800         22 of 60   (extended-grid interior optimum)

Default is 1.150.

USAGE
    cd ~/master-thesis-unipi           # on branch campaign5
    python3 apply_leu_box.py --check           # verify anchors, write nothing
    python3 apply_leu_box.py --m-p 1.150       # apply
    python3 apply_leu_box.py --revert          # restore the .bak files

FLAGS
    --check    verify every anchor and print the planned edits, no writes
    --m-p F    peripheral zoning multiplier used to derive the box cap
    --leu F    the LEU cap in wt% U-235 (default 19.75)
    --revert   restore each patched file from its .bak backup
    --root D   repository root (default: the current directory)

A FRESH CAMPAIGN IS MANDATORY AFTER APPLYING
--------------------------------------------
Changing the box changes the search domain. Designs evaluated under the old
box and designs proposed under the new one are not samples of the same
optimization problem. Start Campaign 6 in a fresh --out and a fresh --workdir.
Resuming out_c5 is possible and the archive stays useful, but the campaign
must then be reported as two domains, not one.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# the edits, as (file, anchor, replacement) triples. Every anchor must occur
# EXACTLY ONCE or the applier refuses to touch the file.
# --------------------------------------------------------------------------
POLICY_MODULE = '''"""
leu_policy.py
=============
Single source of truth for the enrichment policy of the optimization.

Two numbers govern every enrichment decision in this pipeline and they are
defined here so that reactor_optimization.py (the search box) and
openmc_evaluator.py (the audit constraint) can never disagree.

    LEU_CAP_WTPC : float
        Maximum permitted U-235 enrichment anywhere in the as-built core,
        in weight per cent. 19.75 wt% is the conventional LEU (Low Enriched
        Uranium) ceiling, set below the 20 wt% boundary that defines HEU
        (High Enriched Uranium) so that manufacturing tolerance cannot cross
        it.

    M_P_DESIGN : float
        Peripheral zoning multiplier of the loading map the candidate cores
        will use, dimensionless. The zoned core's highest enrichment is
        max(enrich_inner, enrich_outer) * M_P_DESIGN, because
        zoning.assign_zone_designs scales both intra-assembly enrichments of
        a ring by one multiplier.

    E_SEARCH_MAX : float
        Upper bound placed on BOTH enrichment design variables, in weight
        per cent. Derived, not chosen:

            E_SEARCH_MAX = LEU_CAP_WTPC / M_P_DESIGN

        so that no design the optimizer can propose exceeds the LEU cap once
        the peripheral multiplier is applied.

Set M_P_DESIGN to 1.0 to recover the unzoned behaviour, in which the search
box and the LEU cap coincide.
"""
from __future__ import annotations

LEU_CAP_WTPC = {leu!r}
M_P_DESIGN = {mp!r}
E_SEARCH_MAX = LEU_CAP_WTPC / M_P_DESIGN


def max_zoned_enrichment_wtpc(e_inner: float, e_outer: float) -> float:
    """Highest U-235 enrichment anywhere in the zoned core, in weight per cent.

    Mirrors zoning.max_zoned_enrichment, duplicated here so the evaluator can
    audit the enrichment without importing the zoning module.
    """
    return max(float(e_inner), float(e_outer)) * M_P_DESIGN
'''

ANCHOR_OPT_BOUNDS = (
    '        DesignVariable("enrich_inner", 2.0, 19.75, "%"),\n'
    '        DesignVariable("enrich_outer", 2.0, 19.75, "%"),'
)
REPLACE_OPT_BOUNDS = (
    '        # Upper bound is LEU_CAP_WTPC / M_P_DESIGN, so that the highest\n'
    '        # enrichment anywhere in the ZONED core stays at or below the LEU\n'
    '        # (Low Enriched Uranium) cap by construction. See leu_policy.py.\n'
    '        DesignVariable("enrich_inner", 2.0, _leu.E_SEARCH_MAX, "%"),\n'
    '        DesignVariable("enrich_outer", 2.0, _leu.E_SEARCH_MAX, "%"),'
)

ANCHOR_OPT_IMPORT = ('    design that cannot physically be built."""\n'
                     '    from core_geometry import geometry_margin\n')
REPLACE_OPT_IMPORT = ('    design that cannot physically be built."""\n'
                      '    from core_geometry import geometry_margin\n'
                      '    import leu_policy as _leu\n')

ANCHOR_EV_IMPORT = 'import core_geometry as cg\n'
REPLACE_EV_IMPORT = ('import core_geometry as cg\n'
                     'import leu_policy as _leu\n')

ANCHOR_EV_GENR = '            "g_enr":   max(e_in, e_out) - 19.75,        # LEU cap\n'
REPLACE_EV_GENR = (
    '            # LEU cap audited on the AS-BUILT zoned enrichment, not the\n'
    '            # design value: the peripheral ring carries\n'
    '            # max(e_in, e_out) * M_P_DESIGN. Satisfied by construction\n'
    '            # because the search box is LEU_CAP / M_P_DESIGN.\n'
    '            "g_enr":   (_leu.max_zoned_enrichment_wtpc(e_in, e_out)\n'
    '                        - _leu.LEU_CAP_WTPC),\n'
    '            "e_max_zoned": _leu.max_zoned_enrichment_wtpc(e_in, e_out),\n'
)

ANCHOR_CKPT = '        self.evaluator.n_calls = len(ckpt["all_raw"])\n'
REPLACE_CKPT = (
    '        # Bounds guard. load_checkpoint matches variable NAMES but not\n'
    '        # BOUNDS, so a checkpoint written under a wider box loads without\n'
    '        # complaint. Those points stay in the archive on purpose: they are\n'
    '        # valid training data for the surrogate and represent real spent\n'
    '        # evaluations. What changes is that NSGA-II can no longer propose\n'
    '        # designs there, so the search domain and the training domain are\n'
    '        # no longer the same set. Report the campaign accordingly.\n'
    '        if len(self.X):\n'
    '            xl, xu = self.spec.design_space.xl, self.spec.design_space.xu\n'
    '            outside = np.any((self.X < xl - 1e-9) | (self.X > xu + 1e-9),\n'
    '                             axis=1)\n'
    '            if outside.any():\n'
    '                names = self.spec.design_space.names\n'
    '                print(f"!! WARNING: {int(outside.sum())} of {len(self.X)} "\n'
    '                      f"loaded evaluations lie OUTSIDE the current design "\n'
    '                      f"box. They are kept as surrogate training data but "\n'
    '                      f"cannot be proposed again.")\n'
    '                for j, nm in enumerate(names):\n'
    '                    col = self.X[:, j]\n'
    '                    n_j = int(np.sum((col < xl[j] - 1e-9) |\n'
    '                                     (col > xu[j] + 1e-9)))\n'
    '                    if n_j:\n'
    '                        print(f"     {nm}: {n_j} outside "\n'
    '                              f"[{xl[j]:.4g}, {xu[j]:.4g}], "\n'
    '                              f"observed range "\n'
    '                              f"[{col.min():.4g}, {col.max():.4g}]")\n'
    '        self.evaluator.n_calls = len(ckpt["all_raw"])\n'
)

EDITS = [
    ("reactor_optimization.py", ANCHOR_OPT_IMPORT, REPLACE_OPT_IMPORT,
     "import the enrichment policy module"),
    ("reactor_optimization.py", ANCHOR_OPT_BOUNDS, REPLACE_OPT_BOUNDS,
     "tie both enrichment bounds to E_SEARCH_MAX"),
    ("reactor_optimization.py", ANCHOR_CKPT, REPLACE_CKPT,
     "warn when a checkpoint holds points outside the current box"),
    ("openmc_evaluator.py", ANCHOR_EV_IMPORT, REPLACE_EV_IMPORT,
     "import the enrichment policy module"),
    ("openmc_evaluator.py", ANCHOR_EV_GENR, REPLACE_EV_GENR,
     "audit g_enr on the as-built zoned enrichment"),
]


def verify(root: Path):
    """Check that every anchor occurs exactly once. Returns a list of errors."""
    errors = []
    for fname, anchor, _, label in EDITS:
        path = root / fname
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        text = path.read_text()
        n = text.count(anchor)
        if n != 1:
            head = anchor.strip().splitlines()[0][:60]
            errors.append(f"{fname}: anchor for '{label}' found {n} times, "
                          f"expected 1  ({head}...)")
    return errors


def apply(root: Path, leu: float, mp: float):
    (root / "leu_policy.py").write_text(POLICY_MODULE.format(leu=leu, mp=mp))
    print(f"created  leu_policy.py  "
          f"(LEU_CAP_WTPC={leu}, M_P_DESIGN={mp}, "
          f"E_SEARCH_MAX={leu / mp:.4f} wt%)")

    touched = set()
    for fname, anchor, repl, label in EDITS:
        path = root / fname
        if fname not in touched:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
            touched.add(fname)
        text = path.read_text()
        path.write_text(text.replace(anchor, repl, 1))
        print(f"patched  {fname}: {label}")
    print("\nbackups written as <file>.bak")


def revert(root: Path):
    n = 0
    for fname in sorted({f for f, *_ in EDITS}):
        bak = root / (fname + ".bak")
        if bak.is_file():
            shutil.copy2(bak, root / fname)
            print(f"restored {fname} from {bak.name}")
            n += 1
        else:
            print(f"no backup for {fname}, left untouched")
    policy = root / "leu_policy.py"
    if policy.is_file():
        policy.unlink()
        print("removed  leu_policy.py")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify anchors and print the planned edits, "
                         "writing nothing")
    ap.add_argument("--revert", action="store_true",
                    help="restore every patched file from its .bak backup")
    ap.add_argument("--m-p", type=float, default=1.150, dest="mp",
                    help="peripheral zoning multiplier, dimensionless "
                         "(1.0 recovers the unzoned behaviour)")
    ap.add_argument("--leu", type=float, default=19.75,
                    help="LEU cap in wt%% U-235")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"repository: {root}")

    if args.revert:
        revert(root)
        return

    if args.mp <= 0:
        raise SystemExit("--m-p must be positive")
    if not (0.5 <= args.mp <= 2.0):
        raise SystemExit(f"--m-p {args.mp} is outside the sanity window "
                         f"(0.5, 2.0); check the value against the zoning map")

    cap = args.leu / args.mp
    print(f"LEU cap        : {args.leu} wt% U-235")
    print(f"m_P            : {args.mp}")
    print(f"new search box : enrich_inner, enrich_outer in "
          f"[2.0, {cap:.4f}] wt%")

    errors = verify(root)
    if errors:
        print("\nANCHOR CHECK FAILED, nothing was written:")
        for e in errors:
            print("  " + e)
        print("\nThe file has already been patched, or the branch is not "
              "campaign5 at the expected commit. Run --revert first if a "
              "previous run left the tree half-patched.")
        sys.exit(1)
    print(f"\nanchor check   : all {len(EDITS)} anchors found exactly once")

    if args.check:
        print("\nplanned edits:")
        for fname, _, _, label in EDITS:
            print(f"  {fname}: {label}")
        print("  leu_policy.py: created")
        print("\n--check given, nothing written.")
        return

    apply(root, args.leu, args.mp)
    print("\nNEXT STEPS")
    print("  1. python -c \"import leu_policy; print(leu_policy.E_SEARCH_MAX)\"")
    print("  2. python -c \"from reactor_optimization import "
          "example_reactor_problem as p; s=p(); print(s.design_space.xu)\"")
    print("  3. python run_optimization.py --smoke --workdir smoke_leu "
          "--out out_smoke_leu")
    print("  4. Start Campaign 6 in a FRESH --out and --workdir. The search "
          "domain has changed, so old and new evaluations are not samples of "
          "the same problem.")


if __name__ == "__main__":
    main()
