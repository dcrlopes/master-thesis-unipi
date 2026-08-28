#!/usr/bin/env python3
"""
apply_leu_box.py  (v2) -- tie the enrichment SEARCH BOX to the LEU cap after
zoning, with a default that is a numerical no-op.

WHAT CHANGED SINCE v1
---------------------
v1 was written against GitHub commit 4144050. The local tree is ahead: a
local, unpushed fix_kmax_basis.py parameterised the four scalar limits, so
the constraint line now reads

    "g_enr":   max(e_in, e_out) - self.enr_max,   # LEU cap

instead of the hardcoded 19.75. v2 anchors on that line and KEEPS
self.enr_max, so fix_kmax_basis.py stays compatible. v2 also does not touch
the OpenMCEvaluator constructor signature or its call site, both of which
span continuation lines that cannot be anchored safely from the outside.

THE PROBLEM
-----------
Two enrichment quantities exist in this pipeline and only one is bounded.

  1. The DESIGN enrichment, the pair (enrich_inner, enrich_outer) the
     optimizer proposes, bounded by the design box and audited by g_enr.
  2. The AS-BUILT enrichment of the zoned core. zoning.max_zoned_enrichment
     shows the peripheral ring carries max(e_in, e_out) * m_P, because the
     zoning map scales both intra-assembly enrichments of a ring by one
     multiplier.

Nothing bounds the second. campaign5/transfer_summary.json records
n_leu_screen_violations = 10, that is, ten of sixty Campaign 5 designs
exceeded the LEU (Low Enriched Uranium) cap once the peripheral multiplier
was applied, while every one of them satisfied g_enr as written.

THE FIX
-------
The box becomes the enforcement and the constraint becomes the audit.

    E_SEARCH_MAX = LEU_CAP_WTPC / M_P_DESIGN

A box bound is hard, so the optimizer cannot propose a core it is not
allowed to build and no evaluation is spent discovering that. g_enr is then
audited on the physically meaningful quantity:

    g_enr = max(e_in, e_out) * M_P_DESIGN - self.enr_max

and a new e_max_zoned field records the as-built maximum per evaluation.

THE SAFETY PROPERTY
-------------------
M_P_DESIGN defaults to 1.0, at which

    E_SEARCH_MAX = LEU_CAP_WTPC = 19.75  and
    g_enr = max(e_in, e_out) - self.enr_max

which is bit-for-bit the current behaviour. Applying this patch with the
default therefore changes nothing that can be measured, and
verify_leu_box.py proves that against all sixty archived Campaign 5
evaluations before any campaign is launched. Only after that check passes do
you raise m_P.

BLAST RADIUS
------------
Seven edits in three files, plus one new file. Every anchor is a single line
or two adjacent lines, and verify() refuses to write anything unless every
anchor occurs exactly once.

    leu_policy.py            created
    reactor_optimization.py  import, two bounds, checkpoint bounds guard
    openmc_evaluator.py      import, g_enr and e_max_zoned
    run_optimization.py      import, enrichment policy into checkpoint meta

Not touched: the OpenMCEvaluator constructor, its call site, self.enr_max,
the constraint name list, and every other constraint.

USAGE
    cd ~/master-thesis-unipi
    python3 apply_leu_box.py --check                # verify anchors, no writes
    python3 apply_leu_box.py                        # apply at m_P = 1.0
    python3 verify_leu_box.py --checkpoint out_c5/optimization_checkpoint.json
    python3 apply_leu_box.py --revert               # restore from .bak
    python3 apply_leu_box.py --m-p 1.150            # only after verify passes

FLAGS
    --check    verify every anchor and print the planned edits, write nothing
    --m-p F    peripheral zoning multiplier (default 1.0, a no-op)
    --leu F    LEU cap in wt%% U-235 (default 19.75)
    --revert   restore each patched file from its .bak backup
    --root D   repository root (default: the current directory)

CHOOSING M_P LATER

    m_P     box cap [wt%]   Campaign 5 designs outside the box
    1.000       19.750          0 of 60   (no-op, current behaviour)
    1.075       18.372         10 of 60   (transfer_summary map)
    1.150       17.174         17 of 60   (Stage 2 base-grid optimum)
    1.250       15.800         22 of 60   (extended-grid interior optimum)

Raising m_P changes the search domain, so the next campaign needs a fresh
--out and a fresh --workdir. At m_P = 1.0 nothing changes and no fresh
campaign is needed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

POLICY_MODULE = '''"""
leu_policy.py
=============
Single source of truth for the enrichment policy of the optimization.

Two numbers govern every enrichment decision in this pipeline. They are
defined here so that reactor_optimization.py (the search box),
openmc_evaluator.py (the audit constraint) and run_optimization.py (the
campaign provenance) can never disagree.

    LEU_CAP_WTPC : float
        Maximum permitted U-235 enrichment anywhere in the as-built core, in
        weight per cent. 19.75 wt% is the conventional LEU (Low Enriched
        Uranium) ceiling, set below the 20 wt% boundary that defines HEU
        (High Enriched Uranium) so manufacturing tolerance cannot cross it.

    M_P_DESIGN : float
        Peripheral zoning multiplier of the loading map the candidate cores
        will use, dimensionless. The zoned core's highest enrichment is
        max(enrich_inner, enrich_outer) * M_P_DESIGN, because
        zoning.assign_zone_designs scales both intra-assembly enrichments of
        a ring by one multiplier.

    E_SEARCH_MAX : float
        Upper bound on BOTH enrichment design variables, in weight per cent.
        Derived, not chosen:

            E_SEARCH_MAX = LEU_CAP_WTPC / M_P_DESIGN

        so no design the optimizer can propose exceeds the LEU cap once the
        peripheral multiplier is applied.

At M_P_DESIGN = 1.0 the search box and the LEU cap coincide and every
formula below reduces to the unzoned behaviour exactly.
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

# --------------------------------------------------------------------------
# anchors. Every one must occur EXACTLY ONCE or nothing is written.
# --------------------------------------------------------------------------
A_OPT_IMPORT = ('    design that cannot physically be built."""\n'
                '    from core_geometry import geometry_margin\n')
R_OPT_IMPORT = ('    design that cannot physically be built."""\n'
                '    from core_geometry import geometry_margin\n'
                '    import leu_policy as _leu\n')

A_OPT_BOUNDS = ('        DesignVariable("enrich_inner", 2.0, 19.75, "%"),\n'
                '        DesignVariable("enrich_outer", 2.0, 19.75, "%"),')
R_OPT_BOUNDS = (
    '        # Upper bound is LEU_CAP_WTPC / M_P_DESIGN, so the highest\n'
    '        # enrichment anywhere in the ZONED core stays at or below the\n'
    '        # LEU (Low Enriched Uranium) cap by construction. At\n'
    '        # M_P_DESIGN = 1.0 this is exactly 19.75, the previous bound.\n'
    '        # See leu_policy.py.\n'
    '        DesignVariable("enrich_inner", 2.0, _leu.E_SEARCH_MAX, "%"),\n'
    '        DesignVariable("enrich_outer", 2.0, _leu.E_SEARCH_MAX, "%"),')

A_CKPT = '        self.evaluator.n_calls = len(ckpt["all_raw"])\n'
R_CKPT = (
    '        # Bounds guard. load_checkpoint matches variable NAMES but not\n'
    '        # BOUNDS, so a checkpoint written under a wider box loads without\n'
    '        # complaint. Those points stay in the archive on purpose: they\n'
    '        # are valid training data for the surrogate and represent real\n'
    '        # spent evaluations. What changes is that NSGA-II can no longer\n'
    '        # propose designs there, so the search domain and the training\n'
    '        # domain are no longer the same set. Report accordingly.\n'
    '        if len(self.X):\n'
    '            xl, xu = self.spec.design_space.xl, self.spec.design_space.xu\n'
    '            outside = np.any((self.X < xl - 1e-9) | (self.X > xu + 1e-9),\n'
    '                             axis=1)\n'
    '            if outside.any():\n'
    '                print(f"!! WARNING: {int(outside.sum())} of {len(self.X)} "\n'
    '                      f"loaded evaluations lie OUTSIDE the current design "\n'
    '                      f"box. They are kept as surrogate training data but "\n'
    '                      f"cannot be proposed again.")\n'
    '                for j, nm in enumerate(self.spec.design_space.names):\n'
    '                    col = self.X[:, j]\n'
    '                    n_j = int(np.sum((col < xl[j] - 1e-9) |\n'
    '                                     (col > xu[j] + 1e-9)))\n'
    '                    if n_j:\n'
    '                        print(f"     {nm}: {n_j} outside "\n'
    '                              f"[{xl[j]:.4g}, {xu[j]:.4g}], observed "\n'
    '                              f"range [{col.min():.4g}, {col.max():.4g}]")\n'
    '        self.evaluator.n_calls = len(ckpt["all_raw"])\n'
)

A_EV_IMPORT = 'import core_geometry as cg\n'
R_EV_IMPORT = 'import core_geometry as cg\nimport leu_policy as _leu\n'

A_EV_GENR = ('            "g_enr":   max(e_in, e_out) - self.enr_max,'
             '   # LEU cap\n')
R_EV_GENR = (
    '            # LEU cap audited on the AS-BUILT zoned enrichment, not the\n'
    '            # design value: the peripheral ring carries\n'
    '            # max(e_in, e_out) * M_P_DESIGN. Satisfied by construction,\n'
    '            # because the search box is LEU_CAP_WTPC / M_P_DESIGN. At\n'
    '            # M_P_DESIGN = 1.0 this reduces to the previous expression.\n'
    '            "g_enr":   (_leu.max_zoned_enrichment_wtpc(e_in, e_out)\n'
    '                        - self.enr_max),\n'
    '            "e_max_zoned": _leu.max_zoned_enrichment_wtpc(e_in, e_out),\n'
)

A_RUN_IMPORT = '    spec = example_reactor_problem()\n'
R_RUN_IMPORT = ('    import leu_policy as _leu\n'
                '    spec = example_reactor_problem()\n')

A_RUN_META = '                           "geometry": "v2-envelope",\n'
R_RUN_META = (
    '                           "geometry": "v2-envelope",\n'
    '                           "enrichment_policy": {\n'
    '                               "leu_cap_wtpc": _leu.LEU_CAP_WTPC,\n'
    '                               "m_p_design": _leu.M_P_DESIGN,\n'
    '                               "e_search_max_wtpc": _leu.E_SEARCH_MAX},\n'
)

EDITS = [
    ("reactor_optimization.py", A_OPT_IMPORT, R_OPT_IMPORT,
     "import the enrichment policy module"),
    ("reactor_optimization.py", A_OPT_BOUNDS, R_OPT_BOUNDS,
     "tie both enrichment bounds to E_SEARCH_MAX"),
    ("reactor_optimization.py", A_CKPT, R_CKPT,
     "warn when a checkpoint holds points outside the current box"),
    ("openmc_evaluator.py", A_EV_IMPORT, R_EV_IMPORT,
     "import the enrichment policy module"),
    ("openmc_evaluator.py", A_EV_GENR, R_EV_GENR,
     "audit g_enr on the as-built zoned enrichment"),
    ("run_optimization.py", A_RUN_IMPORT, R_RUN_IMPORT,
     "import the enrichment policy module"),
    ("run_optimization.py", A_RUN_META, R_RUN_META,
     "record the enrichment policy in the checkpoint metadata"),
]


def verify(root: Path):
    errors = []
    for fname, anchor, _, label in EDITS:
        path = root / fname
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        n = path.read_text().count(anchor)
        if n != 1:
            head = anchor.strip().splitlines()[0][:64]
            errors.append(f"{fname}: anchor for '{label}' found {n} times, "
                          f"expected 1  ({head}...)")
    return errors


def apply(root: Path, leu: float, mp: float):
    (root / "leu_policy.py").write_text(POLICY_MODULE.format(leu=leu, mp=mp))
    print(f"created  leu_policy.py  (LEU_CAP_WTPC={leu}, M_P_DESIGN={mp}, "
          f"E_SEARCH_MAX={leu / mp:.4f} wt%)")
    touched = set()
    for fname, anchor, repl, label in EDITS:
        path = root / fname
        if fname not in touched:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
            touched.add(fname)
        path.write_text(path.read_text().replace(anchor, repl, 1))
        print(f"patched  {fname}: {label}")
    print("\nbackups written as <file>.bak")


def revert(root: Path):
    for fname in sorted({f for f, *_ in EDITS}):
        bak = root / (fname + ".bak")
        if bak.is_file():
            shutil.copy2(bak, root / fname)
            print(f"restored {fname} from {bak.name}")
        else:
            print(f"no backup for {fname}, left untouched")
    policy = root / "leu_policy.py"
    if policy.is_file():
        policy.unlink()
        print("removed  leu_policy.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify anchors and print planned edits, no writes")
    ap.add_argument("--revert", action="store_true",
                    help="restore every patched file from its .bak backup")
    ap.add_argument("--m-p", type=float, default=1.0, dest="mp",
                    help="peripheral zoning multiplier, dimensionless. "
                         "1.0 is a numerical no-op and is the default.")
    ap.add_argument("--leu", type=float, default=19.75,
                    help="LEU cap in wt%% U-235")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"repository: {root}")

    if args.revert:
        revert(root)
        return

    if not (0.5 <= args.mp <= 2.0):
        raise SystemExit(f"--m-p {args.mp} is outside the sanity window "
                         f"(0.5, 2.0); check it against the zoning map")

    cap = args.leu / args.mp
    print(f"LEU cap        : {args.leu} wt% U-235")
    print(f"m_P            : {args.mp}"
          + ("   (no-op: behaviour is unchanged)" if args.mp == 1.0 else ""))
    print(f"new search box : enrich_inner, enrich_outer in "
          f"[2.0, {cap:.4f}] wt%")

    errors = verify(root)
    if errors:
        print("\nANCHOR CHECK FAILED, nothing was written:")
        for e in errors:
            print("  " + e)
        print("\nEither the tree is already patched, or a local change has "
              "moved an anchor. Run --revert if a previous attempt left "
              "backups, then send me the line the anchor expected.")
        sys.exit(1)
    print(f"\nanchor check   : all {len(EDITS)} anchors found exactly once")

    if args.check:
        print("\nplanned edits:")
        print("  leu_policy.py: created")
        for fname, _, _, label in EDITS:
            print(f"  {fname}: {label}")
        print("\n--check given, nothing written.")
        return

    apply(root, args.leu, args.mp)
    print("\nNEXT STEPS")
    print("  1. python3 -c \"import leu_policy as l; print(l.E_SEARCH_MAX)\"")
    print("  2. python3 verify_leu_box.py "
          "--checkpoint out_c5/optimization_checkpoint.json")
    print("  3. python3 run_optimization.py --smoke --workdir smoke_leu "
          "--out out_smoke_leu")
    if args.mp == 1.0:
        print("  4. m_P is 1.0, so behaviour is unchanged and no fresh "
              "campaign is needed. Raise m_P only after step 2 passes.")
    else:
        print("  4. m_P is not 1.0, so the search domain has changed. "
              "Campaign 6 needs a FRESH --out and --workdir.")


if __name__ == "__main__":
    main()
