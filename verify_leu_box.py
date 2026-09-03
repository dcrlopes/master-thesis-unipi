#!/usr/bin/env python3
"""
verify_leu_box.py -- prove that apply_leu_box.py changed nothing it should not.

WHY A SEPARATE PROOF
--------------------
A patch that is "carefully written" is not the same as a patch that is known
to be correct. This script turns the claim into a measurement, using the
sixty real OpenMC (Open source Monte Carlo particle transport code)
evaluations already in the Campaign 5 archive. No new evaluation is run and
the checkpoint is opened read-only.

FOUR CHECKS
-----------
  1. IMPORTS. leu_policy, reactor_optimization and run_optimization all
     import, and the derived cap satisfies E_SEARCH_MAX * M_P_DESIGN ==
     LEU_CAP_WTPC to within floating-point tolerance.

  2. BOX. Both enrichment design variables carry the derived upper bound,
     and a large Latin Hypercube draw from the box never zones above the
     LEU (Low Enriched Uranium) cap.

  3. NO-OP AT m_P = 1.0. For every archived design, the NEW g_enr formula
     is recomputed and compared against the g_enr stored in the checkpoint.
     At M_P_DESIGN = 1.0 the two must agree to within 1e-12. This is the
     check that licenses applying the patch to a live tree.

  4. IMPACT AT THE CONFIGURED m_P. When M_P_DESIGN is above 1.0 the script
     reports, without asserting anything, how many archived designs now
     violate the zoned LEU audit and how many fall outside the new box.
     Those are consequences to understand, not failures.

The exit status is 0 when every applicable check passes and 1 otherwise, so
this can gate a launch inside an && chain.

USAGE
    python3 verify_leu_box.py --checkpoint out_c5/optimization_checkpoint.json

FLAGS
    --checkpoint PATH   a campaign checkpoint to replay the audit against
    --csv PATH          use a c5_full.csv style table instead of a checkpoint
    --n-lhs N           Latin Hypercube points drawn for the box check
    --tol F             tolerance of the no-op comparison (default 1e-12)
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import sys
from pathlib import Path

import numpy as np

FAIL = []
WARN = []


def ok(label, passed, detail=""):
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
    if not passed:
        FAIL.append(label)
    return passed


def load_designs(checkpoint, csv_path):
    """Return a list of dicts with enrich_inner, enrich_outer and g_enr.

    The checkpoint is opened read-only. Nothing in this script writes to it.
    """
    if csv_path:
        rows = list(csvmod.DictReader(open(csv_path)))
        return [{"enrich_inner": float(r["enrich_inner"]),
                 "enrich_outer": float(r["enrich_outer"]),
                 "g_enr": float(r["g_enr"])} for r in rows], str(csv_path)
    ck = json.loads(Path(checkpoint).read_text())
    out = []
    for r in ck["all_raw"]:
        if "g_enr" not in r:
            continue
        out.append({"enrich_inner": float(r["enrich_inner"]),
                    "enrich_outer": float(r["enrich_outer"]),
                    "g_enr": float(r["g_enr"])})
    return out, str(checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="campaign checkpoint, opened read-only")
    ap.add_argument("--csv", default=None,
                    help="c5_full.csv style table, used instead of a "
                         "checkpoint")
    ap.add_argument("--n-lhs", type=int, default=2000,
                    help="Latin Hypercube points drawn for the box check")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="tolerance of the no-op comparison")
    args = ap.parse_args()

    if not args.checkpoint and not args.csv:
        raise SystemExit("give --checkpoint PATH or --csv PATH")

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # ---- check 1: imports and the derived cap ---------------------------
    print("\n1. imports and derived cap")
    try:
        import leu_policy as leu
    except ImportError as exc:
        print(f"  [FAIL] leu_policy did not import: {exc}")
        print("\nThe patch has not been applied. Run apply_leu_box.py first.")
        sys.exit(1)
    print(f"        LEU_CAP_WTPC = {leu.LEU_CAP_WTPC}")
    print(f"        M_P_DESIGN   = {leu.M_P_DESIGN}")
    print(f"        E_SEARCH_MAX = {leu.E_SEARCH_MAX:.6f} wt%")
    ok("E_SEARCH_MAX * M_P_DESIGN equals the LEU cap",
       abs(leu.E_SEARCH_MAX * leu.M_P_DESIGN - leu.LEU_CAP_WTPC) < 1e-9)

    from reactor_optimization import example_reactor_problem
    spec = example_reactor_problem()
    ok("reactor_optimization imports and builds the problem", True,
       f"{spec.design_space.n} variables, {spec.n_constr} constraints")
    ok("g_enr is still in the constraint list",
       "g_enr" in spec.constraint_names,
       "checkpoint compatibility depends on this")

    # ---- check 2: the search box ----------------------------------------
    print("\n2. search box")
    xu = spec.design_space.xu
    names = spec.design_space.names
    i_in, i_out = names.index("enrich_inner"), names.index("enrich_outer")
    ok("enrich_inner upper bound is E_SEARCH_MAX",
       abs(xu[i_in] - leu.E_SEARCH_MAX) < 1e-9, f"{xu[i_in]:.6f} wt%")
    ok("enrich_outer upper bound is E_SEARCH_MAX",
       abs(xu[i_out] - leu.E_SEARCH_MAX) < 1e-9, f"{xu[i_out]:.6f} wt%")

    X = spec.design_space.lhs(args.n_lhs, seed=12345,
                              accept=spec.exact_ok if spec.exact_constraints
                              else None)
    e_design = X[:, [i_in, i_out]].max()
    e_zoned = e_design * leu.M_P_DESIGN
    ok(f"{args.n_lhs} sampled designs stay under the zoned LEU cap",
       e_zoned <= leu.LEU_CAP_WTPC + 1e-9,
       f"highest design {e_design:.4f} wt%, zoned {e_zoned:.4f} wt%")

    # ---- check 3: no-op at m_P = 1.0 ------------------------------------
    print("\n3. archive replay")
    designs, source = load_designs(args.checkpoint, args.csv)
    print(f"        {len(designs)} archived evaluations from {source}")
    if not designs:
        ok("archive contains g_enr values", False,
           "no usable rows found")
    else:
        old = np.array([d["g_enr"] for d in designs])
        new = np.array([leu.max_zoned_enrichment_wtpc(d["enrich_inner"],
                                                      d["enrich_outer"])
                        - leu.LEU_CAP_WTPC for d in designs])
        dmax = float(np.max(np.abs(new - old)))
        if leu.M_P_DESIGN == 1.0:
            ok("new g_enr reproduces the archived g_enr exactly",
               dmax <= args.tol,
               f"largest absolute difference {dmax:.3e}, tolerance "
               f"{args.tol:.0e}")
            print("        m_P is 1.0, so the patch is a proven no-op. "
                  "Applying it cannot change any result.")
        else:
            print(f"  [INFO] m_P is {leu.M_P_DESIGN}, so a difference is "
                  f"expected and is not a failure.")
            print(f"        mean shift in g_enr  {float(np.mean(new - old)):+.4f}")
            print(f"        largest shift        {dmax:.4f}")

        # ---- check 4: impact at the configured m_P ----------------------
        print("\n4. impact on the existing archive")
        e_max = np.array([max(d["enrich_inner"], d["enrich_outer"])
                          for d in designs])
        n_viol_old = int(np.sum(old > 0))
        n_viol_new = int(np.sum(new > 0))
        n_outside = int(np.sum(e_max > leu.E_SEARCH_MAX + 1e-9))
        print(f"        violated the OLD g_enr audit : {n_viol_old} of "
              f"{len(designs)}")
        print(f"        violate the NEW g_enr audit  : {n_viol_new} of "
              f"{len(designs)}")
        print(f"        outside the NEW search box   : {n_outside} of "
              f"{len(designs)}")
        if leu.M_P_DESIGN == 1.0:
            ok("no archived design falls outside the box at m_P = 1.0",
               n_outside == 0)
        else:
            print("        Those designs stay in the archive as surrogate "
                  "training data. NSGA-II can no longer propose them, so the "
                  "training domain and the search domain differ. Report it.")
            if n_outside:
                WARN.append(f"{n_outside} archived designs outside the new box")

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 70)
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print("  " + f)
        print("\nRun: python3 apply_leu_box.py --revert")
        print("=" * 70)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    for w in WARN:
        print(f"  note: {w}")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
