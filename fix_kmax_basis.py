#!/usr/bin/env python
"""
fix_kmax_basis.py -- put g_kmax on the right quantity.

THE PROBLEM
-----------
openmc_evaluator.py currently writes

    "g_kmin":  1.02 - k_bol,
    "g_kmax":  k_bol - 1.35,

where k_bol is the ASSEMBLY k_inf from the reflective-boundary depletion
model. Excess reactivity and shutdown margin are core-level budgets, so the
cap belongs on the core k_eff, which the evaluator already computes and
stores as keff_core_bol at no additional cost.

This is the same category error that Campaign 4 fixed for peaking, when
F_dh and g_peak moved from the assembly to the core. g_kmax was left behind.

The gap is not small. Measured on c4_full.csv, k_inf minus k_eff,core runs
from 5012 to 7530 pcm with a mean of 6753 pcm, so 1.35 on the assembly is a
very different constraint from 1.35 on the core.

WHAT THIS PATCH DOES
--------------------
1. The four hard-coded limits (1.02, 1.35, 19.75, 2.0) become constructor
   arguments, so the thesis can quote them from one place instead of from a
   comment inside a dict literal.
2. A new argument k_basis selects the quantity g_kmin and g_kmax act on:
       "assembly"  k_inf         (default, unchanged behaviour)
       "core"      keff_core_bol
3. BOTH readings are recorded on every evaluation, always, as the diagnostics
   g_kmax_asm, g_kmax_core, g_kmin_asm, g_kmin_core and k_basis. Any existing
   or future archive can therefore be re-scored either way without rerunning
   anything.
4. Selecting k_basis="core" REQUIRES an explicit k_max. Reusing 1.35 on a
   quantity that sits roughly 6800 pcm lower would silently tighten the
   constraint by that amount, so the script refuses to guess.

The default is unchanged on purpose. A basis change alters the meaning of a
stored g_kmax, so it must not happen silently, and a campaign started under
one basis must not be resumed under the other.

USAGE
-----
    python fix_kmax_basis.py --dry-run
    python fix_kmax_basis.py
    python fix_kmax_basis.py --verify

Idempotent. Backs up to openmc_evaluator.py.kmax_basis.bak.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TARGET = "openmc_evaluator.py"
MARKER = "k_basis"

# --------------------------------------------------------------------------
# Edit 1: constructor gains the limits and the basis switch
# --------------------------------------------------------------------------
OLD_SIG = """                 core_particles: int = 100000,
                 core_batches: int = 170,
                 core_inactive: int = 60,
                 verbose: bool = True):
        super().__init__(spec)"""

NEW_SIG = """                 core_particles: int = 100000,
                 core_batches: int = 170,
                 core_inactive: int = 60,
                 k_basis: str = "assembly",
                 k_max: float | None = None,
                 k_min: float = 1.02,
                 f_max: float = 2.0,
                 enr_max: float = 19.75,
                 verbose: bool = True):
        super().__init__(spec)"""

# --------------------------------------------------------------------------
# Edit 2: validate and store them
# --------------------------------------------------------------------------
OLD_STORE = """        self.core_particles = int(core_particles)
        self.core_batches = int(core_batches)
        self.core_inactive = int(core_inactive)

        self.verbose = verbose"""

NEW_STORE = '''        self.core_particles = int(core_particles)
        self.core_batches = int(core_batches)
        self.core_inactive = int(core_inactive)

        # --- reactivity basis and screening limits --------------------------
        # g_kmin and g_kmax act on ONE of two quantities:
        #   "assembly"  k_inf from the reflective-boundary depletion model
        #   "core"      k_eff of the 2-D core at beginning of life
        # Excess reactivity is a core budget, so "core" is the physically
        # correct basis. "assembly" is the historical default and is the
        # conservative of the two, because k_inf exceeds k_eff always.
        # Both readings are recorded on every evaluation regardless, so an
        # archive can be re-scored either way without rerunning transport.
        if k_basis not in ("assembly", "core"):
            raise ValueError(
                f"k_basis must be 'assembly' or 'core', got {k_basis!r}")
        self.k_basis = k_basis
        if k_max is None:
            if k_basis == "core":
                raise ValueError(
                    "k_basis='core' requires an explicit k_max. The assembly "
                    "value of 1.35 must NOT be carried over: measured on "
                    "c4_full.csv the assembly-to-core gap is 5012 to 7530 pcm "
                    "(mean 6753 pcm), so reusing it would tighten the "
                    "constraint by roughly that amount without saying so. "
                    "Calibrate the core-level budget from the rod-worth study "
                    "and pass it here.")
            k_max = 1.35                      # historical assembly default
        self.k_max = float(k_max)
        self.k_min = float(k_min)
        self.f_max = float(f_max)
        self.enr_max = float(enr_max)
        if self.k_min >= self.k_max:
            raise ValueError(f"k_min ({self.k_min}) must be below k_max "
                             f"({self.k_max})")

        self.verbose = verbose'''

# --------------------------------------------------------------------------
# Edit 3: the constraint block itself
# --------------------------------------------------------------------------
OLD_CONS = '''            "g_kmin":  1.02 - k_bol,                    # need k_bol >= 1.02
            "g_kmax":  k_bol - 1.35,                    # and  k_bol <= 1.35
            "g_enr":   max(e_in, e_out) - 19.75,        # LEU cap
            "g_peak":  core["fdh_core"] - 2.0,          # CORE peaking <= 2.0'''

NEW_CONS = '''            # Reactivity screen. k_ref is the quantity selected by k_basis;
            # both readings follow as diagnostics so the archive can be
            # re-scored either way without rerunning transport.
            "g_kmin":  self.k_min - k_ref,
            "g_kmax":  k_ref - self.k_max,
            "g_enr":   max(e_in, e_out) - self.enr_max,   # LEU cap
            "g_peak":  core["fdh_core"] - self.f_max,     # CORE peaking'''

# --------------------------------------------------------------------------
# Edit 4: resolve k_ref before the dict, and record both readings after it
# --------------------------------------------------------------------------
OLD_PRE = """        e_in = design["enrich_inner"]
        e_out = design["enrich_outer"]
        res = {"""

NEW_PRE = """        e_in = design["enrich_inner"]
        e_out = design["enrich_outer"]
        # the quantity the reactivity screen acts on, see k_basis
        k_core_bol = float(core["keff_core"])
        k_ref = k_bol if self.k_basis == "assembly" else k_core_bol
        res = {"""

OLD_DIAG = '''            "keff_core_bol": core["keff_core"],   # free Route-B closure check'''

NEW_DIAG = '''            "keff_core_bol": k_core_bol,          # free Route-B closure check
            # both readings of the reactivity screen, always recorded
            "k_basis":      self.k_basis,
            "k_max_used":   self.k_max,
            "g_kmax_asm":   k_bol - self.k_max,
            "g_kmax_core":  k_core_bol - self.k_max,
            "g_kmin_asm":   self.k_min - k_bol,
            "g_kmin_core":  self.k_min - k_core_bol,
            "dk_asm_core_pcm": 1.0e5 * (k_bol - k_core_bol),'''

EDITS = [("constructor signature", OLD_SIG, NEW_SIG),
         ("limit validation", OLD_STORE, NEW_STORE),
         ("k_ref resolution", OLD_PRE, NEW_PRE),
         ("constraint block", OLD_CONS, NEW_CONS),
         ("diagnostics", OLD_DIAG, NEW_DIAG)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=TARGET)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
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
            sys.exit(f"anchor for '{name}' matched {n} times, expected 1. "
                     f"The file has drifted from the version this patch was "
                     f"written against. Patch by hand rather than forcing it.")
        print(f"  {name:24s} anchor found")
        out = out.replace(old, new)

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    bak = p.with_suffix(p.suffix + ".kmax_basis.bak")
    shutil.copy2(p, bak)
    p.write_text(out)
    print(f"\npatched {p}")
    print(f"backup  {bak}")
    print("\nDefault behaviour is UNCHANGED (k_basis='assembly', k_max=1.35).")
    print("To run on the core basis, pass k_basis='core' AND an explicit")
    print("k_max in run_optimization.py. Do not resume a campaign across a")
    print("basis change: the stored g_kmax would mix two different meanings.")


if __name__ == "__main__":
    main()
