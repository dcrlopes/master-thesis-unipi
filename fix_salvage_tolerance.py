#!/usr/bin/env python3
"""Widen the burnup-cap tolerance in salvage_k_histories.py.

The label is reconstructed through a burnup-to-days-and-back round trip, so
a run that truly reached the ceiling returns a value marginally below it.
A tolerance of 1e-6 rejected those, and every ceiling-stopped design was
misclassified as one whose reactivity peak never reached the target. Half a
step is the right scale for this comparison."""
import py_compile, shutil
from pathlib import Path

P = Path("salvage_k_histories.py"); s = P.read_text()
A = "        at_cap = rows[-1][0] >= args.cap_label - 1e-6"
B = ("        # round-trip through days loses the last digits, so compare\n"
     "        # at the scale of a depletion step, not at machine precision\n"
     "        at_cap = rows[-1][0] >= args.cap_label - 0.5")
if B in s:
    raise SystemExit("already patched.")
assert s.count(A) == 1, "anchor not unique, do not force"
shutil.copy(P, "salvage_k_histories.py.tol.bak")
P.write_text(s.replace(A, B))
py_compile.compile(str(P), doraise=True)
print("patched salvage_k_histories.py (backup .tol.bak)")
