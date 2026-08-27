#!/usr/bin/env python3
"""Add a three-way status column to salvage_k_histories.py.

The single censored flag conflates two different outcomes. A design stopped
at the burnup cap while still above its target has an unknown cycle length
that is at least the burnup reached. A design whose reactivity peak never
rose above its target cannot be made critical at all and has no cycle. The
evaluator scores the second at zero, so the two must not share a label."""
import py_compile, shutil
from pathlib import Path

P = Path("salvage_k_histories.py"); s = P.read_text()
if "never_critical" in s:
    raise SystemExit("already patched.")

A = '''    ap.add_argument("--q-spec", type=float, default=9.9827)'''
B = '''    ap.add_argument("--q-spec", type=float, default=9.9827)
    ap.add_argument("--cap-label", type=float, default=75.0,
                    help="burnup ceiling in LABEL units, used to tell a run "
                         "stopped at the cap from one that stopped because "
                         "its peak never reached the target")'''
assert s.count(A) == 1
s = s.replace(A, B)

A2 = '''        eoc = crossing(bu_t, kk, kt) if kt is not None else None
        censored = eoc is None'''
B2 = '''        eoc = crossing(bu_t, kk, kt) if kt is not None else None
        at_cap = rows[-1][0] >= args.cap_label - 1e-6
        if eoc is not None:
            status = "crossed"
        elif at_cap:
            status = "censored_at_cap"
        else:
            status = "never_critical"
        censored = status == "censored_at_cap"'''
assert s.count(A2) == 1
s = s.replace(A2, B2)

A3 = '''            censored=censored,'''
B3 = '''            status=status,
            censored=censored,'''
assert s.count(A3) == 1
s = s.replace(A3, B3)

A4 = '''        tail = (f"censored at true {rec['bu_true_final']}"
                if censored else
                f"EOC true {rec['eoc_true_bu']} MWd/kg = "
                f"{rec['eoc_true_efpd']} EFPD")'''
B4 = '''        tail = {"crossed": f"EOC true {rec['eoc_true_bu']} MWd/kg = "
                           f"{rec['eoc_true_efpd']} EFPD",
                "censored_at_cap": f"censored at true "
                                   f"{rec['bu_true_final']}",
                "never_critical": "never reached its target, no cycle",
                }[status]'''
assert s.count(A4) == 1
s = s.replace(A4, B4)

shutil.copy(P, "salvage_k_histories.py.status.bak")
P.write_text(s); py_compile.compile(str(P), doraise=True)
print("patched salvage_k_histories.py (backup .status.bak)")
