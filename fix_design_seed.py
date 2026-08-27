#!/usr/bin/env python3
"""Make _design_seed tolerant of non-numeric bookkeeping keys.

zoning.zone_designs labels each ring variant with a string key "zone", which
the model builders already ignore by filtering on numeric type. The seed
helper did not filter and called float() on every value, so a ring depletion
raised ValueError before any transport ran.

Non-numeric values are now hashed as strings. Designs containing only
numeric values produce exactly the same JSON key as before, so every seed
used by Campaigns 1 to 5 is unchanged."""
import py_compile, shutil
from pathlib import Path

P = Path("openmc_evaluator.py"); s = P.read_text()
if "_seed_value" in s:
    raise SystemExit("already patched.")

A = '''    key = _json.dumps({k: round(float(v), 10)
                       for k, v in sorted(design.items())}) + salt'''
B = '''    def _seed_value(v):
        # Numeric values keep their previous representation exactly, so the
        # seeds of every all-numeric campaign design are unchanged. Labels
        # such as zoning's "zone" key are hashed as strings instead of
        # raising, which also gives each ring variant its own stream.
        try:
            return round(float(v), 10)
        except (TypeError, ValueError):
            return str(v)

    key = _json.dumps({k: _seed_value(v)
                       for k, v in sorted(design.items())}) + salt'''
assert s.count(A) == 1, "seed anchor not unique, do not force"
shutil.copy(P, "openmc_evaluator.py.seed.bak")
P.write_text(s.replace(A, B))
py_compile.compile(str(P), doraise=True)
print("patched openmc_evaluator.py (backup .seed.bak)")
