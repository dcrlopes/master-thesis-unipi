#!/usr/bin/env python3
"""Force reaction rates into the depletion results files so chunked
restarts are exact, and refuse loudly if they are ever missing.

OpenMC 0.15.3 defaults write_rates to False. A restart loaded from such a
file reuses EMPTY rates for its first step, which then depletes by decay
only. Verified 27 Aug 2026 by test_chunking.py: the chunked arm lagged the
single-chunk arm by one full step per restart, with xenon-decay spikes on
the dead points. Patches openmc_evaluator.py and test_chunking.py."""
import py_compile, shutil
from pathlib import Path

GUARD = '''                last_rates = getattr(results[-1], "rates", None)
                import numpy as _np
                if last_rates is None or getattr(last_rates, "size", 0) == 0 \\
                        or float(_np.abs(_np.asarray(last_rates)).sum()) == 0.0:
                    raise RuntimeError(
                        "depletion results carry no reaction rates: a "
                        "restarted chunk would deplete its first step by "
                        "decay only (the write_rates trap).")
'''
ev = Path("openmc_evaluator.py"); s = ev.read_text()
if "write_rates=True" in s:
    raise SystemExit("openmc_evaluator.py already patched.")
A = """            try:
                os.chdir(cdir)
                integrator.integrate()
                results = openmc.deplete.Results("depletion_results.h5")
"""
B = """            try:
                os.chdir(cdir)
                # OpenMC 0.15.3 stopped writing reaction rates into the
                # results file by default. A restart loaded from such a file
                # reuses EMPTY rates for its first step, which then depletes
                # by decay only (verified 27 Aug 2026, test_chunking.py).
                # Force the rates into the file so prev_results is exact.
                try:
                    integrator.integrate(write_rates=True)
                except TypeError:      # older OpenMC without the kwarg
                    integrator.integrate()
                results = openmc.deplete.Results("depletion_results.h5")
""" + GUARD
assert s.count(A) == 1, "evaluator anchor not unique, do not force"
shutil.copy(ev, "openmc_evaluator.py.rates.bak")
ev.write_text(s.replace(A, B)); py_compile.compile(str(ev), doraise=True)
print("patched openmc_evaluator.py (backup .rates.bak)")

tc = Path("test_chunking.py"); s = tc.read_text()
if "write_rates=True" not in s:
    A2 = """            os.chdir(cdir)
            integ.integrate()
"""
    B2 = """            os.chdir(cdir)
            try:
                integ.integrate(write_rates=True)
            except TypeError:
                integ.integrate()
"""
    assert s.count(A2) == 1, "test anchor not unique"
    shutil.copy(tc, "test_chunking.py.rates.bak")
    tc.write_text(s.replace(A2, B2)); py_compile.compile(str(tc), doraise=True)
    print("patched test_chunking.py")
