#!/usr/bin/env python3
"""
test_core3d_hook.py -- the bookkeeping of the Campaign 10 hook (the cycle-length
constraint read from the eight-layer core depletion), without OpenMC.

Loads only core3d_cycle_fields and _core3d_selftest from openmc_evaluator.py,
because that module imports openmc at the top. Seconds to run.

    python test_core3d_hook.py
"""
import ast
import sys

src = open("openmc_evaluator.py", encoding="utf-8").read()
ns = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("core3d_cycle_fields", "_core3d_selftest"):
        exec(compile(ast.Module([node], []), "openmc_evaluator.py", "exec"), ns)
assert "core3d_cycle_fields" in ns and "_core3d_selftest" in ns, "hook functions not found"
ns["_core3d_selftest"]()
# the driver must accept the flag and refuse it together with the fitted ratio
drv = open("run_optimization.py", encoding="utf-8").read()
assert "--cycle-core3d" in drv and "cycle_core3d_meta" in drv and "ev.c9_cycle_core3d" in drv
assert drv.count("cycle_core3d_meta = None") == 1, "the meta default must be set once, outside the c9 block"
print("driver wiring OK")
sys.exit(0)
