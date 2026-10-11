#!/usr/bin/env python3
"""
test_core3d_hook.py -- the bookkeeping of the Campaign 10 hook (the cycle-length
constraint read from the eight-layer core depletion), the evaluation cache and
the subprocess result slimming, without OpenMC.

Loads only the pure functions from openmc_evaluator.py, because that module
imports openmc at the top. Seconds to run.

    python test_core3d_hook.py
"""
import ast
import json
import sys
import tempfile
from pathlib import Path

src = open("openmc_evaluator.py", encoding="utf-8").read()
ns = {"json": json, "Path": Path}
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("core3d_cycle_fields", "_core3d_selftest"):
        exec(compile(ast.Module([node], []), "openmc_evaluator.py", "exec"), ns)
    if isinstance(node, ast.ClassDef) and node.name == "OpenMCEvaluator":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in ("eval_cache_key", "_eval_cache_get", "_eval_cache_put"):
                fn = item
                fn.decorator_list = []                       # drop @staticmethod for the free-function test
                exec(compile(ast.Module([fn], []), "openmc_evaluator.py", "exec"), ns)
            if isinstance(item, ast.Assign) and any(getattr(t, "id", "") == "_EVAL_CACHE" for t in item.targets):
                exec(compile(ast.Module([item], []), "openmc_evaluator.py", "exec"), ns)
assert "core3d_cycle_fields" in ns and "_core3d_selftest" in ns, "hook functions not found"
ns["_core3d_selftest"]()

# the evaluation cache: put, get, and a miss on a different design
class Stub:
    _EVAL_CACHE = ns["_EVAL_CACHE"]
    def __init__(self, wd): self.workdir = Path(wd)
    eval_cache_key = staticmethod(ns["eval_cache_key"])
    _eval_cache_get = ns["_eval_cache_get"]
    _eval_cache_put = ns["_eval_cache_put"]

with tempfile.TemporaryDirectory() as wd:
    ev = Stub(wd)
    d1 = {"enrich_inner": 4.5, "enrich_outer": 4.5, "gd_wt": 2.0, "pitch": 1.26, "refl_thick": 5.66, "gd_pins": 16.0}
    assert ev._eval_cache_get(d1) is None
    ev._eval_cache_put(d1, {"cycle_length": 1900.0, "peaking": 1.5, "k_hist_3d": [1.08, 1.0]})
    assert ev._eval_cache_get(d1)["cycle_length"] == 1900.0
    assert ev._eval_cache_get({**d1, "gd_wt": 2.0000001}) is None, "a different design must miss"
    assert ev._eval_cache_get({**d1, "gd_wt": 2.0 + 1e-12}) is not None, "rounding at 1e-9 must hit"
    assert len(json.loads((Path(wd) / Stub._EVAL_CACHE).read_text())) == 1
print("eval cache OK")

# the subprocess result slimming keeps what the hook reads and drops the bulk
sub = open("core3d_subprocess.py", encoding="utf-8").read()
ns2 = {"json": json}
for node in ast.parse(sub).body:
    if isinstance(node, ast.FunctionDef) and node.name == "slim":
        exec(compile(ast.Module([node], []), "core3d_subprocess.py", "exec"), ns2)
import numpy as np
dep = {"efpd": np.float64(1900.0), "k_hist": [np.float64(1.08)], "states": [{"fz": 1.44}], "materials": [1] * 96,
       "info": {"n_fuel_segments": 14}, "solve_times": [1, 2], "sigma_efpd": 12.0, "seed": np.int64(7)}
s = ns2["slim"](dep)
assert "materials" not in s and "info" not in s and "solve_times" not in s
assert s["efpd"] == 1900.0 and s["states"][0]["fz"] == 1.44 and s["seed"] == 7
out = ns["core3d_cycle_fields"]({"cycle_length": 2300.0}, s, 1826.0)
assert out["cycle_length"] == 1900.0 and out["fz_bol_3d"] == 1.44
print("subprocess slimming OK")

# the driver must accept the flags and refuse --cycle-core3d together with the fitted ratio
drv = open("run_optimization.py", encoding="utf-8").read()
for s_ in ("--cycle-core3d", "--cycle-max-burnup", "cycle_core3d_meta", "ev.c9_cycle_core3d", '"subprocess": True'):
    assert s_ in drv, s_
assert drv.count("cycle_core3d_meta = None") == 1, "the meta default must be set once, outside the c9 block"
assert "core3d_subprocess.py" in src and "_evaluate_one_transport" in src
print("driver wiring OK")
sys.exit(0)
