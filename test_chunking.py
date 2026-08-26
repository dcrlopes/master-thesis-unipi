#!/usr/bin/env python3
"""Is the two-branch k structure caused by the chunked restarts?

Depletes idx 7 through the evaluator's OWN model preparation, once as a
single uninterrupted chunk of six 4 MWd/kgHM steps and once as three chunks
of two steps covering the identical burnup points. Model construction,
volumes, depletable flags, seed, pin tallies, operator options and the
power normalisation are copied from OpenMCEvaluator._cycle_length line for
line, so the only difference between the two arms is the restart split.

Reading the verdict:
  arm A and arm B share the same seed and the same model, so their points
  at 4 and 8 MWd/kg are the same computation and must agree to within the
  parallel-reduction jitter of a few pcm. Divergence begins only after
  arm B's first restart. If arm B then alternates around arm A by the
  roughly 2000 pcm seen in the campaign histories, the restart chunking is
  the cause. If both arms alternate together, the cause is in the
  integrator or the power normalisation instead.

Cost: about 9 transport solves per arm at the campaign transport settings.
Run detached:
  setsid nohup python -u test_chunking.py > chunk_test.log 2>&1 < /dev/null &
"""
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "64")

import numpy as np
import openmc
import openmc.deplete

import reactor_model as rm
from reactor_optimization import example_reactor_problem
from openmc_evaluator import OpenMCEvaluator, _design_seed

ck = json.loads(Path("out_c5/optimization_checkpoint.json").read_text())
dv = ck["design_variables"]
design = {k: float(ck["all_raw"][7][k]) for k in dv}
transport = dict(ck.get("meta", {}).get("transport",
                 dict(particles=16000, batches=120, inactive=30)))
ev = OpenMCEvaluator(example_reactor_problem(), k_target=1.05,
                     transport=transport, workdir="chunk_test")
print(f"idx 7, transport {transport}, q_spec {ev.spec_power:.3f} W/gHM",
      flush=True)


def build():
    """Model prepared exactly as _cycle_length prepares it."""
    model, fuel_cells, _lat = rm.make_assembly_model(
        design, ev.op, ev.geo, bc="reflective", pin_tally=True,
        **ev.transport)
    model.settings.seed = _design_seed(design)
    pin_vol = math.pi * ev.geo.fuel_or ** 2 * ev.geo.active_height
    counts = Counter(c.fill.id for c in fuel_cells)
    id2mat = {m.id: m for m in model.materials}
    for mat_id, npins in counts.items():
        m = id2mat[mat_id]
        m.volume = npins * pin_vol
        m.depletable = True
    return model


def run(tag, chunks_of_steps):
    """Deplete with the given chunk split, returning merged (burnup, k)."""
    root = Path("chunk_test") / tag
    if root.exists():
        shutil.rmtree(root)          # rerun safety: never append to old h5
    model = build()
    prev, power_w, out = None, None, []
    for ci, steps in enumerate(chunks_of_steps):
        opd = openmc.deplete.CoupledOperator(
            model, prev_results=prev, diff_burnable_mats=False)
        if power_w is None:
            # captured ONCE, exactly like state["power_w"] in the evaluator
            power_w = ev.spec_power * opd.heavy_metal
        days = [s * 1000.0 / ev.spec_power for s in steps]
        integ = openmc.deplete.PredictorIntegrator(
            opd, days, power=power_w, timestep_units="d")
        cdir = root / f"dep_{ci:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(cdir)
            integ.integrate()
            res = openmc.deplete.Results("depletion_results.h5")
        finally:
            os.chdir(cwd)
        prev = res
        t, k = res.get_keff(time_units="d")
        t = np.asarray(t, dtype=float)
        kv = np.asarray(k, dtype=float)[:, 0]
        real = kv > 0.0              # zero-filled entries are placeholders
        out.extend(zip(t[real] * ev.spec_power / 1000.0, kv[real]))
    seen, merged = set(), []
    for b, kk in sorted(out):
        r = round(b, 6)
        if r in seen:
            continue
        seen.add(r)
        merged.append((b, kk))
    return merged


S = 4.0
print("\n--- arm A: ONE chunk, six steps of 4 MWd/kgHM ---", flush=True)
a = run("single", [[S] * 6])
for b, k in a:
    print(f"  {b:6.1f} MWd/kg   k={k:.5f}", flush=True)

print("\n--- arm B: THREE chunks of two steps, same burnups ---", flush=True)
bb = run("chunked", [[S, S], [S, S], [S, S]])
for b, k in bb:
    print(f"  {b:6.1f} MWd/kg   k={k:.5f}", flush=True)

print("\n--- comparison at common burnups ---")
da = {round(x, 3): y for x, y in a}
for b, k in bb:
    r = round(b, 3)
    if r in da:
        print(f"  {b:6.1f}: single {da[r]:.5f}   chunked {k:.5f}   "
              f"diff {(k - da[r]) * 1e5:+.0f} pcm")
print("\npoints at 4 and 8 MWd/kg are the identity check (same seed, same "
      "model, no restart yet): expect a few pcm at most there.")
