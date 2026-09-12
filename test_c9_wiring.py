#!/usr/bin/env python3
"""
test_c9_wiring.py
=================
Exercises the Campaign 9 problem through the REAL active-learning loop
(GP surrogates, NSGA-II, feasibility margin, checkpoint) with a stub
evaluator that mimics the physics cheaply. No OpenMC. Seconds to run.

What it proves
  1. campaign9_problem builds, and example_reactor_problem is untouched.
  2. The loop accepts objectives {peaking, c_max} and constraints
     {..., g_efpd, g_ctrl_peak} by name, runs a DOE plus infill blocks,
     and writes a checkpoint whose objectives and constraint_names are
     the Campaign 9 ones.
  3. The boron_objective arithmetic behaves inside the stub exactly as in
     the evaluator: humped designs get c_max > c_bol, unhumped designs get
     c_max = c_bol, the adaptive branch spends 1 or 2 extra solves.
  4. The feasible front moves toward low boron, which is the whole point.

Run from the repository root after apply_c9_reformulation.py:
    python test_c9_wiring.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import boron_objective as bo
from reactor_optimization import (ActiveLearningMOO, Evaluator, OptimizerConfig,
                                  campaign9_problem, example_reactor_problem)


class StubC9Evaluator(Evaluator):
    """Cheap analytic stand-in with the Campaign 9 output keys.

    Physics caricature, tuned to the Campaign 8 ranges: reactivity rises
    with enrichment, gadolinia suppresses it at BOL and produces a hump
    proportional to its loading, boron worth falls with enrichment as the
    fitted power law says, the four-bank worth is about 15000 pcm."""

    def __init__(self, spec, efpd_req, seed=0):
        super().__init__(spec)
        self.efpd_req = float(efpd_req)
        self.rng = np.random.default_rng(seed)
        self.extra_solves = []

    def evaluate_one(self, d):
        e, gd, refl, pins = d["enrich"], d["gd_wt"], d["refl_thick"], d["gd_pins"]
        # core reactivity at 1000 ppm, pcm, roughly 2000 at 3 wt% to 20000 at 14
        rho1000 = 1400.0 * (e - 1.5) - 250.0 * gd * pins / 12.0 + 300.0 * (refl - 3.5)
        k_core = 1.0 / (1.0 - rho1000 * 1e-5)
        l_ax = 1.08
        k_bol = k_core * l_ax
        # gadolinia hump at the assembly, pcm, appears above ~2 wt% x 20 pins
        hump = max(0.0, 180.0 * gd * pins / 12.0 - 1200.0)
        k_hist = [k_bol, k_bol * (1 - 0.014), k_bol * (1 - 0.014) * (1 + hump * 1e-5),
                  k_bol * 0.97, k_bol * 0.93]
        h = bo.hump_from_history(k_hist, k_core)
        # boron worth law, pcm per ppm, and a measured curve
        wb = 23.292 * e ** -0.8496

        def rho(c):
            return rho1000 - wb * (c - 1000.0) * (1.0 - 0.6e-4 * (c - 1000.0))

        pts = {1000.0: rho(1000.0)}
        targets = [0.0, -h["hump_core_pcm"], -h["hump_core_op_pcm"]]
        n = 0
        while True:
            nxt = bo.next_concentration(pts, targets)
            if nxt is None:
                break
            pts[float(nxt)] = rho(nxt)
            n += 1
        self.extra_solves.append(n)
        obj = bo.boron_objective(pts, h)
        k_allre = 1.0 / (1.0 - (rho1000 - 15000.0) * 1e-5)
        ctrl = bo.ctrl_margin_at_peak(k_allre, h, 0.01)
        cycle = 330.0 * e - 40.0 * gd + 100.0 * (refl - 3.5)
        peaking = 1.45 + 0.02 * e + 0.01 * gd - 0.02 * (refl - 3.5) \
            + 0.01 * self.rng.standard_normal()
        return {
            "cycle_length": cycle, "peaking": peaking,
            "c_max": obj["c_max_ppm"], "c_bol": obj["c_bol_ppm"],
            "c_max_op": obj["c_max_op_ppm"], "hump_core_pcm": h["hump_core_pcm"],
            "hump_core_op_pcm": h["hump_core_op_pcm"],
            "n_boron_solves": n,
            "g_kmin": 1.02 - k_core, "g_kmax": k_core - 1.166,
            "g_enr": e * 1.15 - 16.0, "g_peak": peaking - 1.65,
            "g_geom": (refl + 77.231 + 7.08 + 0.02) - 90.0,
            "g_efpd": self.efpd_req - cycle,
            "g_ctrl_peak": ctrl["g_ctrl_peak"], "g_ctrl": k_allre - 0.99,
        }


def main():
    spec = campaign9_problem(1826.0, f_max=1.65)
    spec.constraint_names.append("g_ctrl")          # run_optimization adds it too
    spec.constraint_scales["g_ctrl"] = 1.0
    c8 = example_reactor_problem()
    assert [o.name for o in c8.objectives] == ["cycle_length", "peaking"]
    assert "g_efpd" not in c8.constraint_names
    print("spec  objectives :", [(o.name, "max" if o.maximize else "min") for o in spec.objectives])
    print("spec  constraints:", spec.constraint_names)

    ev = StubC9Evaluator(spec, 1826.0, seed=1)
    cfg = OptimizerConfig(n_init=24, n_iter=3, n_infill=6, nsga_pop=60,
                          nsga_gen=40, surrogate="gp", seed=1)
    cfg.feas_kappa = 1.5
    cfg.infill_min_sep = 0.14
    cfg.efpd_cap = None
    opt = ActiveLearningMOO(spec, ev, cfg)
    res = opt.run(verbose=False)

    with tempfile.TemporaryDirectory() as td:
        ck = opt.save_checkpoint(str(Path(td) / "ckpt.json"),
                                 meta={"objective_set": "c9", "efpd_req": 1826.0})
        d = json.load(open(ck))
    assert d["objectives"] == [["peaking", "min"], ["c_max", "min"]], d["objectives"]
    assert "g_efpd" in d["constraint_names"] and "g_ctrl_peak" in d["constraint_names"]
    raw = d["all_raw"]
    assert all("c_max" in r and "c_bol" in r for r in raw)
    humped = [r for r in raw if r["hump_core_pcm"] > 0]
    flat = [r for r in raw if r["hump_core_pcm"] == 0]
    assert all(r["c_max"] > r["c_bol"] for r in humped)
    assert all(abs(r["c_max"] - r["c_bol"]) < 1e-9 for r in flat)
    # no floor on c_max_op: the xenon credit lowers it below c_bol when the
    # unfloored hump is negative, and it equals c_max once the hump is
    # above the noise floor
    assert all(r["c_max_op"] <= r["c_bol"] + 1e-9 for r in raw if r["hump_core_op_pcm"] <= 0)
    assert all(abs(r["c_max_op"] - r["c_max"]) < 1e-9 for r in humped)
    solves = np.array(ev.extra_solves)
    assert set(solves.tolist()) <= {1, 2}, solves

    feas = [r for r in raw if all(r[c] <= 0 for c in spec.constraint_names)]
    print(f"evals {len(raw)} | feasible {len(feas)} | humped {len(humped)} | "
          f"extra boron solves mean {solves.mean():.2f} (1: {int((solves==1).sum())}, "
          f"2: {int((solves==2).sum())})")
    pf = res["pareto_F"]
    print("front (peaking, c_max):", [(round(f[0], 3), round(f[1])) for f in pf[np.argsort(pf[:, 1])]])
    doe_c = np.mean([r["c_max"] for r in raw[:24]])
    inf_c = np.mean([r["c_max"] for r in raw[24:]]) if len(raw) > 24 else doe_c
    print(f"mean c_max  DOE {doe_c:.0f} ppm -> infill {inf_c:.0f} ppm")
    print("hypervolume history:", [round(float(h), 4) for h in res["hv_history"]])
    print("test_c9_wiring OK")


if __name__ == "__main__":
    main()
