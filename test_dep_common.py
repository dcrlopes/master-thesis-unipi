#!/usr/bin/env python3
"""
test_dep_common.py -- dep_common.run_adaptive must be the evaluator's own
_cycle_length, step for step. No OpenMC: openmc.deplete is replaced by a
stub that returns k from a synthetic reactivity curve, with the cumulative
restart semantics of OpenMC 0.15. Both routines are driven on the same
curve and every returned quantity is compared.

Three curves are used: a gadolinium hump crossing the target, a design
that never reaches the target (0 EFPD), and one censored at the cap.

    python test_dep_common.py
"""
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, ".")

# ------------------------------------------------------------- stub openmc --
SPEC = 9.9834          # W/gHM, the real core value
STATE = {"curve": None, "bu": [0.0]}


class _Res:
    rates = np.ones(3)


class FakeResults:
    def __init__(self, path=None):
        self.bu = list(STATE["bu"])

    def __getitem__(self, i):
        return _Res()

    def get_keff(self):
        k = np.array([[STATE["curve"](b), 2.0e-3] for b in self.bu])
        return np.array(self.bu) * 1000.0 / SPEC, k


class FakeOperator:
    def __init__(self, model, prev_results=None, diff_burnable_mats=False):
        self.heavy_metal = 12345.0


class FakeIntegrator:
    def __init__(self, op, days, power=None, timestep_units="d"):
        self.days = list(days)

    def integrate(self, write_rates=True):
        for d in self.days:
            STATE["bu"].append(STATE["bu"][-1] + d * SPEC / 1000.0)


dep = types.ModuleType("openmc.deplete")
dep.CoupledOperator, dep.PredictorIntegrator, dep.Results = FakeOperator, FakeIntegrator, FakeResults
omc = MagicMock(name="openmc"); omc.__version__ = "stub"
omc.deplete = dep
sys.modules["openmc"], sys.modules["openmc.deplete"] = omc, dep
import os
os.environ.setdefault("OPENMC_CHAIN_FILE", "/dev/null")

import dep_common                                          # noqa: E402
import openmc_evaluator as oe                              # noqa: E402
import reactor_model as rm                                 # noqa: E402
from reactor_optimization import campaign9_problem         # noqa: E402


class FakeMat:
    def __init__(self, i):
        self.id = i


class FakeCell:
    def __init__(self, m):
        self.fill = m


class FakeModel:
    def __init__(self):
        self.settings = types.SimpleNamespace(seed=None)
        self.materials = [FakeMat(1), FakeMat(2)]


def fake_assembly(design, op, geo, bc="reflective", pin_tally=False, **tr):
    m = FakeModel()
    cells = [FakeCell(m.materials[0])] * 240 + [FakeCell(m.materials[1])] * 24
    return m, cells, None


rm.make_assembly_model = fake_assembly


def curves():
    hump = lambda B: 1.03 + 0.04 * np.exp(-((B - 9.0) / 4.0) ** 2) - 0.0032 * B + (0.02 if B == 0 else 0)
    dud = lambda B: 0.98 - 0.002 * B + (0.02 if B == 0 else 0)
    slow = lambda B: 1.20 - 0.0005 * B + (0.02 if B == 0 else 0)
    late = lambda B: 1.05 + 0.04 * np.exp(-((B - 9.0) / 4.0) ** 2) - 0.0011 * B + (0.02 if B == 0 else 0)
    return dict(hump=(hump, 1.03), late=(late, 1.03), dud=(dud, 1.03), censored=(slow, 1.03))


def main():
    spec = campaign9_problem(1826.0)
    design = {"enrich_inner": 4.46, "enrich_outer": 4.46, "gd_wt": 4.42,
              "pitch": 1.26, "refl_thick": 5.66, "gd_pins": 20}
    with tempfile.TemporaryDirectory() as td:
        ev = oe.OpenMCEvaluator(spec, k_target=1.03, workdir=td, verbose=False,
                                k_basis="core", k_max=1.166, max_burnup=40.0)
        for name, (curve, kt) in curves().items():
            STATE["curve"], STATE["bu"] = curve, [0.0]
            efpd, k_bol, kt_used, cen, bu_eoc, n = ev._cycle_length(design, Path(td) / f"ev_{name}")
            ref = dict(efpd=efpd, bu_eoc=bu_eoc, censored=cen, n_solves=n,
                       k_hist=ev._last_k_hist, bu_hist=ev._last_bu_hist)
            STATE["curve"], STATE["bu"] = curve, [0.0]
            got = dep_common.run_adaptive(
                FakeModel(), kt, ev.spec_power, bol_steps=ev.bol_steps, dep_step=ev.dep_step,
                chunk_steps=ev.chunk_steps, max_burnup=ev.max_burnup,
                case=Path(td) / f"dc_{name}", verbose=False)
            for key in ref:
                assert got[key] == ref[key], (name, key, got[key], ref[key])
            print(f"{name:9s} EFPD {efpd:7.1f}  B_EOC {bu_eoc:5.2f}  censored {cen}  "
                  f"solves {n}  chunks {got['chunks']}  identical")
        assert curves()["dud"] and ref["efpd"] >= 0
    print("test_dep_common OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
