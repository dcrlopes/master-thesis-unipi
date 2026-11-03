#!/usr/bin/env python3
"""
test_c12_loop.py -- the two Campaign 12 changes inside the active-learning loop,
with the Campaign 9 stub evaluator of test_c9_wiring.py. No OpenMC, about a
minute.

  A  exploration pick   with explore_sep above any separation the front can
                        give, the last n_explore picks of the batch come from
                        the seeded space-filling sample: they satisfy the
                        analytic constraints, differ from the uncertainty
                        picks, the first picks are unchanged, and a second
                        propose() with the same batch number returns the same
                        batch (the relaunch replay of run_c10.sh depends on it)
  B  confirmation       with confirm_front, every design of a batch that enters
                        the feasible front is passed once to confirm_one; its
                        rows of F and G are rebuilt from the confirmed record;
                        a design the confirmation makes infeasible is reported
                        in confirm_dropped and is not on the front, and the
                        design it uncovers is confirmed in the same batch

    python test_c12_loop.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from reactor_optimization import ActiveLearningMOO, OptimizerConfig, campaign9_problem
from test_c9_wiring import StubC9Evaluator


class ConfirmingStub(StubC9Evaluator):
    """confirm_one shortens the cycle by 400 d, so a design near the minimum
    becomes infeasible, and counts its calls per design."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.confirm_calls = {}

    def confirm_one(self, design, rec):
        key = tuple(round(float(design[v]), 9) for v in ("enrich", "gd_wt", "refl_thick", "gd_pins"))
        self.confirm_calls[key] = self.confirm_calls.get(key, 0) + 1
        cycle = float(rec["cycle_length"]) - 400.0
        return {**rec, "cycle_length": cycle, "g_efpd": self.efpd_req - cycle,
                "cycle_length_proxy": rec["cycle_length"], "confirmed_3d": True}


def make(ev_cls, **kw):
    spec = campaign9_problem(1826.0, f_max=1.65)
    spec.constraint_names.append("g_ctrl")
    spec.constraint_scales["g_ctrl"] = 1.0
    cfg = OptimizerConfig(n_init=24, n_iter=kw.pop("n_iter", 2), n_infill=6, nsga_pop=40, nsga_gen=20,
                          surrogate="gp", seed=1)
    cfg.feas_kappa, cfg.infill_min_sep, cfg.efpd_cap = 1.5, 0.14, None
    for k, v in kw.items():
        assert hasattr(cfg, k), f"OptimizerConfig has no field {k}"
        setattr(cfg, k, v)
    return spec, ActiveLearningMOO(spec, ev_cls(spec, 1826.0, seed=1), cfg)


def main():
    # ---- A. exploration pick ------------------------------------------------
    spec, a = make(StubC9Evaluator, n_iter=0)
    a.run(verbose=False)                                   # the design of experiments only
    assert len(a.X) == 24
    a.cfg.explore_sep = 0.0
    X0, p0 = a.propose(0, verbose=False)
    assert p0["acq"]["n_explore"] == 0 and p0["acq"]["relax_depth"] >= 0
    a.cfg.explore_sep, a.cfg.n_explore = 1.0, 2            # 1.0 is above any separation: always triggers
    X1, p1 = a.propose(0, verbose=False)
    X2, _ = a.propose(0, verbose=False)
    assert p1["acq"]["n_explore"] == 2, p1["acq"]
    assert np.allclose(X1, X2), "the exploration pick must be reproducible for the same batch number"
    assert np.allclose(X0[:4], X1[:4]), "the uncertainty picks before the exploration picks must be unchanged"
    for x in X1[4:]:
        d = spec.design_space.as_dict(x)
        assert spec.exact_ok(d), "an exploration pick violates an analytic constraint"
        assert np.all(x >= spec.design_space.xl - 1e-12) and np.all(x <= spec.design_space.xu + 1e-12)
        assert not any(np.allclose(x, y) for y in X0), "an exploration pick repeats an uncertainty pick"
    print(f"A  exploration  : 2 of 6 picks replaced, reproducible, inside the box and the analytic constraints")

    # ---- B. confirmation in the loop ----------------------------------------
    spec, b = make(ConfirmingStub, n_iter=2, confirm_front=True)
    b.run(verbose=False)
    inf = [p for p in b.phase_log if p["stage"] == "infill"]
    assert len(inf) == 2 and all("n_confirmed" in p and "confirm_dropped" in p and "t_confirm_s" in p for p in inf)
    n_conf = sum(p["n_confirmed"] for p in inf)
    assert n_conf == sum(b.evaluator.confirm_calls.values()) and n_conf >= 1, (n_conf, b.evaluator.confirm_calls)
    assert all(v == 1 for v in b.evaluator.confirm_calls.values()), "a design was confirmed twice"
    for i, r in enumerate(b.raw):
        if r.get("confirmed_3d"):
            assert r["cycle_length"] == r["cycle_length_proxy"] - 400.0
            G_row = [float(r[c]) / spec.g_scale(c) for c in spec.constraint_names]
            assert np.allclose(b.G[i], G_row), "G row not rebuilt from the confirmed record"
            assert np.allclose(b.F[i], [o.to_min(r[o.name]) for o in spec.objectives])
    dropped = [i for p in inf for i in p["confirm_dropped"]]
    feas = b._feasible_mask()
    assert all(not feas[i] for i in dropped), "a dropped design is still feasible"
    # the whole feasible front is confirmed, the designs uncovered by a drop included
    idx_f = np.flatnonzero(feas)
    front = idx_f[b._nondominated(b.F[feas])]
    assert all(b.raw[i].get("confirmed_3d") for i in front), "an unconfirmed design is on the front"
    print(f"B  confirmation : {n_conf} design(s) confirmed over 2 batches, {len(dropped)} left the front, "
          f"rows rebuilt, front members all confirmed")
    print("test_c12_loop OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
