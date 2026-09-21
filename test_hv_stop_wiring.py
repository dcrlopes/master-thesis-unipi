#!/usr/bin/env python3
"""
test_hv_stop_wiring.py
======================
Drives the REAL active-learning loop (GP surrogates, NSGA-II, feasibility
margin, checkpoint) with the Campaign 9 stub evaluator of test_c9_wiring.py
through every branch of the stopping logic. No OpenMC. About two minutes.

The design of experiments is not seeded (DesignSpace.lhs ignores its seed
argument under pymoo 0.6.2), so every assertion below is written to hold for
any draw: it checks the MEANING of the stop, never a particular number.

  A  rule off        runs exactly n_iter iterations, stop by "iters", and the
                     health indicators are in phase_log
  B  rule on         fires at the FIRST iteration where the rule is met, not
                     earlier and not later, and never exceeds the budget
  C  budget binds    24 + 2 x 6 = 36 <= 40 < 42, stop by "cap" at 36
  D  censored front  a censored front member blocks the rule until the budget
  E  resume          the quiet count carries across a checkpoint and a resume
  F  checkpoint      the "stop" block and the indicators survive JSON

Run from the repository root after apply_hv_stop.py:
    python test_hv_stop_wiring.py
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import hv_stop as hvs
from reactor_optimization import ActiveLearningMOO, OptimizerConfig, campaign9_problem
from test_c9_wiring import StubC9Evaluator

HEALTH = ("n_candidates", "n_margin_feasible", "relax_depth", "n_infill_feasible", "hv_gain")


class CensoredStub(StubC9Evaluator):
    """Every design censored, so condition 2 of the rule can never hold."""
    def evaluate_one(self, d):
        r = super().evaluate_one(d); r["censored"] = True
        return r


class PlainStub(StubC9Evaluator):
    def evaluate_one(self, d):
        r = super().evaluate_one(d); r["censored"] = False
        return r


def make(ev_cls, **kw):
    spec = campaign9_problem(1826.0, f_max=1.65)
    spec.constraint_names.append("g_ctrl"); spec.constraint_scales["g_ctrl"] = 1.0
    cfg = OptimizerConfig(n_init=24, n_iter=kw.pop("n_iter", 3), n_infill=6,
                          nsga_pop=40, nsga_gen=20, surrogate="gp", seed=1)
    cfg.feas_kappa, cfg.infill_min_sep, cfg.efpd_cap = 1.5, 0.14, None
    for k, v in kw.items():
        assert hasattr(cfg, k), f"OptimizerConfig has no field {k}: is the patch applied?"
        setattr(cfg, k, v)
    return ActiveLearningMOO(spec, ev_cls(spec, 1826.0, seed=1), cfg)


def first_firing(history, tol, patience):
    """Index of the first iteration at which condition 1 holds, or None."""
    for i in range(1, len(history)):
        if hvs.quiet_run(history[:i + 1], tol) >= patience:
            return i
    return None


def main():
    # ---- A. rule off: behaviour of every earlier campaign ----------------
    a = make(PlainStub, n_iter=2)
    a.run(verbose=False)
    assert len(a.history) == 3 and len(a.X) == 36, (len(a.history), len(a.X))
    assert a.stop_info["by"] == "iters" and a.stop_info["stop"] is False
    inf = [p for p in a.phase_log if p["stage"] == "infill"]
    assert len(inf) == 2 and all(k in p for p in inf for k in HEALTH), inf[-1].keys()
    assert all(0 <= p["n_infill_feasible"] <= 6 and p["relax_depth"] >= 0 for p in inf)
    print(f"A  rule off     : {len(a.X)} evaluations, stop by {a.stop_info['by']}, "
          f"indicators {[(p['n_margin_feasible'], p['relax_depth'], p['n_infill_feasible']) for p in inf]}")

    # ---- B. rule on, loose tolerance so it fires inside the budget --------
    tol, pat, cap = 0.05, 3, 96
    b = make(PlainStub, n_iter=12, hv_stop=True, hv_tol=tol, hv_patience=pat, max_evals=cap)
    b.run(verbose=False)
    assert len(b.X) <= cap
    fire = first_firing(b.history, tol, pat)
    if b.stop_info["by"] == "rule":
        assert fire == len(b.history) - 1, (fire, len(b.history))   # first firing, not later
        assert b.stop_info["quiet"] >= pat and b.stop_info["censored"] == 0
        assert all(g is not None and g < tol for g in hvs.gains(b.history)[-pat:])
        assert len(b.X) == 24 + 6 * (len(b.history) - 1)
    else:
        assert b.stop_info["by"] == "cap" and fire is None, (b.stop_info, fire)
    print(f"B  rule on      : {len(b.X)} evaluations, stop by {b.stop_info['by']}, gains % "
          f"{[None if g is None else round(100 * g, 2) for g in hvs.gains(b.history)]}")

    # ---- C. the budget binds: tolerance so tight the rule cannot fire -----
    c = make(PlainStub, n_iter=12, hv_stop=True, hv_tol=-1.0, hv_patience=3, max_evals=40)
    c.run(verbose=False)
    assert c.stop_info["by"] == "cap" and len(c.X) == 36, (c.stop_info, len(c.X))
    print(f"C  budget 40    : {len(c.X)} evaluations, stop by {c.stop_info['by']}")

    # ---- D. a censored front blocks the rule ------------------------------
    d = make(CensoredStub, n_iter=12, hv_stop=True, hv_tol=10.0, hv_patience=1, max_evals=42)
    d.run(verbose=False)
    assert d.stop_info["by"] == "cap" and len(d.X) == 42, (d.stop_info, len(d.X))
    assert d._n_front_censored() >= 1
    assert hvs.quiet_run(d.history, 10.0) >= 1, "condition 1 held, only the censoring blocked"
    print(f"D  censored     : {len(d.X)} evaluations, stop by {d.stop_info['by']}, "
          f"{d._n_front_censored()} censored on the front")

    # ---- E. resume: the quiet count must carry across the checkpoint -------
    with tempfile.TemporaryDirectory() as td:
        ck = str(Path(td) / "ckpt.json")
        e1 = make(PlainStub, n_iter=2, hv_stop=True, hv_tol=10.0, hv_patience=3, max_evals=96)
        e1.checkpoint_path = ck
        e1.run(verbose=False)
        assert e1.stop_info["by"] == "iters" and hvs.quiet_run(e1.history, 10.0) == 2
        e1.save_checkpoint(ck, meta={"objective_set": "c9"})
        e2 = make(PlainStub, n_iter=12, hv_stop=True, hv_tol=10.0, hv_patience=3, max_evals=96)
        e2.checkpoint_path = ck
        n = e2.load_checkpoint(ck)
        assert n == 36
        e2.run(verbose=False)
        # two quiet iterations came from the first block, so ONE more suffices
        assert e2.stop_info["by"] == "rule" and len(e2.X) == 42, (e2.stop_info, len(e2.X))
        # ---- F. what is on disk
        disk = json.load(open(ck))
        assert disk["stop"]["by"] == "rule" and disk["stop"]["stop"] is True
        assert isinstance(disk["stop"]["warnings"], list)
        assert len(disk["stop"]["notes"]) == 3, disk["stop"]["notes"]
        last = [p for p in disk["phase_log"] if p["stage"] == "infill"][-1]
        assert all(k in last for k in HEALTH)
        assert len(disk["hv_history"]) == 4 and len(disk["all_raw"]) == 42
    print(f"E  resume       : 36 loaded, rule fired after one more iteration at {len(e2.X)}")
    print(f"F  checkpoint   : stop block on disk, reason: {disk['stop']['reason']}")
    print("test_hv_stop_wiring OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
