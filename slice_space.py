#!/usr/bin/env python3
"""
slice_space.py -- freeze design variables and enumerate explicit design
lists with the PRODUCTION optimizer. Written for the Step 3 verification
of the surrogate-assisted search against an exhaustive grid.

Two features, both wired into run_optimization.py by apply_slice_flags.py:

  --freeze NAME=VALUE     remove NAME from the search box and inject VALUE
                          into every design dict at the single vector-to-
                          design choke point (DesignSpace.as_dict). The
                          evaluator, the zoning path, the k_target lookup,
                          the log line and the archive all see a complete
                          design, exactly as in a four-variable run.
                          Repeatable: --freeze refl_thick=4.2889 --freeze
                          gd_pins=12.

  --eval-list PATH.json   ENUMERATION MODE: evaluate the listed designs one
                          at a time, write the checkpoint after every
                          evaluation, and skip any design already in the
                          archive (so a --resume continues where it
                          stopped). No design of experiments, no surrogate,
                          no infill. The archive it writes has the same
                          format as a campaign checkpoint, so every
                          post-processing script reads it unchanged.

WHY A SUBCLASS OF DesignSpace AND NOT A NEW PROBLEM SPEC
    Every consumer of the design dict keys (openmc_evaluator.evaluate_one,
    zoning.evaluator_design_map, the case log line, save_checkpoint and
    load_checkpoint) stays byte-for-byte untouched. Only the vector the
    optimizer manipulates gets shorter.

WHAT CANNOT BE FROZEN
    The enrichment. The LEU box, the g_enr limit and the meta record all
    key on the "enrich" variable in run_optimization.py, so freezing it
    would need edits in three places for no benefit in this study. Use
    --enr-box-low / --enr-box-high to narrow it instead.

The eval-list file is a JSON list of dicts keyed by the LIVE design
variable names, e.g. [{"enrich": 3.0, "gd_wt": 0.0}, ...], or a dict with
a "designs" key holding that list (valgrid_make_list.py writes the second
form, with the frozen values and the grid definition alongside).

    python slice_space.py --selftest      analytic evaluator, no OpenMC
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from reactor_optimization import DesignSpace, ProblemSpec

# keys that DesignSpace.as_dict DERIVES rather than reads from the vector;
# they may be frozen (overridden) without being a design variable
DERIVED_KEYS = ("pitch",)
# variable names that must stay live (see module docstring)
NEVER_FROZEN = ("enrich", "enrich_inner", "enrich_outer")


class FrozenDesignSpace(DesignSpace):
    """A DesignSpace whose as_dict() completes every design with fixed
    values for the frozen variables. The frozen values WIN over the derived
    defaults of the parent (so pitch can be frozen at a non-default value),
    while the derivation rules of the parent (enrich_inner, enrich_outer,
    pitch from enrich) still run for everything not frozen."""

    def __init__(self, variables, frozen: dict):
        super().__init__(list(variables))
        self.frozen = {str(k): float(v) for k, v in frozen.items()}

    def as_dict(self, x: Sequence[float]) -> dict:
        d = super().as_dict(x)
        d.update(self.frozen)
        return d


def parse_freeze(items) -> dict:
    """['refl_thick=4.2889', 'gd_pins=12'] -> {'refl_thick': 4.2889,
    'gd_pins': 12.0}. Raises on a malformed item."""
    out: dict = {}
    for s in items or []:
        if "=" not in s:
            raise ValueError(f"--freeze expects NAME=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"--freeze expects NAME=VALUE, got {s!r}")
        try:
            out[k] = float(v)
        except ValueError as exc:
            raise ValueError(f"--freeze {s!r}: value is not a number") from exc
    return out


def freeze_variables(spec: ProblemSpec, frozen: dict) -> ProblemSpec:
    """Return a NEW ProblemSpec whose design space is `spec`'s with the
    frozen variables removed and their values injected by as_dict().
    Objectives, constraint names, exact constraints and constraint scales
    are carried over unchanged."""
    if not frozen:
        return spec
    ds = spec.design_space
    names = list(ds.names)
    unknown = [k for k in frozen if k not in names and k not in DERIVED_KEYS]
    if unknown:
        raise ValueError(f"cannot freeze {unknown}: not design variables "
                         f"{names} and not derived keys {list(DERIVED_KEYS)}")
    bad = [k for k in frozen if k in NEVER_FROZEN]
    if bad:
        raise ValueError(f"cannot freeze {bad}: the enrichment must stay a "
                         "live variable (LEU box, g_enr and the meta record "
                         "key on it). Narrow it with --enr-box-low / "
                         "--enr-box-high instead.")
    keep = []
    for v in ds.variables:
        if v.name in frozen:
            val = float(frozen[v.name])
            if not (v.low - 1e-12 <= val <= v.high + 1e-12):
                raise ValueError(f"frozen {v.name}={val:g} lies outside its "
                                 f"search box [{v.low:g}, {v.high:g}]")
        else:
            keep.append(v)
    if not keep:
        raise ValueError("every design variable is frozen: nothing to search")
    new_ds = FrozenDesignSpace(keep, frozen)
    return ProblemSpec(new_ds, list(spec.objectives),
                       list(spec.constraint_names),
                       exact_constraints=dict(spec.exact_constraints),
                       constraint_scales=dict(spec.constraint_scales))


def load_eval_list(path, names) -> np.ndarray:
    """Read the design list and return an (n, n_var) array in the order of
    the LIVE variable names. Extra keys in the file are ignored, missing
    keys raise."""
    obj = json.loads(Path(path).read_text())
    designs = obj["designs"] if isinstance(obj, dict) else obj
    if not designs:
        raise ValueError(f"{path}: empty design list")
    rows = []
    for i, d in enumerate(designs):
        missing = [n for n in names if n not in d]
        if missing:
            raise ValueError(f"{path}: design {i} lacks {missing} "
                             f"(live variables are {names})")
        rows.append([float(d[n]) for n in names])
    return np.array(rows, dtype=float)


def run_eval_list(opt, list_path, ckpt_path=None, meta=None, verbose=True):
    """Evaluate every listed design with the optimizer's truth evaluator,
    one at a time, checkpointing after each. Designs already present in
    opt.X (within 1e-9) are skipped, so a --resume finishes an interrupted
    list. Returns opt.results(), like ActiveLearningMOO.run()."""
    names = opt.spec.design_space.names
    X = load_eval_list(list_path, names)
    xl = np.asarray(opt.spec.design_space.xl, dtype=float)
    xu = np.asarray(opt.spec.design_space.xu, dtype=float)
    outside = np.any((X < xl - 1e-9) | (X > xu + 1e-9), axis=1)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} of {len(X)} listed designs "
                         f"lie outside the search box for {names}: "
                         f"xl={xl.tolist()} xu={xu.tolist()}")
    n_done = n_skip = n_geom = 0
    t_start = time.time()
    if verbose:
        print(f"[enumerate] {len(X)} designs listed in {list_path} | "
              f"live variables {names} | archive holds {len(opt.X)}")
    for i, x in enumerate(X):
        if len(opt.X) and np.any(
                np.all(np.isclose(opt.X, x, rtol=1e-9, atol=1e-9), axis=1)):
            n_skip += 1
            continue
        d = opt.spec.design_space.as_dict(x)
        if not opt.spec.exact_ok(d):
            n_geom += 1
            if verbose:
                print(f"[enumerate] {i + 1}/{len(X)} fails an analytic "
                      f"constraint, not evaluated: "
                      f"{ {n: round(float(d[n]), 4) for n in names} }")
            continue
        t0 = time.perf_counter()
        F, G, raw = opt.evaluator.evaluate(x[None, :])
        t_ev = time.perf_counter() - t0
        opt._add(np.atleast_2d(x), F, G, raw)
        t_hv0 = time.perf_counter()
        opt.history.append(opt._hv())
        t_hv = time.perf_counter() - t_hv0
        opt.phase_log.append(dict(
            stage="enumerate", iteration=len(opt.phase_log), n_eval=1,
            list_index=int(i), t_eval_s=t_ev, t_hv_s=t_hv, t_ckpt_s=None))
        n_done += 1
        if ckpt_path:
            t_ck0 = time.perf_counter()
            opt.save_checkpoint(ckpt_path, meta=meta)
            opt.phase_log[-1]["t_ckpt_s"] = time.perf_counter() - t_ck0
        if verbose:
            elapsed = (time.time() - t_start) / 60.0
            print(f"[enumerate] {i + 1}/{len(X)} done this run={n_done} "
                  f"skipped={n_skip} archive={len(opt.X)} "
                  f"HV={opt.history[-1]:.4g} elapsed={elapsed:.1f} min")
    if verbose:
        print(f"[enumerate] finished: {n_done} evaluated, {n_skip} already "
              f"in the archive, {n_geom} rejected by analytic constraints, "
              f"{len(opt.X)} in the archive")
    return opt.results()


# --------------------------------------------------------------------------- #
# self-test on the analytic evaluator (no OpenMC)                             #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import tempfile
    from reactor_optimization import (example_reactor_problem,
                                      AnalyticEvaluator, OptimizerConfig,
                                      ActiveLearningMOO)
    spec4 = example_reactor_problem()
    frozen = parse_freeze(["refl_thick=4.2889", "gd_pins=12"])
    spec2 = freeze_variables(spec4, frozen)
    assert spec2.design_space.names == ["enrich", "gd_wt"], spec2.design_space.names
    d = spec2.design_space.as_dict([5.0, 3.0])
    for k, v in (("enrich", 5.0), ("enrich_inner", 5.0), ("enrich_outer", 5.0),
                 ("pitch", 1.26), ("refl_thick", 4.2889), ("gd_pins", 12.0),
                 ("gd_wt", 3.0)):
        assert abs(d[k] - v) < 1e-12, (k, d[k], v)
    assert spec2.exact_ok(d), "g_geom should pass at refl 4.2889"
    # frozen pitch overrides the derived default
    dp = freeze_variables(spec4, {"pitch": 1.2598}).design_space.as_dict(
        [5.0, 3.0, 4.0, 12.0])
    assert abs(dp["pitch"] - 1.2598) < 1e-12
    # guards
    for bad in ({"enrich": 5.0}, {"nope": 1.0}, {"refl_thick": 9.0}):
        try:
            freeze_variables(spec4, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"freeze_variables accepted {bad}")
    # enumeration on the analytic evaluator: 6 designs, resume skips them
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "list.json"
        grid = [{"enrich": e, "gd_wt": g} for e in (3.0, 5.0, 7.0)
                for g in (0.0, 4.0)]
        lst.write_text(json.dumps({"designs": grid}))
        ck = Path(td) / "ckpt.json"
        opt = ActiveLearningMOO(spec2, AnalyticEvaluator(spec2, noise=0.0),
                                OptimizerConfig(n_init=4, n_iter=0,
                                                n_infill=2, seed=1))
        res = run_eval_list(opt, lst, str(ck), {"frozen": frozen},
                            verbose=False)
        assert len(res["all_raw"]) == 6, len(res["all_raw"])
        assert all(abs(r["refl_thick"] - 4.2889) < 1e-12 for r in res["all_raw"])
        assert opt.evaluator.n_calls == 6
        ckd = json.loads(ck.read_text())
        assert ckd["design_variables"] == ["enrich", "gd_wt"]
        assert ckd["meta"]["frozen"] == frozen
        assert len(ckd["phase_log"]) == 6
        # resume: load the checkpoint into a fresh optimizer, rerun the list
        opt2 = ActiveLearningMOO(spec2, AnalyticEvaluator(spec2, noise=0.0),
                                 OptimizerConfig(n_init=4, n_iter=0, seed=1))
        opt2.load_checkpoint(str(ck))
        res2 = run_eval_list(opt2, lst, str(ck), {"frozen": frozen},
                             verbose=False)
        assert len(res2["all_raw"]) == 6 and opt2.evaluator.n_calls == 6
        # the normal active-learning path still works on the 2-D space
        opt3 = ActiveLearningMOO(spec2, AnalyticEvaluator(spec2, noise=0.0),
                                 OptimizerConfig(n_init=6, n_iter=1,
                                                 n_infill=2, nsga_pop=20,
                                                 nsga_gen=10, seed=1))
        res3 = opt3.run(verbose=False)
        assert len(res3["all_raw"]) == 8
        assert res3["all_X"].shape == (8, 2)
    print("slice_space selftest OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
