#!/usr/bin/env python3
r"""
hv_stop.py -- the stopping rule of the active-learning loop, as pure arithmetic.

No OpenMC, no pymoo, no numpy. Imported by reactor_optimization.py once
apply_hv_stop.py has been applied, and usable on any archived checkpoint.

THE RULE, as worded in methodology.tex (empirical stopping rule, both required)
    1. The hypervolume gain stays below 1 % for three consecutive iterations.
    2. No censored design remains on the Pareto front.

    gain_i = (HV_i - HV_{i-1}) / HV_{i-1},  i >= 1, HV_0 being the DOE value.

    An iteration whose previous hypervolume is zero has no defined gain. It
    never counts as quiet, so the rule cannot fire on an empty feasible front.

THE CAP
    A budget on real evaluations. The loop does not START an iteration whose
    infill batch would take the archive past the cap, so the cap is never
    exceeded: 24 + 12 x 6 = 96 stops after the twelfth iteration.

HEALTH (reported, never used to decide)
    methodology.tex calls the rule necessary and not sufficient, and names
    two indicators to inspect before trusting it. A third is added here.
      margin-feasible candidates   zero means the batch came from the fallback
                                   ordering, not from uncertainty (WARNING)
      diversity relaxation depth   a deep cascade means few new designs are
                                   left, which supports convergence (note)
      feasible infill designs      a zero gain with zero feasible infill is
                                   the Campaign 6 false firing, a zero gain
                                   with feasible infill is a converged search
    health_warnings() and health_notes() turn them into plain sentences.

USAGE
    python hv_stop.py --selftest
    python hv_stop.py --checkpoint out_c9/optimization_checkpoint.json
"""
from __future__ import annotations

import argparse
import json
import sys

TOL = 0.01
PATIENCE = 3


def gains(history):
    """Relative gain per iteration. None where the previous value is not > 0."""
    out = []
    for prev, cur in zip(history[:-1], history[1:]):
        out.append((cur - prev) / prev if prev > 0 else None)
    return out


def quiet_run(history, tol=TOL):
    """Number of consecutive quiet iterations at the END of the history."""
    n = 0
    for g in reversed(gains(history)):
        if g is None or g >= tol:
            break
        n += 1
    return n


def cap_blocks(n_evals, n_infill, max_evals):
    """True when starting one more iteration would exceed the budget."""
    return max_evals is not None and n_evals + n_infill > max_evals


def decide(history, n_front_censored, tol=TOL, patience=PATIENCE):
    """Evaluate the two-condition rule after an iteration.

    history           hypervolume per iteration, DOE first
    n_front_censored  censored designs on the current feasible Pareto front
    Returns dict(stop, quiet, patience, last_gain, censored, reason).
    """
    q = quiet_run(history, tol)
    g = gains(history)
    last = g[-1] if g else None
    cond1 = q >= patience
    cond2 = n_front_censored == 0
    if cond1 and cond2:
        reason = (f"rule met: gain below {100 * tol:g} % for {q} consecutive "
                  f"iterations and no censored design on the front")
    elif cond1:
        reason = (f"gain quiet for {q} iterations but {n_front_censored} censored "
                  f"design(s) on the front, continuing")
    else:
        reason = f"{q} of {patience} quiet iterations, continuing"
    return dict(stop=bool(cond1 and cond2), quiet=q, patience=patience,
                last_gain=last, censored=int(n_front_censored), reason=reason)


def _last_infill(phase_log, patience):
    return [p for p in phase_log if p.get("stage") == "infill"][-patience:]


def health_warnings(phase_log, patience=PATIENCE):
    """Conditions that UNDERMINE a firing of the rule, as plain sentences."""
    W = []
    for p in _last_infill(phase_log, patience):
        it = p.get("iteration")
        if p.get("n_margin_feasible") == 0:
            W.append(f"iteration {it}: no candidate passed the feasibility margin, the "
                     f"batch came from the fallback ordering and not from uncertainty")
        if p.get("n_infill_feasible") == 0:
            W.append(f"iteration {it}: no infill design was feasible, so its zero "
                     f"gain says nothing about convergence")
    return W


def health_notes(phase_log, patience=PATIENCE):
    """Neutral indicators. A deep cascade SUPPORTS convergence: the surrogate
    front had few designs left that were new at the requested separation."""
    N = []
    for p in _last_infill(phase_log, patience):
        if p.get("relax_depth") is not None:
            N.append(f"iteration {p.get('iteration')}: {p.get('n_margin_feasible')} of "
                     f"{p.get('n_candidates')} margin-feasible, diversity relaxed "
                     f"{p['relax_depth']} time(s), {p.get('n_infill_feasible')} feasible infill")
    return N


# ------------------------------------------------------------------ selftest --
def selftest():
    # the real Campaign 9 history, from out_c9/optimization_checkpoint.json
    c9 = [879.0689093428063, 1244.5303455391681, 1291.447080207337, 1291.563218334697,
          1318.4927430812443, 1318.4927430812443, 1318.4927430812443]
    g = gains(c9)
    assert [round(100 * x, 3) for x in g] == [41.574, 3.77, 0.009, 2.085, 0.0, 0.0], g
    assert quiet_run(c9) == 2
    assert not decide(c9, 0)["stop"], "C9 has two quiet iterations, not three"
    assert decide(c9 + [c9[-1] * 1.004], 0)["stop"], "a third quiet iteration fires the rule"
    assert not decide(c9 + [c9[-1]], 1)["stop"], "a censored front member blocks the rule"
    assert not decide(c9 + [c9[-1] * 1.02], 0)["stop"], "a 2 % gain resets the count"
    assert quiet_run(c9[:4]) == 1, "0.009 % then the 2.1 % jump: the count was reset"
    # an empty feasible front never fires
    assert not decide([0.0, 0.0, 0.0, 0.0, 0.0], 0)["stop"]
    assert gains([0.0, 5.0]) == [None] and quiet_run([0.0, 5.0, 5.0, 5.0, 5.0]) == 3
    # exactly at the tolerance is NOT quiet ("stays below 1 %")
    assert quiet_run([100.0, 101.0]) == 0 and quiet_run([100.0, 100.99]) == 1
    # the cap: 24 + 12 x 6 = 96 runs the twelfth iteration and blocks the thirteenth
    assert not cap_blocks(90, 6, 96) and cap_blocks(96, 6, 96) and cap_blocks(92, 6, 96)
    assert not cap_blocks(10 ** 6, 6, None)
    # health
    log = [dict(stage="DOE", iteration=0),
           dict(stage="infill", iteration=1, n_margin_feasible=40, n_infill_feasible=3, relax_depth=0),
           dict(stage="infill", iteration=2, n_margin_feasible=0, n_infill_feasible=0, relax_depth=4)]
    W = health_warnings(log)
    assert len(W) == 2 and all("iteration 2" in w for w in W), W
    assert len(health_notes(log)) == 2 and "relaxed 4" in health_notes(log)[1]
    old = [dict(stage="infill", iteration=1)]
    assert health_warnings(old) == [] and health_notes(old) == [], "old logs carry no indicators"
    print("selftest: Campaign 9 gains (%)", [round(100 * x, 3) for x in g])
    print("selftest: Campaign 9 ends with 2 quiet iterations, the rule asks for 3")
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.checkpoint:
        ap.error("--checkpoint or --selftest")
    d = json.load(open(a.checkpoint))
    raw, cn = d["all_raw"], d["constraint_names"]
    keys = [(o[0], -1.0 if o[1] == "max" else 1.0) for o in d["objectives"]]
    feas = [r for r in raw if all(r[c] <= 1e-9 for c in cn)]
    F = [[s * r[k] for k, s in keys] for r in feas]
    front = [i for i, f in enumerate(F)
             if not any(all(x <= y for x, y in zip(h, f)) and any(x < y for x, y in zip(h, f))
                        for j, h in enumerate(F) if j != i)]
    n_cen = sum(bool(feas[i].get("censored")) for i in front)
    for i, g in enumerate(gains(d["hv_history"]), 1):
        print(f"  iteration {i:2d}: HV {d['hv_history'][i]:12.4f}  gain "
              + (f"{100 * g:8.3f} %" if g is not None else "   undefined"))
    r = decide(d["hv_history"], n_cen, a.tol, a.patience)
    print(f"  front members {len(front)}, censored on the front {n_cen}")
    print(f"  {r['reason']}")
    for n in health_notes(d.get("phase_log", []), a.patience):
        print(f"  note    {n}")
    for w in health_warnings(d.get("phase_log", []), a.patience):
        print(f"  WARNING {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
