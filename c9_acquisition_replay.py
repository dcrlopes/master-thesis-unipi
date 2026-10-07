#!/usr/bin/env python3
"""
c9_acquisition_replay.py -- replay the acquisition of Campaign 9 from the
archive as it stood before each infill block, under the rule the campaign
used and under the alternatives, and compare the batches each rule would
have chosen. No transport: the surrogates, NSGA-II and the ranking only.

WHAT IS REPLAYED
    For block k the optimizer is seeded with the first n_k evaluations of the
    archive (24, 30, ..., 54), the surrogates are fitted as the loop fits
    them, NSGA-II runs with the campaign settings and seed (1 + k), and
    ActiveLearningMOO.propose() picks the batch. Under the rule the campaign
    used ("asrun") the picks must reproduce the six designs the campaign then
    evaluated; that is the fidelity check of the replay.

RULES
    asrun         margin rule, kappa 1.5 on every GP constraint (Campaign 9)
    pof           probability-of-feasibility weighting, no gate
    boron         margin rule with the MTC boron limit as a loop constraint
    pof_boron     both
    kappa_efpd    margin rule, kappa 0.5 on g_efpd and 1.5 elsewhere
    gate_no_efpd  margin rule, the gate applied to every constraint except
                  the cycle length, which the surrogate predicts worst
    iso_efpd      as run, but the constraint surrogate of the cycle length
                  uses an isotropic Matern 5/2 (one length-scale) instead of
                  ARD: the leave-one-out study of 7 Oct 2026 found it twice
                  as accurate on the feasible designs (surrogate_kernel_cv.py)

    For every rule the fitted constraint surrogate of each block also
    predicts g_efpd of the six designs the campaign actually evaluated in
    that block, so the report counts how many of the designs that turned out
    below 1826 d the surrogate would have declared feasible (mean <= 0) and
    margin-feasible (mean + kappa sd <= 0). That is the walk-forward
    decision test of the constraint surrogate, independent of the picks.

WHAT IS REPORTED, per rule and block
    the six picks with their predicted objectives, the predicted probability
    of feasibility, how many sit below 1 wt% gadolinia, how many are
    predicted above the 2763 ppm MTC boron limit, the spread of the batch
    (mean and minimum pairwise separation in the unit design box, the
    measure the loop uses), and for "asrun" the number of picks that match
    the designs actually evaluated. The chosen designs of the other rules
    were never evaluated, so the comparison is on these predicted
    quantities and on the spread only.

USAGE (repository root, a few minutes per rule)
    python c9_acquisition_replay.py --rules asrun
    python c9_acquisition_replay.py --rules asrun pof boron pof_boron kappa_efpd gate_no_efpd
    python c9_acquisition_replay.py --rules pof --blocks 1 --nsga-gen 50    # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import core_geometry as cg
from reactor_optimization import (ActiveLearningMOO, GPSurrogate, OptimizerConfig,
                                  campaign9_problem)

RULES = {
    "asrun":        dict(acq_rule="margin", boron=False),
    "pof":          dict(acq_rule="pof", boron=False),
    "boron":        dict(acq_rule="margin", boron=True),
    "pof_boron":    dict(acq_rule="pof", boron=True),
    "kappa_efpd":   dict(acq_rule="margin", boron=False, feas_kappa_map={"g_efpd": 0.5}),
    "gate_no_efpd": dict(acq_rule="margin", boron=False,
                         margin_constraints=("g_kmin", "g_kmax", "g_enr", "g_peak", "g_ctrl_peak", "g_ctrl")),
    "iso_efpd":     dict(acq_rule="margin", boron=False, iso_constraints=("g_efpd",)),
}


class GPSurrogateKernels(GPSurrogate):
    """GPSurrogate with an isotropic Matern 5/2 on chosen output columns.
    The loop uses one factory for the objective and the constraint
    surrogates, so the choice is keyed by the number of output columns:
    iso_by_width = {n_constraints: {column indices}}."""
    def __init__(self, iso_by_width):
        super().__init__()
        self.iso_by_width = iso_by_width

    def fit(self, X, Y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        Xs = self.xscaler.fit_transform(X)
        self.models = []
        iso = self.iso_by_width.get(Y.shape[1], set())
        for j in range(Y.shape[1]):
            ls = 1.0 if j in iso else np.ones(X.shape[1])
            kernel = (ConstantKernel(1.0, (1e-3, 1e4))
                      * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5)
                      + WhiteKernel(1e-3, (1e-8, 1e1)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4, random_state=0)
            gp.fit(Xs, Y[:, j])
            self.models.append(gp)
        return self
BORON_LIMIT = 2763.0


class _NoEvaluator:
    n_calls = 0

    def evaluate(self, X):
        raise RuntimeError("the replay never evaluates")


def build(ck, rule, nsga_gen=None):
    meta = ck["meta"]
    lim = meta["limits"]
    c9 = meta["campaign9"]
    pol = meta["surrogate_policy"]
    spec = campaign9_problem(float(c9["efpd_req"]), f_max=float(lim["f_max"]))
    spec.constraint_names.append("g_ctrl")
    spec.constraint_scales["g_ctrl"] = 1.0
    spec.constraint_scales.update({"g_kmin": float(lim["k_min"]), "g_kmax": float(lim["k_max"]),
                                   "g_enr": float(lim["enr_max"]), "g_peak": float(lim["f_max"]),
                                   "g_geom": cg.R_VESSEL_INNER - cg.VESSEL_CLEARANCE_CM})
    if RULES[rule]["boron"]:
        spec.constraint_names.append("g_boron")
        spec.constraint_scales["g_boron"] = float(c9["boron_ceiling_ppm"])
    assert list(spec.constraint_names[:len(ck["constraint_names"])]) == list(ck["constraint_names"]), \
        (spec.constraint_names, ck["constraint_names"])
    cfg = OptimizerConfig(n_init=24, n_iter=1, n_infill=6, nsga_pop=int(pol["nsga_pop"]),
                          nsga_gen=int(nsga_gen or pol["nsga_gen"]), surrogate="gp", seed=1)
    cfg.infill_min_sep = float(pol["infill_min_sep"])
    cfg.feas_kappa = float(pol["feas_kappa"])
    cfg.efpd_cap = pol.get("efpd_cap_efpd")
    cfg.acq_rule = RULES[rule]["acq_rule"]
    cfg.feas_kappa_map = RULES[rule].get("feas_kappa_map")
    cfg.margin_constraints = RULES[rule].get("margin_constraints")
    return spec, cfg


def blocks_of(ck):
    n0, out = 0, []
    for p in ck["phase_log"]:
        m = int(p["n_eval"])
        if p["stage"] == "DOE":
            n0 += m
        elif p["stage"] == "infill":
            out.append((n0, n0 + m))
            n0 += m
    return out


def spread(X, xl, xu):
    span = np.where(xu > xl, xu - xl, 1.0)
    n = X.shape[1]
    d = [float(np.linalg.norm((a - b) / span)) / np.sqrt(n) for a, b in combinations(X, 2)]
    return (float(np.mean(d)), float(np.min(d))) if d else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--rules", nargs="+", default=["asrun"], choices=list(RULES))
    ap.add_argument("--blocks", nargs="*", type=int, default=None, help="1-based; default all")
    ap.add_argument("--nsga-gen", type=int, default=None, help="override, for a smoke test")
    ap.add_argument("--out", default="figs_c9_acq")
    a = ap.parse_args()

    ck = json.loads(Path(a.checkpoint).read_text())
    raw = ck["all_raw"]
    names = ck["design_variables"]
    blocks = blocks_of(ck)
    sel = [b - 1 for b in a.blocks] if a.blocks else list(range(len(blocks)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"checkpoint": a.checkpoint, "blocks": blocks, "rules": {}}
    igd, ic = names.index("gd_wt"), None

    for rule in a.rules:
        spec, cfg = build(ck, rule, a.nsga_gen)
        xl, xu = np.asarray(spec.design_space.xl, float), np.asarray(spec.design_space.xu, float)
        per_block = []
        print(f"\n=== rule {rule}: {RULES[rule]}")
        for k in sel:
            s, e = blocks[k]
            opt = ActiveLearningMOO(spec, _NoEvaluator(), cfg)
            opt._seed_from_raw(raw[:s])
            iso = RULES[rule].get("iso_constraints")
            if iso:
                cols = {spec.constraint_names.index(n) for n in iso}
                opt._new_surrogate = lambda cols=cols, n=len(spec.constraint_names): GPSurrogateKernels({n: cols})
            t0 = time.time()
            Xinf, info = opt.propose(it=k, verbose=False)
            f_mu, _ = info["obj_sur"].predict(Xinf)
            g_mu, g_sd = info["con_sur"].predict(Xinf)
            from scipy.stats import norm
            cols = [j for j, n in enumerate(spec.constraint_names) if n != "g_geom"]
            pof = np.prod(norm.cdf(-np.atleast_2d(g_mu)[:, cols] / np.maximum(np.atleast_2d(g_sd)[:, cols], 1e-12)), axis=1)
            f_mu = np.atleast_2d(f_mu)
            actual = np.array([[float(r[n]) for n in names] for r in raw[s:e]])
            match = sum(any(np.allclose(x, y, rtol=1e-6, atol=1e-6) for y in actual) for x in Xinf)
            mean_sep, min_sep = spread(Xinf, xl, xu)
            # walk-forward decision test on the designs the campaign evaluated in this block
            je = spec.constraint_names.index("g_efpd")
            ga_mu, ga_sd = info["con_sur"].predict(actual)
            ga_mu = np.atleast_2d(ga_mu)[:, je] * spec.g_scale("g_efpd")
            ga_sd = np.atleast_2d(ga_sd)[:, je] * spec.g_scale("g_efpd")
            g_true = np.array([float(r["g_efpd"]) for r in raw[s:e]])
            kap = float(cfg.feas_kappa)
            short = g_true > 0                                   # below 1826 d in the assembly depletion
            dec = dict(n_short=int(short.sum()),
                       short_pred_feasible=int(np.sum(short & (ga_mu <= 0))),
                       short_pred_margin_feasible=int(np.sum(short & (ga_mu + kap * ga_sd <= 0))),
                       ok_pred_feasible=int(np.sum(~short & (ga_mu <= 0))),
                       rmse_g_efpd=float(np.sqrt(np.mean((ga_mu - g_true) ** 2))),
                       pred_g_efpd=[float(v) for v in ga_mu], true_g_efpd=[float(v) for v in g_true])
            row = dict(block=k + 1, n_train=s, seconds=round(time.time() - t0, 1),
                       picks=[dict(zip(names, map(float, x))) for x in Xinf],
                       predicted_peaking=[float(v) for v in f_mu[:, 0]],
                       predicted_c_max=[float(v) for v in f_mu[:, 1]],
                       pof=[float(v) for v in pof],
                       n_low_gd=int((Xinf[:, igd] < 1.0).sum()),
                       n_pred_above_boron=int((f_mu[:, 1] > BORON_LIMIT).sum()),
                       mean_sep=mean_sep, min_sep=min_sep,
                       match_actual=int(match), decisions=dec,
                       gate=info["acq"].get("gate_columns"),
                       n_eligible=int(np.sum(info["acq"]["eligible"])))
            per_block.append(row)
            print(f"  block {k + 1} (train {s}): match {match}/6 | low-gd {row['n_low_gd']} | pred>2763 ppm "
                  f"{row['n_pred_above_boron']} | pof mean {pof.mean():.2f} | sep mean {mean_sep:.3f} min {min_sep:.3f} "
                  f"| eligible {row['n_eligible']}/300 | {row['seconds']} s")
            print(f"     actual batch: {dec['n_short']} of 6 below 1826 d; of those predicted feasible {dec['short_pred_feasible']}, "
                  f"margin-feasible {dec['short_pred_margin_feasible']}; g_efpd rmse {dec['rmse_g_efpd']:.0f} d; pred/true: "
                  + ", ".join(f"{pv:+.0f}/{tv:+.0f}" for pv, tv in zip(dec['pred_g_efpd'], dec['true_g_efpd'])))
            for x, fp, fc, p in zip(Xinf, f_mu[:, 0], f_mu[:, 1], pof):
                print("     " + "  ".join(f"{n} {v:7.3f}" for n, v in zip(names, x)) + f"  F {fp:.3f}  c_max {fc:5.0f}  pof {p:.2f}")
        summary = dict(mean_sep=float(np.mean([b["mean_sep"] for b in per_block])),
                       min_sep=float(np.mean([b["min_sep"] for b in per_block])),
                       low_gd=int(sum(b["n_low_gd"] for b in per_block)),
                       pred_above_boron=int(sum(b["n_pred_above_boron"] for b in per_block)),
                       pof_mean=float(np.mean([np.mean(b["pof"]) for b in per_block])),
                       match_actual=int(sum(b["match_actual"] for b in per_block)),
                       n_picks=6 * len(per_block),
                       n_short=int(sum(b["decisions"]["n_short"] for b in per_block)),
                       short_pred_feasible=int(sum(b["decisions"]["short_pred_feasible"] for b in per_block)),
                       short_pred_margin_feasible=int(sum(b["decisions"]["short_pred_margin_feasible"] for b in per_block)),
                       rmse_g_efpd=float(np.sqrt(np.mean([(pv - tv) ** 2 for b in per_block for pv, tv in
                                                           zip(b["decisions"]["pred_g_efpd"], b["decisions"]["true_g_efpd"])]))))
        results["rules"][rule] = dict(config=RULES[rule], blocks=per_block, summary=summary)
        print(f"  summary {rule}: {summary}")

    (out / "c9_acquisition_replay.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"\n{'rule':14s} {'picks':>5s} {'match':>5s} {'low-gd':>6s} {'>2763':>5s} {'pof':>5s} {'sep mean':>8s} {'sep min':>7s}")
    for rule, r in results["rules"].items():
        m = r["summary"]
        print(f"{rule:14s} {m['n_picks']:5d} {m['match_actual']:5d} {m['low_gd']:6d} {m['pred_above_boron']:5d} "
              f"{m['pof_mean']:5.2f} {m['mean_sep']:8.3f} {m['min_sep']:7.3f}")
    print(f"wrote {out}/c9_acquisition_replay.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
