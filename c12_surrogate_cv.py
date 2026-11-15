#!/usr/bin/env python3
r"""
c12_surrogate_cv.py -- which surrogate predicts the two constraints that
decided Campaign 12, g_margin and g_kmin, and the two objectives, on the 117
designs of its checkpoint. No transport, scikit-learn only.

WHY
    In Campaign 12 the acquisition was decided by two constraint surrogates.
    g_margin (reactivity margin at the mission burnup) excluded every candidate
    with a sd of about 270 pcm on the boundary. g_kmin (beginning-of-life core
    reactivity) predicted four designs on the boundary with a sd of about
    1800 pcm, and they violate by 2068 to 2825 pcm. A surrogate with a smaller
    and calibrated sd on these two outputs changes the batches.

SURROGATES
    the Gaussian processes of surrogate_kernel_cv.py (gp_m52_ard is the campaign)
    gp_m52_ard_inv   the campaign kernel with the gadolinia inventory
                     gd_wt x gd_pins as a fifth input
    quad             quadratic response surface, Bayesian ridge regression
    rf               random forest, 500 trees, sd = spread of the trees
    gbr_q            gradient boosting on the 16, 50 and 84 % quantiles
    mlp_ens          the MLP ensemble of the code base, 5 networks
    gpc              Gaussian-process classifier of the sign of the constraint
                     (decisions only, no regression metrics)

VALIDATION SCHEMES
    loo           every design predicted from the other 116
    walkforward   every infill batch predicted from the designs evaluated
                  before it (99, 105, 111): the surrogate at decision time

METRICS
    RMSE, R^2, Spearman, coverage and s_z as in surrogate_kernel_cv.py, on all
    designs and on the feasible ones. For the two constraints, the feasibility
    decisions: wrong by the mean rule (false positive = predicted feasible,
    measured violated) and by the margin rule mean + kappa sd <= 0 at the
    campaign kappa (g_margin 1.20, g_kmin 1.11). The margin rule is also scored
    after recalibration: the sd of design i is multiplied by the s_z of the
    other designs, so an over-confident model is not favoured by its small sd.

    The stored constraint values of the 15 confirmed designs are full-fidelity
    two-seed means and the others one proxy depletion, as in the campaign.

USAGE (repository root, about 20 min on 8 threads)
    python c12_surrogate_cv.py
    python c12_surrogate_cv.py --variants gp_m52_ard rf --outputs g_kmin
Results are merged into <out>/surrogate_cv.json; surrogate_cv.md is rewritten.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from surrogate_kernel_cv import GP, MLP, metrics

CKPT = "results_wks720_2026-11-15/c12/out_c12/optimization_checkpoint.json"
OUTPUTS = {"g_margin": ("g_margin [pcm]", 1e5), "g_kmin": ("g_kmin [pcm]", 1e5),
           "peaking": ("F_dH [-]", 1.0), "c_max": ("c_max [ppm]", 1.0)}
CONSTRAINTS = ("g_margin", "g_kmin")


class Inv:
    """A surrogate with the gadolinia inventory gd_wt x gd_pins appended to the inputs."""
    def __init__(self, base, i_wt, i_pins):
        self.base, self.i, self.j = base, i_wt, i_pins

    def _x(self, X):
        X = np.atleast_2d(X)
        return np.column_stack([X, X[:, self.i] * X[:, self.j]])

    def fit(self, X, y):
        self.base.fit(self._x(X), y)
        return self

    def predict(self, X):
        return self.base.predict(self._x(X))


class Quad:
    def fit(self, X, y):
        from sklearn.linear_model import BayesianRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        self.m = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), BayesianRidge()).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(np.atleast_2d(X), return_std=True)


class RF:
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        self.m = RandomForestRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=1).fit(X, y)
        return self

    def predict(self, X):
        P = np.array([t.predict(np.atleast_2d(X)) for t in self.m.estimators_])
        return P.mean(axis=0), np.maximum(P.std(axis=0), 1e-12)


class GBQ:
    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingRegressor
        self.m = [GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=300, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, random_state=0).fit(X, y) for q in (0.16, 0.5, 0.84)]
        self.floor = 0.01 * float(np.std(y))      # the quantile models can cross, which would give a zero sd
        return self

    def predict(self, X):
        lo, mid, hi = (m.predict(np.atleast_2d(X)) for m in self.m)
        return mid, np.maximum(np.abs(hi - lo) / 2.0, self.floor)


class GPC:
    """Classifier of the sign: predict() returns (0.5 - P(feasible), None), so the mean rule is P >= 0.5."""
    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        from sklearn.preprocessing import StandardScaler
        self.xs = StandardScaler()
        lab = (y <= 0).astype(int)
        self.const = None if 0 < lab.sum() < len(lab) else float(lab[0])
        if self.const is None:
            k = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=2.5)
            self.m = GaussianProcessClassifier(kernel=k, n_restarts_optimizer=2, random_state=0).fit(self.xs.fit_transform(X), lab)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        p = np.full(len(X), self.const) if self.const is not None else self.m.predict_proba(self.xs.transform(X))[:, 1]
        return 0.5 - p, np.full(len(X), np.nan)


def variants(names):
    i, j = names.index("gd_wt"), names.index("gd_pins")
    return {
        "gp_m52_ard": lambda: GP("m52", True), "gp_m52_iso": lambda: GP("m52", False),
        "gp_m32_ard": lambda: GP("m32", True), "gp_m12_ard": lambda: GP("m12", True),
        "gp_rbf_ard": lambda: GP("rbf", True), "gp_rq_iso": lambda: GP("rq", False),
        "gp_m52_ard_inv": lambda: Inv(GP("m52", True), i, j),
        "quad": Quad, "rf": RF, "gbr_q": GBQ, "mlp_ens": MLP, "gpc": GPC,
    }


LABEL = {"gp_m52_ard": "GP Matern 5/2 ARD (campaign)", "gp_m52_iso": "GP Matern 5/2 iso", "gp_m32_ard": "GP Matern 3/2 ARD",
         "gp_m12_ard": "GP Matern 1/2 ARD", "gp_rbf_ard": "GP SE ARD", "gp_rq_iso": "GP rational quadratic iso",
         "gp_m52_ard_inv": "GP Matern 5/2 ARD + Gd inventory", "quad": "Quadratic response surface",
         "rf": "Random forest", "gbr_q": "Gradient boosting, quantiles", "mlp_ens": "MLP ensemble", "gpc": "GP classifier"}


def _one(make, X, y, tr, te):
    warnings.filterwarnings("ignore")
    a, b = make().fit(X[tr], y[tr]).predict(X[te])
    return np.ravel(a), np.ravel(b)


def predict_folds(make, X, y, folds, n_jobs):
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(make, X, y, tr, te) for tr, te in folds)
    mu, sd = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for (tr, te), (a, b) in zip(folds, res):
        mu[te], sd[te] = a, b
    return mu, sd


def jackknife_scale(y, mu, sd):
    """sd of design i multiplied by the s_z of the other designs."""
    z2 = ((y - mu) / np.maximum(sd, 1e-12)) ** 2
    n = len(y)
    return sd * np.sqrt((z2.sum() - z2) / max(n - 1, 1))


def decisions(y, mu, sd, kappa):
    bad, ok = y > 0, y <= 0
    out = dict(n=int(len(y)), n_violated=int(bad.sum()),
               mean_fp=int(np.sum(bad & (mu <= 0))), mean_fn=int(np.sum(ok & (mu > 0))))
    if np.all(np.isfinite(sd)):
        for tag, s in (("margin", sd), ("margin_recal", jackknife_scale(y, mu, sd))):
            adm = mu + kappa * s <= 0
            out[tag + "_fp"], out[tag + "_fn"] = int(np.sum(bad & adm)), int(np.sum(ok & ~adm))
        out["sz_recal"] = float(np.sqrt(np.mean(((y - mu) / np.maximum(jackknife_scale(y, mu, sd), 1e-12)) ** 2)))
    return out


def score(oname, y, mu, sd, feas, kappa):
    te = np.isfinite(mu)
    res = {}
    if np.all(np.isfinite(sd[te])):
        res["all"] = metrics(y[te], mu[te], sd[te])
        if (te & feas).sum() > 2:
            res["feasible"] = metrics(y[te & feas], mu[te & feas], sd[te & feas])
    if oname in CONSTRAINTS:
        res["decisions"] = decisions(y[te], mu[te], sd[te], kappa)
    return res


def render(R):
    L = ["# Campaign 12: comparison of surrogates on the constraints and the objectives", ""]
    for scheme, title in (("loo", "Leave-one-out, 117 designs"), ("walkforward", "Walk-forward, the 18 infill designs")):
        for oname, (olab, _) in OUTPUTS.items():
            rows = R.get(scheme, {}).get(oname)
            if not rows:
                continue
            con = oname in CONSTRAINTS
            L += ["", f"### {title}: {olab}", "",
                  "| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible |"
                  + (" Mean rule FP/FN | Margin rule FP/FN | Margin rule recalibrated FP/FN |" if con else ""),
                  "|---|---|---|---|---|---|---|" + ("---|---|---|" if con else "")]
            for v, res in rows.items():
                m, f, d = res.get("all"), res.get("feasible"), res.get("decisions", {})
                fmt = (lambda x: f"{x:.4f}") if oname == "peaking" else (lambda x: f"{x:.0f}")
                reg = (f"{fmt(m['rmse'])} | {m['r2']:.3f} | {m['spearman']:.3f} | {m['cov2']:.2f} | {m['sz']:.2f} | "
                       f"{fmt(f['rmse']) if f else '--'}") if m else "-- | -- | -- | -- | -- | --"
                dec = ""
                if con:
                    pair = lambda k: f"{d[k + '_fp']}/{d[k + '_fn']}" if k + "_fp" in d else "--"  # noqa: E731
                    dec = f" {pair('mean')} | {pair('margin')} | {pair('margin_recal')} |"
                L.append(f"| {LABEL.get(v, v)} | {reg} |{dec}")
            if con:
                d0 = next(iter(rows.values()))["decisions"]
                L += ["", f"{d0['n']} designs, {d0['n_violated']} violate the constraint. FP: admitted, measured violated. FN: rejected, measured satisfied."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--outputs", nargs="*", default=list(OUTPUTS), choices=list(OUTPUTS))
    ap.add_argument("--schemes", nargs="*", default=["loo", "walkforward"], choices=["loo", "walkforward"])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="figs_c12_cv")
    a = ap.parse_args(argv)

    D = json.loads(Path(a.checkpoint).read_text())
    A, names, CN = D["all_raw"], D["design_variables"], D["constraint_names"]
    kmap = D["meta"]["surrogate_policy"]["feas_kappa_map"]
    X = np.array([[r[v] for v in names] for r in A], float)
    feas = np.array([all(r[c] <= 0 for c in CN) for r in A])
    n = len(A)
    V = variants(names)
    todo = a.variants or list(V)
    n0, wf = 0, []
    for p in D["phase_log"]:
        m = int(p["n_eval"])
        if p["stage"] != "DOE":
            wf.append((np.arange(n0), np.arange(n0, n0 + m)))
        n0 += m
    folds = {"loo": [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)], "walkforward": wf}

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    store = out / "surrogate_cv.json"
    R = json.loads(store.read_text()) if store.exists() else {}
    print(f"{n} designs, {int(feas.sum())} feasible, variables {names}, kappa {kmap}")
    for scheme in a.schemes:
        for oname in a.outputs:
            olab, unit = OUTPUTS[oname]
            y = np.array([r[oname] for r in A], float) * unit
            for v in todo:
                if v == "gpc" and oname not in CONSTRAINTS:
                    continue
                t0 = time.time()
                mu, sd = predict_folds(V[v], X, y, folds[scheme], a.jobs)
                res = score(oname, y, mu, sd, feas, float(kmap.get(oname, 1.5)))
                res["seconds"] = time.time() - t0
                R.setdefault(scheme, {}).setdefault(oname, {})[v] = res
                store.write_text(json.dumps(R, indent=1))
                m, d = res.get("all"), res.get("decisions")
                print(f"  {scheme:11s} {olab:15s} {v:15s} "
                      + (f"RMSE {m['rmse']:.4g} R2 {m['r2']:.3f} s_z {m['sz']:.2f} " if m else "")
                      + (f"| mean {d['mean_fp']}/{d['mean_fn']} margin {d.get('margin_fp', '-')}/{d.get('margin_fn', '-')} "
                         f"recal {d.get('margin_recal_fp', '-')}/{d.get('margin_recal_fn', '-')} " if d else "")
                      + f"[{res['seconds']:.0f} s]", flush=True)
    (out / "surrogate_cv.md").write_text(render(R), encoding="utf-8")
    print(f"wrote {out}/surrogate_cv.json and surrogate_cv.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
