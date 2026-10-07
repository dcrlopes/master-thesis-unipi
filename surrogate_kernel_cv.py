#!/usr/bin/env python3
"""
surrogate_kernel_cv.py -- leave-one-out comparison of surrogate models on the
Campaign 8 and Campaign 9 data, for the paper. No transport.

The thesis (Section sec:bg-gp) states that the kernel, a Matern 5/2 with one
length-scale per variable and a white-noise term, was not compared against
other kernels. This script makes the comparison.

  gp_m12_iso, gp_m12_ard   Matern 1/2 (exponential), isotropic / ARD
  gp_m32_iso, gp_m32_ard   Matern 3/2
  gp_m52_iso, gp_m52_ard   Matern 5/2 (the campaign kernel is gp_m52_ard)
  gp_rbf_iso, gp_rbf_ard   squared exponential
  gp_rq_iso                rational quadratic (scikit-learn has no ARD form)
  gp_m52_iso_n0,  gp_m52_ard_n0    Matern 5/2 with the noise fixed at zero
                                   (1e-8 in standardised units): the
                                   interpolating Gaussian process
  gp_m52_iso_nmc, gp_m52_ard_nmc   Matern 5/2 with the noise fixed at the
                                   measured Monte Carlo standard deviation
                                   of the output (MC_SIGMA below)
  mlp_ens                  reactor_optimization.MLPEnsembleSurrogate, 5 networks

Every Gaussian process keeps the campaign settings otherwise (ConstantKernel
x kernel + WhiteKernel, standardised inputs, normalize_y, 4 restarts). Each
design is held out once and predicted from the others, exactly as
c8_surrogate_cv.py and c9_surrogate_cv.py do for the campaign kernel.

Outputs scored: the two objectives of each campaign and the cycle length
(objective in Campaign 8, constraint in Campaign 9), on all designs and on
the feasible ones. Metrics: RMSE, R^2, Spearman, coverage of the 1 and 2
sigma intervals, standardised residual s_z (1 means calibrated).

usage (repository root):
    python surrogate_kernel_cv.py                          # every variant, both campaigns
    python surrogate_kernel_cv.py --variants gp_m12_iso gp_rq_iso
Results are merged into figs_surrogate_cv/kernel_cv.json, so a later call
with other variants adds to the table; kernel_cv.md is rewritten from it.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel
from sklearn.preprocessing import StandardScaler

from zoning import spearman

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ARCHIVES = {"C8": "out_c8/optimization_checkpoint.json", "C9": "out_c9/optimization_checkpoint.json"}
OUTPUTS = {"C8": [("cycle_length", "Cycle length [d]"), ("peaking", "F_dH [-]")],
           "C9": [("peaking", "F_dH [-]"), ("c_max", "c_max [ppm]"), ("cycle_length", "Cycle length [d]")]}
# measured seed-to-seed standard deviation of each output at the campaign transport
# settings: F_dH from the Campaign 8 core solves scaled to 100 000 x 170 (thesis,
# Section sec:res-c8-post), cycle length and c_max pooled over the 5 replicated
# designs of c9_dep_replicas (23 degrees of freedom)
MC_SIGMA = {"peaking": 0.014, "cycle_length": 29.1, "c_max": 27.0}


class GP:
    """One GaussianProcessRegressor, campaign settings, chosen kernel and noise.
    noise: "fit" (the campaign), "none" (fixed at 1e-8) or "mc" (fixed at sigma_mc)."""
    def __init__(self, kind, ard, noise="fit", sigma_mc=None):
        self.kind, self.ard, self.noise, self.sigma_mc = kind, ard, noise, sigma_mc
        self.xs = StandardScaler()

    def _kernel(self, d, s_y):
        ls = np.ones(d) if self.ard else 1.0
        lb = (1e-2, 1e3)
        if self.kind == "rq":
            if self.ard:
                raise ValueError("scikit-learn's RationalQuadratic kernel is isotropic only")
            base = RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=lb, alpha_bounds=(1e-2, 1e3))
        else:
            base = {"m12": Matern(length_scale=ls, length_scale_bounds=lb, nu=0.5),
                    "m32": Matern(length_scale=ls, length_scale_bounds=lb, nu=1.5),
                    "m52": Matern(length_scale=ls, length_scale_bounds=lb, nu=2.5),
                    "rbf": RBF(length_scale=ls, length_scale_bounds=lb)}[self.kind]
        if self.noise == "fit":
            white = WhiteKernel(1e-3, (1e-8, 1e1))
        elif self.noise == "none":
            white = WhiteKernel(1e-8, "fixed")
        else:                                   # the output is standardised by s_y inside the regressor
            white = WhiteKernel(max((float(self.sigma_mc) / s_y) ** 2, 1e-8), "fixed")
        return ConstantKernel(1.0, (1e-3, 1e4)) * base + white

    def fit(self, X, y):
        Xs = self.xs.fit_transform(X)
        s_y = float(np.std(y)) or 1.0
        self.gp = GaussianProcessRegressor(kernel=self._kernel(X.shape[1], s_y), normalize_y=True,
                                           n_restarts_optimizer=4, random_state=0).fit(Xs, y)
        return self

    def predict(self, X):
        return self.gp.predict(self.xs.transform(np.atleast_2d(X)), return_std=True)


class MLP:
    """The MLP ensemble of the code base, one output at a time."""
    def fit(self, X, y):
        from reactor_optimization import MLPEnsembleSurrogate
        # the ensemble's predict squeezes a single output to 1-D before the
        # inverse scaling and fails; two identical columns keep it 2-D
        self.m = MLPEnsembleSurrogate().fit(X, np.column_stack([y, y]))
        return self

    def predict(self, X):
        m, s = self.m.predict(np.atleast_2d(X))
        m, s = np.atleast_2d(m), np.atleast_2d(s)
        return m[:, 0], s[:, 0]


# every factory takes the measured Monte Carlo sigma of the output, used by the _nmc variants only
VARIANTS = {
    "gp_m12_iso": lambda s=None: GP("m12", False), "gp_m12_ard": lambda s=None: GP("m12", True),
    "gp_m32_iso": lambda s=None: GP("m32", False), "gp_m32_ard": lambda s=None: GP("m32", True),
    "gp_m52_iso": lambda s=None: GP("m52", False), "gp_m52_ard": lambda s=None: GP("m52", True),
    "gp_rbf_iso": lambda s=None: GP("rbf", False), "gp_rbf_ard": lambda s=None: GP("rbf", True),
    "gp_rq_iso": lambda s=None: GP("rq", False),
    "gp_m52_iso_n0": lambda s=None: GP("m52", False, "none"), "gp_m52_ard_n0": lambda s=None: GP("m52", True, "none"),
    "gp_m52_iso_nmc": lambda s=None: GP("m52", False, "mc", s), "gp_m52_ard_nmc": lambda s=None: GP("m52", True, "mc", s),
    "mlp_ens": lambda s=None: MLP(),
}
LABEL = {"gp_m12_iso": "Matern 1/2 iso", "gp_m12_ard": "Matern 1/2 ARD", "gp_m32_iso": "Matern 3/2 iso",
         "gp_m32_ard": "Matern 3/2 ARD", "gp_m52_iso": "Matern 5/2 iso", "gp_m52_ard": "Matern 5/2 ARD (campaign)",
         "gp_rbf_iso": "SE iso", "gp_rbf_ard": "SE ARD", "gp_rq_iso": "Rational quadratic iso",
         "gp_m52_iso_n0": "Matern 5/2 iso, no noise", "gp_m52_ard_n0": "Matern 5/2 ARD, no noise",
         "gp_m52_iso_nmc": "Matern 5/2 iso, noise = MC", "gp_m52_ard_nmc": "Matern 5/2 ARD, noise = MC",
         "mlp_ens": "MLP ensemble"}


def loo(make, X, y):
    n = len(y)
    mu, sd = np.zeros(n), np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        a, b = make().fit(X[m], y[m]).predict(X[i:i + 1])
        mu[i], sd[i] = float(np.ravel(a)[0]), float(np.ravel(b)[0])
    return mu, sd


def metrics(y, mu, sd):
    e = y - mu
    z = e / np.maximum(sd, 1e-12)
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean(e ** 2))),
                r2=float(1 - np.sum(e ** 2) / ss) if ss > 0 else float("nan"),
                spearman=float(spearman(mu.tolist(), y.tolist())),
                cov1=float(np.mean(np.abs(z) <= 1)), cov2=float(np.mean(np.abs(z) <= 2)),
                sz=float(np.sqrt(np.mean(z ** 2))))


def num(oname, v):
    return f"{v:.4f}" if oname == "peaking" else f"{v:.0f}"


def render(R):
    L = ["# Leave-one-out comparison of surrogates", "",
         f"Fixed-noise variants use the measured Monte Carlo standard deviation: {MC_SIGMA}."]
    for tag in R:
        key = "C9" if "9" in tag else "C8"
        for oname, olab in OUTPUTS[key]:
            if oname not in R[tag]:
                continue
            L += ["", f"### {tag}, {olab}", "",
                  "| Surrogate | Designs | RMSE | R^2 | Spearman | Cov. 1 sigma | Cov. 2 sigma | s_z |", "|---|---|---|---|---|---|---|---|"]
            for v in VARIANTS:
                res = R[tag][oname].get(v)
                if not res:
                    continue
                for sub in ("all", "feasible"):
                    m = res[sub]
                    L.append(f"| {LABEL[v] if sub == 'all' else ''} | {sub} {m['n']} | {num(oname, m['rmse'])} | {m['r2']:.3f} | "
                             f"{m['spearman']:.3f} | {m['cov1']:.2f} | {m['cov2']:.2f} | {m['sz']:.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", nargs="*", default=None, help="checkpoints; default C8 and C9")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default="figs_surrogate_cv")
    a = ap.parse_args()
    arch = ARCHIVES if a.archives is None else {Path(p).parent.name: p for p in a.archives}
    out = Path(a.out); out.mkdir(exist_ok=True)
    store = out / "kernel_cv.json"
    R = json.loads(store.read_text()) if store.exists() else {}
    for tag, path in arch.items():
        D = json.load(open(path)); A = D["all_raw"]; names = D["design_variables"]; CN = D["constraint_names"]
        X = np.array([[r[v] for v in names] for r in A], float)
        feas = np.array([all(r[c] is not None and r[c] <= 0 for c in CN) for r in A])
        key = "C9" if "c9" in path else "C8"
        print(f"== {tag}: {len(A)} designs, {int(feas.sum())} feasible, variables {names}")
        R.setdefault(tag, {})
        for oname, olab in OUTPUTS[key]:
            y = np.array([r[oname] for r in A], float)
            R[tag].setdefault(oname, {})
            for v in a.variants:
                t0 = time.time()
                mu, sd = loo(lambda v=v, s=MC_SIGMA.get(oname): VARIANTS[v](s), X, y)
                res = {"all": metrics(y, mu, sd), "feasible": metrics(y[feas], mu[feas], sd[feas]), "seconds": time.time() - t0}
                R[tag][oname][v] = res
                store.write_text(json.dumps(R, indent=1))          # keep what is done if the run is stopped
                print(f"  {olab:18s} {v:15s} all: RMSE {res['all']['rmse']:.4g} R2 {res['all']['r2']:.3f} "
                      f"rho {res['all']['spearman']:.3f} s_z {res['all']['sz']:.2f} | feasible: RMSE {res['feasible']['rmse']:.4g} "
                      f"R2 {res['feasible']['r2']:.3f} rho {res['feasible']['spearman']:.3f} [{res['seconds']:.0f} s]", flush=True)
    (out / "kernel_cv.md").write_text(render(R), encoding="utf-8")
    print(f"wrote {out}/kernel_cv.json and kernel_cv.md")


if __name__ == "__main__":
    main()
