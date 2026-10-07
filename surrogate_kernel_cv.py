#!/usr/bin/env python3
"""
surrogate_kernel_cv.py -- leave-one-out comparison of surrogate models on the
Campaign 8 and Campaign 9 data, for the paper. No transport.

The thesis (Section sec:bg-gp) states that the kernel, a Matern 5/2 with one
length-scale per variable and a white-noise term, was not compared against
other kernels. This script makes the comparison the online review asked for:

  gp_m32_iso, gp_m32_ard   Matern 3/2, isotropic / one length-scale per variable
  gp_m52_iso, gp_m52_ard   Matern 5/2 (the campaign kernel is gp_m52_ard)
  gp_rbf_iso, gp_rbf_ard   squared exponential
  mlp_ens                  reactor_optimization.MLPEnsembleSurrogate, 5 networks

Every Gaussian process keeps the campaign settings otherwise (ConstantKernel
x kernel + WhiteKernel, standardised inputs, normalize_y, 4 restarts). Each
design is held out once and predicted from the others, exactly as
c8_surrogate_cv.py and c9_surrogate_cv.py do for the campaign kernel.

Outputs scored: the two objectives of each campaign and the cycle length
(objective in Campaign 8, constraint in Campaign 9), on all designs and on
the feasible ones. Metrics: RMSE, R^2, Spearman, coverage of the 1 and 2
sigma intervals, standardised residual s_z (1 means calibrated).

usage (repository root, a few minutes):
    python surrogate_kernel_cv.py                      # both campaigns
    python surrogate_kernel_cv.py --archives out_c9/optimization_checkpoint.json
Writes figs_surrogate_cv/kernel_cv.json and kernel_cv.md.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from zoning import spearman

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ARCHIVES = {"C8": "out_c8/optimization_checkpoint.json", "C9": "out_c9/optimization_checkpoint.json"}
OUTPUTS = {"C8": [("cycle_length", "Cycle length [d]"), ("peaking", "F_dH [-]")],
           "C9": [("peaking", "F_dH [-]"), ("c_max", "c_max [ppm]"), ("cycle_length", "Cycle length [d]")]}


class GP:
    """One GaussianProcessRegressor per output, campaign settings, chosen kernel."""
    def __init__(self, kind, ard):
        self.kind, self.ard = kind, ard
        self.xs = StandardScaler()

    def _kernel(self, d):
        ls = np.ones(d) if self.ard else 1.0
        base = {"m32": Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=1.5),
                "m52": Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5),
                "rbf": RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3))}[self.kind]
        return ConstantKernel(1.0, (1e-3, 1e4)) * base + WhiteKernel(1e-3, (1e-8, 1e1))

    def fit(self, X, y):
        Xs = self.xs.fit_transform(X)
        self.gp = GaussianProcessRegressor(kernel=self._kernel(X.shape[1]), normalize_y=True,
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


VARIANTS = {"gp_m32_iso": lambda: GP("m32", False), "gp_m32_ard": lambda: GP("m32", True),
            "gp_m52_iso": lambda: GP("m52", False), "gp_m52_ard": lambda: GP("m52", True),
            "gp_rbf_iso": lambda: GP("rbf", False), "gp_rbf_ard": lambda: GP("rbf", True),
            "mlp_ens": lambda: MLP()}


def loo(make, X, y):
    n = len(y)
    mu, sd = np.zeros(n), np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        a, b = make().fit(X[m], y[m]).predict(X[i:i + 1])
        mu[i], sd[i] = float(a[0]), float(b[0])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", nargs="*", default=None, help="checkpoints; default C8 and C9")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default="figs_surrogate_cv")
    a = ap.parse_args()
    arch = ARCHIVES if a.archives is None else {Path(p).parent.name: p for p in a.archives}
    out = Path(a.out); out.mkdir(exist_ok=True)
    R, lines = {}, []
    for tag, path in arch.items():
        D = json.load(open(path)); A = D["all_raw"]; names = D["design_variables"]; CN = D["constraint_names"]
        X = np.array([[r[v] for v in names] for r in A], float)
        feas = np.array([all(r[c] is not None and r[c] <= 0 for c in CN) for r in A])
        key = "C9" if "c9" in path else "C8"
        print(f"== {tag}: {len(A)} designs, {int(feas.sum())} feasible, variables {names}")
        R[tag] = {}
        for oname, olab in OUTPUTS[key]:
            y = np.array([r[oname] for r in A], float)
            R[tag][oname] = {}
            lines.append(f"\n### {tag}, {olab}\n")
            lines.append("| Surrogate | Designs | RMSE | R^2 | Spearman | Cov. 1 sigma | Cov. 2 sigma | s_z | Time [s] |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for v in a.variants:
                t0 = time.time()
                mu, sd = loo(VARIANTS[v], X, y)
                dt = time.time() - t0
                res = {"all": metrics(y, mu, sd), "feasible": metrics(y[feas], mu[feas], sd[feas]), "seconds": dt}
                R[tag][oname][v] = res
                for sub, m in (("all", res["all"]), ("feasible", res["feasible"])):
                    fmt = "{:.4g}" if oname == "peaking" else "{:.0f}"
                    lines.append(f"| {v if sub == 'all' else ''} | {sub} {m['n']} | {fmt.format(m['rmse'])} | {m['r2']:.3f} | "
                                 f"{m['spearman']:.3f} | {m['cov1']:.2f} | {m['cov2']:.2f} | {m['sz']:.2f} | "
                                 f"{dt:.0f} |" if sub == "all" else
                                 f"|  | {sub} {m['n']} | {fmt.format(m['rmse'])} | {m['r2']:.3f} | {m['spearman']:.3f} | "
                                 f"{m['cov1']:.2f} | {m['cov2']:.2f} | {m['sz']:.2f} |  |")
                print(f"  {olab:18s} {v:12s} all: RMSE {res['all']['rmse']:.4g} R2 {res['all']['r2']:.3f} "
                      f"rho {res['all']['spearman']:.3f} s_z {res['all']['sz']:.2f} | feasible: RMSE {res['feasible']['rmse']:.4g} "
                      f"R2 {res['feasible']['r2']:.3f} rho {res['feasible']['spearman']:.3f} [{dt:.0f} s]", flush=True)
    (out / "kernel_cv.json").write_text(json.dumps(R, indent=1))
    (out / "kernel_cv.md").write_text("# Leave-one-out comparison of surrogates\n" + "\n".join(lines) + "\n")
    print(f"wrote {out}/kernel_cv.json and kernel_cv.md")


if __name__ == "__main__":
    main()
