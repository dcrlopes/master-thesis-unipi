#!/usr/bin/env python3
"""
variable_importance.py -- rank the five design variables by importance, per
response and per decision perspective, from the campaign archive.

TWO COMPLEMENTARY MEASURES
--------------------------
1. STANDARDISED REGRESSION coefficients: each variable is scaled to zero
   mean / unit variance, then a linear model is fitted per response
   (k_BOL, F_dH, EFPD, and a logistic model for feasibility). The absolute
   standardised coefficient is "how many response-sigmas one design-sigma
   buys" -- directly comparable across variables of different units. Cheap,
   transparent, but linear-only.

2. PERMUTATION IMPORTANCE on a Gaussian-Process (GP) surrogate: a GP with
   automatic relevance determination (one length-scale per variable) is
   fitted per response; each column is then shuffled and the drop in
   cross-validated R^2 measures how much the model relied on it. Captures
   nonlinearity and interactions; needs enough points (>=30 recommended).
   The inverse learned length-scales are reported too -- the GP's own
   internal ranking.

The script prints one ranked table per response, then aggregates them into
the three decision perspectives (constraint-limitation, safety-relevant,
operations/refuelling) with explicit weights, so the mapping from raw
sensitivities to perspectives is auditable rather than asserted.

USAGE
  conda activate openmc-env       # needs scikit-learn
  python variable_importance.py \
      --checkpoint out_c3_atf75/optimization_checkpoint.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--n-perm", type=int, default=30,
                help="permutation repeats per variable (default 30)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
ck = json.loads(Path(args.checkpoint).read_text())
dv = ck["design_variables"]
con = ck.get("constraint_names", [])
raw = [r for r in ck["all_raw"] if "k_bol" in r]

X = np.array([[float(r[k]) for k in dv] for r in raw])
resp = {
    "k_bol": np.array([float(r["k_bol"]) for r in raw]),
    "F_dH": np.array([float(r["peaking"]) for r in raw]),
    "EFPD": np.array([float(r["cycle_length"]) for r in raw]),
}
feas = np.array([all(float(r.get(c, 0.0)) <= 1e-9 for c in con)
                 for r in raw], dtype=float)

mu, sd = X.mean(0), X.std(0)
sd[sd == 0] = 1.0
Z = (X - mu) / sd
n, p = Z.shape
print(f"{n} evaluations, {p} variables: {dv}")
print(f"feasible fraction: {feas.mean():.2f}")

# --------------------------------------------------------------------------- #
# 1. standardised linear coefficients (+ logistic for feasibility)             #
# --------------------------------------------------------------------------- #
def std_linear(y):
    ys = (y - y.mean()) / y.std()
    A = np.c_[np.ones(n), Z]
    b, *_ = np.linalg.lstsq(A, ys, rcond=None)
    r2 = 1 - ((ys - A @ b) ** 2).sum() / (ys ** 2).sum()
    return b[1:], r2


def std_logistic(y, iters=500, lr=0.5):
    """Plain Newton-free gradient logistic fit on standardised inputs."""
    A = np.c_[np.ones(n), Z]
    w = np.zeros(p + 1)
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-A @ w))
        w += lr * A.T @ (y - pr) / n
    pr = 1 / (1 + np.exp(-A @ w))
    acc = ((pr > 0.5) == (y > 0.5)).mean()
    return w[1:], acc


lin = {}
print("\n== standardised linear coefficients (|beta| = sigmas of response "
      "per sigma of variable) ==")
for name, y in resp.items():
    b, r2 = std_linear(y)
    lin[name] = np.abs(b)
    order = np.argsort(-np.abs(b))
    print(f"{name:6s} (R2={r2:.3f}): "
          + "  ".join(f"{dv[i]}={b[i]:+.3f}" for i in order))
bl, acc = std_logistic(feas)
lin["feasibility"] = np.abs(bl)
order = np.argsort(-np.abs(bl))
print(f"feasib (acc={acc:.2f}): "
      + "  ".join(f"{dv[i]}={bl[i]:+.3f}" for i in order))

# --------------------------------------------------------------------------- #
# 2. GP permutation importance                                                 #
# --------------------------------------------------------------------------- #
perm = {}
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.model_selection import KFold

    print("\n== GP permutation importance (drop in CV R^2 when the column "
          "is shuffled) ==")
    for name, y in resp.items():
        ys = (y - y.mean()) / y.std()
        kern = (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(p), nu=2.5,
                       length_scale_bounds=(1e-2, 1e3))
                + WhiteKernel(1e-2, (1e-6, 1.0)))
        kf = KFold(n_splits=min(5, n), shuffle=True, random_state=0)
        base, drops = [], np.zeros(p)
        for tr, te in kf.split(Z):
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=False,
                                          n_restarts_optimizer=2,
                                          random_state=0).fit(Z[tr], ys[tr])
            yhat = gp.predict(Z[te])
            ss = ((ys[te] - ys[te].mean()) ** 2).sum()
            b = 1 - ((ys[te] - yhat) ** 2).sum() / ss
            base.append(b)
            for j in range(p):
                d = 0.0
                for _ in range(args.n_perm):
                    Zp = Z[te].copy()
                    Zp[:, j] = rng.permutation(Zp[:, j])
                    yp = gp.predict(Zp)
                    d += b - (1 - ((ys[te] - yp) ** 2).sum() / ss)
                drops[j] += d / args.n_perm
        drops /= kf.get_n_splits()
        perm[name] = np.maximum(drops, 0)
        # ARD length-scales from a full-data fit: small scale = important
        gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=3,
                                      random_state=0).fit(Z, ys)
        ls = gp.kernel_.k1.k2.length_scale
        inv = (1 / np.asarray(ls))
        order = np.argsort(-drops)
        print(f"{name:6s} (CV R2={np.mean(base):.3f}): "
              + "  ".join(f"{dv[i]}={drops[i]:.3f}" for i in order))
        print(f"        ARD 1/lengthscale: "
              + "  ".join(f"{dv[i]}={inv[i]:.2f}"
                          for i in np.argsort(-inv)))
except ImportError:
    print("\n(scikit-learn not available in this environment -- "
          "GP permutation importance skipped; linear results above stand)")

# --------------------------------------------------------------------------- #
# 3. perspective aggregation                                                   #
# --------------------------------------------------------------------------- #
def norm(v):
    v = np.asarray(v, float)
    return v / v.sum() if v.sum() > 0 else v


base_meas = perm if perm else lin
persp = {
    # which responses matter, and with what weight, for each viewpoint
    "constraint-limitation": {"feasibility": 0.6, "k_bol": 0.4},
    "safety-relevant":       {"F_dH": 0.6, "k_bol": 0.4},
    "operations/refuelling": {"EFPD": 0.7, "F_dH": 0.3},
}
print("\n== perspective ranking (weighted, normalised) ==")
print("   weights:", {k: v for k, v in persp.items()})
for pname, wts in persp.items():
    agg = np.zeros(p)
    for rname, w in wts.items():
        src = base_meas.get(rname, lin.get(rname))
        if src is not None:
            agg += w * norm(src)
    order = np.argsort(-agg)
    print(f"{pname:22s}: "
          + "  >  ".join(f"{dv[i]} ({agg[i]:.2f})" for i in order))

print("\nCaveats printed for the record: importance is measured WITHIN the "
      "sampled bounds\n(a variable can be safety-critical yet rank low if "
      "its range was narrow); the 2D BOL\nmodel sees peaking and reactivity "
      "only -- MTC, shutdown margin and transient response\nare outside its "
      "scope; refl_thick acts on feasibility indirectly through the "
      "k_target\ntable (Route B), so its k_bol coefficient understates its "
      "physical role.")
