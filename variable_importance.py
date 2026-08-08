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

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Kernel bounds are deliberately WIDE. A length-scale driven to its upper
    # bound means the GP found the variable IRRELEVANT (flat in that
    # direction) -- informative, not an error. With a tight bound the
    # optimiser stops AT the wall and sklearn emits ConvergenceWarning; wide
    # bounds let it converge to the true (large) value. Likewise the noise
    # floor: k_bol at 16000x120 is so well converged that the GP legitimately
    # wants near-zero noise.
    LS_BOUNDS = (1e-2, 1e6)
    NOISE_BOUNDS = (1e-10, 1.0)
    SAT = 1e5          # length-scale above this => treat as irrelevant

    print("\n== GP permutation importance ==")
    print("   R^2 is computed ONCE on pooled OUT-OF-FOLD predictions, not "
          "per fold:\n   with --max-burnup 75 many designs pin at exactly the "
          "cap, so a single\n   fold can hold only capped designs -> zero "
          "within-fold variance -> -inf.")
    for name, y in resp.items():
        spread = y.std()
        if spread == 0:
            print(f"{name:6s}: constant response, skipped")
            continue
        ys = (y - y.mean()) / spread
        n_tied = int((np.abs(y - y.max()) < 1e-6).sum())
        kern = (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(p), nu=2.5,
                       length_scale_bounds=LS_BOUNDS)
                + WhiteKernel(1e-2, NOISE_BOUNDS))
        kf = KFold(n_splits=min(5, n), shuffle=True, random_state=0)

        oof = np.full(n, np.nan)                       # out-of-fold baseline
        oof_perm = np.full((p, n), np.nan)             # ... per permuted column
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for tr, te in kf.split(Z):
                gp = GaussianProcessRegressor(kernel=kern, normalize_y=False,
                                              n_restarts_optimizer=2,
                                              random_state=0).fit(Z[tr], ys[tr])
                oof[te] = gp.predict(Z[te])
                for j in range(p):
                    acc = np.zeros(len(te))
                    for _ in range(args.n_perm):
                        Zp = Z[te].copy()
                        Zp[:, j] = rng.permutation(Zp[:, j])
                        acc += gp.predict(Zp)
                    oof_perm[j, te] = acc / args.n_perm

        ss_tot = ((ys - ys.mean()) ** 2).sum()         # pooled: never zero
        r2 = 1 - ((ys - oof) ** 2).sum() / ss_tot
        drops = np.array([r2 - (1 - ((ys - oof_perm[j]) ** 2).sum() / ss_tot)
                          for j in range(p)])
        perm[name] = np.maximum(drops, 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=3,
                                          random_state=0).fit(Z, ys)
        ls = np.asarray(gp.kernel_.k1.k2.length_scale, dtype=float)
        noise = float(gp.kernel_.k2.noise_level)
        irrel = [dv[i] for i in range(p) if ls[i] > SAT]

        flag = ""
        if r2 < 0:
            flag = "  ** worse than the mean: too few points, ignore this row **"
        elif n_tied > 0.25 * n:
            flag = (f"  ** {n_tied}/{n} designs tied at the maximum "
                    f"(objective saturated) **")
        print(f"{name:6s} (pooled out-of-fold R2={r2:+.3f}){flag}")
        print("        importance: "
              + "  ".join(f"{dv[i]}={drops[i]:.3f}"
                          for i in np.argsort(-drops)))
        print(f"        fitted noise={noise:.2e}   "
              + ("GP-irrelevant (length-scale saturated): "
                 + ", ".join(irrel) if irrel else "no irrelevant variables"))
except ImportError:
    print("\n(scikit-learn not available in this environment -- "
          "GP permutation importance skipped; linear results above stand)")

# --------------------------------------------------------------------------- #
# 3. perspective aggregation                                                   #
# --------------------------------------------------------------------------- #
def norm(v):
    """Normalise to sum 1, treating NaN/inf as zero so one unusable response
    cannot poison an entire perspective ranking."""
    v = np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=0.0, neginf=0.0)
    v = np.maximum(v, 0.0)
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
