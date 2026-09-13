#!/usr/bin/env python3
r"""
c9_step0.py
===========
Step 0 validation of the Campaign 9 search, objective-agnostic. No OpenMC.
About a minute.

Two instruments, mirroring c8_surrogate_cv.py and c8_front_stability.py
but reading the objective keys from the archive instead of assuming
cycle length and peaking.

A. Leave-one-out cross-validation of the surrogate.
   The same kernel the framework fits (constant x ARD Matern 5/2 + white
   noise, targets standardised) is refitted 60 times with one design
   held out. Reports R2, RMSE and the 2-sigma coverage per objective, on
   the full archive and on the feasible subset.

B. Front stability under the measurement noise.
   Each archived objective is perturbed B times by its measurement noise
   and the feasible non-dominated set is recomputed. Reports how often
   each design sits on the front, the hypervolume ratio between the
   perturbed and the nominal fronts, and whether the front membership is
   robust.

   Noise defaults: sigma_F = 0.010 (the framework placeholder) and
   sigma_c = 10 ppm (44 pcm eigenvalue noise across two solves at a
   worth near 6.5 pcm/ppm). Override with --sigma.

Usage
  python c9_step0.py --checkpoint out_c9/optimization_checkpoint.json \
      --manifest c9_post/c9_front.json --out c9_post
  python c9_step0.py --selftest        (C8 archive, C8 objectives)
"""
import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def gp():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    return lambda dim: GaussianProcessRegressor(
        kernel=(ConstantKernel(1.0, (1e-3, 1e4))
                * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                + WhiteKernel(1e-3, (1e-8, 1e1))),
        normalize_y=True, n_restarts_optimizer=2, random_state=0)


def loo(X, y, make):
    n = len(y); mu = np.zeros(n); sd = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        g = make(X.shape[1]).fit(X[m], y[m])
        m_, s_ = g.predict(X[i:i + 1], return_std=True)
        mu[i], sd[i] = float(m_[0]), float(s_[0])
    r = y - mu
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(r2=float(r2), rmse=float(np.sqrt((r ** 2).mean())),
                cov2=float(np.mean(np.abs(r) <= 2 * sd)), n=int(n))


def nondom(F):
    n = len(F); keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False; break
    return np.where(keep)[0]


def hv2(F, ref):
    """2-objective hypervolume, minimisation, exact."""
    P = F[nondom(F)] if len(F) else F
    if not len(P): return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, prev = 0.0, ref[1]
    for x, y in P:
        if x >= ref[0] or y >= ref[1]: continue
        hv += (ref[0] - x) * (prev - y); prev = y
    return float(hv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--manifest", default="c9_post/c9_front.json")
    ap.add_argument("--out", default="c9_post")
    ap.add_argument("--objectives", nargs=2, default=None,
                    help="override the archive's objective keys (both minimised)")
    ap.add_argument("--sigma", nargs=2, type=float, default=[0.010, 10.0],
                    help="measurement noise per objective, in objective units")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    d = json.load(open(a.checkpoint))
    raw, cn, dv = d["all_raw"], d["constraint_names"], d["design_variables"]
    keys = a.objectives or [o[0] for o in d["objectives"]]
    sign = [1.0, 1.0] if a.objectives else [(-1.0 if o[1] == "max" else 1.0) for o in d["objectives"]]
    X = np.array([[x[v] for v in dv] for x in raw], float)
    lo, hi = X.min(0), X.max(0); Xn = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    F = np.array([[s * x[k] for s, k in zip(sign, keys)] for x in raw], float)
    feas = np.array([all(x[c] <= 0 for c in cn) for x in raw])
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"objectives": keys, "n": len(raw), "n_feasible": int(feas.sum())}
    L = []

    # ---- A. LOO CV
    make = gp()
    L.append(f"A. leave-one-out GP cross-validation, {len(raw)} designs")
    rep["loo"] = {}
    for j, k in enumerate(keys):
        full = loo(Xn, F[:, j], make)
        sub = loo(Xn[feas], F[feas, j], make) if feas.sum() >= 8 else None
        rep["loo"][k] = dict(all=full, feasible=sub)
        L.append(f"   {k:<10} all: R2 {full['r2']:+.3f} rmse {full['rmse']:.4g} cov2 {full['cov2']:.2f}"
                 + (f" | feasible: R2 {sub['r2']:+.3f} rmse {sub['rmse']:.4g} cov2 {sub['cov2']:.2f}" if sub else ""))

    # ---- B. front stability
    rng = np.random.default_rng(a.seed)
    Ff = F[feas]; ids = np.where(feas)[0]
    nom = set(ids[nondom(Ff)].tolist())
    ref = Ff.max(0) * 1.05 + 1e-9
    hv0 = hv2(Ff, ref)
    count = np.zeros(len(raw)); hvs = []
    sig = np.array(a.sigma)
    for _ in range(a.B):
        Fp = Ff + rng.normal(0, 1, Ff.shape) * sig
        nd = ids[nondom(Fp)]
        count[nd] += 1; hvs.append(hv2(Fp, ref) / hv0 if hv0 > 0 else 0.0)
    freq = count / a.B
    L.append("")
    L.append(f"B. front stability, {a.B} perturbations, sigma {sig.tolist()} in objective units")
    L.append(f"   nominal front {sorted(nom)}  hypervolume ratio perturbed/nominal "
             f"{np.mean(hvs):.3f} +/- {np.std(hvs):.3f}")
    L.append(f"   {'id':>3} {'freq':>5}  on nominal front")
    stable = []
    for i in np.argsort(-freq):
        if freq[i] < 0.05: break
        stable.append(int(i))
        L.append(f"   {i:>3} {freq[i]:5.2f}  {'yes' if i in nom else 'no'}")
    robust = [i for i in nom if freq[i] >= 0.5]
    L.append(f"   nominal members retained in >=50% of resamples: {len(robust)} of {len(nom)}")
    rep["stability"] = dict(nominal_front=sorted(int(i) for i in nom),
                            hv_ratio_mean=float(np.mean(hvs)), hv_ratio_sd=float(np.std(hvs)),
                            membership_freq={int(i): float(freq[i]) for i in stable},
                            robust=sorted(int(i) for i in robust))
    txt = "\n".join(L); print(txt)
    (out / "c9_step0.txt").write_text(txt + "\n")
    (out / "c9_step0.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}/c9_step0.txt and c9_step0.json")
    return 0


def selftest():
    rc = main(["--checkpoint", "out_c8/optimization_checkpoint.json", "--out", "c9_step0_selftest",
               "--sigma", "30", "0.010", "--B", "300"])
    r = json.load(open("c9_step0_selftest/c9_step0.json"))
    assert r["objectives"] == ["cycle_length", "peaking"]
    assert r["loo"]["cycle_length"]["all"]["r2"] > 0.9, r["loo"]           # C8 gave 0.988
    assert set(r["stability"]["nominal_front"]) == {47, 42, 23, 29, 21, 44, 59, 1}
    import shutil; shutil.rmtree("c9_step0_selftest", ignore_errors=True)
    print("\nselftest OK: C8 LOO R2 and front reproduced")
    return rc


if __name__ == "__main__":
    sys.exit(main())
