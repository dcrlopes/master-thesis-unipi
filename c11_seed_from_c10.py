#!/usr/bin/env python3
"""
c11_seed_from_c10.py -- the design of experiments of Campaign 11: the 58
evaluations of Campaign 10 with their three-dimensional cycle length
recomputed under the corrected end-of-cycle criterion.

The criterion k_3D(BOL) x k_target / k_bol used the campaign assembly k_bol
(4000 x 60, 264 pcm between seeds), which put about 43 d of error into every
cycle length (paper_finds.md Section 12). For every design that has a
high-statistics k_bol in repro_c9/kbol_c10_<i>_hifi.json (20 000 x 160,
c9_repro_check.py) the cycle length, the constraint g_efpd and the censoring
are recomputed from the stored k history with the corrected ratio. Designs
without one are listed in c11_kbol_jobs.txt for run_kbol_queue.sh.

Also computes the kappa of the margin rule for g_efpd and g_kmin from the
measured leave-one-out error of the Matern 1/2 constraint surrogate:
kappa = RMSE(LOO) / mean(sigma_GP), so that kappa x sigma is the error the
surrogate actually makes, written to c11_kappa.json.

usage:  python c11_seed_from_c10.py [--require-all]
writes c11_seed.json (evaluator cache format, for --doe-from-cache),
       c11_kbol_jobs.txt, c11_kappa.json
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import core_geometry as cg

CKPT = "out_c10/optimization_checkpoint.json"
CACHE = "openmc_runs_c10/eval_cache.json"
EFPD_REQ = 1826.0
SPEC_POWER = 9.983354024602374      # W/gHM, the campaign value (c9_dep_core3d runs.json)
VARS = ["enrich", "gd_wt", "refl_thick", "gd_pins"]


def corrected_cycle(rec, k_hifi):
    bu, k = rec["bu_hist_3d"], rec["k_hist_3d"]
    ratio = rec["k_target"] / k_hifi
    k_eoc = k[0] * ratio
    b = cg.eoc_crossing_burnup(bu, k, k_eoc)
    censored = False
    if b is None:
        if k[-1] > k_eoc:
            b, censored = bu[-1], True          # still above the criterion at the horizon
        else:
            b = 0.0                             # never above it
    efpd = b * 1000.0 / SPEC_POWER
    return dict(cycle_length=efpd, efpd_3d=efpd, g_efpd=EFPD_REQ - efpd, censored_3d=censored,
                k_bol_hifi=k_hifi, k_target_ratio_3d=ratio, k_bol_campaign=rec["k_bol"],
                kbol_correction="cycle length recomputed with the high-statistics k_bol (c11_seed_from_c10.py)")


def kappa_from_loo(X, G, names, nu=0.5):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    Xs = (X - X.mean(0)) / X.std(0)
    out = {}
    for name in names:
        y = G[name]
        pred, std = np.zeros(len(y)), np.zeros(len(y))
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            kern = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(np.ones(X.shape[1]), (1e-2, 1e3), nu=nu) + WhiteKernel(1e-3, (1e-8, 1e1))
            gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=4, random_state=0).fit(Xs[m], y[m])
            p, s = gp.predict(Xs[i:i + 1], return_std=True); pred[i], std[i] = p[0], s[0]
        rmse = float(np.sqrt(((y - pred) ** 2).mean()))
        out[name] = dict(rmse=rmse, mean_sigma=float(std.mean()), kappa=rmse / float(std.mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-all", action="store_true", help="exit 2 if any design lacks a high-statistics k_bol")
    ap.add_argument("--no-kappa", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    ck = json.load(open(CKPT)); R = ck["all_raw"]
    cache = json.load(open(CACHE))
    by_design = {tuple(round(json.loads(key)[v], 6) for v in VARS): (key, val) for key, val in cache.items()}
    seed, missing, corrected = {}, [], 0
    for i, r in enumerate(R):
        key, val = by_design[tuple(round(r[v], 6) for v in VARS)]
        rec = dict(val)
        p = Path(f"repro_c9/kbol_c10_{i}_hifi.json")
        if p.exists() and rec.get("k_hist_3d"):
            rec.update(corrected_cycle(rec, float(json.load(open(p))["k"][0]))); corrected += 1
        elif not p.exists():
            missing.append(i)
        seed[key] = rec
    Path("c11_kbol_jobs.txt").write_text("".join(f"{CKPT}:{i}\n" for i in missing))
    Path("c11_seed.json").write_text(json.dumps(seed))
    print(f"{len(seed)} designs in c11_seed.json, {corrected} with the corrected criterion, "
          f"{len(missing)} without a high-statistics k_bol (c11_kbol_jobs.txt)")
    if missing and a.require_all:
        print("not all designs are corrected; run the k_bol queue first", file=sys.stderr); return 2
    if not a.no_kappa:
        X = np.array([[r[v] for v in VARS] for r in R])
        G = {"g_efpd": np.array([seed[by_design[tuple(round(r[v], 6) for v in VARS)][0]]["g_efpd"] for r in R]),
             "g_kmin": np.array([r["g_kmin"] for r in R])}
        kap = kappa_from_loo(X, G, ["g_efpd", "g_kmin"])
        for n, k in kap.items():
            print(f"  {n}: leave-one-out RMSE {k['rmse']:.4g}, mean GP sigma {k['mean_sigma']:.4g} -> kappa {k['kappa']:.2f}")
        Path("c11_kappa.json").write_text(json.dumps(kap, indent=1))
        print("  suggested: --feas-kappa-per " + " ".join(f"{n}={k['kappa']:.2f}" for n, k in kap.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
