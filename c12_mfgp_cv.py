#!/usr/bin/env python3
r"""
c12_mfgp_cv.py -- two-fidelity Gaussian processes (autoregressive co-kriging,
Kennedy and O'Hagan, in the recursive form of Le Gratiet) on the Campaign 12
checkpoint. No transport, scikit-learn only.

FIDELITY PAIRS IN THE CHECKPOINT
    g_kmin    low:  g_kmin_asm, the infinite-lattice assembly solve (median 11 s)
              high: g_kmin_core, the core solve (median 110 s), 117 designs
    peaking   low:  peaking_asm          high: peaking (core), 117 designs
    g_margin  low:  the proxy depletion (g_margin_proxy, or g_margin where the
                    design was not confirmed), 117 designs
              high: the full-fidelity two-seed depletion, the 15 confirmed designs

MODEL
    y_H(x) = rho * y_L(x) + delta(x), with y_L and delta independent GPs
    (Matern 5/2 ARD, the campaign kernel). rho is the least-squares slope of
    y_H on y_L at the high-fidelity training designs.
    sf         the campaign GP on the high-fidelity data only (the baseline)
    ar1        the low fidelity of the predicted design is unknown: a candidate
               not yet evaluated. mu = rho mu_L + mu_d, var = rho^2 s_L^2 + s_d^2
    ar1_known  the low fidelity of the predicted design is computed first (an
               assembly solve before the candidate is accepted, or the proxy
               before the confirmation): mu = rho y_L + mu_d, var = s_d^2
    proxy      g_margin only: the proxy value itself, sd = RMS of (high - low)
               on the other confirmed designs (what the campaign assumes)

VALIDATION
    g_kmin, peaking: leave-one-out (117) and walk-forward (the 18 infill designs).
    g_margin: leave-one-out on the 15 confirmed designs. The 14 others give the
    high-fidelity training data, the low fidelity is trained on the other 116.
    These 15 designs were selected near the front, so the result holds there.

USAGE (repository root, a few minutes)
    python c12_mfgp_cv.py
    python c12_mfgp_cv.py --selftest
Writes <out>/mfgp_cv.json and mfgp_cv.md.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from c12_surrogate_cv import decisions
from surrogate_kernel_cv import GP, metrics

CKPT = "results_wks720_2026-11-15/c12/out_c12/optimization_checkpoint.json"
PAIRS = {"g_kmin": ("g_kmin_asm", "g_kmin_core", 1e5), "peaking": ("peaking_asm", "peaking", 1.0)}


class AR1:
    def fit(self, X, yl, yh, il):
        """X, yl: low-fidelity designs. yh at X[il], the high-fidelity designs (nested)."""
        warnings.filterwarnings("ignore")
        self.gl = GP("m52", True).fit(X, yl)
        A = np.column_stack([yl[il], np.ones(len(il))])
        self.rho = float(np.linalg.lstsq(A, yh, rcond=None)[0][0])
        self.gd = GP("m52", True).fit(X[il], yh - self.rho * yl[il])
        return self

    def predict(self, X, yl_known=None):
        md, sd = self.gd.predict(X)
        if yl_known is not None:
            return self.rho * yl_known + md, sd
        ml, sl = self.gl.predict(X)
        return self.rho * ml + md, np.sqrt((self.rho * sl) ** 2 + sd ** 2)


def _fold(X, yl, yh, hf, tr, te):
    """tr, te: indices of designs. Low fidelity on tr, high fidelity on tr & hf."""
    warnings.filterwarnings("ignore")
    il = np.array([k for k, i in enumerate(tr) if hf[i]])
    m = AR1().fit(X[tr], yl[tr], yh[tr][il], il)
    sf = GP("m52", True).fit(X[tr][il], yh[tr][il])
    out = {"sf": sf.predict(X[te]), "ar1": m.predict(X[te]), "ar1_known": m.predict(X[te], yl[te])}
    d = yh[tr][il] - yl[tr][il]
    out["proxy"] = (yl[te], np.full(len(te), np.sqrt(np.mean(d ** 2))))
    out["rho"] = (np.full(len(te), m.rho), np.zeros(len(te)))
    return out


def cv(X, yl, yh, hf, folds, jobs):
    res = Parallel(n_jobs=jobs)(delayed(_fold)(X, yl, yh, hf, tr, te) for tr, te in folds)
    P = {}
    for (tr, te), r in zip(folds, res):
        for k, (a, b) in r.items():
            mu, sd = P.setdefault(k, (np.full(len(yh), np.nan), np.full(len(yh), np.nan)))
            mu[te], sd[te] = a, b
    return P


def scored(name, yh, P, te, feas, kappa, models):
    R = {}
    for k in models:
        mu, sd = P[k]
        r = {"all": metrics(yh[te], mu[te], sd[te])}
        if (te & feas).sum() > 2:
            r["feasible"] = metrics(yh[te & feas], mu[te & feas], sd[te & feas])
        if kappa:
            r["decisions"] = decisions(yh[te], mu[te], sd[te], kappa)
        R[k] = r
    R["rho"] = [float(np.nanmin(P["rho"][0][te])), float(np.nanmax(P["rho"][0][te]))]
    return R


LABEL = {"sf": "GP on the high fidelity only (campaign)", "ar1": "Co-kriging, low fidelity predicted",
         "ar1_known": "Co-kriging, low fidelity computed", "proxy": "Proxy value as the prediction"}


def render(R):
    L = ["# Campaign 12: two-fidelity Gaussian processes", ""]
    for key, rows in R.items():
        unit = "" if key.startswith("peaking") else " [pcm]"
        fmt = (lambda x: f"{x:.4f}") if key.startswith("peaking") else (lambda x: f"{x:.0f}")
        con = any("decisions" in v for v in rows.values() if isinstance(v, dict))
        L += ["", f"### {key}{unit}, rho between {rows['rho'][0]:.3f} and {rows['rho'][1]:.3f}", "",
              "| Model | RMSE | R^2 | Cov. 2 sigma | s_z | RMSE feasible |" + (" Mean rule FP/FN | Margin rule FP/FN |" if con else ""),
              "|---|---|---|---|---|---|" + ("---|---|" if con else "")]
        for k, r in rows.items():
            if k == "rho":
                continue
            m, f, d = r["all"], r.get("feasible"), r.get("decisions", {})
            line = (f"| {LABEL[k]} | {fmt(m['rmse'])} | {m['r2']:.3f} | {m['cov2']:.2f} | {m['sz']:.2f} | "
                    f"{fmt(f['rmse']) if f else '--'} |")
            if con:
                line += f" {d['mean_fp']}/{d['mean_fn']} | {d.get('margin_fp', '-')}/{d.get('margin_fn', '-')} |"
            L.append(line)
        if con:
            d0 = next(v for v in rows.values() if isinstance(v, dict))["decisions"]
            L += ["", f"{d0['n']} designs, {d0['n_violated']} violate the constraint."]
    return "\n".join(L) + "\n"


def selftest():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(60, 2))
    yl = np.sin(6 * X[:, 0]) + X[:, 1]
    yh = 2.0 * yl + 0.3 * X[:, 0] + 0.01 * rng.normal(size=60)
    hf = np.zeros(60, bool)
    hf[::3] = True
    folds = [(np.delete(np.arange(60), i), np.array([i])) for i in np.flatnonzero(hf)]
    P = cv(X, yl, yh, hf, folds, 1)
    te = hf
    e = {k: np.sqrt(np.mean((yh[te] - P[k][0][te]) ** 2)) for k in ("sf", "ar1", "ar1_known")}
    assert abs(np.nanmean(P["rho"][0][te]) - 2.0) < 0.1, P["rho"][0][te]
    assert e["ar1_known"] <= 1.05 * e["ar1"] and e["ar1"] < e["sf"], e
    print("selftest passed", {k: round(v, 4) for k, v in e.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="figs_c12_cv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    D = json.loads(Path(a.checkpoint).read_text())
    A, names, CN = D["all_raw"], D["design_variables"], D["constraint_names"]
    kmap = D["meta"]["surrogate_policy"]["feas_kappa_map"]
    X = np.array([[r[v] for v in names] for r in A], float)
    feas = np.array([all(r[c] <= 0 for c in CN) for r in A])
    n = len(A)
    loo = [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
    n0, wf = 0, []
    for p in D["phase_log"]:
        m = int(p["n_eval"])
        if p["stage"] != "DOE":
            wf.append((np.arange(n0), np.arange(n0, n0 + m)))
        n0 += m

    R, every = {}, np.ones(n, bool)
    for out, (lo, hi, unit) in PAIRS.items():
        yl = np.array([r[lo] for r in A], float) * unit
        yh = np.array([r[hi] for r in A], float) * unit
        for scheme, folds in (("loo", loo), ("walkforward", wf)):
            P = cv(X, yl, yh, every, folds, a.jobs)
            te = np.isfinite(P["sf"][0])
            R[f"{out}, {scheme}"] = scored(out, yh, P, te, feas, kmap.get(out), ("sf", "ar1", "ar1_known"))
            print(f"{out:8s} {scheme:11s} " + " ".join(f"{k} {v['all']['rmse']:.4g}/{v['all']['sz']:.2f}"
                                                     for k, v in R[f'{out}, {scheme}'].items() if k != "rho"), flush=True)

    hf = np.array([bool(r.get("confirmed_3d")) for r in A])
    yl = np.array([r.get("g_margin_proxy", r["g_margin"]) if r.get("confirmed_3d") else r["g_margin"] for r in A], float) * 1e5
    yh = np.array([r["g_margin"] for r in A], float) * 1e5
    folds = [(np.delete(np.arange(n), i), np.array([i])) for i in np.flatnonzero(hf)]
    P = cv(X, yl, yh, hf, folds, a.jobs)
    R["g_margin, leave-one-out on the confirmed designs"] = scored(
        "g_margin", yh, P, hf, feas, kmap.get("g_margin"), ("sf", "ar1", "ar1_known", "proxy"))
    print("g_margin " + " ".join(f"{k} {v['all']['rmse']:.0f}/{v['all']['sz']:.2f}"
                                 for k, v in R["g_margin, leave-one-out on the confirmed designs"].items() if k != "rho"))

    o = Path(a.out)
    o.mkdir(exist_ok=True)
    (o / "mfgp_cv.json").write_text(json.dumps(R, indent=1))
    (o / "mfgp_cv.md").write_text(render(R), encoding="utf-8")
    print(f"wrote {o}/mfgp_cv.json and mfgp_cv.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
