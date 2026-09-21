#!/usr/bin/env python3
r"""
c10_fz_gate.py -- is the axial peaking factor F_z a usable optimisation
objective for this core? Archive-only, no transport, seconds to run.

WHY
    None of the four design variables (enrich, gd_wt, refl_thick, gd_pins)
    acts along z. The fuel is axially uniform and reactor_model.Operating
    carries one fuel and one moderator temperature, so at beginning of life
    the axial shape is the fundamental mode of a 120 cm column with grid
    depressions. On the seven designs of axial_c9 F_z spans 1.438 to 1.446.
    This script measures whether that is design signal or Monte Carlo noise,
    on as many designs as axial_shape_c9.py has solved.

WHAT IT REPORTS
    1. F_z per design and seed, with the SAME estimator as the thesis table:
       derive() is imported from axial_shape_c9.py, not re-implemented.
    2. Seed standard deviation of a single solve, pooled from every design
       that has two or more seeds (paired differences).
    3. Variance decomposition of the first-seed F_z across designs:
       var_observed = var_true + var_seed. Signal-to-noise = sd_true / sd_seed.
    4. Spearman rank correlation of F_z with each design variable and with
       the archived objectives.
    5. Leave-one-out R2 of a Gaussian process on F_z, with the SAME kernel
       and scaling as c9_step0.py (gp() and loo() are imported from it), next
       to the same number for the archived radial factor, as the yardstick.
    6. Whether the product F_dH x F_z reorders the designs against F_dH alone.
    7. A verdict against two thresholds, both printed, both overridable.

INPUT
    <axial-dir>/d<idx>/<STATE>_s<seed>.npz   written by axial_shape_c9.py
    <checkpoint>                             out_c9/optimization_checkpoint.json

USAGE (any machine with numpy, scipy, scikit-learn, from the repository root)
    python c10_fz_gate.py --selftest
    python c10_fz_gate.py --axial-dir axial_c9_all \
        --checkpoint out_c9/optimization_checkpoint.json --out c10_gate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SNR_MIN = 2.0      # sd_true / sd_seed below this: a single solve cannot rank designs
R2_MIN = 0.50      # leave-one-out R2 below this: the surrogate has nothing to learn


def load_fz(axial_dir: Path, state: str):
    """{idx: {seed: dict(fz, fdh, ao)}} with the estimator of axial_shape_c9."""
    from axial_shape_c9 import derive
    out = {}
    for ddir in sorted(axial_dir.glob("d*")):
        m = re.fullmatch(r"d(\d+)", ddir.name)
        if not m:
            continue
        idx = int(m.group(1))
        for npz in sorted(ddir.glob(f"{state}_s*.npz")):
            seed = int(re.fullmatch(rf"{state}_s(\d+)\.npz", npz.name).group(1))
            z = np.load(npz)
            dv = derive(z["asm"], z["pin"], z["ax"], z["edges"], 48.0)
            out.setdefault(idx, {})[seed] = dict(fz=dv["fz"], fdh=dv["fdh"], ao=dv["ao"])
    return out


def pooled_seed_sd(fz):
    """Single-solve s.d. from designs with >= 2 seeds. Returns (sd, dof)."""
    ss, dof = 0.0, 0
    for seeds in fz.values():
        v = np.array([s["fz"] for s in seeds.values()])
        if len(v) >= 2:
            ss += ((v - v.mean()) ** 2).sum(); dof += len(v) - 1
    return (float(np.sqrt(ss / dof)), dof) if dof else (float("nan"), 0)


def analyse(fz, ckpt, seed_sd_override=None):
    from scipy.stats import spearmanr
    from c9_step0 import gp, loo
    raw, cn, dv = ckpt["all_raw"], ckpt["constraint_names"], ckpt["design_variables"]
    idx = sorted(fz)
    first = np.array([fz[i][min(fz[i])]["fz"] for i in idx])       # one solve per design,
    fdh3 = np.array([fz[i][min(fz[i])]["fdh"] for i in idx])       # as a campaign would see it
    X = np.array([[raw[i][v] for v in dv] for i in idx], float)
    feas = np.array([all(raw[i][c] <= 0 for c in cn) for i in idx])

    sd_seed, dof = pooled_seed_sd(fz)
    if seed_sd_override is not None:
        sd_seed, dof = float(seed_sd_override), 0
    var_obs = float(first.var(ddof=1)) if len(first) > 1 else float("nan")
    var_true = var_obs - sd_seed ** 2
    sd_true = float(np.sqrt(var_true)) if var_true > 0 else 0.0
    rep = dict(n_designs=len(idx), n_feasible=int(feas.sum()), designs=idx,
               fz_min=float(first.min()), fz_max=float(first.max()), fz_mean=float(first.mean()),
               fz_span_pct=float(100 * (first.max() - first.min()) / first.mean()),
               fdh3d_span_pct=float(100 * (fdh3.max() - fdh3.min()) / fdh3.mean()),
               sd_seed=sd_seed, sd_seed_dof=dof, sd_observed=float(np.sqrt(var_obs)),
               sd_true=sd_true, snr=float(sd_true / sd_seed) if sd_seed > 0 else float("nan"))

    rep["spearman"] = {}
    cols = [(v, X[:, j]) for j, v in enumerate(dv)]
    cols += [(k, np.array([raw[i][k] for i in idx], float)) for k in ("peaking", "c_max", "keff_core_bol")
             if all(k in raw[i] for i in idx)]
    for name, col in cols:
        if np.ptp(col) > 0 and len(idx) >= 4:
            r, p = spearmanr(col, first)
            rep["spearman"][name] = dict(rho=float(r), p=float(p))

    rep["loo"] = None
    if len(idx) >= 12:
        lo, hi = X.min(0), X.max(0); Xn = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
        make = gp()
        rep["loo"] = dict(fz=loo(Xn, first, make), fdh3d=loo(Xn, fdh3, make))

    fq = fdh3 * first
    rep["fq_vs_fdh_spearman"] = float(spearmanr(fq, fdh3)[0]) if len(idx) >= 4 else None
    rep["fq_max_rank_shift"] = int(np.abs(np.argsort(np.argsort(fq)) - np.argsort(np.argsort(fdh3))).max())

    r2 = rep["loo"]["fz"]["r2"] if rep["loo"] else None
    rep["thresholds"] = dict(snr_min=SNR_MIN, r2_min=R2_MIN)
    rep["usable_objective"] = bool(rep["snr"] >= SNR_MIN and (r2 is None or r2 >= R2_MIN))
    rep["table"] = [dict(idx=int(i), **{v: float(raw[i][v]) for v in dv}, feasible=bool(f),
                         fz=[float(fz[i][s]["fz"]) for s in sorted(fz[i])], fdh3d=float(h))
                    for i, f, h in zip(idx, feas, fdh3)]
    return rep


def report(rep):
    L = [f"designs with an axial solve : {rep['n_designs']}  ({rep['n_feasible']} feasible in the archive)",
         f"F_z, first seed             : {rep['fz_min']:.4f} to {rep['fz_max']:.4f}, mean {rep['fz_mean']:.4f}, "
         f"span {rep['fz_span_pct']:.2f} %",
         f"3D F_dH on the same solves  : span {rep['fdh3d_span_pct']:.1f} %",
         f"seed s.d. of one solve      : {rep['sd_seed']:.4f}  ({rep['sd_seed_dof']} degrees of freedom)",
         f"observed s.d. across designs: {rep['sd_observed']:.4f}",
         f"implied true s.d.           : {rep['sd_true']:.4f}",
         f"signal-to-noise, one solve  : {rep['snr']:.2f}   (threshold {SNR_MIN})", ""]
    L.append("Spearman rank correlation with F_z")
    for k, v in rep["spearman"].items():
        L.append(f"  {k:14s} rho {v['rho']:+.3f}   p {v['p']:.3f}")
    if rep["loo"]:
        a, b = rep["loo"]["fz"], rep["loo"]["fdh3d"]
        L += ["", f"leave-one-out R2, F_z       : {a['r2']:+.3f}   rmse {a['rmse']:.4f}   (threshold {R2_MIN})",
              f"leave-one-out R2, 3D F_dH   : {b['r2']:+.3f}   rmse {b['rmse']:.4f}   (yardstick)"]
    else:
        L += ["", "leave-one-out R2            : skipped, fewer than 12 designs"]
    L += ["", f"F_dH x F_z against F_dH     : Spearman {rep['fq_vs_fdh_spearman']:.3f}, "
              f"largest rank shift {rep['fq_max_rank_shift']} places", "",
          "VERDICT: F_z " + ("IS" if rep["usable_objective"] else "is NOT")
          + " a usable objective under the two thresholds above."]
    return "\n".join(L)


def selftest():
    """Synthetic: a flat noisy F_z must fail the gate, a real trend must pass."""
    rng = np.random.default_rng(0)
    dv = ["enrich", "gd_wt", "refl_thick", "gd_pins"]
    lo, hi = np.array([2, 0, 2.0, 12]), np.array([17.17, 8, 5.66, 40])
    X = lo + rng.uniform(size=(40, 4)) * (hi - lo)
    raw = [dict(zip(dv, x), peaking=1.5 + 0.02 * x[0], c_max=1500.0 + 100 * x[0],
                keff_core_bol=1.0 + 0.01 * x[0], g=-1.0) for x in X]
    ck = dict(all_raw=raw, constraint_names=["g"], design_variables=dv)

    def fake(trend):
        return {i: {s: dict(fz=1.44 + trend * (X[i, 0] - 9.0) + rng.normal(0, 0.0026),
                            fdh=1.5 + 0.02 * X[i, 0] + rng.normal(0, 0.01), ao=-0.02)
                    for s in (1, 2)} for i in range(40)}
    flat, real = analyse(fake(0.0), ck), analyse(fake(0.004), ck)
    assert abs(flat["sd_seed"] - 0.0026) < 0.0008, flat["sd_seed"]
    assert not flat["usable_objective"], flat["snr"]
    assert real["usable_objective"] and real["spearman"]["enrich"]["rho"] > 0.9, real["snr"]
    print(f"selftest: flat case  snr {flat['snr']:.2f}, LOO R2 {flat['loo']['fz']['r2']:+.2f} -> rejected")
    print(f"selftest: trend case snr {real['snr']:.2f}, LOO R2 {real['loo']['fz']['r2']:+.2f} -> accepted")
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axial-dir", default="axial_c9_all")
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--state", default="ARO")
    ap.add_argument("--seed-sd", type=float, default=None,
                    help="single-solve seed s.d. of F_z, if no design has two seeds")
    ap.add_argument("--out", default="c10_gate")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ad = Path(a.axial_dir)
    if not ad.is_dir():
        print(f"FAIL: {ad} not found"); return 2
    fz = load_fz(ad, a.state)
    if len(fz) < 3:
        print(f"FAIL: only {len(fz)} designs with a {a.state} solve in {ad}"); return 2
    rep = analyse(fz, json.loads(Path(a.checkpoint).read_text()), a.seed_sd)
    txt = report(rep)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "fz_gate.json").write_text(json.dumps(rep, indent=1))
    (out / "fz_gate.txt").write_text(txt + "\n")
    print(txt); print(f"\nwrote {out}/fz_gate.json and {out}/fz_gate.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
