#!/usr/bin/env python3
"""
paper_margin_constraint.py -- a reactivity margin at the mission burnup as the
cycle-length constraint of the loop, replayed on the Campaign 10 three-
dimensional k histories. No transport.

The loop constrains g_efpd = 1826 - EFPD_3D, where EFPD_3D comes from the
crossing of k_3D(B) with the end-of-cycle criterion k_eoc = k_3D(0) x
k_target / k_bol. That target has a cliff: a design whose k_3D never reaches
k_eoc has EFPD_3D = 0 next to designs at 1900 d. The alternative proposed on
26 Sep 2026 (Kanban note "Campaign 10") is the margin at the mission burnup,
    g_margin = k_eoc - k_3D(B_mission),   B_mission = 1826 d x P_spec / 1000,
which is smooth in the design variables and has the same sign as g_efpd.
Both targets are read from the same stored histories, so the comparison is
exact. Leave-one-out with the campaign kernel (Matern 5/2 ARD) and with the
Matern 1/2 ARD of paper_finds.md Section 9, on the designs whose history
reaches B_mission.

usage:  python paper_margin_constraint.py [--checkpoint out_c10/optimization_checkpoint.json]
                                          [--hifi-kbol]   use repro_c9/kbol_c10_*_hifi.json where present
writes figs_paper/margin_constraint.md and .json
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

VARS = ["enrich", "gd_wt", "refl_thick", "gd_pins"]
EFPD_REQ = 1826.0
SPEC_POWER = 9.983354024602374      # W/gHM, c9_dep_core3d runs.json


def loo(X, y, nu):
    """Leave-one-out mean and std with the campaign GP settings."""
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    pred, std = np.zeros(len(y)), np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        kern = (ConstantKernel(1.0, (1e-3, 1e4)) * Matern(np.ones(X.shape[1]), (1e-2, 1e3), nu=nu)
                + WhiteKernel(1e-3, (1e-8, 1e1)))
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=4, random_state=0).fit(Xs[m], y[m])
        p, s = gp.predict(Xs[i:i + 1], return_std=True)
        pred[i], std[i] = p[0], s[0]
    return pred, std


def metrics(y, pred, std, feasible_true, feasible_pred):
    r = y - pred
    out = dict(rmse=float(np.sqrt((r ** 2).mean())), r2=float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()),
               spearman=float(spearmanr(y, pred).correlation), s_z=float(np.sqrt(((r / std) ** 2).mean())),
               cov2=float((np.abs(r) <= 2 * std).mean()))
    tp = int((feasible_true & feasible_pred).sum()); fp = int((~feasible_true & feasible_pred).sum())
    fn = int((feasible_true & ~feasible_pred).sum()); tn = int((~feasible_true & ~feasible_pred).sum())
    out.update(verdict_accuracy=(tp + tn) / len(y), fp=fp, fn=fn)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c10/optimization_checkpoint.json")
    ap.add_argument("--hifi-kbol", action="store_true")
    ap.add_argument("--out", default="figs_paper")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    R = json.load(open(a.checkpoint))["all_raw"]
    b_mission = EFPD_REQ * SPEC_POWER / 1000.0
    rows = []
    for i, r in enumerate(R):
        bu, k = r.get("bu_hist_3d"), r.get("k_hist_3d")
        if not bu or not k:
            continue
        kbol = r["k_bol"]
        p = Path(f"repro_c9/kbol_c10_{i}_hifi.json")
        if a.hifi_kbol and p.exists():
            kbol = float(json.load(open(p))["k"][0])
        k_eoc = k[0] * r["k_target"] / kbol
        if bu[-1] < b_mission:                      # history too short for the margin (cycle 0 designs)
            k_m, reach = None, False
        else:
            k_m, reach = float(np.interp(b_mission, bu, k)), True
        rows.append(dict(i=i, x=[r[v] for v in VARS], efpd=float(r["efpd_3d"]), censored=bool(r.get("censored_3d")),
                         g_margin_pcm=(None if k_m is None else 1e5 * (k_eoc - k_m)), reach=reach,
                         k_last_below=(k[-1] < k_eoc)))
    n_all = len(rows); n_cliff = sum(1 for w in rows if w["efpd"] == 0.0)
    use = [w for w in rows if w["reach"]]
    X = np.array([w["x"] for w in use]); efpd = np.array([w["efpd"] for w in use]); gm = np.array([w["g_margin_pcm"] for w in use])
    feas = efpd >= EFPD_REQ
    print(f"{n_all} designs with a 3D history, {n_cliff} with a cycle of 0 (the cliff), "
          f"{len(use)} whose history reaches B_mission = {b_mission:.2f} MWd/kgHM; feasible {int(feas.sum())}")
    print(f"sign agreement of the two targets on these: {int(((gm <= 0) == feas).sum())} of {len(use)}")
    res = {"n_designs": n_all, "n_cliff": n_cliff, "n_used": len(use), "b_mission": b_mission, "hifi_kbol": a.hifi_kbol, "loo": {}}
    lines = ["| Target | Kernel | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | Verdict accuracy | FP | FN |", "|---|---|---|---|---|---|---|---|---|---|"]
    for nu, kname in ((2.5, "Matern 5/2 ARD (campaign)"), (0.5, "Matern 1/2 ARD")):
        for tname, y, fp_true, pred_feas in (("cycle length [d]", efpd, feas, lambda p: p >= EFPD_REQ),
                                             ("margin at B_mission [pcm]", gm, gm <= 0, lambda p: p <= 0)):
            pred, std = loo(X, y, nu)
            m = metrics(y, pred, std, fp_true, pred_feas(pred))
            res["loo"][f"{tname}|{kname}"] = m
            lines.append(f"| {tname} | {kname} | {m['rmse']:.1f} | {m['r2']:.3f} | {m['spearman']:.3f} | {m['cov2']:.2f} | "
                         f"{m['s_z']:.2f} | {m['verdict_accuracy']:.2f} | {m['fp']} | {m['fn']} |")
            print(lines[-1])
    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / "margin_constraint.json").write_text(json.dumps(res, indent=1))
    (out / "margin_constraint.md").write_text(
        f"# Reactivity margin at the mission burnup as the constraint target\n\n"
        f"Campaign 10 three-dimensional histories, {n_all} designs, {n_cliff} with a cycle of 0, {len(use)} reaching "
        f"B_mission = {b_mission:.2f} MWd/kgHM. k_bol {'high statistics where available' if a.hifi_kbol else 'campaign record'}. "
        f"Leave-one-out, one design out at a time.\n\n" + "\n".join(lines) + "\n")
    print(f"wrote {out}/margin_constraint.md")


if __name__ == "__main__":
    main()
