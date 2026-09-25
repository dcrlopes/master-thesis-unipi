#!/usr/bin/env python3
"""
c9_surrogate_cv.py -- cross-validation of the Gaussian-process surrogates on
the Campaign 9 archive, with the same protocols and metrics as
c8_surrogate_cv.py, so the two campaigns are reported alike. No OpenMC.

WHAT IS VALIDATED
    The surrogate the campaign used, reactor_optimization.GPSurrogate, trained
    on the matrices the loop trains on (reactor_optimization._seed_from_raw):
        F = (peaking, c_max)                   both minimised
        G = every constraint / its own scale   CONSTRAINT-NORM
    over every archived design, feasible and infeasible, as in the loop.

WHAT C8_SURROGATE_CV DOES NOT COVER HERE
    In Campaign 9 the cycle length is a constraint, g_efpd = E_req - E, so its
    surrogate decides feasibility rather than an objective. It is reported in
    physical units (EFPD), next to k_core, k_ALLRE, the operating-maximum
    control margin and F_dH, and the feasibility the loop would have predicted
    is compared with the truth, with the mean rule and with the margin rule
    g_mean + kappa g_std <= 0 at the campaign's own kappa.

PROTOCOLS
    loo          leave-one-out over the N designs: the surrogate at the end.
    walkforward  each infill block predicted from everything before it:
                 the surrogate at decision time, on the designs it chose.

METRICS, per output in physical units
    RMSE, MAE, R^2, Spearman rank correlation of predicted and true, coverage
    of the one- and two-sigma predictive intervals (nominal 0.683 and 0.954),
    and s_z, the standard deviation of the standardised residual (1 for a
    calibrated predictor, above 1 over-confident).

OUTPUT (--out, default figs_c9_cv/)
    c9_surrogate_cv.json   every number
    c9_cv_table.tex        booktabs table, leave-one-out, physical units
    c9_cv_loo.pdf/.png     predicted against true, four panels, one-sigma bars

USAGE (repository root)
    python c9_surrogate_cv.py --check
    python c9_surrogate_cv.py                    # about 3 to 6 min
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import core_geometry as cg
from c8_surrogate_cv import (metrics, loo, walkforward, blocks_from_phase_log,
                             setup_mpl, save)


def panel(ax, y, mu, sd, feas, label, unit):
    """Predicted against true with one-sigma bars. Every label starts with a
    capital letter, as for every figure produced by this work."""
    lo = min(y.min(), (mu - sd).min())
    hi = max(y.max(), (mu + sd).max())
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", c="0.5", lw=0.8)
    ax.errorbar(y[~feas], mu[~feas], yerr=sd[~feas], fmt="o", ms=3.2, mfc="white", mec="0.35",
                ecolor="0.6", elinewidth=0.6, capsize=0, label="Infeasible")
    ax.errorbar(y[feas], mu[feas], yerr=sd[feas], fmt="o", ms=3.6, c="tab:blue", ecolor="tab:blue",
                elinewidth=0.7, capsize=0, label="Feasible")
    ax.set_xlabel(f"True {label} [{unit}]")
    ax.set_ylabel(f"Predicted {label} [{unit}]")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")


def load_archive(path):
    ck = json.loads(Path(path).read_text())
    names, cons, raw = ck["design_variables"], ck["constraint_names"], ck["all_raw"]
    meta = ck.get("meta", {}) or {}
    lim = meta.get("limits", {}) or {}
    efpd_req = float((meta.get("campaign9") or {}).get("efpd_req", 1826.0))
    scales = {"g_kmin": float(lim.get("k_min", 1.02)),
              "g_kmax": float(lim.get("k_max", 1.166)),
              "g_enr": float(lim.get("enr_max", 16.0)),
              "g_peak": float(lim.get("f_max", 1.65)),
              "g_geom": cg.R_VESSEL_INNER - cg.VESSEL_CLEARANCE_CM,
              "g_efpd": efpd_req,
              "g_ctrl_peak": 1.0,
              "g_ctrl": 1.0}
    missing = [c for c in cons if c not in scales]
    if missing:
        raise SystemExit(f"no scale for {missing}")
    X = np.array([[float(r[n]) for n in names] for r in raw])
    F = np.array([[float(r["peaking"]), float(r["c_max"])] for r in raw])
    G = np.array([[float(r[c]) / scales[c] for c in cons] for r in raw])
    feas = np.all(G <= 1e-9, axis=1)
    return ck, names, cons, raw, meta, scales, efpd_req, X, F, G, feas


def physical(cons, scales, efpd_req, margin, G, muG, sdG):
    """(true, mean, sd) in physical units for the constraint surrogates."""
    out = {}

    def col(c):
        return cons.index(c)
    if "g_efpd" in cons:
        j, s = col("g_efpd"), scales["g_efpd"]
        out["cycle length [EFPD]"] = (efpd_req - G[:, j] * s, efpd_req - muG[:, j] * s, sdG[:, j] * s)
    j, s = col("g_kmax"), scales["g_kmax"]
    out["k_core"] = (G[:, j] * s + s, muG[:, j] * s + s, sdG[:, j] * s)
    if "g_ctrl" in cons:
        j = col("g_ctrl")
        out["k_ALLRE"] = (G[:, j] + 1.0 - margin, muG[:, j] + 1.0 - margin, sdG[:, j])
    if "g_ctrl_peak" in cons:
        j = col("g_ctrl_peak")
        out["g_ctrl_peak [dk]"] = (G[:, j], muG[:, j], sdG[:, j])
    j, s = col("g_peak"), scales["g_peak"]
    out["F_dH (from g_peak)"] = (G[:, j] * s + s, muG[:, j] * s + s, sdG[:, j] * s)
    return out


def confusion(pred, true):
    return dict(tp=int((pred & true).sum()), fp=int((pred & ~true).sum()),
                fn=int((~pred & true).sum()), tn=int((~pred & ~true).sum()),
                accuracy=float((pred == true).mean()))


def fmt(v, nd):
    return "---" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def table(results, path):
    rows = [("$F_{\\Delta H}$", "peaking F_dH [-]", 3, 3),
            ("$c_\\mathrm{max}$, ppm", "c_max [ppm]", 0, 3),
            ("Cycle length, EFPD", "cycle length [EFPD]", 0, 3),
            ("$k_\\mathrm{eff}^\\mathrm{core}$", "k_core", 4, 3),
            ("$k_\\mathrm{RE}$", "k_ALLRE", 4, 3)]
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"  \toprule",
         r"  Output & Designs & RMSE & $R^2$ & $\rho_\mathrm{S}$ & Cov.\ $1\sigma$ & Cov.\ $2\sigma$ & $s_z$ \\",
         r"  \midrule"]
    for label, key, nd, _ in rows:
        if key not in results["loo"]:
            continue
        for sub, name in (("all", f"all {results['n']}"), ("feasible", f"feasible {results['n_feasible']}")):
            m = results["loo"][key].get(sub, {})
            if m.get("n", 0) < 3:
                continue
            L.append(f"  {label if sub == 'all' else ''} & {name} & {fmt(m['rmse'], nd)} & {fmt(m['r2'], 3)} & "
                     f"{fmt(m['spearman'], 3)} & {fmt(m['cov1'], 2)} & {fmt(m['cov2'], 2)} & {fmt(m['z_std'], 2)} \\\\")
    L += [r"  \bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--out", default="figs_c9_cv")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-walkforward", action="store_true")
    a = ap.parse_args()

    ck, names, cons, raw, meta, scales, efpd_req, X, F, G, feas = load_archive(a.checkpoint)
    n = len(X)
    kappa = float((meta.get("surrogate_policy") or {}).get("feas_kappa", 1.0))
    margin = float((meta.get("ctrl_screen") or {}).get("margin_pcm", 1000.0)) * 1e-5
    exact_idx = {cons.index(c) for c in ("g_geom",) if c in cons}
    blocks = blocks_from_phase_log(ck, n)
    print(f"archive {a.checkpoint}: {n} designs, {int(feas.sum())} feasible, mission {efpd_req:.0f} d")
    print(f"variables {names} | constraints {cons}")
    print(f"scales {scales}")
    print(f"walk-forward blocks {blocks} | kappa {kappa} | control margin {margin * 1e5:.0f} pcm")
    if a.check:
        return 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("leave-one-out on the objectives (2 columns)")
    muF, sdF = loo(X, F)
    print(f"leave-one-out on the constraints ({G.shape[1]} columns)")
    muG, sdG = loo(X, G)

    subsets = {"all": np.ones(n, bool), "feasible": feas}
    results = {"checkpoint": a.checkpoint, "n": n, "n_feasible": int(feas.sum()), "efpd_req": efpd_req,
               "variables": names, "constraints": cons, "scales": scales, "kappa": kappa,
               "ctrl_margin_dk": margin, "blocks": blocks, "loo": {}, "walkforward": {}}
    results["loo"]["peaking F_dH [-]"] = {k: metrics(F[m, 0], muF[m, 0], sdF[m, 0]) for k, m in subsets.items()}
    results["loo"]["c_max [ppm]"] = {k: metrics(F[m, 1], muF[m, 1], sdF[m, 1]) for k, m in subsets.items()}
    phys = physical(cons, scales, efpd_req, margin, G, muG, sdG)
    for k, (y, mu, sd) in phys.items():
        results["loo"][k] = {s: metrics(y[m], mu[m], sd[m]) for s, m in subsets.items()}
    gp_cols = [j for j in range(G.shape[1]) if j not in exact_idx]
    pred_mean = np.all(muG[:, gp_cols] <= 0.0, axis=1)
    pred_marg = (muG[:, gp_cols] + kappa * sdG[:, gp_cols]).max(axis=1) <= 0.0
    results["loo"]["feasibility, mean rule"] = confusion(pred_mean, feas)
    results["loo"][f"feasibility, margin rule kappa {kappa:g}"] = confusion(pred_marg, feas)
    if "g_efpd" in cons:
        j = cons.index("g_efpd")
        results["loo"]["cycle-length verdict, mean rule"] = confusion(muG[:, j] <= 0.0, G[:, j] <= 1e-9)

    print("\nleave-one-out, physical units")
    for k, v in results["loo"].items():
        if "all" in v:
            r, f = v["all"], v.get("feasible", {})
            print(f"  {k:<24s} all {r['n']:2d}: RMSE {r['rmse']:.4g}  R2 {r['r2']:.3f}  rho_S {r['spearman']:.3f}  "
                  f"cov1 {r['cov1']:.2f}  cov2 {r['cov2']:.2f}  s_z {r['z_std']:.2f}"
                  + (f"  | feasible {f['n']}: RMSE {f['rmse']:.4g}  R2 {f['r2']:.3f}  rho_S {f['spearman']:.3f}  "
                     f"cov2 {f['cov2']:.2f}  s_z {f['z_std']:.2f}" if f.get("n", 0) >= 3 else ""))
        else:
            print(f"  {k:<42s} {v}")

    if not a.no_walkforward and blocks:
        print("\nwalk-forward over the infill blocks")
        wf = walkforward(X, F, G, blocks, kappa, exact_idx)
        pooled = {"peaking F_dH [-]": ([], [], []), "c_max [ppm]": ([], [], [])}
        per_block = []
        for w in wf:
            s, e = w["start"], w["end"]
            for c, key in ((0, "peaking F_dH [-]"), (1, "c_max [ppm]")):
                pooled[key][0].extend(F[s:e, c]); pooled[key][1].extend(w["muF"][:, c]); pooled[key][2].extend(w["sdF"][:, c])
            per_block.append(dict(block=w["block"], n_train=w["n_train"], n_pred=e - s,
                                  true_feasible=int(w["true_feas"].sum()),
                                  predicted_feasible_margin=int(w["pred_feas_marg"].sum()),
                                  agree_margin=int((w["pred_feas_marg"] == w["true_feas"]).sum())))
        results["walkforward"]["pooled"] = {k: metrics(*v) for k, v in pooled.items()}
        results["walkforward"]["per_block"] = per_block
        for k, m in results["walkforward"]["pooled"].items():
            print(f"  pooled {k:<18s} RMSE {m['rmse']:.4g}  R2 {m['r2']:.3f}  rho_S {m['spearman']:.3f}  "
                  f"cov2 {m['cov2']:.2f}  s_z {m['z_std']:.2f}")
        for b in per_block:
            print(f"  block {b['block']}: trained on {b['n_train']}, {b['n_pred']} predicted, "
                  f"{b['true_feasible']} truly feasible, {b['predicted_feasible_margin']} predicted feasible "
                  f"by the margin rule, {b['agree_margin']} verdicts agree")

    Path(out / "c9_surrogate_cv.json").write_text(json.dumps(results, indent=1))
    table(results, out / "c9_cv_table.tex")

    plt = setup_mpl()
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 8.0))
    panel(axs[0, 0], F[:, 0], muF[:, 0], sdF[:, 0], feas, r"core $F_{\Delta H}$", "-")
    panel(axs[0, 1], F[:, 1], muF[:, 1], sdF[:, 1], feas, r"$c_\mathrm{max}$", "ppm")
    if "cycle length [EFPD]" in phys:
        y, mu, sd = phys["cycle length [EFPD]"]
        panel(axs[1, 0], y, mu, sd, feas, "cycle length", "EFPD")
    if "k_ALLRE" in phys:
        y, mu, sd = phys["k_ALLRE"]
        panel(axs[1, 1], y, mu, sd, feas, r"$k_\mathrm{RE}$", "-")
    save(fig, out, "c9_cv_loo")
    print(f"\nwrote {out}/c9_surrogate_cv.json, c9_cv_table.tex, c9_cv_loo.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
