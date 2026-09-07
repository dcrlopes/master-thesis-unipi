#!/usr/bin/env python3
"""
c8_surrogate_cv.py -- cross-validation of the Gaussian Process surrogate on
the Campaign 8 archive. No OpenMC: numpy, scikit-learn, pymoo (for the
import of reactor_optimization) and matplotlib.

WHAT IS VALIDATED
    Exactly the surrogate the campaign used: reactor_optimization.GPSurrogate
    (one anisotropic Matern-5/2 GP per output, StandardScaler on X,
    normalize_y, WhiteKernel, 4 restarts, random_state 0), trained on the
    SAME matrices the loop trains on:
        F  = (-cycle_length, peaking)             minimise space
        G  = every constraint / its own limit     CONSTRAINT-NORM
    over ALL archive designs, feasible and infeasible alike, because that
    is what ActiveLearningMOO.run() does at every iteration.

TWO PROTOCOLS
    loo          leave-one-out over the N archive designs: N fits per
                 matrix, each predicting the held-out design. Measures the
                 surrogate at the END of the campaign.
    walkforward  for every infill block k of the archive (from phase_log),
                 train on every evaluation BEFORE the block, predict the
                 block. Measures the surrogate AT DECISION TIME, on the very
                 designs it chose. Also reports whether the acquisition's
                 margin-feasibility rule (g_mean + kappa g_std <= 0, kappa
                 from meta) agreed with the truth for those designs.

METRICS (per output, in physical units)
    RMSE, MAE, R^2 (1 - SS_res / SS_tot), Spearman rank correlation,
    coverage of the 1-sigma and 2-sigma predictive intervals (nominal
    0.683 and 0.954 for a calibrated Gaussian), and the standard deviation
    of the standardised residual z = (y - mu) / sigma (1.0 if calibrated,
    above 1 over-confident, below 1 under-confident).
    Objectives are reported on three subsets: all designs, designs with a
    non-zero cycle (the k_bol >= k_target set), feasible designs. The
    cycle-length zeros of the sub-k_target designs are a discontinuity the
    GP must model, and the campaign trained through it, so it is reported
    rather than hidden.

CONSTRAINT COLUMNS IN PHYSICAL UNITS
    g_kmax  -> k_core   = g * k_max + k_max
    g_kmin  -> k_core   = k_min - g * k_min      (same k, second reading)
    g_ctrl  -> k_ALLRE  = g + (1 - margin)
    g_peak  -> F_dH     = g * f_max + f_max      (same F as objective 2)
    g_enr, g_geom are closed-form in the design and are listed only to
    show the GP reproduces them (the loop overwrites g_geom exactly).

OUTPUT (--out, default figs_c8_cv/)
    c8_cv_loo.pdf/.png            predicted vs true, four panels, 1-sigma bars
    c8_cv_walkforward.pdf/.png    per-block RMSE and coverage
    c8_surrogate_cv.json          every number
    c8_cv_table.tex               booktabs table (LOO, physical units)

    python c8_surrogate_cv.py --check       load, report sizes, no fits
    python c8_surrogate_cv.py               everything (about 2 to 5 min)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")   # sklearn ConvergenceWarning on tiny data

import core_geometry as cg                           # noqa: E402
from reactor_optimization import GPSurrogate         # noqa: E402


# ----------------------------------------------------------------------------
def load_archive(path):
    ck = json.loads(Path(path).read_text())
    names = ck["design_variables"]
    cons = ck["constraint_names"]
    raw = ck["all_raw"]
    meta = ck.get("meta", {})
    lim = meta.get("limits", {})
    scales = {"g_kmin": float(lim.get("k_min", 1.02)),
              "g_kmax": float(lim.get("k_max", 1.35)),
              "g_enr": float(lim.get("enr_max", 19.75)),
              "g_peak": float(lim.get("f_max", 2.0)),
              "g_geom": cg.R_VESSEL_INNER - cg.VESSEL_CLEARANCE_CM,
              "g_ctrl": 1.0}
    X = np.array([[float(r[n]) for n in names] for r in raw])
    F = np.array([[-float(r["cycle_length"]), float(r["peaking"])] for r in raw])
    G = np.array([[float(r[c]) / scales[c] for c in cons] for r in raw])
    feas = np.all(G <= 1e-9, axis=1)
    return ck, names, cons, raw, meta, scales, X, F, G, feas


def blocks_from_phase_log(ck, n):
    """[(start, end), ...] of the infill blocks, from phase_log; the DOE is
    everything before the first infill block."""
    pl = ck.get("phase_log", [])
    blocks, start = [], 0
    for p in pl:
        m = int(p.get("n_eval", 0))
        if p.get("stage") == "DOE":
            start += m
        elif p.get("stage") == "infill":
            blocks.append((start, start + m))
            start += m
    if not blocks or start != n:
        # fall back to the profile defaults
        n0 = 24
        blocks = [(s, min(s + 6, n)) for s in range(n0, n, 6)]
    return blocks


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def metrics(y, mu, sd):
    y, mu, sd = np.asarray(y, float), np.asarray(mu, float), np.asarray(sd, float)
    n = len(y)
    if n < 3:
        return dict(n=int(n))
    res = y - mu
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    z = res / np.where(sd > 0, sd, np.nan)
    return dict(
        n=int(n),
        rmse=float(math.sqrt(ss_res / n)),
        mae=float(np.abs(res).mean()),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        spearman=spearman(y, mu),
        cov1=float(np.mean(np.abs(res) <= sd)),
        cov2=float(np.mean(np.abs(res) <= 2.0 * sd)),
        z_std=float(np.nanstd(z)),
        sigma_mean=float(np.mean(sd)),
        bias=float(res.mean()),
    )


# ----------------------------------------------------------------------------
def loo(X, Y, verbose=True):
    """Leave-one-out with GPSurrogate on the columns of Y at once (as the
    loop fits them). Returns (mu, sd) arrays of Y's shape."""
    n = len(X)
    mu = np.zeros_like(Y, dtype=float)
    sd = np.zeros_like(Y, dtype=float)
    t0 = time.time()
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        gp = GPSurrogate().fit(X[m], Y[m])
        mi, si = gp.predict(X[i:i + 1])
        mu[i], sd[i] = mi[0], si[0]
        if verbose and (i + 1) % 10 == 0:
            print(f"    loo {i + 1}/{n}  {time.time() - t0:.0f} s")
    return mu, sd


def walkforward(X, F, G, blocks, kappa, exact_idx, verbose=True):
    out = []
    for b, (s, e) in enumerate(blocks):
        gpF = GPSurrogate().fit(X[:s], F[:s])
        gpG = GPSurrogate().fit(X[:s], G[:s])
        muF, sdF = gpF.predict(X[s:e])
        muG, sdG = gpG.predict(X[s:e])
        gp_cols = [j for j in range(G.shape[1]) if j not in exact_idx]
        s_marg = (muG[:, gp_cols] + kappa * sdG[:, gp_cols]).max(axis=1)
        pred_feas_mean = np.all(muG <= 0.0, axis=1)
        pred_feas_marg = s_marg <= 0.0
        true_feas = np.all(G[s:e] <= 1e-9, axis=1)
        out.append(dict(block=b + 1, start=s, end=e, n_train=s,
                        muF=muF, sdF=sdF, muG=muG, sdG=sdG,
                        pred_feas_mean=pred_feas_mean,
                        pred_feas_marg=pred_feas_marg, true_feas=true_feas))
        if verbose:
            print(f"    walk-forward block {b + 1}: train {s}, predict "
                  f"{e - s}, true feasible {int(true_feas.sum())}, "
                  f"margin-feasible predicted {int(pred_feas_marg.sum())}")
    return out


# ----------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9.5,
        "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.framealpha": 0.9})
    return plt


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    print(f"  wrote {name}.pdf/.png")


def panel(ax, y, mu, sd, feas, label, unit):
    lo = min(y.min(), (mu - sd).min())
    hi = max(y.max(), (mu + sd).max())
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", c="0.5", lw=0.8)
    ax.errorbar(y[~feas], mu[~feas], yerr=sd[~feas], fmt="o", ms=3.2,
                mfc="white", mec="0.35", ecolor="0.6", elinewidth=0.6,
                capsize=0, label="infeasible")
    ax.errorbar(y[feas], mu[feas], yerr=sd[feas], fmt="o", ms=3.6,
                c="tab:blue", ecolor="tab:blue", elinewidth=0.7, capsize=0,
                label="feasible")
    ax.set_xlabel(f"true {label}{unit}")
    ax.set_ylabel(f"predicted {label}{unit}")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--out", default="figs_c8_cv")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-walkforward", action="store_true")
    a = ap.parse_args()

    (ck, names, cons, raw, meta, scales, X, F, G, feas) = load_archive(a.checkpoint)
    n = len(X)
    kappa = float(meta.get("surrogate_policy", {}).get("feas_kappa", 1.0))
    margin = float(meta.get("ctrl_screen", {}).get("margin_pcm", 1000.0)) * 1e-5
    exact_names = ["g_geom"]                    # overwritten exactly in the loop
    exact_idx = {cons.index(c) for c in exact_names if c in cons}
    blocks = blocks_from_phase_log(ck, n)
    nonzero = F[:, 0] < 0.0                     # cycle_length > 0
    print(f"archive {a.checkpoint}: {n} designs, {int(feas.sum())} feasible, "
          f"{int(nonzero.sum())} with a non-zero cycle")
    print(f"variables {names} | constraints {cons}")
    print(f"scales {scales}")
    print(f"blocks (walk-forward) {blocks} | kappa {kappa} | ctrl margin "
          f"{margin * 1e5:.0f} pcm")
    if a.check:
        return 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- leave-one-out ----------------------------------------
    print("LOO on the objectives F (2 columns)")
    muF, sdF = loo(X, F)
    print("LOO on the constraints G (%d columns)" % G.shape[1])
    muG, sdG = loo(X, G)

    # physical conversions
    y_cyc, m_cyc, s_cyc = -F[:, 0], -muF[:, 0], sdF[:, 0]
    y_pk, m_pk, s_pk = F[:, 1], muF[:, 1], sdF[:, 1]
    phys = {}
    j = cons.index("g_kmax")
    kmax = scales["g_kmax"]
    phys["k_core (from g_kmax)"] = ((G[:, j] * kmax + kmax),
                                    (muG[:, j] * kmax + kmax), sdG[:, j] * kmax, "")
    if "g_ctrl" in cons:
        j = cons.index("g_ctrl")
        phys["k_ALLRE (from g_ctrl)"] = ((G[:, j] + 1.0 - margin),
                                         (muG[:, j] + 1.0 - margin), sdG[:, j], "")
    j = cons.index("g_kmin")
    kmin = scales["g_kmin"]
    phys["k_core (from g_kmin)"] = ((kmin - G[:, j] * kmin),
                                    (kmin - muG[:, j] * kmin), sdG[:, j] * kmin, "")
    j = cons.index("g_peak")
    fmax = scales["g_peak"]
    phys["F_dH (from g_peak)"] = ((G[:, j] * fmax + fmax),
                                  (muG[:, j] * fmax + fmax), sdG[:, j] * fmax, "")
    for c in ("g_enr", "g_geom"):
        if c in cons:
            j = cons.index(c)
            phys[f"{c} (closed form, normalised)"] = (G[:, j], muG[:, j], sdG[:, j], "")

    results = {"n": n, "n_feasible": int(feas.sum()), "n_nonzero_cycle": int(nonzero.sum()),
               "variables": names, "constraints": cons, "scales": scales,
               "kappa": kappa, "ctrl_margin_dk": margin, "blocks": blocks,
               "loo": {}, "walkforward": {}}
    subsets = {"all": np.ones(n, bool), "nonzero_cycle": nonzero, "feasible": feas}
    results["loo"]["cycle_length [EFPD]"] = {k: metrics(y_cyc[m], m_cyc[m], s_cyc[m])
                                             for k, m in subsets.items()}
    results["loo"]["peaking F_dH [-]"] = {k: metrics(y_pk[m], m_pk[m], s_pk[m])
                                          for k, m in subsets.items()}
    for k, (y, mu, sd, _) in phys.items():
        results["loo"][k] = {"all": metrics(y, mu, sd), "feasible": metrics(y[feas], mu[feas], sd[feas])}
    # LOO feasibility classification from the constraint GPs
    pred_feas_mean = np.all(muG <= 0.0, axis=1)
    gp_cols = [j for j in range(G.shape[1]) if j not in exact_idx]
    s_marg = (muG[:, gp_cols] + kappa * sdG[:, gp_cols]).max(axis=1)
    pred_feas_marg = s_marg <= 0.0
    def confusion(pred, true):
        return dict(tp=int((pred & true).sum()), fp=int((pred & ~true).sum()),
                    fn=int((~pred & true).sum()), tn=int((~pred & ~true).sum()),
                    accuracy=float((pred == true).mean()))
    results["loo"]["feasibility_from_mean"] = confusion(pred_feas_mean, feas)
    results["loo"][f"feasibility_with_margin_kappa_{kappa:g}"] = confusion(pred_feas_marg, feas)

    print("\nLOO summary (physical units)")
    for k, v in results["loo"].items():
        if isinstance(v, dict) and "all" in v:
            r_all = v["all"]; r_f = v.get("feasible", {})
            print(f"  {k:<34s} all: RMSE {r_all.get('rmse', float('nan')):.4g} "
                  f"R2 {r_all.get('r2', float('nan')):.3f} cov1 {r_all.get('cov1', float('nan')):.2f} "
                  f"cov2 {r_all.get('cov2', float('nan')):.2f} z_std {r_all.get('z_std', float('nan')):.2f}"
                  + (f" | feasible: RMSE {r_f.get('rmse', float('nan')):.4g} "
                     f"R2 {r_f.get('r2', float('nan')):.3f}" if r_f.get("n", 0) >= 3 else ""))
    print(f"  feasibility (mean rule)   {results['loo']['feasibility_from_mean']}")
    print(f"  feasibility (margin rule) {results['loo'][f'feasibility_with_margin_kappa_{kappa:g}']}")

    # ---------------- walk-forward -------------------------------------------
    if not a.no_walkforward and blocks:
        print("\nwalk-forward over the infill blocks")
        wf = walkforward(X, F, G, blocks, kappa, exact_idx)
        allF_y, allF_mu, allF_sd = [], [], []
        per_block = []
        for w in wf:
            s, e = w["start"], w["end"]
            yc, mc, sc = -F[s:e, 0], -w["muF"][:, 0], w["sdF"][:, 0]
            yp, mp, sp = F[s:e, 1], w["muF"][:, 1], w["sdF"][:, 1]
            per_block.append(dict(
                block=w["block"], n_train=w["n_train"], n_pred=e - s,
                cycle=metrics(yc, mc, sc), peaking=metrics(yp, mp, sp),
                feas_true=int(w["true_feas"].sum()),
                feas_pred_mean=int(w["pred_feas_mean"].sum()),
                feas_pred_margin=int(w["pred_feas_marg"].sum()),
                margin_rule=confusion(w["pred_feas_marg"], w["true_feas"]),
                cycle_pred_vs_true=[(float(t), float(m), float(sd_))
                                    for t, m, sd_ in zip(yc, mc, sc)],
                peaking_pred_vs_true=[(float(t), float(m), float(sd_))
                                      for t, m, sd_ in zip(yp, mp, sp)]))
            allF_y.append(np.column_stack([yc, yp]))
            allF_mu.append(np.column_stack([mc, mp]))
            allF_sd.append(np.column_stack([sc, sp]))
        Yw, Mw, Sw = (np.vstack(allF_y), np.vstack(allF_mu), np.vstack(allF_sd))
        results["walkforward"] = dict(
            blocks=per_block,
            pooled_cycle=metrics(Yw[:, 0], Mw[:, 0], Sw[:, 0]),
            pooled_peaking=metrics(Yw[:, 1], Mw[:, 1], Sw[:, 1]),
            pooled_margin_rule=confusion(
                np.concatenate([w["pred_feas_marg"] for w in wf]),
                np.concatenate([w["true_feas"] for w in wf])))
        pc, pp = results["walkforward"]["pooled_cycle"], results["walkforward"]["pooled_peaking"]
        print(f"  pooled cycle  : RMSE {pc['rmse']:.1f} EFPD, R2 {pc['r2']:.3f}, cov1 {pc['cov1']:.2f}, cov2 {pc['cov2']:.2f}, z_std {pc['z_std']:.2f}")
        print(f"  pooled peaking: RMSE {pp['rmse']:.4f}, R2 {pp['r2']:.3f}, cov1 {pp['cov1']:.2f}, cov2 {pp['cov2']:.2f}, z_std {pp['z_std']:.2f}")
        print(f"  pooled margin rule: {results['walkforward']['pooled_margin_rule']}")

    (out / "c8_surrogate_cv.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"  wrote c8_surrogate_cv.json")

    # ---------------- figures ------------------------------------------------
    plt = setup_mpl()
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.0))
    panel(axs[0, 0], y_cyc, m_cyc, s_cyc, feas, "cycle length", " [EFPD]")
    panel(axs[0, 1], y_pk, m_pk, s_pk, feas, r"$F_{\Delta H}$", "")
    yk, mk, sk, _ = phys["k_core (from g_kmax)"]
    panel(axs[1, 0], yk, mk, sk, feas, r"$k_\mathrm{core}$", "")
    if "k_ALLRE (from g_ctrl)" in phys:
        ya, ma, sa, _ = phys["k_ALLRE (from g_ctrl)"]
        panel(axs[1, 1], ya, ma, sa, feas, r"$k_\mathrm{RE}$ (all regulating banks)", "")
    else:
        axs[1, 1].axis("off")
    axs[0, 0].legend(loc="upper left")
    for ax, t in zip(axs.ravel(), "abcd"):
        ax.set_title(f"({t})", loc="left")
    fig.tight_layout()
    save(fig, out, "c8_cv_loo")

    if results["walkforward"]:
        pb = results["walkforward"]["blocks"]
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
        b = [p["block"] for p in pb]
        axs[0].bar(b, [p["cycle"]["rmse"] for p in pb], color="tab:blue", width=0.6)
        axs[0].set_xlabel("infill block")
        axs[0].set_ylabel("RMSE of cycle length [EFPD]")
        ax2 = axs[0].twinx()
        ax2.plot(b, [p["peaking"]["rmse"] for p in pb], "s-", c="tab:red", ms=4)
        ax2.set_ylabel(r"RMSE of $F_{\Delta H}$", color="tab:red")
        ax2.grid(False)
        axs[1].plot(b, [p["cycle"]["cov2"] for p in pb], "o-", c="tab:blue", label="cycle length")
        axs[1].plot(b, [p["peaking"]["cov2"] for p in pb], "s-", c="tab:red", label=r"$F_{\Delta H}$")
        axs[1].axhline(0.954, ls="--", c="0.5", lw=0.8)
        axs[1].set_ylim(0, 1.05)
        axs[1].set_xlabel("infill block")
        axs[1].set_ylabel(r"coverage of the $2\sigma$ interval")
        axs[1].legend(loc="lower right")
        for ax, t in zip(axs, "ab"):
            ax.set_title(f"({t})", loc="left")
        fig.tight_layout()
        save(fig, out, "c8_cv_walkforward")

    # ---------------- LaTeX table --------------------------------------------
    rows = [("Cycle length", "EFPD", results["loo"]["cycle_length [EFPD]"]["all"],
             results["loo"]["cycle_length [EFPD]"]["feasible"]),
            (r"$F_{\Delta H}$", "--", results["loo"]["peaking F_dH [-]"]["all"],
             results["loo"]["peaking F_dH [-]"]["feasible"]),
            (r"$k_\mathrm{core}$", "--", results["loo"]["k_core (from g_kmax)"]["all"],
             results["loo"]["k_core (from g_kmax)"]["feasible"])]
    if "k_ALLRE (from g_ctrl)" in results["loo"]:
        rows.append((r"$k_\mathrm{RE}$", "--", results["loo"]["k_ALLRE (from g_ctrl)"]["all"],
                     results["loo"]["k_ALLRE (from g_ctrl)"]["feasible"]))

    def fmt(v, nd):
        return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"

    lines = [
        r"% c8_cv_table.tex -- written by c8_surrogate_cv.py. Leave-one-out",
        r"% cross-validation of the Gaussian Process surrogate on the Campaign 8",
        r"% archive. Paste into results_c8_v2.tex. Requires booktabs, threeparttable.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{threeparttable}",
        r"  \caption[Leave-one-out validation of the surrogate]{Leave-one-out cross-validation of the Gaussian Process surrogate on the Campaign 8 archive.}",
        r"  \label{tab:c8-cv-loo}",
        r"  \begin{tabular}{llrrrrrrr}",
        r"    \toprule",
        r"    Quantity & Subset & $n$ & RMSE & $R^2$ & $\rho_\mathrm{S}$ & Cov.\ $1\sigma$ & Cov.\ $2\sigma$ & $s_z$ \\",
        r"    \midrule",
    ]
    for label, unit, r_all, r_f in rows:
        nd = 0 if unit == "EFPD" else 4
        u = f" [{unit}]" if unit != "--" else ""
        lines.append(f"    {label}{u} & all & {r_all['n']} & {fmt(r_all['rmse'], nd)} & "
                     f"{fmt(r_all['r2'], 3)} & {fmt(r_all['spearman'], 3)} & "
                     f"{fmt(r_all['cov1'], 2)} & {fmt(r_all['cov2'], 2)} & {fmt(r_all['z_std'], 2)} \\\\")
        if r_f.get("n", 0) >= 3:
            lines.append(f"     & feasible & {r_f['n']} & {fmt(r_f['rmse'], nd)} & "
                         f"{fmt(r_f['r2'], 3)} & {fmt(r_f['spearman'], 3)} & "
                         f"{fmt(r_f['cov1'], 2)} & {fmt(r_f['cov2'], 2)} & {fmt(r_f['z_std'], 2)} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}\footnotesize",
        r"    \item RMSE is the root mean square error of the held-out prediction. $R^2$ is the coefficient of determination. $\rho_\mathrm{S}$ is the Spearman rank correlation between predicted and true values. Cov.\ $1\sigma$ and Cov.\ $2\sigma$ are the fractions of true values inside the one-sigma and two-sigma predictive intervals, with nominal values 0.683 and 0.954. $s_z$ is the standard deviation of the standardised residual, equal to 1 for a calibrated predictor.",
        f"    \\item Source: c8\\_surrogate\\_cv.py on the archive of {n} designs, of which {int(feas.sum())} are feasible.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    (out / "c8_cv_table.tex").write_text("\n".join(lines) + "\n")
    print("  wrote c8_cv_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
