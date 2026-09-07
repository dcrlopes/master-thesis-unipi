#!/usr/bin/env python3
"""
valgrid_compare.py -- the Step 3 verdict: does the surrogate-assisted search
recover the Pareto front that exhaustive enumeration finds on the same
two-variable slice, and at what fraction of the cost?

INPUTS  two archives written by run_optimization.py on the SAME slice
    --grid  the enumeration (--eval-list, 35 nodes by default)
    --opt   the search (--n-init 8 --iters 2 --n-infill 6, 20 evaluations)
The script REFUSES to compare archives whose meta differ in frozen
variables, limits, transport, core transport, schedule or k_target: a
mismatch means the two runs evaluated different problems.

METRICS (objective space normalised by the range of the union of both
fronts, so both objectives count equally)
    HV ratio        HV(opt front) / HV(grid front) at a shared reference
                    point, the nadir of the union of the two fronts plus
                    10 % of the range (standard practice), and also at the
                    physical reference (0 EFPD, F_dH = f_max). A ratio
                    above 1 is possible: the search is continuous and the
                    grid is not, so the search can find points between
                    grid nodes that dominate them.
    IGD             mean distance from each grid-front point to its nearest
                    opt-front point: how well the search COVERS the
                    enumerated front (0 is perfect).
    GD              mean distance from each opt-front point to its nearest
                    grid-front point: how far the search strays FROM it.
    eps+            additive epsilon indicator I_eps+(opt, grid): the
                    smallest eps such that every grid-front point is
                    weakly dominated by some opt-front point shifted by eps
                    in both normalised objectives. Reported in physical
                    units too. The reverse indicator is also given.
    coverage        fraction of grid-front points eps-dominated by the opt
                    front at eps = --tol (default 0.02 normalised), and the
                    fraction strictly dominated.
    union front     the front of BOTH archives together, with the number of
                    members from each side: opt members that no grid node
                    dominates are genuinely new points of the front.
    surrogate       the GP trained on the opt archive (exactly as the loop
                    trains it) predicts every grid node: RMSE, R^2 and
                    coverage on the 35 truth values it never saw, plus the
                    feasibility classification map.

OUTPUT (--out, default figs_valgrid/)
    valgrid_objective.pdf/.png    both archives and both fronts
    valgrid_design.pdf/.png       the (enrich, gd_wt) slice: grid
                                  feasibility map, search picks, fronts
    valgrid_surrogate.pdf/.png    GP trained on the search, tested on the grid
    valgrid_compare.json          every number
    valgrid_table.tex             booktabs summary table

    python valgrid_compare.py --check
    python valgrid_compare.py --grid out_valgrid_grid/optimization_checkpoint.json \\
                              --opt  out_valgrid_opt/optimization_checkpoint.json
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import core_geometry as cg                           # noqa: E402
from pymoo.indicators.hv import HV                   # noqa: E402
from reactor_optimization import GPSurrogate         # noqa: E402


# ----------------------------------------------------------------------------
def load(path):
    ck = json.loads(Path(path).read_text())
    raw = ck["all_raw"]
    names = ck["design_variables"]
    cons = ck["constraint_names"]
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
    return dict(ck=ck, raw=raw, names=names, cons=cons, meta=meta,
                scales=scales, X=X, F=F, G=G, feas=feas, path=path)


def same_problem(a, b):
    keys = ("frozen", "limits", "transport", "core_transport", "schedule",
            "k_target", "geometry", "enr_box_high")
    diffs = []
    for k in keys:
        va, vb = a["meta"].get(k), b["meta"].get(k)
        if json.dumps(va, sort_keys=True) != json.dumps(vb, sort_keys=True):
            diffs.append((k, va, vb))
    if a["names"] != b["names"]:
        diffs.append(("design_variables", a["names"], b["names"]))
    if a["cons"] != b["cons"]:
        diffs.append(("constraint_names", a["cons"], b["cons"]))
    ea = a["meta"].get("enrichment_policy", {}).get("e_box_used_wtpc")
    eb = b["meta"].get("enrichment_policy", {}).get("e_box_used_wtpc")
    if ea != eb:
        diffs.append(("e_box_used_wtpc", ea, eb))
    return diffs


def nondominated(F):
    n = len(F)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dom[i] = False
        if dom.any():
            nd[i] = False
    return nd


def front_of(d):
    idx = np.flatnonzero(d["feas"])
    if len(idx) == 0:
        return idx
    return idx[nondominated(d["F"][idx])]


def hv(F, ref):
    if len(F) == 0:
        return 0.0
    keep = np.all(F < ref, axis=1)
    return float(HV(ref_point=ref)(F[keep])) if keep.any() else 0.0


def dist_matrix(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def eps_plus(A, B):
    """Additive epsilon indicator I(A, B): min eps such that every b in B is
    weakly dominated by some a in A shifted by eps (minimisation)."""
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    # for each b: min over a of max over objectives of (a - b); then max over b
    return float(np.max([np.min(np.max(A - b, axis=1)) for b in B]))


def metrics(y, mu, sd):
    y, mu, sd = map(lambda v: np.asarray(v, float), (y, mu, sd))
    n = len(y)
    if n < 3:
        return dict(n=int(n))
    res = y - mu
    ss_res = float((res ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(n=int(n), rmse=float(math.sqrt(ss_res / n)),
                mae=float(np.abs(res).mean()),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                cov1=float(np.mean(np.abs(res) <= sd)),
                cov2=float(np.mean(np.abs(res) <= 2 * sd)),
                z_std=float(np.nanstd(res / np.where(sd > 0, sd, np.nan))),
                bias=float(res.mean()))


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="out_valgrid_grid/optimization_checkpoint.json")
    ap.add_argument("--opt", default="out_valgrid_opt/optimization_checkpoint.json")
    ap.add_argument("--out", default="figs_valgrid")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="eps for the coverage count, normalised objective units")
    ap.add_argument("--force", action="store_true",
                    help="compare even if the archives' meta differ (reported)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    g, o = load(a.grid), load(a.opt)
    diffs = same_problem(g, o)
    print(f"grid: {len(g['X'])} designs, {int(g['feas'].sum())} feasible | "
          f"opt: {len(o['X'])} designs, {int(o['feas'].sum())} feasible")
    print(f"live variables {g['names']} | frozen {g['meta'].get('frozen')}")
    if diffs:
        print("META MISMATCH between the two archives:")
        for k, va, vb in diffs:
            print(f"  {k}: grid {va!r} | opt {vb!r}")
        if not a.force:
            raise SystemExit("the archives describe different problems; "
                             "stop, or pass --force to compare anyway")
    else:
        print("meta agree: same slice, same limits, same fidelity, same schedule")
    if a.check:
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fg, fo = front_of(g), front_of(o)
    Fg, Fo = g["F"][fg], o["F"][fo]
    print(f"grid front: {len(fg)} members {fg.tolist()}")
    print(f"opt  front: {len(fo)} members {fo.tolist()}")
    if len(fg) == 0 or len(fo) == 0:
        raise SystemExit("one of the archives has no feasible design; "
                         "nothing to compare")

    # union front and provenance
    Fu = np.vstack([Fg, Fo])
    src = np.array(["grid"] * len(fg) + ["opt"] * len(fo))
    ndu = nondominated(Fu)
    n_union_grid = int((ndu & (src == "grid")).sum())
    n_union_opt = int((ndu & (src == "opt")).sum())
    # opt front points not dominated by ANY feasible grid node
    Fg_all = g["F"][g["feas"]]
    new_pts = [int(i) for i, f in zip(fo, Fo)
               if not np.any(np.all(Fg_all <= f, axis=1) & np.any(Fg_all < f, axis=1))]
    # grid nodes dominated by the opt front
    dom_by_opt = np.array([np.any(np.all(Fo <= f, axis=1) & np.any(Fo < f, axis=1))
                           for f in Fg_all])

    # normalisation by the union of the fronts
    lo, hi = Fu.min(axis=0), Fu.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    Ng, No = (Fg - lo) / rng, (Fo - lo) / rng
    # reference points
    ref_nadir = hi + 0.10 * rng
    fmax = g["scales"]["g_peak"]
    ref_phys = np.array([0.0, fmax])
    hv_g_n, hv_o_n = hv(Fg, ref_nadir), hv(Fo, ref_nadir)
    hv_g_p, hv_o_p = hv(Fg, ref_phys), hv(Fo, ref_phys)
    D = dist_matrix(Ng, No)
    igd = float(D.min(axis=1).mean())          # grid -> nearest opt
    gd = float(D.min(axis=0).mean())           # opt -> nearest grid
    igd_max = float(D.min(axis=1).max())
    e_og = eps_plus(No, Ng)                    # how far opt must shift to cover grid
    e_go = eps_plus(Ng, No)
    cov_eps = float(np.mean([np.any(np.all(No <= q + a.tol, axis=1)) for q in Ng]))
    cov_strict = float(np.mean([np.any(np.all(No <= q, axis=1) & np.any(No < q, axis=1))
                                for q in Ng]))
    cost_ratio = len(o["X"]) / len(g["X"])
    t_g = sum(float(r.get("t_eval_s", 0.0)) for r in g["raw"]) / 3600.0
    t_o = sum(float(r.get("t_eval_s", 0.0)) for r in o["raw"]) / 3600.0

    print(f"HV ratio (nadir + 10 %): {hv_o_n / hv_g_n if hv_g_n > 0 else float('nan'):.4f}   "
          f"(physical ref: {hv_o_p / hv_g_p if hv_g_p > 0 else float('nan'):.4f})")
    print(f"IGD {igd:.4f} (max {igd_max:.4f}) | GD {gd:.4f} | "
          f"eps+(opt,grid) {e_og:.4f} = {e_og * rng[0]:.0f} EFPD / {e_og * rng[1]:.4f} F_dH | "
          f"eps+(grid,opt) {e_go:.4f}")
    print(f"coverage of the grid front: strict {cov_strict:.2f}, within tol "
          f"{a.tol:g}: {cov_eps:.2f}")
    print(f"union front: {n_union_grid} from the grid, {n_union_opt} from the "
          f"search | search points no grid node dominates: {new_pts}")
    print(f"feasible grid nodes dominated by the search front: "
          f"{int(dom_by_opt.sum())}/{len(Fg_all)}")
    print(f"cost: {len(o['X'])} vs {len(g['X'])} evaluations "
          f"(ratio {cost_ratio:.2f}), wall {t_o:.1f} h vs {t_g:.1f} h")

    # -------- surrogate trained on the search, tested on the grid ------------
    gpF = GPSurrogate().fit(o["X"], o["F"])
    gpG = GPSurrogate().fit(o["X"], o["G"])
    muF, sdF = gpF.predict(g["X"])
    muG, sdG = gpG.predict(g["X"])
    kappa = float(o["meta"].get("surrogate_policy", {}).get("feas_kappa", 1.0))
    exact_idx = {g["cons"].index(c) for c in ("g_geom",) if c in g["cons"]}
    gp_cols = [j for j in range(g["G"].shape[1]) if j not in exact_idx]
    pred_mean = np.all(muG <= 0, axis=1)
    pred_marg = (muG[:, gp_cols] + kappa * sdG[:, gp_cols]).max(axis=1) <= 0
    y_c, m_c, s_c = -g["F"][:, 0], -muF[:, 0], sdF[:, 0]
    y_p, m_p, s_p = g["F"][:, 1], muF[:, 1], sdF[:, 1]
    fe = g["feas"]
    sur = dict(
        cycle_all=metrics(y_c, m_c, s_c), cycle_feasible=metrics(y_c[fe], m_c[fe], s_c[fe]),
        peaking_all=metrics(y_p, m_p, s_p), peaking_feasible=metrics(y_p[fe], m_p[fe], s_p[fe]),
        feasibility_mean=dict(tp=int((pred_mean & fe).sum()), fp=int((pred_mean & ~fe).sum()),
                              fn=int((~pred_mean & fe).sum()), tn=int((~pred_mean & ~fe).sum())),
        feasibility_margin=dict(tp=int((pred_marg & fe).sum()), fp=int((pred_marg & ~fe).sum()),
                                fn=int((~pred_marg & fe).sum()), tn=int((~pred_marg & ~fe).sum())),
        kappa=kappa)
    print(f"surrogate on the grid: cycle RMSE {sur['cycle_all']['rmse']:.0f} EFPD "
          f"(R2 {sur['cycle_all']['r2']:.3f}), peaking RMSE {sur['peaking_all']['rmse']:.4f} "
          f"(R2 {sur['peaking_all']['r2']:.3f}); feasibility mean rule "
          f"{sur['feasibility_mean']}, margin rule {sur['feasibility_margin']}")

    res = dict(
        grid=dict(path=a.grid, n=len(g["X"]), n_feasible=int(g["feas"].sum()),
                  front=fg.tolist(), wall_h=t_g),
        opt=dict(path=a.opt, n=len(o["X"]), n_feasible=int(o["feas"].sum()),
                 front=fo.tolist(), wall_h=t_o,
                 hv_history=o["ck"].get("hv_history")),
        frozen=g["meta"].get("frozen"), names=g["names"],
        normalisation=dict(lo=lo.tolist(), hi=hi.tolist(), range=rng.tolist()),
        hv=dict(ref_nadir=ref_nadir.tolist(), grid=hv_g_n, opt=hv_o_n,
                ratio=hv_o_n / hv_g_n if hv_g_n > 0 else None,
                ref_phys=ref_phys.tolist(), grid_phys=hv_g_p, opt_phys=hv_o_p,
                ratio_phys=hv_o_p / hv_g_p if hv_g_p > 0 else None),
        igd=igd, igd_max=igd_max, gd=gd,
        eps_plus=dict(opt_covers_grid=e_og, grid_covers_opt=e_go,
                      opt_covers_grid_efpd=e_og * rng[0], opt_covers_grid_fdh=e_og * rng[1]),
        coverage=dict(tol=a.tol, within_tol=cov_eps, strict=cov_strict),
        union_front=dict(from_grid=n_union_grid, from_opt=n_union_opt,
                         opt_points_no_grid_node_dominates=new_pts),
        grid_feasible_nodes_dominated_by_opt=int(dom_by_opt.sum()),
        cost_ratio=cost_ratio, surrogate_on_grid=sur, meta_diffs=diffs)
    (out / "valgrid_compare.json").write_text(json.dumps(res, indent=2, default=float))
    print("  wrote valgrid_compare.json")

    # -------- figures ----------------------------------------------------------
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(-g["F"][~fe, 0], g["F"][~fe, 1], s=22, facecolors="none",
               edgecolors="0.55", label="grid node, infeasible")
    ax.scatter(-g["F"][fe, 0], g["F"][fe, 1], s=22, c="0.55", label="grid node, feasible")
    og = np.argsort(-Fg[:, 0])
    ax.plot(-Fg[og, 0], Fg[og, 1], "-", c="0.3", lw=1.0, label="enumerated front")
    ax.scatter(-o["F"][:, 0], o["F"][:, 1], s=26, marker="^", c="tab:orange",
               edgecolors="k", linewidths=0.4, label="search evaluations")
    oo = np.argsort(-Fo[:, 0])
    ax.plot(-Fo[oo, 0], Fo[oo, 1], "-", c="crimson", lw=1.2, label="recovered front")
    ax.scatter(-Fo[:, 0], Fo[:, 1], s=40, marker="^", c="crimson", edgecolors="k",
               linewidths=0.4, zorder=4)
    ax.set_xlabel("cycle length [EFPD]")
    ax.set_ylabel(r"$F_{\Delta H}$")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "valgrid_objective")

    ie, ig = g["names"].index("enrich"), g["names"].index("gd_wt")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(g["X"][~fe, ie], g["X"][~fe, ig], s=60, marker="s", facecolors="white",
               edgecolors="0.5", label="grid node, infeasible")
    ax.scatter(g["X"][fe, ie], g["X"][fe, ig], s=60, marker="s", c="0.6", label="grid node, feasible")
    ax.scatter(g["X"][fg, ie], g["X"][fg, ig], s=110, marker="s", facecolors="none",
               edgecolors="k", linewidths=1.4, label="enumerated front")
    ax.scatter(o["X"][~o["feas"], ie], o["X"][~o["feas"], ig], s=34, marker="^",
               facecolors="white", edgecolors="tab:orange", label="search, infeasible")
    ax.scatter(o["X"][o["feas"], ie], o["X"][o["feas"], ig], s=34, marker="^",
               c="tab:orange", edgecolors="k", linewidths=0.4, label="search, feasible")
    ax.scatter(o["X"][fo, ie], o["X"][fo, ig], s=70, marker="^", c="crimson",
               edgecolors="k", linewidths=0.6, label="recovered front")
    # annotate which constraint fails at each infeasible node
    for i in np.flatnonzero(~fe):
        viol = [c.replace("g_", "") for c in g["cons"] if float(g["raw"][i][c]) > 1e-9]
        ax.annotate(",".join(viol), (g["X"][i, ie], g["X"][i, ig]), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points", color="0.4")
    ax.set_xlabel("enrichment [wt% U-235]")
    ax.set_ylabel(r"Gd$_2$O$_3$ [wt%]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5,
              ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, "valgrid_design")

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, y, mu, sd, lab in ((axs[0], y_c, m_c, s_c, "cycle length [EFPD]"),
                               (axs[1], y_p, m_p, s_p, r"$F_{\Delta H}$")):
        lo_, hi_ = min(y.min(), (mu - sd).min()), max(y.max(), (mu + sd).max())
        pad = 0.04 * (hi_ - lo_ if hi_ > lo_ else 1.0)
        ax.plot([lo_ - pad, hi_ + pad], [lo_ - pad, hi_ + pad], "-", c="0.5", lw=0.8)
        ax.errorbar(y[~fe], mu[~fe], yerr=sd[~fe], fmt="o", ms=3.2, mfc="white", mec="0.35",
                    ecolor="0.6", elinewidth=0.6, label="infeasible node")
        ax.errorbar(y[fe], mu[fe], yerr=sd[fe], fmt="o", ms=3.6, c="tab:blue",
                    elinewidth=0.7, label="feasible node")
        ax.set_xlabel(f"true {lab}")
        ax.set_ylabel(f"predicted {lab}")
        ax.set_xlim(lo_ - pad, hi_ + pad)
        ax.set_ylim(lo_ - pad, hi_ + pad)
        ax.set_aspect("equal", adjustable="box")
    axs[0].legend(loc="upper left")
    for ax, t in zip(axs, "ab"):
        ax.set_title(f"({t})", loc="left")
    fig.tight_layout()
    save(fig, out, "valgrid_surrogate")

    # -------- table ------------------------------------------------------------
    def f(v, nd=3):
        return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"
    lines = [
        r"% valgrid_table.tex -- written by valgrid_compare.py.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{threeparttable}",
        r"  \caption[Search against exhaustive enumeration]{Surrogate-assisted search against exhaustive enumeration on the two-variable slice.}",
        r"  \label{tab:valgrid}",
        r"  \begin{tabular}{lrr}",
        r"    \toprule",
        r"    Quantity & Enumeration & Search \\",
        r"    \midrule",
        f"    Truth evaluations & {len(g['X'])} & {len(o['X'])} \\\\",
        f"    Feasible designs & {int(g['feas'].sum())} & {int(o['feas'].sum())} \\\\",
        f"    Front members & {len(fg)} & {len(fo)} \\\\",
        f"    Hypervolume, nadir reference & {f(hv_g_n, 1)} & {f(hv_o_n, 1)} \\\\",
        f"    Hypervolume ratio & 1 & {f(hv_o_n / hv_g_n if hv_g_n else float('nan'))} \\\\",
        f"    Union-front members & {n_union_grid} & {n_union_opt} \\\\",
        r"    \midrule",
        f"    IGD, normalised & \\multicolumn{{2}}{{r}}{{{f(igd)}}} \\\\",
        f"    GD, normalised & \\multicolumn{{2}}{{r}}{{{f(gd)}}} \\\\",
        f"    $I_{{\\varepsilon+}}$(search, enumeration), normalised & \\multicolumn{{2}}{{r}}{{{f(e_og)}}} \\\\",
        f"    $I_{{\\varepsilon+}}$ in cycle length [EFPD] & \\multicolumn{{2}}{{r}}{{{f(e_og * rng[0], 0)}}} \\\\",
        f"    $I_{{\\varepsilon+}}$ in $F_{{\\Delta H}}$ & \\multicolumn{{2}}{{r}}{{{f(e_og * rng[1], 4)}}} \\\\",
        f"    Enumerated-front coverage, strict / within {a.tol:g} & \\multicolumn{{2}}{{r}}{{{f(cov_strict, 2)} / {f(cov_eps, 2)}}} \\\\",
        f"    Wall-clock [h] & {f(t_g, 1)} & {f(t_o, 1)} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}\footnotesize",
        r"    \item Objectives are normalised by the range of the union of the two fronts before the distance indicators are computed. IGD is the inverted generational distance from the enumerated front to the recovered front. GD is the generational distance from the recovered front to the enumerated front. $I_{\varepsilon+}$ is the additive epsilon indicator, the smallest shift that makes the recovered front weakly dominate every enumerated-front point. The hypervolume reference point is the nadir of the union of the two fronts plus ten per cent of the range.",
        f"    \\item Slice: {', '.join(f'{k} = {v:g}' for k, v in (g['meta'].get('frozen') or {}).items())}. Source: valgrid\\_compare.py.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    (out / "valgrid_table.tex").write_text("\n".join(lines) + "\n")
    print("  wrote valgrid_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
