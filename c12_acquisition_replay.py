#!/usr/bin/env python3
r"""
c12_acquisition_replay.py -- replay of the three acquisitions of Campaign 12
under the rule the campaign used and under two alternatives. No transport:
the surrogates, NSGA-II and the ranking only.

WHY
    In Campaigns 11 and 12 the margin rule (mean + kappa sd <= 0 on every
    constraint surrogate) admitted 0 of 300 candidates in every batch, so
    every batch was ordered by the smallest predicted violation. The replay
    shows the cause and tests the two remedies.

RULES
    asrun             the campaign: NSGA-II constrained on the surrogate mean,
                      the margin rule applied to its population as a filter
    pof               the existing probability-of-feasibility ranking
                      (acq_rule "pof"): uncertainty weighted by the product of
                      Phi(-mean / sd) over the constraint surrogates
    margin_in_search  NSGA-II constrained on mean + kappa sd (the option
                      margin_in_search added to reactor_optimization.py on
                      15 Nov 2026), then the campaign ranking

WHAT IS REPORTED, per rule and batch
    how many candidates the rule admits and which constraint excludes the
    others (count of candidates with mean + kappa sd > 0, and the median
    predicted value and sd, in pcm for the reactivity constraints); the six
    picks; and a judgement of the picks by the surrogates fitted on all 117
    designs (the best model available without transport): the expected number
    of feasible picks (sum of the probabilities of feasibility) and the
    hypervolume gain of the picks whose judged constraints are all satisfied,
    on the reference (f_max, boron limit), over the measured feasible front
    of the designs evaluated before the batch.
    The designs the campaign evaluated are judged the same way and listed
    with their measured feasibility and hypervolume gain. They are in the
    training data of the judging surrogates, so their judgement is in-sample.

    The replay on another machine does not reproduce the campaign picks
    design by design (scikit-learn 1.7.2 against 1.9.0 changes the fitted
    hyperparameters in the last digits and NSGA-II amplifies it), so the
    "asrun" row is the rule, not the six designs of the campaign.

USAGE (repository root, about 4 min per rule)
    python c12_acquisition_replay.py
    python c12_acquisition_replay.py --rules pof --nsga-gen 40      # quick check
Writes <out>/acquisition_replay.json and .md (default figs_c12_acq/).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm

import core_geometry as cg
import hv_stop as hvs
from reactor_optimization import ActiveLearningMOO, OptimizerConfig, campaign9_problem

CKPT = "results_wks720_2026-11-15/c12/out_c12/optimization_checkpoint.json"
RULES = {"asrun": dict(acq_rule="margin", margin_in_search=False),
         "pof": dict(acq_rule="pof", margin_in_search=False),
         "margin_in_search": dict(acq_rule="margin", margin_in_search=True)}
PCM = ("g_margin", "g_kmin", "g_kmax", "g_ctrl", "g_ctrl_peak")


class _NoEvaluator:
    n_calls = 0

    def evaluate(self, X):
        raise RuntimeError("the replay never evaluates")


def build(ck, rule, nsga_gen=None):
    meta = ck["meta"]
    lim, c9, pol = meta["limits"], meta["campaign9"], meta["surrogate_policy"]
    spec = campaign9_problem(float(c9["efpd_req"]), f_max=float(lim["f_max"]))
    spec.constraint_scales.update({"g_kmin": float(lim["k_min"]), "g_kmax": float(lim["k_max"]),
                                   "g_enr": float(lim["enr_max"]), "g_peak": float(lim["f_max"]),
                                   "g_geom": cg.R_VESSEL_INNER - cg.VESSEL_CLEARANCE_CM})
    spec.constraint_names.append("g_ctrl"); spec.constraint_scales["g_ctrl"] = 1.0
    spec.constraint_names.append("g_boron"); spec.constraint_scales["g_boron"] = float(c9["boron_ceiling_ppm"])
    spec.constraint_names[spec.constraint_names.index("g_efpd")] = "g_margin"
    spec.constraint_scales["g_margin"] = float(c9["cycle_core3d"]["margin_scale"])
    assert list(spec.constraint_names) == list(ck["constraint_names"]), spec.constraint_names
    cfg = OptimizerConfig(n_init=24, n_iter=1, n_infill=6, nsga_pop=int(pol["nsga_pop"]),
                          nsga_gen=int(nsga_gen or pol["nsga_gen"]), surrogate="gp", seed=1)
    cfg.infill_min_sep, cfg.feas_kappa = float(pol["infill_min_sep"]), float(pol["feas_kappa"])
    cfg.efpd_cap, cfg.feas_kappa_map, cfg.margin_constraints = None, pol["feas_kappa_map"], None
    cfg.con_kernel_nu = 2.5
    cfg.explore_sep, cfg.n_explore = float(pol["explore_sep"]), int(pol["n_explore"])
    cfg.acq_rule, cfg.margin_in_search = RULES[rule]["acq_rule"], RULES[rule]["margin_in_search"]
    return spec, cfg


def blocks_of(ck):
    n0, out = 0, []
    for p in ck["phase_log"]:
        m = int(p["n_eval"])
        if p["stage"] != "DOE":
            out.append((n0, n0 + m))
        n0 += m
    return out


def pof_of(con, X, gp_cols):
    m, s = (np.atleast_2d(v) for v in con.predict(X))
    return np.prod(norm.cdf(-m[:, gp_cols] / np.maximum(s[:, gp_cols], 1e-12)), axis=1), m


def judge(X, obj, con, gp_cols, front_before, ref):
    """Expected number of feasible designs and hypervolume gain of the judged-feasible ones."""
    p, gm = pof_of(con, X, gp_cols)
    f = np.atleast_2d(obj.predict(X)[0])
    ok = np.all(gm[:, gp_cols] <= 0.0, axis=1)
    hv0 = hvs.hv2d(front_before, ref)
    hv1 = hvs.hv2d(list(front_before) + [list(v) for v in f[ok]], ref)
    return dict(expected_feasible=float(p.sum()), judged_feasible=int(ok.sum()),
                hv_gain_pct=float(100.0 * (hv1 - hv0) / hv0) if hv0 > 0 else None,
                pof=[float(v) for v in p], objectives=f.tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    ap.add_argument("--nsga-gen", type=int, default=None, help="override, for a quick check")
    ap.add_argument("--out", default="figs_c12_acq")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    ck = json.loads(Path(a.checkpoint).read_text())
    raw, meta = ck["all_raw"], ck["meta"]
    ref = (float(meta["limits"]["f_max"]), float(meta["campaign9"]["boron_ceiling_ppm"]))
    blocks = blocks_of(ck)
    spec, cfg = build(ck, "asrun", a.nsga_gen)
    names, cn = spec.design_space.names, list(spec.constraint_names)
    scale = np.array([spec.g_scale(c) for c in cn])
    gp_cols = [j for j, c in enumerate(cn) if c not in spec.exact_constraints]
    kvec = np.array([float(cfg.feas_kappa_map.get(c, cfg.feas_kappa)) for c in cn])
    feasible = lambda r: all(float(r[c]) <= 1e-9 for c in cn)  # noqa: E731
    front = lambda n: [[r["peaking"], r["c_max"]] for r in raw[:n] if feasible(r)]  # noqa: E731

    # the judging surrogates: fitted on every evaluated design
    full = ActiveLearningMOO(spec, _NoEvaluator(), cfg)
    full._seed_from_raw(raw)
    j_obj = full._new_surrogate().fit(full.X, full.F)
    j_con = full._new_surrogate(constraints=True).fit(full.X, full.G)

    R = {"checkpoint": a.checkpoint, "reference": ref, "blocks": blocks, "rules": {}, "campaign": []}
    L = [f"# Campaign 12 acquisition replay ({a.checkpoint})", "",
         f"Reference of the hypervolume: F_dH {ref[0]}, c_max {ref[1]:.0f} ppm. Judged by the surrogates fitted on all {len(raw)} designs.", ""]

    for k, (s, e) in enumerate(blocks):
        X = np.array([[float(r[n]) for n in names] for r in raw[s:e]])
        J = judge(X, j_obj, j_con, gp_cols, front(s), ref)
        hv0, hv1 = hvs.hv2d(front(s), ref), hvs.hv2d(front(e), ref)
        R["campaign"].append(dict(batch=k + 1, measured_feasible=sum(feasible(r) for r in raw[s:e]),
                                  measured_hv_gain_pct=100.0 * (hv1 - hv0) / hv0, **J))

    for rule in a.rules:
        spec, cfg = build(ck, rule, a.nsga_gen)
        rows = []
        print(f"\n=== rule {rule}")
        for k, (s, e) in enumerate(blocks):
            opt = ActiveLearningMOO(spec, _NoEvaluator(), cfg)
            opt._seed_from_raw(raw[:s])
            Xinf, info = opt.propose(it=k, verbose=False)
            cand, con, acq = info["cand"], info["con_sur"], info["acq"]
            gm, gs = (np.atleast_2d(v) for v in con.predict(cand))
            upper = gm + kvec * gs
            excl = {}
            for j in gp_cols:
                u = scale[j] * (1e5 if cn[j] in PCM else 1.0)
                excl[cn[j]] = dict(n_mean_violated=int((gm[:, j] > 0).sum()), n_margin_violated=int((upper[:, j] > 0).sum()),
                                   median_mean=float(np.median(gm[:, j]) * u), median_sd=float(np.median(gs[:, j]) * u),
                                   unit="pcm" if cn[j] in PCM else "native")
            own_p, _ = pof_of(con, Xinf, gp_cols)
            row = dict(batch=k + 1, n_train=s, n_candidates=int(len(cand)),
                       n_admitted=int(acq["n_margin_feasible"]), relax_depth=int(acq["relax_depth"]),
                       n_explore=int(acq["n_explore"]), constraints=excl,
                       picks=[dict(zip(names, map(float, x))) for x in Xinf],
                       own_expected_feasible=float(own_p.sum()),
                       judged=judge(Xinf, j_obj, j_con, gp_cols, front(s), ref))
            rows.append(row)
            jd = row["judged"]
            print(f"  batch {k + 1}: admitted {row['n_admitted']}/{row['n_candidates']}, relaxed {row['relax_depth']}x, "
                  f"expected feasible {row['own_expected_feasible']:.1f} (own) {jd['expected_feasible']:.1f} (judged), "
                  f"judged gain {jd['hv_gain_pct']:.2f} %", flush=True)
        R["rules"][rule] = rows

    L += ["| Rule | Batch | Admitted | Relaxations | Expected feasible, own model | Expected feasible, judged | Judged feasible | Judged HV gain [%] |",
          "|---|---|---|---|---|---|---|---|"]
    for rule, rows in R["rules"].items():
        for r in rows:
            j = r["judged"]
            L.append(f"| {rule} | {r['batch']} | {r['n_admitted']}/{r['n_candidates']} | {r['relax_depth']} | "
                     f"{r['own_expected_feasible']:.1f} | {j['expected_feasible']:.1f} | {j['judged_feasible']} | {j['hv_gain_pct']:.2f} |")
    for c in R["campaign"]:
        L.append(f"| campaign, measured | {c['batch']} | 0/300 | | | {c['expected_feasible']:.1f} (in-sample) | "
                 f"{c['measured_feasible']} measured | {c['measured_hv_gain_pct']:.2f} measured |")
    L += ["", "Candidates with mean + kappa sd > 0, per constraint surrogate (median predicted value and sd of the candidates):", "",
          "| Rule | Batch | " + " | ".join(cn[j] for j in gp_cols) + " |", "|---|---|" + "---|" * len(gp_cols)]
    for rule, rows in R["rules"].items():
        for r in rows:
            L.append(f"| {rule} | {r['batch']} | " + " | ".join(
                f"{v['n_margin_violated']} ({v['median_mean']:+.0f} ± {v['median_sd']:.0f})" for v in r["constraints"].values()) + " |")
    L += ["", "Picks (enrichment wt%, gadolinia wt%, reflector cm, gadolinia pins -> judged F_dH, c_max, probability of feasibility):", ""]
    for rule, rows in R["rules"].items():
        for r in rows:
            L.append(f"- {rule}, batch {r['batch']}:")
            for p, f, q in zip(r["picks"], r["judged"]["objectives"], r["judged"]["pof"]):
                L.append("  - " + ", ".join(f"{p[n]:.2f}" for n in names) + f" -> {f[0]:.3f}, {f[1]:.0f} ppm, {q:.2f}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "acquisition_replay.json").write_text(json.dumps(R, indent=1))
    (out / "acquisition_replay.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/acquisition_replay.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
