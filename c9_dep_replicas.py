#!/usr/bin/env python3
r"""
c9_dep_replicas.py -- check 1: noise and fidelity of the assembly depletion
that gives the Campaign 9 cycle length and, through the hump, c_max.

WHAT IT RUNS, per front design
    tier "campaign"    4000 x 60 (20 inactive), the archive fidelity
                       replica 1 = the campaign seed. It must reproduce the
                       archived cycle length exactly (the thesis says every
                       archived evaluation is reproducible; this measures it)
                       replicas 2..R = fresh seeds (salted design hash)
    tier "production"  16000 x 120 (30 inactive), the Campaign 5 tier
                       replica 1 = the campaign seed, more with --hi-reps
    Only the depletion is rerun. The peaking factor, the reactivity window
    and the boron points are beginning-of-life core solves at 100000 x 170
    and do not depend on the depletion fidelity: they are read from the
    archive, and c_max is recomputed from the new hump with the archived
    boron points, exactly as the campaign did (boron_objective.py).

    The depletion is OpenMCEvaluator._cycle_length itself, with the
    evaluator configured from the checkpoint metadata, so the schedule,
    the k-target table and the chunking are those of the campaign. Only
    the seed and the transport settings differ between runs.

WHAT --analyse REPORTS
    per design       archived value, replica-1 reproduction error, mean and
                     s.d. over the campaign-tier replicas, the propagated
                     bracketing sigma (thesis eq. sigma-efpd) of every run,
                     the production-tier value and the fidelity shift
    pooled           seed s.d. of EFPD and c_max at campaign fidelity, the
                     mean fidelity shift with its standard error, and a
                     paired t statistic over the designs
    front            whether the (F_dH, c_max) non-dominated set changes
                     under the replica mean and under the production value,
                     the EFPD order of the front, and the margin of every
                     design over the 1826 d floor at each tier
    files            <out>/summary.json, summary.txt, summary_table.tex

COST (wks720, 64 threads, from the archived depletion times)
    campaign tier    about 8 min per replica per design
    production tier  up to 8 x that (histories x 8), about 60 min per design
    default plan     5 designs x (4 + 1) runs, about 7.5 h

USAGE (repository root, openmc-env)
    python c9_dep_replicas.py --selftest
    python c9_dep_replicas.py --dry-run
    python c9_dep_replicas.py --threads 64
    python c9_dep_replicas.py --analyse
Resumable: every finished run is in <out>/runs.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

CAMPAIGN = dict(particles=4000, batches=60, inactive=20)
EFPD_REQ = 1826.0


# --------------------------------------------------------------- inputs ----
def load_ckpt(path):
    ck = json.loads(Path(path).read_text())
    return ck, ck["all_raw"], ck.get("meta", {}) or {}


def front_designs(front_json, ckpt):
    p = Path(front_json)
    if p.is_file():
        f = json.loads(p.read_text()).get("front")
        if f:
            return [int(i) for i in f]
    raw, cn = ckpt["all_raw"], ckpt["constraint_names"]
    feas = [i for i, r in enumerate(raw) if all(r[c] <= 0 for c in cn)]
    F = np.array([[raw[i]["peaking"], raw[i]["c_max"]] for i in feas])
    nd = [feas[i] for i in range(len(feas))
          if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(len(feas)) if j != i)]
    return sorted(nd)


def design_of(ckpt, idx):
    """The dict the campaign hashed for its seed: design_space.as_dict of the
    archived design variables. Identical keys, identical seed."""
    from reactor_optimization import campaign9_problem
    spec = campaign9_problem(float(ckpt["meta"]["campaign9"]["efpd_req"]))
    x = [ckpt["all_raw"][idx][v] for v in ckpt["design_variables"]]
    return spec.design_space.as_dict(np.asarray(x, float)), spec


def make_evaluator(ckpt, spec, workdir, transport):
    import openmc_evaluator as oe
    meta = ckpt["meta"]
    lim, ct, sch = meta["limits"], meta["core_transport"], meta["schedule"]
    return oe.OpenMCEvaluator(
        spec, k_target=meta["k_target"], transport=dict(transport),
        workdir=str(workdir), bol_steps=sch["bol_steps"], dep_step=sch["dep_step"],
        chunk_steps=sch["chunk_steps"], max_burnup=sch["max_burnup"],
        core_particles=ct["particles"], core_batches=ct["batches"], core_inactive=ct["inactive"],
        k_basis=lim["k_basis"], k_max=lim["k_max"], k_min=lim["k_min"],
        f_max=lim["f_max"], enr_max=lim["enr_max"], verbose=False)


def plan_runs(designs, reps, hi_reps, hi_tr, skip_rep1):
    runs = []
    for idx in designs:
        for r in range(1, reps + 1):
            if r == 1 and skip_rep1:
                continue
            runs.append(dict(idx=idx, tier="campaign", rep=r, transport=dict(CAMPAIGN),
                             salt="" if r == 1 else f"dep_rep{r}"))
        for r in range(1, hi_reps + 1):
            runs.append(dict(idx=idx, tier="production", rep=r, transport=dict(hi_tr),
                             salt="" if r == 1 else f"dep_rep{r}"))
    return runs


def key_of(run):
    return f"d{run['idx']}_{run['tier']}_r{run['rep']}"


def estimate_minutes(runs, raw):
    tot = 0.0
    for run in runs:
        t = float(raw[run["idx"]]["t_deplete_s"]) / 60.0
        ratio = (run["transport"]["particles"] * run["transport"]["batches"]) / (
            CAMPAIGN["particles"] * CAMPAIGN["batches"])
        run["est_min"] = t * ratio
        tot += run["est_min"]
    return tot


# ----------------------------------------------------------------- run ----
def do_run(ev, run, design, rec, meta, out):
    import dep_common as dc
    import openmc_evaluator as oe
    ev.transport = dict(run["transport"])
    salt = run["salt"]
    orig = oe._design_seed
    oe._design_seed = lambda d, salt_="": orig(d, salt=salt)      # what _cycle_length calls
    case = out / "work" / key_of(run)
    t0 = time.time()
    try:
        efpd, k_bol, kt, censored, bu_eoc, n = ev._cycle_length(design, case)
    finally:
        oe._design_seed = orig
    k_hist, bu_hist = list(ev._last_k_hist), list(ev._last_bu_hist)
    try:
        sdmap = dc.k_sd_from_case(case)
    except Exception as exc:                  # never lose a finished depletion
        print(f"      WARNING k_sd_from_case failed: {exc!r}", flush=True)
        sdmap = {}
    ksd = [sdmap.get(v) for v in k_hist]
    sig, ib, slope = dc.bracket_sigma_efpd(bu_hist, k_hist, ksd, kt, ev.spec_power)
    c9 = meta["campaign9"]
    obj = dc.hump_and_cmax(k_hist, float(rec["keff_core_bol"]), rec["boron_points"],
                           float(c9["hump_noise_pcm"]))
    return dict(
        key=key_of(run), idx=run["idx"], tier=run["tier"], rep=run["rep"], salt=salt,
        seed=int(orig(design, salt=salt)), transport=run["transport"],
        efpd=float(efpd), bu_eoc=float(bu_eoc), censored=bool(censored),
        k_target=float(kt), k_bol=float(k_bol), n_solves=int(n),
        k_hist=k_hist, k_sd=ksd, bu_hist=bu_hist,
        sigma_efpd=sig, bracket=ib, slope_pcm_per_mwdkg=slope,
        hump_asm_pcm=obj["hump_asm_pcm"], hump_core_pcm=obj["hump_core_pcm"],
        hump_core_op_pcm=obj["hump_core_op_pcm"], l_ax=obj["l_ax"], k_peak=obj["k_peak"],
        c_max=obj["c_max"], c_max_status=obj["c_max_status"], c_bol=obj["c_bol_ppm"],
        g_efpd=float(c9["efpd_req"]) - float(efpd),
        wall_s=time.time() - t0)


# ------------------------------------------------------------- analysis ----
def nondominated(F):
    F = np.asarray(F, float); n = len(F)
    return [i for i in range(n) if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                                           for j in range(n) if j != i)]


def analyse(done, designs, raw, hi_tr):
    R = {}
    for idx in designs:
        rec = raw[idx]
        camp = sorted([d for d in done.values() if d["idx"] == idx and d["tier"] == "campaign"],
                      key=lambda d: d["rep"])
        prod = sorted([d for d in done.values() if d["idx"] == idx and d["tier"] == "production"],
                      key=lambda d: d["rep"])
        row = dict(idx=idx, archive=dict(efpd=rec["cycle_length"], c_max=rec["c_max"],
                                        peaking=rec["peaking"], hump_asm_pcm=rec["hump_asm_pcm"],
                                        bu_eoc=rec["bu_eoc_mwd_kg"], n_solves=rec["n_dep_solves"]),
                   n_campaign=len(camp), n_production=len(prod))
        r1 = next((d for d in camp if d["rep"] == 1), None)
        row["rep1_efpd_error"] = (r1["efpd"] - rec["cycle_length"]) if r1 else None
        row["rep1_cmax_error"] = (r1["c_max"] - rec["c_max"]) if r1 else None
        for tier, runs in (("campaign", camp), ("production", prod)):
            if not runs:
                continue
            e = np.array([d["efpd"] for d in runs]); c = np.array([d["c_max"] for d in runs])
            h = np.array([d["hump_asm_pcm"] for d in runs])
            sig = [d["sigma_efpd"] for d in runs if d["sigma_efpd"] is not None]
            row[tier] = dict(
                n=len(runs), efpd=e.tolist(), c_max=c.tolist(), hump_asm_pcm=h.tolist(),
                efpd_mean=float(e.mean()), efpd_sd=float(e.std(ddof=1)) if len(e) > 1 else None,
                c_max_mean=float(c.mean()), c_max_sd=float(c.std(ddof=1)) if len(c) > 1 else None,
                sigma_efpd_propagated=(float(np.mean(sig)) if sig else None),
                censored=[d["censored"] for d in runs], wall_min=float(sum(d["wall_s"] for d in runs) / 60))
        if "campaign" in row and "production" in row:
            row["shift_efpd"] = row["production"]["efpd_mean"] - row["campaign"]["efpd_mean"]
            row["shift_c_max"] = row["production"]["c_max_mean"] - row["campaign"]["c_max_mean"]
        R[idx] = row

    # pooled statistics
    pooled = {}
    ss, dof = 0.0, 0
    ssc = 0.0
    for row in R.values():
        t = row.get("campaign")
        if t and t["n"] > 1:
            e = np.array(t["efpd"]); c = np.array(t["c_max"])
            ss += ((e - e.mean()) ** 2).sum(); ssc += ((c - c.mean()) ** 2).sum(); dof += t["n"] - 1
    pooled["seed_sd_efpd"] = math.sqrt(ss / dof) if dof else None
    pooled["seed_sd_c_max"] = math.sqrt(ssc / dof) if dof else None
    pooled["seed_dof"] = dof
    shifts = np.array([row["shift_efpd"] for row in R.values() if "shift_efpd" in row])
    if len(shifts):
        pooled["shift_efpd_mean"] = float(shifts.mean())
        pooled["shift_efpd_se"] = float(shifts.std(ddof=1) / math.sqrt(len(shifts))) if len(shifts) > 1 else None
        pooled["shift_t"] = (float(shifts.mean() / (shifts.std(ddof=1) / math.sqrt(len(shifts))))
                             if len(shifts) > 1 and shifts.std(ddof=1) > 0 else None)
        pooled["shift_n"] = int(len(shifts))
    hist_ratio = (hi_tr["particles"] * hi_tr["batches"]) / (CAMPAIGN["particles"] * CAMPAIGN["batches"])
    pooled["expected_sd_ratio"] = 1.0 / math.sqrt(hist_ratio)

    # front under each reading of c_max
    fronts = {}
    for label, getter in (("archive", lambda r: r["archive"]["c_max"]),
                          ("campaign_mean", lambda r: r.get("campaign", {}).get("c_max_mean")),
                          ("production", lambda r: r.get("production", {}).get("c_max_mean"))):
        pts = [(idx, R[idx]["archive"]["peaking"], getter(R[idx])) for idx in designs
               if getter(R[idx]) is not None]
        if len(pts) == len(designs):
            nd = nondominated([[p, c] for _, p, c in pts])
            fronts[label] = sorted(pts[i][0] for i in nd)
    order = {}
    for label, getter in (("archive", lambda r: r["archive"]["efpd"]),
                          ("campaign_mean", lambda r: r.get("campaign", {}).get("efpd_mean")),
                          ("production", lambda r: r.get("production", {}).get("efpd_mean"))):
        v = [(idx, getter(R[idx])) for idx in designs if getter(R[idx]) is not None]
        if v:
            order[label] = [i for i, _ in sorted(v, key=lambda t: -t[1])]
            order[label + "_margin_d"] = {int(i): float(e - EFPD_REQ) for i, e in v}
    return dict(designs=designs, per_design=R, pooled=pooled, fronts=fronts, efpd_order=order)


def fmt(v, f="{:.1f}"):
    return "   --" if v is None else f.format(v)


def report(S):
    L = []
    P = S["pooled"]
    L.append("=== Check 1: replicas and fidelity of the assembly depletion")
    L.append(f"{'design':>7} {'EFPD arch':>9} {'rep1 err':>8} {'EFPD mean':>9} {'sd':>6} {'sig prop':>8} "
             f"{'EFPD prod':>9} {'shift':>6} | {'c_max arch':>10} {'mean':>6} {'sd':>5} {'prod':>6} {'shift':>6}")
    for idx in S["designs"]:
        r = S["per_design"][idx]; c = r.get("campaign", {}); p = r.get("production", {})
        L.append(f"  C9-{idx:<3d} {r['archive']['efpd']:9.1f} {fmt(r['rep1_efpd_error'], '{:+8.3f}'):>8} "
                 f"{fmt(c.get('efpd_mean')):>9} {fmt(c.get('efpd_sd')):>6} {fmt(c.get('sigma_efpd_propagated')):>8} "
                 f"{fmt(p.get('efpd_mean')):>9} {fmt(r.get('shift_efpd'), '{:+.1f}'):>6} | "
                 f"{r['archive']['c_max']:10.0f} {fmt(c.get('c_max_mean'), '{:.0f}'):>6} "
                 f"{fmt(c.get('c_max_sd'), '{:.0f}'):>5} {fmt(p.get('c_max_mean'), '{:.0f}'):>6} "
                 f"{fmt(r.get('shift_c_max'), '{:+.0f}'):>6}")
    L.append("")
    L.append(f"pooled seed s.d. at campaign fidelity : EFPD {fmt(P['seed_sd_efpd'])} d, "
             f"c_max {fmt(P['seed_sd_c_max'], '{:.0f}')} ppm  ({P['seed_dof']} degrees of freedom)")
    if "shift_efpd_mean" in P:
        L.append(f"fidelity shift, production - campaign : {P['shift_efpd_mean']:+.1f} d, "
                 f"s.e. {fmt(P['shift_efpd_se'])} d, t {fmt(P['shift_t'], '{:.2f}')} over {P['shift_n']} designs")
    L.append(f"expected s.d. ratio production/campaign: {P['expected_sd_ratio']:.2f} (histories)")
    L.append("")
    for k, v in S["fronts"].items():
        L.append(f"(F_dH, c_max) front, c_max from {k:14s}: {v}")
    for k, v in S["efpd_order"].items():
        if not k.endswith("_margin_d"):
            m = S["efpd_order"][k + "_margin_d"]
            L.append(f"EFPD order, {k:14s}: {v}   margins over {EFPD_REQ:.0f} d: "
                     + ", ".join(f"C9-{i} {m[i]:+.0f}" for i in v))
    return "\n".join(L)


def latex_table(S):
    rows = []
    for idx in S["designs"]:
        r = S["per_design"][idx]; c = r.get("campaign", {}); p = r.get("production", {})
        rows.append(f"    C9-{idx} & {r['archive']['efpd']:.0f} & {fmt(c.get('efpd_mean'), '{:.0f}')} & "
                    f"{fmt(c.get('efpd_sd'), '{:.0f}')} & {fmt(p.get('efpd_mean'), '{:.0f}')} & "
                    f"{fmt(r.get('shift_efpd'), '{:+.0f}')} & {r['archive']['c_max']:.0f} & "
                    f"{fmt(c.get('c_max_sd'), '{:.0f}')} & {fmt(p.get('c_max_mean'), '{:.0f}')} \\\\")
    P = S["pooled"]
    return "\n".join([
        "\\begin{table}[htbp]", "  \\centering", "  \\small",
        "  \\caption{Campaign 9 front: replicas and production fidelity of the assembly depletion. "
        "Cycle length in d, boron in ppm.}",
        "  \\label{tab:c9-dep-replicas}",
        "  \\begin{tabular}{@{}lrrrrrrrr@{}}", "    \\toprule",
        "    Design & EFPD & mean & s.d. & prod. & shift & $c_\\mathrm{max}$ & s.d. & prod. \\\\",
        "    & archive & \\multicolumn{2}{c}{$4\\,000\\times60$} & $16\\,000\\times120$ & & archive & "
        "$4\\,000\\times60$ & $16\\,000\\times120$ \\\\",
        "    \\midrule", *rows, "    \\bottomrule", "  \\end{tabular}", "\\end{table}",
        f"% pooled seed s.d.: EFPD {fmt(P['seed_sd_efpd'])} d, c_max {fmt(P['seed_sd_c_max'], '{:.0f}')} ppm, "
        f"{P['seed_dof']} dof; shift {fmt(P.get('shift_efpd_mean'), '{:+.1f}')} +- {fmt(P.get('shift_efpd_se'))} d"])


# ------------------------------------------------------------- selftest ----
def selftest():
    raw = {1: dict(cycle_length=1900.0, c_max=1400.0, peaking=1.50, hump_asm_pcm=0.0, bu_eoc_mwd_kg=19.0, n_dep_solves=8),
           2: dict(cycle_length=1850.0, c_max=1700.0, peaking=1.49, hump_asm_pcm=300.0, bu_eoc_mwd_kg=18.5, n_dep_solves=8)}
    done = {}
    rng = np.random.default_rng(1)
    for idx in (1, 2):
        for r in range(1, 5):
            e = raw[idx]["cycle_length"] + (0.0 if r == 1 else rng.normal(0, 25))
            done[f"d{idx}_campaign_r{r}"] = dict(idx=idx, tier="campaign", rep=r, efpd=e, c_max=raw[idx]["c_max"] + (0 if r == 1 else rng.normal(0, 30)),
                                                hump_asm_pcm=0.0, sigma_efpd=27.0, censored=False, wall_s=460.0)
        done[f"d{idx}_production_r1"] = dict(idx=idx, tier="production", rep=1, efpd=raw[idx]["cycle_length"] + 40.0,
                                            c_max=raw[idx]["c_max"] - 20.0, hump_asm_pcm=0.0, sigma_efpd=9.0, censored=False, wall_s=3700.0)
    S = analyse(done, [1, 2], raw, dict(particles=16000, batches=120, inactive=30))
    assert S["per_design"][1]["rep1_efpd_error"] == 0.0
    assert S["pooled"]["seed_dof"] == 6 and 5 < S["pooled"]["seed_sd_efpd"] < 60
    assert abs(S["pooled"]["shift_efpd_mean"] - 40.0) < 30, S["pooled"]
    assert S["fronts"]["archive"] == [1, 2] and set(S["efpd_order"]["archive"]) == {1, 2}
    txt = report(S); tex = latex_table(S)
    assert "C9-1" in txt and "\\toprule" in tex
    runs = plan_runs([1, 2], 4, 1, dict(particles=16000, batches=120, inactive=30), False)
    assert len(runs) == 10 and runs[0]["salt"] == "" and runs[1]["salt"] == "dep_rep2"
    assert len(plan_runs([1], 4, 1, {}, True)) == 4
    tot = estimate_minutes(runs, {1: dict(t_deplete_s=468.0), 2: dict(t_deplete_s=468.0)})
    assert abs(tot - (8 * 7.8 + 2 * 7.8 * 8)) < 1.0, tot
    print("selftest OK")
    return 0


# ----------------------------------------------------------------- main ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--front-json", default="c9_post/c9_front.json")
    ap.add_argument("--designs", type=int, nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=4, help="campaign-tier replicas, replica 1 = campaign seed")
    ap.add_argument("--hi-reps", type=int, default=1, help="production-tier replicas")
    ap.add_argument("--hi-particles", type=int, default=16000)
    ap.add_argument("--hi-batches", type=int, default=120)
    ap.add_argument("--hi-inactive", type=int, default=30)
    ap.add_argument("--skip-rep1", action="store_true", help="skip the campaign-seed reproduction")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="c9_dep_replicas")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ckpt, raw, meta = load_ckpt(a.checkpoint)
    designs = a.designs or front_designs(a.front_json, ckpt)
    hi_tr = dict(particles=a.hi_particles, batches=a.hi_batches, inactive=a.hi_inactive)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = out / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    runs = plan_runs(designs, a.reps, a.hi_reps, hi_tr, a.skip_rep1)
    tot = estimate_minutes(runs, raw)
    todo = [r for r in runs if key_of(r) not in done]

    if a.analyse:
        S = analyse(done, designs, raw, hi_tr)
        (out / "summary.json").write_text(json.dumps(S, indent=1))
        txt = report(S); (out / "summary.txt").write_text(txt + "\n")
        (out / "summary_table.tex").write_text(latex_table(S) + "\n")
        print(txt); print(f"\nwrote {out}/summary.json, summary.txt, summary_table.tex")
        return 0

    print(f"check 1: designs {designs}, {len(runs)} runs, {len(todo)} to do, "
          f"{len(done)} cached in {store}")
    print(f"  schedule {meta.get('schedule')}  target table {meta.get('k_target')}")
    for r in runs:
        print(f"  {key_of(r):24s} {r['transport']}  salt {r['salt']!r:12s} "
              f"est {r['est_min']:5.1f} min  {'cached' if key_of(r) in done else ''}")
    print(f"  total estimate {tot / 60:.1f} h, remaining "
          f"{sum(r['est_min'] for r in todo) / 60:.1f} h (production tier scaled by histories, upper bound)")
    if a.dry_run:
        return 0

    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import openmc
    print(f"  openmc {openmc.__version__}, chain {os.environ.get('OPENMC_CHAIN_FILE')}")
    for run in todo:
        idx = run["idx"]
        design, spec = design_of(ckpt, idx)
        ev = make_evaluator(ckpt, spec, out / "work", run["transport"])
        print(f"[{key_of(run)}] start {time.strftime('%H:%M:%S')}  est {run['est_min']:.0f} min", flush=True)
        res = do_run(ev, run, design, raw[idx], meta, out)
        done[res["key"]] = res
        store.write_text(json.dumps(done, indent=1))
        err = res["efpd"] - raw[idx]["cycle_length"]
        print(f"[{key_of(run)}] EFPD {res['efpd']:7.1f} (archive {raw[idx]['cycle_length']:7.1f}, "
              f"{err:+.2f})  sigma {fmt(res['sigma_efpd'])} d  c_max {res['c_max']:.0f} "
              f"(archive {raw[idx]['c_max']:.0f})  hump_asm {res['hump_asm_pcm']:+.0f} pcm  "
              f"[{res['n_solves']} solves, {res['wall_s'] / 60:.1f} min]", flush=True)
    print("all runs done; run --analyse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
