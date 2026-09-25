#!/usr/bin/env python3
r"""
axial_ratio_model.py -- the axial-loss ratio, fitted on every design depleted
with eight axial layers, as a design-dependent floor on the assembly cycle
length for the Campaign 9 continuation.

WHAT
    ratio = E_3D / E_asm for each measured design. A linear model on named
    quantities of the evaluation record,

        ratio_hat(q) = b0 + sum_j b_j q_j ,

    every q_j clamped to the range of the fitted designs, so the model never
    extrapolates, and the requirement on the assembly cycle length carries a
    margin of kappa leave-one-out rms:

        E_req(q) = E_mission / max(ratio_hat(q) - kappa s_loo, r_min) .

    The evaluator then writes g_efpd = E_req - E_asm, and seed_c9_axial.py
    writes the same for every archived record, using the measured 3D cycle
    where a record has one.

WHY THIS FORM (25 Sep 2026, eleven measured designs)
    The constant floor of 2270 d admitted design 72 of the first
    continuation, which resolved to 1782 d against the 1826 d mission. The
    gadolinia-only fit of the archive, 0.8957 - 0.0189 Gd, admits it too
    (54 d leave-one-out). Over the quantities the campaign already stores,
    the gadolinia content together with the assembly reactivity hump gives
    38 d leave-one-out and the absorber inventory gd_wt x gd_pins gives 52 d.
    Run --report for the table. No single-variable proxy of this accuracy
    replaces the eight-layer confirmation, which is why run_c9_axial.sh
    confirms every new front member and refits between blocks.

MEASUREMENTS
    Every runs.json under the RUN_DIRS prefixes written by c9_dep_core3d.py
    with eight layers and the relative end of cycle, not right-censored. The
    record of a measured design is the archive entry of the checkpoint its
    directory maps to, and its four design variables are stored in the
    model, so the same design is recognised as measured in any later
    checkpoint by its variables rather than by its index.

USAGE (repository root)
    python axial_ratio_model.py --report
    python axial_ratio_model.py --fit gd_wt hump_asm_pcm --kappa 1.0 --out axial_ratio_model.json
    python axial_ratio_model.py --predict axial_ratio_model.json out_c9f/optimization_checkpoint.json 69 70 72
    python axial_ratio_model.py --selftest
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MISSION = 1826.0
R_MIN = 0.5
VARS = ("enrich", "gd_wt", "refl_thick", "gd_pins")
# run-directory prefix -> the checkpoint whose indices its runs.json uses
RUN_DIRS = {"c9_dep_core3d": "out_c9/optimization_checkpoint.json",
            "c9f_dep_core3d": "out_c9f/optimization_checkpoint.json",
            "c9a_dep_core3d": "out_c9a/optimization_checkpoint.json"}
DERIVED = {"inventory": lambda q: float(q["gd_wt"]) * float(q["gd_pins"])}
CANDIDATES = [[], ["gd_wt"], ["gd_pins"], ["inventory"], ["gd_wt", "gd_pins"], ["enrich"],
              ["refl_thick"], ["k_bol"], ["keff_core_bol"], ["hump_asm_pcm"], ["peaking"],
              ["peaking_asm"], ["c_max"], ["cycle_length"], ["gd_wt", "hump_asm_pcm"],
              ["inventory", "k_bol"], ["gd_pins", "hump_asm_pcm"],
              ["enrich", "gd_wt", "refl_thick", "gd_pins"]]


def quantity(q, name):
    if name in DERIVED:
        return DERIVED[name](q)
    if name not in q:
        raise KeyError(f"axial ratio model: regressor {name!r} is not in the record")
    return float(q[name])


def measurements(run_dirs=RUN_DIRS, root="."):
    """One row per measured design: the archive record plus ratio, efpd_3d,
    sigma_3d, source and x."""
    root = Path(root)
    rows, seen = [], set()
    for prefix, ckpath in run_dirs.items():
        p = root / ckpath
        if not p.exists():
            continue
        raw = json.loads(p.read_text())["all_raw"]
        for d in sorted(root.glob(prefix + "*")):
            rj = d / "runs.json"
            if not d.is_dir() or not rj.is_file():
                continue
            for r in json.loads(rj.read_text()).values():
                if int(r.get("layers", 0)) != 8 or r.get("eoc_mode", "relative") != "relative":
                    continue
                if r.get("censored"):
                    continue
                rec = raw[int(r["idx"])]
                x = [float(rec[v]) for v in VARS]
                key = tuple(round(v, 6) for v in x)
                if key in seen:
                    continue
                seen.add(key)
                q = dict(rec)
                E = float(rec["cycle_length"])
                q.update(source=f"{d.name}/d{r['idx']}", ratio=float(r["efpd"]) / E,
                         efpd_3d=float(r["efpd"]), sigma_3d=float(r.get("sigma_efpd") or 0.0), x=x)
                rows.append(q)
    return rows


def _X(rows, regs):
    cols = [np.ones(len(rows))] + [np.array([quantity(q, r) for q in rows]) for r in regs]
    return np.column_stack(cols)


def fit(rows, regs, kappa=1.0, mission=MISSION):
    y = np.array([q["ratio"] for q in rows])
    n = len(y)
    if n < len(regs) + 3:
        raise SystemExit(f"{n} measured designs cannot support {len(regs)} regressors")
    X = _X(rows, regs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        bi, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        pred[i] = X[i] @ bi
    E = np.array([float(q["cycle_length"]) for q in rows])
    wrong = [q["source"] for i, q in enumerate(rows)
             if ((pred[i] * E[i]) >= mission) != ((y[i] * E[i]) >= mission)]
    return dict(
        regressors=list(regs), intercept=float(b[0]), coef=[float(v) for v in b[1:]],
        ranges={r: [float(X[:, j + 1].min()), float(X[:, j + 1].max())] for j, r in enumerate(regs)},
        kappa=float(kappa), s_fit=float(np.sqrt(np.mean((X @ b - y) ** 2))),
        s_loo=float(np.sqrt(np.mean((pred - y) ** 2))), loo_wrong_verdicts=wrong, n=int(n),
        mission_efpd=float(mission), r_min=R_MIN,
        measured=[dict(source=q["source"], x=q["x"], ratio=float(q["ratio"]), efpd_3d=q["efpd_3d"],
                       sigma_3d=q["sigma_3d"], cycle_length=float(q["cycle_length"])) for q in rows],
        fitted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def predict(model, q):
    r = model["intercept"]
    clamped = []
    for name, b in zip(model["regressors"], model["coef"]):
        v = quantity(q, name)
        lo, hi = model["ranges"][name]
        if v < lo or v > hi:
            clamped.append(name)
            v = min(max(v, lo), hi)
        r += b * v
    return float(r), clamped


def requirement(model, q, mission=None):
    """The per-design floor on the assembly cycle length, and the fields the
    evaluator records with it."""
    mission = float(model.get("mission_efpd", MISSION) if mission is None else mission)
    r, clamped = predict(model, q)
    margin = float(model["kappa"]) * float(model["s_loo"])
    r_eff = max(r - margin, float(model.get("r_min", R_MIN)))
    return dict(axial_ratio_pred=r, axial_ratio_margin=margin, axial_clamped=clamped,
                axial_req_efpd=mission / r_eff,
                cycle_length_axial=float(q["cycle_length"]) * r)


def measured_lookup(model, q, tol=1e-6):
    x = [float(q[v]) for v in VARS]
    for m in model["measured"]:
        if all(abs(a - b) <= tol * max(1.0, abs(b)) for a, b in zip(x, m["x"])):
            return m
    return None


def load(path, kappa=None):
    m = json.loads(Path(path).read_text())
    if kappa is not None:
        m["kappa"] = float(kappa)
    return m


def describe(m):
    terms = " ".join(f"{b:+.6f} {r}" for r, b in zip(m["regressors"], m["coef"]))
    return (f"ratio = {m['intercept']:.4f} {terms}  (n {m['n']}, LOO {m['s_loo']:.4f}, "
            f"kappa {m['kappa']:g}, margin {m['kappa'] * m['s_loo']:.4f}, clamped to the fitted ranges)")


def summary(m):
    return {k: m[k] for k in ("regressors", "intercept", "coef", "ranges", "kappa", "s_loo", "n",
                              "mission_efpd", "fitted_utc")}


def report(rows, mission=MISSION):
    print(f"{len(rows)} measured designs: " + " ".join(q["source"] for q in rows))
    print(f"\n{'regressors':38s} {'fit rms':>8s} {'LOO rms':>8s} {'LOO d':>6s}  "
          f"wrong LOO verdicts against {mission:.0f} d")
    for regs in CANDIDATES:
        label = " + ".join(regs) or "constant"
        try:
            m = fit(rows, regs, mission=mission)
        except (KeyError, SystemExit) as exc:
            print(f"{label:38s}  skipped: {exc}")
            continue
        print(f"{label:38s} {m['s_fit']:8.4f} {m['s_loo']:8.4f} {m['s_loo'] * 2290:6.0f}  "
              f"{len(m['loo_wrong_verdicts'])}: {' '.join(m['loo_wrong_verdicts'])}")


def selftest():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(12):
        gd, hump, pins = rng.uniform(2, 7), rng.uniform(-1800, 1200), rng.uniform(12, 30)
        ratio = 0.95 - 0.03 * gd - 2e-5 * hump
        rows.append(dict(enrich=5.0, gd_wt=gd, refl_thick=5.66, gd_pins=pins, hump_asm_pcm=hump,
                         cycle_length=2300.0, ratio=ratio, efpd_3d=2300.0 * ratio, sigma_3d=8.0,
                         source=f"t/d{i}", x=[5.0, gd, 5.66, pins]))
    m = fit(rows, ["gd_wt", "hump_asm_pcm"], kappa=1.0)
    assert abs(m["intercept"] - 0.95) < 1e-9 and abs(m["coef"][0] + 0.03) < 1e-9, "fit does not recover the coefficients"
    assert m["s_loo"] < 1e-9, "noise-free data must give zero leave-one-out error"
    q = dict(enrich=5.0, gd_wt=9.0, refl_thick=5.66, gd_pins=20.0, hump_asm_pcm=0.0, cycle_length=2300.0)
    req = requirement(m, q)
    assert req["axial_clamped"] == ["gd_wt"], "a regressor outside the fitted range must be clamped"
    hi = m["ranges"]["gd_wt"][1]
    assert abs(req["axial_ratio_pred"] - (0.95 - 0.03 * hi)) < 1e-9, "clamping must use the range edge"
    assert abs(req["axial_req_efpd"] - MISSION / req["axial_ratio_pred"]) < 1e-6, "zero margin with zero LOO"
    found = measured_lookup(m, dict(zip(VARS, rows[3]["x"])))
    assert found is not None and found["source"] == "t/d3", "measured design not recognised by its variables"
    assert measured_lookup(m, dict(zip(VARS, [5.0, 1.0, 5.66, 12.0]))) is None, "an unmeasured design must not match"
    inv = fit(rows, ["inventory"])
    assert inv["regressors"] == ["inventory"] and inv["n"] == 12, "derived regressor"
    print("selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--mission", type=float, default=MISSION)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--fit", nargs="*", metavar="REGRESSOR")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--out", default="axial_ratio_model.json")
    ap.add_argument("--predict", nargs="+", metavar="ARG",
                    help="MODEL CHECKPOINT IDX [IDX ...]")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.predict:
        model = load(a.predict[0])
        raw = json.loads(Path(a.predict[1]).read_text())["all_raw"]
        print(describe(model))
        for i in (int(s) for s in a.predict[2:]):
            q = raw[i]
            m = measured_lookup(model, q)
            req = requirement(model, q, a.mission)
            E = float(q["cycle_length"])
            line = (f"  {i:3d}: E_asm {E:6.0f} d  ratio_hat {req['axial_ratio_pred']:.3f}  "
                    f"E_req {req['axial_req_efpd']:6.0f} d  g_efpd {req['axial_req_efpd'] - E:+6.0f}"
                    + (f"  clamped {req['axial_clamped']}" if req["axial_clamped"] else ""))
            if m:
                line += (f"  | measured ratio {m['ratio']:.3f}, {m['efpd_3d']:.0f} d, "
                         f"margin {m['efpd_3d'] - a.mission:+.0f} d")
            print(line)
        return 0
    rows = measurements(root=a.root)
    if a.report:
        report(rows, a.mission)
        return 0
    if a.fit is not None:
        m = fit(rows, a.fit, kappa=a.kappa, mission=a.mission)
        Path(a.out).write_text(json.dumps(m, indent=1))
        print(describe(m))
        print(f"  wrong leave-one-out verdicts: {m['loo_wrong_verdicts'] or 'none'}")
        print("  measured: " + " ".join(x["source"] for x in m["measured"]))
        print(f"wrote {a.out}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
