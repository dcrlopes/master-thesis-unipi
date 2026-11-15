#!/usr/bin/env python3
r"""
c12_objective_replicas.py -- the two objectives of the near-front designs of
Campaign 12, evaluated again with independent seeds, to decide which designs
are front members once the Monte Carlo noise of the objectives is counted.

WHY
    The full-fidelity confirmation of the loop repeats the cycle margin, not
    the objectives. F_dH and c_max of every design are one beginning-of-life
    core solve each (100 000 x 170), and the stored value of a front member
    was selected for being low. C12-112 displaced C10-56 and C11-89 by 0.009
    to 0.023 in F_dH and 11 to 66 ppm, the size of the seed noise of F_dH.

WHAT IT EXECUTES, per design and replica (fresh seed, salt "-objrepN")
    the unrodded zoned core at the reference boron, at --ref-transport
    (default 400 000 x 170, four times the campaign histories), which gives
    F_dH and the reactivity at 1000 ppm, and the boron solves the campaign
    loop would add (boron_objective.next_concentration), at the campaign core
    statistics. c_max is boron_objective.boron_objective of those points.
    The hump of the design is read from the checkpoint: every design selected
    here has hump_core_pcm = 0, so c_max is the beginning-of-life critical
    concentration and no depletion enters it. A design with a positive hump
    is reported with a warning, because its depletion noise is not sampled.

DESIGNS
    --designs, or by default the feasible designs that no other feasible
    design dominates by more than --eps (F_dH 0.01, c_max 15 ppm).

WHAT --analyse REPORTS (no transport)
    per design: stored value, replica mean and its standard error (pooled
    seed s.d. over the designs), for both objectives; the pooled seed s.d.;
    stored minus replica mean for the stored front and for the others (the
    selection effect); the front of the replica means; and for each design
    the probability of being non-dominated when the means are sampled from
    their standard errors.

USAGE (repository root)
    python c12_objective_replicas.py --selftest
    python c12_objective_replicas.py --dry-run
    python c12_objective_replicas.py --threads 64          # wks720, openmc-env
    python c12_objective_replicas.py --analyse
Resumable: every finished replica is in <out>/runs.json.
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

CKPT = "out_c12/optimization_checkpoint.json"
HUMP_KEYS = ("k_peak", "bu_index", "hump_asm_pcm", "hump_xe_pcm", "l_ax", "hump_core_pcm", "hump_core_op_pcm")


def feasible(raw, cn):
    return [i for i, r in enumerate(raw) if all(float(r[c]) <= 0 for c in cn)]


def nondominated(F, eps=(0.0, 0.0)):
    """Rows of F no other row dominates by more than eps (minimisation)."""
    F = np.asarray(F, float)
    e = np.asarray(eps, float)
    out = []
    for i in range(len(F)):
        dom = any(j != i and np.all(F[j] <= F[i] - e) and (np.any(F[j] < F[i] - e) or np.any(e > 0))
                  for j in range(len(F)))
        if not dom:
            out.append(i)
    return out


def select(ck, eps):
    raw = ck["all_raw"]
    feas = feasible(raw, ck["constraint_names"])
    F = [[raw[i]["peaking"], raw[i]["c_max"]] for i in feas]
    return sorted(feas[i] for i in nondominated(F, eps))


def key_of(idx, rep):
    return f"d{idx}_r{rep}"


# ----------------------------------------------------------------- run ----
def do_run(ev, design, rec, rep, ref_tr, work):
    import boron_objective as bo
    import openmc_evaluator as oe
    orig = oe._design_seed
    oe._design_seed = lambda d, salt="": orig(d, salt=f"{salt}-objrep{rep}")
    camp = (ev.core_particles, ev.core_batches, ev.core_inactive)
    ev.workdir = Path(work) / key_of(rec["_idx"], rep)
    ev.workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        ev.core_particles, ev.core_batches, ev.core_inactive = ref_tr
        core = ev._bol_core_peaking(design, ev.workdir)
        ev.core_particles, ev.core_batches, ev.core_inactive = camp
        hump = {k: rec.get(k) for k in HUMP_KEYS}
        points = {bo.PPM_REF: bo.rho_pcm(float(core["keff_core"]))}
        targets = [0.0, -float(hump["hump_core_pcm"]), -float(hump["hump_core_op_pcm"])]
        while True:
            nxt = bo.next_concentration(points, targets, top=float(ev.c9_ppm_top), step=float(ev.c9_ppm_step))
            if nxt is None:
                break
            points[float(nxt)] = bo.rho_pcm(float(oe._boron_solve(ev, design, nxt)["keff"]))
        obj = bo.boron_objective(points, hump)
    finally:
        oe._design_seed = orig
        ev.core_particles, ev.core_batches, ev.core_inactive = camp
    return dict(key=key_of(rec["_idx"], rep), idx=rec["_idx"], rep=rep, peaking=float(core["fdh_core"]),
                keff_core=float(core["keff_core"]), entropy_conv=core.get("entropy_conv"),
                c_max=float(obj["c_max_ppm"]), c_max_status=obj["c_max_status"], c_bol=float(obj["c_bol_ppm"]),
                boron_points=obj["boron_points"], ref_transport=list(ref_tr), wall_s=time.time() - t0)


# ------------------------------------------------------------- analysis ----
def analyse(done, designs, raw, n_draw=20000, seed=1):
    rows, ss, dof = {}, np.zeros(2), 0
    for idx in designs:
        v = np.array([[d["peaking"], d["c_max"]] for d in done.values() if d["idx"] == idx])
        if len(v) == 0:
            continue
        rows[idx] = dict(n=len(v), stored=[raw[idx]["peaking"], raw[idx]["c_max"]], mean=v.mean(axis=0).tolist(),
                         values=v.tolist(), hump_core_pcm=float(raw[idx].get("hump_core_pcm") or 0.0),
                         confirmed=bool(raw[idx].get("confirmed_3d")))
        if len(v) > 1:
            ss += ((v - v.mean(axis=0)) ** 2).sum(axis=0)
            dof += len(v) - 1
    sd = np.sqrt(ss / dof) if dof else np.array([np.nan, np.nan])
    ids = sorted(rows)
    for idx in ids:
        rows[idx]["se"] = (sd / math.sqrt(rows[idx]["n"])).tolist()
    S = np.array([rows[i]["stored"] for i in ids])
    M = np.array([rows[i]["mean"] for i in ids])
    E = np.array([rows[i]["se"] for i in ids])
    f_stored = [ids[i] for i in nondominated(S)]
    f_mean = [ids[i] for i in nondominated(M)]
    prob = np.zeros(len(ids))
    if dof:
        rng = np.random.default_rng(seed)
        for _ in range(n_draw):
            prob[nondominated(M + E * rng.standard_normal(M.shape))] += 1
        prob /= n_draw
    for i, idx in enumerate(ids):
        rows[idx]["p_front"] = float(prob[i]) if dof else None
    on = np.array([i in f_stored for i in ids])
    shift = S - M                                   # stored minus replica mean
    sel = {name: dict(n=int(m.sum()), peaking=float(shift[m, 0].mean()) if m.any() else None,
                      c_max=float(shift[m, 1].mean()) if m.any() else None)
           for name, m in (("stored_front", on), ("others", ~on))}
    return dict(designs=ids, per_design=rows, seed_sd=dict(peaking=float(sd[0]), c_max=float(sd[1]), dof=int(dof)),
                front_stored=f_stored, front_replica_mean=f_mean, selection=sel)


def report(S):
    sd = S["seed_sd"]
    L = ["=== Campaign 12: the objectives of the near-front designs under independent seeds",
         f"{'design':>7} {'n':>2} | {'F_dH':>6} {'mean':>6} {'s.e.':>6} | {'c_max':>6} {'mean':>6} {'s.e.':>5} | "
         f"{'P(front)':>8}  notes"]
    for idx in S["designs"]:
        r = S["per_design"][idx]
        note = ("stored front " if idx in S["front_stored"] else "") + ("confirmed " if r["confirmed"] else "") \
            + ("HUMP > 0, depletion noise not sampled" if r["hump_core_pcm"] > 0 else "")
        p = "    --" if r["p_front"] is None else f"{r['p_front']:8.2f}"
        L.append(f"  {idx:>5d} {r['n']:>2d} | {r['stored'][0]:6.3f} {r['mean'][0]:6.3f} {r['se'][0]:6.4f} | "
                 f"{r['stored'][1]:6.0f} {r['mean'][1]:6.0f} {r['se'][1]:5.1f} | {p}  {note}")
    L += ["", f"pooled seed s.d. of one replica: F_dH {sd['peaking']:.4f}, c_max {sd['c_max']:.1f} ppm ({sd['dof']} degrees of freedom)",
          f"front of the stored values : {S['front_stored']}", f"front of the replica means : {S['front_replica_mean']}"]
    for name, v in S["selection"].items():
        if v["n"]:
            L.append(f"stored minus replica mean, {name:12s} ({v['n']:2d} designs): F_dH {v['peaking']:+.4f}, c_max {v['c_max']:+.1f} ppm")
    return "\n".join(L)


def selftest():
    rng = np.random.default_rng(3)
    raw = {0: dict(peaking=1.50, c_max=2200.0), 1: dict(peaking=1.52, c_max=1800.0), 2: dict(peaking=1.60, c_max=1500.0),
           3: dict(peaking=1.61, c_max=1900.0, hump_core_pcm=500.0)}
    done = {}
    for idx, r in raw.items():
        for rep in range(1, 5):
            done[key_of(idx, rep)] = dict(idx=idx, peaking=r["peaking"] + rng.normal(0, 0.005), c_max=r["c_max"] + rng.normal(0, 5.0))
    S = analyse(done, [0, 1, 2, 3], raw, n_draw=2000)
    assert S["front_stored"] == [0, 1, 2] and S["front_replica_mean"] == [0, 1, 2], S
    assert S["seed_sd"]["dof"] == 12 and 0.002 < S["seed_sd"]["peaking"] < 0.01
    assert S["per_design"][0]["p_front"] > 0.99 and S["per_design"][3]["p_front"] < 0.01
    assert nondominated([[1.50, 2000.0], [1.505, 2010.0]]) == [0]
    assert nondominated([[1.50, 2000.0], [1.505, 2010.0]], eps=(0.01, 15.0)) == [0, 1]
    assert "HUMP > 0" in report(S)
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--designs", type=int, nargs="*", default=None)
    ap.add_argument("--eps", type=float, nargs=2, default=(0.01, 15.0), metavar=("F_DH", "PPM"))
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--ref-transport", default="400000,170,60", help="particles,batches,inactive of the reference core solve")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="c12_objective_replicas")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ck = json.loads(Path(a.checkpoint).read_text())
    raw, meta = ck["all_raw"], ck["meta"]
    designs = a.designs or select(ck, a.eps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = out / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    if a.analyse:
        S = analyse(done, designs, raw)
        (out / "summary.json").write_text(json.dumps(S, indent=1))
        txt = report(S)
        (out / "summary.txt").write_text(txt + "\n")
        print(txt)
        return 0

    ref_tr = tuple(int(v) for v in a.ref_transport.split(","))
    ct = meta["core_transport"]
    ratio = ref_tr[0] * ref_tr[1] / (ct["particles"] * ct["batches"])
    todo = [(i, r) for i in designs for r in range(1, a.reps + 1) if key_of(i, r) not in done]
    est = sum(float(raw[i]["t_core_bol_s"]) * (ratio + len(raw[i]["boron_points"]) - 1) for i, _ in todo) / 3600
    print(f"designs {designs}\n{len(designs) * a.reps} replicas, {len(todo)} to do, reference solve {ref_tr}, estimate {est:.1f} h")
    if a.dry_run:
        return 0

    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import c9_dep_replicas as dr
    for idx, rep in todo:
        design, spec = dr.design_of(ck, idx)
        ev = dr.make_evaluator(ck, spec, out / "work", meta["transport"])
        ev.c9_ppm_step, ev.c9_ppm_top = (float(v) for v in meta["campaign9"]["boron_points_ppm"][1:])
        res = do_run(ev, design, {**raw[idx], "_idx": idx}, rep, ref_tr, out / "work")
        done[res["key"]] = res
        store.write_text(json.dumps(done, indent=1))
        print(f"[{res['key']}] {time.strftime('%H:%M:%S')}  F_dH {res['peaking']:.4f} (stored {raw[idx]['peaking']:.4f})  "
              f"c_max {res['c_max']:.0f} (stored {raw[idx]['c_max']:.0f})  [{res['wall_s'] / 60:.1f} min]", flush=True)
    print("all replicas done; --analyse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
