#!/usr/bin/env python3
r"""
c9_dep_core2d.py -- check 2: deplete the zoned two-dimensional core itself
and compare with the assembly proxy that gave the archived cycle length.

WHAT IS DIFFERENT FROM THE CAMPAIGN
    campaign   one reflective 17x17 assembly at the core-average enrichment
               depletes; the core enters only through the leakage target
               k_target(t_refl) = LF_2D x L_ax and the BOL core solves
    here       the 32-assembly zoned core of make_core_model, the model of
               every archived core solve, depletes with a vacuum radial
               boundary. Each ring's UO2 and UO2-Gd2O3 materials deplete as
               ONE lump each (inner and outer blocks of the assembly are
               separate materials, twelve in all), so the
               rings burn at the rates their own power shares dictate and
               the gadolinium burns out in the local flux of each ring.
               Per-pin depletion (32 x 264 materials) is out of reach.
    end of     the two-dimensional core is at end of cycle when
    cycle      k_eff(2D) = L_ax, the axial leakage factor of the k-target
               table (1.0289), so that the three-dimensional core is
               critical. This is the same statement as the campaign's
               k_inf = k_target with the leakage factor held at its BOL
               value, and it is exact where the campaign's is calibrated
    schedule   the campaign's BOL block and 4 MWd/kgHM marching steps, at
               the same specific power (W/gHM), so burnup means the same
    boron      1000 ppm throughout, as in the campaign
    hump       measured on the core directly, rho(k_peak) - rho(k_BOL),
               where the campaign carried the assembly hump through L_ax
    c_max      the archived BOL boron points of the design with the
               measured core hump (boron_objective.py), no new core solves

WHAT IT REPORTS, per design
    EFPD_core vs archive (assembly proxy), with the propagated sigma of both
    k_eff(core, BOL) of this run vs the archived 100000 x 170 value (pcm)
    hump_core measured vs the archived L_ax x H_asm
    c_max recomputed vs archive, margin over the 1826 d floor
    F_dH and the ring power shares at every depletion state, from the
    per-solve statepoints: the radial peaking evolves with burnup, which no
    campaign solve measured

COST    unknown until --estimate: it runs the first two steps
        (0.5 + 0.5 MWd/kgHM, three transport solves) at the chosen
        fidelity and projects a full cycle from the archived solve count.

USAGE (repository root, openmc-env)
    python c9_dep_core2d.py --selftest
    python c9_dep_core2d.py --dry-run
    python c9_dep_core2d.py --estimate --threads 64
    python c9_dep_core2d.py --threads 64                # C9-47 by default
    python c9_dep_core2d.py --analyse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EFPD_REQ = 1826.0
DEFAULT_TR = dict(particles=20000, batches=160, inactive=60)


def load_ckpt(path):
    ck = json.loads(Path(path).read_text())
    return ck, ck["all_raw"], ck.get("meta", {}) or {}


def design_of(ckpt, idx):
    from reactor_optimization import campaign9_problem
    spec = campaign9_problem(float(ckpt["meta"]["campaign9"]["efpd_req"]))
    x = [ckpt["all_raw"][idx][v] for v in ckpt["design_variables"]]
    return spec.design_space.as_dict(np.asarray(x, float))


def lax_of(meta, override=None):
    if override is not None:
        return float(override), "command line"
    t = json.loads(Path(meta["k_target"]).read_text())
    return float(t["axial_leakage_factor"]), f"{meta['k_target']} axial_leakage_factor"


# ------------------------------------------------------------- the model ----
def build(design, tr, seed, geo, op):
    """The zoned 2D core with the pin-fission mesh tally of core_bol_solve and
    the six ring fuel materials marked depletable with their volumes."""
    import openmc
    import reactor_model as rm
    import zoning as zn
    import dep_common as dc

    dmap = zn.evaluator_design_map(design)
    model, _ = rm.make_core_model(design, op, geo, design_map=dmap, **tr)
    model.settings.seed = int(seed)

    NL = geo.lattice
    pitch = design.get("pitch", 1.26)
    rmap = zn.ring_map(); ny, nx = rmap.shape
    half = nx * NL * pitch / 2.0
    mesh = openmc.RegularMesh()
    mesh.dimension = (nx * NL, ny * NL)
    mesh.lower_left = (-half, -half); mesh.upper_right = (half, half)
    t = openmc.Tally(name="core_pin_fission")
    t.filters = [openmc.MeshFilter(mesh)]; t.scores = ["fission"]
    model.tallies = openmc.Tallies([t])

    lat = next(c.fill for c in model.geometry.root_universe.cells.values()
               if hasattr(c.fill, "universes"))
    rows = dc.mark_depletable(model, [(lat, geo.active_height)], geo)
    # the base assembly of make_core_model is never placed when the zoning map
    # covers all 32 positions, but its fuel materials are still in
    # model.materials and arrive depletable: switch them off
    off = dc.unmark_unused(model, rows)
    # label each depletable material by ring from its name: UO2_<e> carries
    # the ring enrichment (two decimals), UGd_<e>_<gd> the derated one (one
    # decimal), so the two are matched against different values
    zones = {}
    for (i, j), d in dmap.items():
        zones.setdefault(d.get("zone", "?"), float(d["enrich_inner"]))
    red = max(0.0, 1.0 - 0.05 * float(design.get("gd_wt", 0.0)))
    for r in rows:
        e = float(r["name"].split("_")[1])
        r["gd"] = r["name"].startswith("UGd")
        cand = {z: (max(0.2, ez * red) if r["gd"] else ez) for z, ez in zones.items()}
        z, ez = min(cand.items(), key=lambda t: abs(t[1] - e))
        r["zone"] = z if abs(ez - e) < (0.06 if r["gd"] else 0.006) else "?"
    return model, rows, (ny, nx, NL, rmap), off


def read_states(case, shape, inactive):
    """F_dH, ring shares and entropy convergence of every per-solve
    statepoint under case, in solve order."""
    import openmc
    import zoning as zn
    import dep_common as dc
    ny, nx, NL, rmap = shape
    out = []
    for p in dc.statepoints_in_order(case):
        with openmc.StatePoint(str(p)) as sp:
            k = float(sp.keff.nominal_value)
            v = sp.get_tally(name="core_pin_fission").get_values(scores=["fission"]).reshape(ny * NL, nx * NL)
            H = np.asarray(getattr(sp, "entropy", []), float)
        f = np.ma.masked_equal(v, 0.0)
        conv = None
        if H.size:
            tail = H[inactive + (len(H) - inactive) // 2:]
            mu, sd = float(tail.mean()), float(tail.std(ddof=1))
            Hs = np.convolve(H, np.ones(3) / 3.0, mode="same"); Hs[0], Hs[-1] = H[0], H[-1]
            bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
            conv = int(bad[-1]) + 2 if len(bad) else 1
        out.append(dict(file=str(p), keff=k, fdh=float((f / f.mean()).max()),
                        ring_shares=zn.ring_power_shares(v, rmap, NL), entropy_conv=conv))
    return out


# ----------------------------------------------------------------- run ----
def run_design(idx, ckpt, raw, meta, tr, lax, out, threads, estimate=False, mode="relative"):
    import reactor_model as rm
    import dep_common as dc
    from openmc_evaluator import _design_seed

    op, geo = rm.Operating(), rm.Geometry17x17()
    design = design_of(ckpt, idx)
    seed = _design_seed(design, salt="core2d")
    model, rows, shape, off = build(design, tr, seed, geo, op)
    spec_power = rm.core_specific_power_w_per_g(op, geo)
    sch = meta["schedule"]
    case = out / ("estimate" if estimate else "work") / f"d{idx}"
    if estimate:
        import shutil
        shutil.rmtree(case, ignore_errors=True)
        sched = dict(bol_steps=[0.5, 0.5], dep_step=0.5, chunk_steps=1, max_burnup=1.0)
    else:
        sched = dict(bol_steps=sch["bol_steps"], dep_step=sch["dep_step"],
                     chunk_steps=sch["chunk_steps"], max_burnup=sch["max_burnup"])
    rec = raw[idx]
    ratio = float(rec["k_target"]) / float(rec["k_bol"])
    eoc = dict(k_target_ratio=ratio) if mode == "relative" else dict(k_target=lax)
    print(f"[d{idx}] zoned 2D core, {len(rows)} depletable materials, seed {seed}, "
          f"transport {tr}, end of cycle "
          + (f"relative, ratio {ratio:.5f} (assembly loses "
             f"{1e5 * (1 / rec['k_target'] - 1 / rec['k_bol']):.0f} pcm)"
             if mode == "relative" else f"absolute at k = {lax:.4f}"), flush=True)
    for r in sorted(rows, key=lambda r: (r["zone"], r["gd"], -r["pins"])):
        print(f"      {r['name']:16s} ring {r['zone']}  pins {r['pins']:5d}  {r['volume_cm3']:10.1f} cm3")
    for r in off:
        print(f"      not placed, depletion switched off: {r['name']} (id {r['id']})")
    if any(r["zone"] == "?" for r in rows):
        raise RuntimeError("a depletable material could not be assigned to a ring")
    t0 = time.time()
    res = dc.run_adaptive(model, eoc.get("k_target"), spec_power, case=case, verbose=True,
                          k_target_ratio=eoc.get("k_target_ratio"), **sched)
    wall = time.time() - t0
    errors = {}

    def guarded(name, fn, default):
        try:
            return fn()
        except Exception as exc:              # hours of transport are on disk: never lose them
            errors[name] = repr(exc)
            print(f"      WARNING {name} failed: {exc!r}", flush=True)
            return default
    times = guarded("solve_times", lambda: dc.solve_times(case), [])
    if estimate:
        n_full = int(raw[idx]["n_dep_solves"])
        pc = dc.project_cost(times, n_full, wall)
        pc.update(idx=idx, transport=tr, wall_estimate_s=wall, k_bol=res["k_hist"][0],
                  k_core_bol_archive=float(raw[idx]["keff_core_bol"]))
        return pc
    sdmap = guarded("k_sd_from_case", lambda: dc.k_sd_from_case(case), {})
    ksd = [sdmap.get(v) for v in res["k_hist"]]
    k_eoc = res["k_target"]
    sig, ib, slope = dc.bracket_sigma_efpd(res["bu_hist"], res["k_hist"], ksd, k_eoc, spec_power)
    c9 = meta["campaign9"]
    obj = dc.hump_and_cmax(res["k_hist"], res["k_hist"][0], raw[idx]["boron_points"],
                           float(c9["hump_noise_pcm"]))
    states = guarded("read_states", lambda: read_states(case, shape, tr["inactive"]), [])
    aligned = len(states) == len(res["k_hist"])
    return dict(
        idx=idx, seed=int(seed), transport=tr, k_eoc=k_eoc, eoc_mode=mode,
        k_target_ratio=ratio, spec_power=spec_power,
        materials=rows, materials_off=off, efpd=res["efpd"], bu_eoc=res["bu_eoc"], censored=res["censored"],
        k_hist=res["k_hist"], k_sd=ksd, bu_hist=res["bu_hist"], n_solves=res["n_solves"],
        sigma_efpd=sig, bracket=ib, slope_pcm_per_mwdkg=slope,
        hump_core_pcm=obj["hump_core_pcm"], hump_core_op_pcm=obj["hump_core_op_pcm"],
        hump_asm_pcm_here=obj["hump_asm_pcm"], k_peak=obj["k_peak"], bu_index=obj["bu_index"],
        c_max=obj["c_max"], c_max_status=obj["c_max_status"], c_bol=obj["c_bol_ppm"],
        g_efpd=EFPD_REQ - res["efpd"], states=states, states_aligned=aligned,
        solve_times=times, wall_s=wall, post_errors=errors)


# ------------------------------------------------------------- analysis ----
def analyse(done, raw):
    R = {}
    for key, d in done.items():
        idx = d["idx"]; rec = raw[idx]
        row = dict(idx=idx, efpd_core=d["efpd"], efpd_archive=rec["cycle_length"],
                   d_efpd=d["efpd"] - rec["cycle_length"], sigma_core=d["sigma_efpd"],
                   censored=d["censored"], n_solves=d["n_solves"],
                   k_bol_core=d["k_hist"][0], k_bol_archive=rec["keff_core_bol"],
                   dk_bol_pcm=1e5 * (d["k_hist"][0] - rec["keff_core_bol"]),
                   k_bol_sd_pcm=(1e5 * d["k_sd"][0]) if d["k_sd"][0] else None,
                   hump_core_measured=d["hump_core_pcm"], hump_core_op_measured=d["hump_core_op_pcm"],
                   hump_core_archive=rec["hump_core_pcm"], hump_core_op_archive=rec["hump_core_op_pcm"],
                   c_max_core=d["c_max"], c_max_archive=rec["c_max"], d_c_max=d["c_max"] - rec["c_max"],
                   margin_core_d=d["efpd"] - EFPD_REQ, margin_archive_d=rec["cycle_length"] - EFPD_REQ,
                   wall_h=d["wall_s"] / 3600.0)
        if d.get("states") and d.get("states_aligned"):
            st = d["states"]
            fdh = [s["fdh"] for s in st]
            row["bu_last"] = d["bu_hist"][-1]
            row["fdh_bol"], row["fdh_max"], row["fdh_eoc"] = fdh[0], max(fdh), fdh[-1]
            row["fdh_archive"] = rec["peaking"]
            row["fdh_trajectory"] = [(b, f) for b, f in zip(d["bu_hist"], fdh)]
            row["ring_shares_bol"] = st[0]["ring_shares"]
            row["ring_shares_eoc"] = st[-1]["ring_shares"]
            row["entropy_conv_max"] = max((s["entropy_conv"] or 0) for s in st)
        R[idx] = row
    return R


def report(R):
    L = ["=== Check 2: zoned 2D core depletion against the assembly proxy"]
    for idx, r in sorted(R.items()):
        L.append(f"C9-{idx}")
        L.append(f"  cycle length   core {r['efpd_core']:7.1f} d (sigma {r['sigma_core'] if r['sigma_core'] is None else round(r['sigma_core'], 1)})   "
                 f"archive {r['efpd_archive']:7.1f} d   difference {r['d_efpd']:+.1f} d   "
                 f"margins over {EFPD_REQ:.0f}: core {r['margin_core_d']:+.0f}, archive {r['margin_archive_d']:+.0f}"
                 + ("   CENSORED" if r["censored"] else ""))
        L.append(f"  k_eff BOL      this run {r['k_bol_core']:.5f} vs archive {r['k_bol_archive']:.5f}: "
                 f"{r['dk_bol_pcm']:+.0f} pcm (run s.d. {r['k_bol_sd_pcm'] if r['k_bol_sd_pcm'] is None else round(r['k_bol_sd_pcm'])} pcm)")
        L.append(f"  hump (core)    measured {r['hump_core_measured']:+.0f} pcm (no floor {r['hump_core_op_measured']:+.0f})   "
                 f"archive L_ax x H_asm {r['hump_core_archive']:+.0f} (no floor {r['hump_core_op_archive']:+.0f})")
        L.append(f"  c_max          core {r['c_max_core']:.0f} ppm   archive {r['c_max_archive']:.0f} ppm   difference {r['d_c_max']:+.0f} ppm")
        if "fdh_bol" in r:
            L.append(f"  (the values marked 'last' are at the last computed state, B = {r['bu_last']:.1f} MWd/kgHM)")
            L.append(f"  F_dH           BOL {r['fdh_bol']:.3f} (archive {r['fdh_archive']:.3f})   max over run {r['fdh_max']:.3f}   last {r['fdh_eoc']:.3f}   "
                     f"entropy conv. batch max {r['entropy_conv_max']}")
            L.append(f"  ring shares    BOL {r['ring_shares_bol']}   last {r['ring_shares_eoc']}")
        L.append(f"  solves {r['n_solves']}, wall {r['wall_h']:.2f} h")
    return "\n".join(L)


def selftest():
    done = {"d47": dict(idx=47, efpd=1900.0, sigma_efpd=30.0, censored=False, n_solves=8,
                        k_hist=[1.0280, 1.0100, 1.0150, 1.0120, 1.0050, 0.9950, 1.0290, 1.0200],
                        k_sd=[1e-3] * 8, bu_hist=[0, .5, 1.5, 3.5, 7.5, 13.5, 17.5, 21.5],
                        hump_core_pcm=0.0, hump_core_op_pcm=-100.0, c_max=1400.0, wall_s=7200.0,
                        states=[dict(fdh=1.55 - 0.01 * i, ring_shares={"C": 0.1, "M": 0.4, "P": 0.5}, entropy_conv=10) for i in range(8)],
                        states_aligned=True)}
    raw = {47: dict(cycle_length=1946.7, keff_core_bol=1.02728, hump_core_pcm=0.0, hump_core_op_pcm=8.6,
                    c_max=1405.9, peaking=1.545)}
    R = analyse(done, raw)
    assert abs(R[47]["d_efpd"] + 46.7) < 1e-9 and abs(R[47]["dk_bol_pcm"] - 72.0) < 0.1
    assert R[47]["fdh_bol"] == 1.55 and R[47]["fdh_eoc"] == 1.48
    txt = report(R); assert "C9-47" in txt and "ring shares" in txt
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--designs", type=int, nargs="*", default=[47])
    ap.add_argument("--particles", type=int, default=DEFAULT_TR["particles"])
    ap.add_argument("--batches", type=int, default=DEFAULT_TR["batches"])
    ap.add_argument("--inactive", type=int, default=DEFAULT_TR["inactive"])
    ap.add_argument("--lax", type=float, default=None, help="k_eff at end of cycle; default the table's axial factor")
    ap.add_argument("--eoc-mode", choices=["relative", "absolute"], default="relative",
                    help="relative (default): end of cycle when the core has lost the same "
                         "reactivity as the campaign assembly, k = k(BOL) x k_target/k_inf(BOL). "
                         "absolute: end of cycle at the fixed k below")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="c9_dep_core2d")
    ap.add_argument("--estimate", action="store_true", help="two short steps, then project the cost")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ckpt, raw, meta = load_ckpt(a.checkpoint)
    tr = dict(particles=a.particles, batches=a.batches, inactive=a.inactive)
    lax, src = lax_of(meta, a.lax)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = out / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    if a.analyse:
        R = analyse(done, raw)
        (out / "summary.json").write_text(json.dumps(R, indent=1))
        txt = report(R); (out / "summary.txt").write_text(txt + "\n"); print(txt)
        return 0
    print(f"check 2: designs {a.designs}, transport {tr}, end of cycle {a.eoc_mode} "
          f"(absolute value would be {lax:.4f} from {src}), "
          f"schedule {meta.get('schedule')}, cached {sorted(done)}")
    for idx in a.designs:
        r = raw[idx]
        print(f"  C9-{idx}: archive EFPD {r['cycle_length']:.0f} d, {r['n_dep_solves']} depletion solves, "
              f"k_core_BOL {r['keff_core_bol']:.4f}, c_max {r['c_max']:.0f} ppm")
    if a.dry_run:
        return 0
    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import openmc
    chain = os.environ.get("OPENMC_CHAIN_FILE")
    if not chain:
        raise SystemExit("OPENMC_CHAIN_FILE is not set")
    openmc.config["chain_file"] = chain
    print(f"  openmc {openmc.__version__}, chain {chain}")
    if a.estimate:
        for idx in a.designs:
            pc = run_design(idx, ckpt, raw, meta, tr, lax, out, a.threads, estimate=True, mode=a.eoc_mode)
            (out / f"estimate_d{idx}.json").write_text(json.dumps(pc, indent=1))
            if pc.get("ok"):
                print(f"[d{idx}] ESTIMATE fresh solve {pc['fresh_s']:.0f} s, depleted solve {pc['depleted_s']:.0f} s, "
                      f"operator overhead {pc['overhead_s']:.0f} s per solve, "
                      f"{pc['n_solves']} solves -> about {pc['projected_h']:.1f} h for the full cycle "
                      f"(k_BOL here {pc['k_bol']:.5f}, archive {pc['k_core_bol_archive']:.5f})")
                print(f"PROJECTED_HOURS {pc['projected_h']:.2f}")
            else:
                print(f"[d{idx}] ESTIMATE failed: no statepoints found")
        return 0
    for idx in a.designs:
        if f"d{idx}" in done:
            print(f"[d{idx}] cached"); continue
        res = run_design(idx, ckpt, raw, meta, tr, lax, out, a.threads, mode=a.eoc_mode)
        done[f"d{idx}"] = res
        store.write_text(json.dumps(done, indent=1))
        print(f"[d{idx}] EFPD {res['efpd']:.1f} d (archive {raw[idx]['cycle_length']:.1f}), "
              f"c_max {res['c_max']:.0f} (archive {raw[idx]['c_max']:.0f}), hump_core {res['hump_core_pcm']:+.0f} pcm, "
              f"[{res['n_solves']} solves, {res['wall_s'] / 3600:.2f} h]", flush=True)
    print("done; run --analyse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
