#!/usr/bin/env python3
r"""
c9_dep_asm3d.py -- depletion of ONE fuel assembly in three dimensions, with
one set of fuel materials per axial layer. Candidate proxy of the eight-layer
core depletion (c9_dep_core3d.py) for the cycle length inside the loop.

MODEL
    radial      the 17 x 17 assembly of the campaign (reactor_model.
                build_assembly_universe, enrichment e, no ring multiplier)
                in a box with REFLECTIVE boundaries, as the campaign assembly
    axial       --axial structure (default): the axial stack of the core
                model (hardware3d: nozzles, end caps, plenum, grids, water
                above and below), vacuum at the top and at the bottom
                --axial bare: the active fuel only, vacuum at its two ends.
                No axial reflector, so more axial leakage. A sensitivity case
    layers      the active fuel is cut into --layers equal slabs, each with
                its own materials, exactly as c9_dep_core3d.fuel_cuts
    end of      relative, as the core depletion: the cycle ends when the model
    cycle       has lost the reactivity the campaign assembly loses,
                k = k(BOL) x k_target / k_inf(BOL) of the stored evaluation
    schedule    the depletion schedule stored in the checkpoint

The only difference from the campaign assembly depletion is the axial
direction, and the only difference from the core depletion is the radial one.

NOT RUN on the machine where it was written (no OpenMC there). Use --estimate
first: two short steps, a projected cost, and a check that the model builds.

USAGE (repository root, openmc-env)
    python c9_dep_asm3d.py --selftest
    python c9_dep_asm3d.py --dry-run --designs 47
    python c9_dep_asm3d.py --estimate --designs 47 --layers 8 --threads 64 --out asm3d_L8
    python c9_dep_asm3d.py --designs 47 --layers 8 --threads 64 --out asm3d_L8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from c9_dep_core3d import EFPD_REQ, design_of, fuel_cuts, layer_burnup, load_ckpt

DEFAULT_TR = dict(particles=10000, batches=100, inactive=50)


def axial_figures(ax, edges):
    """F_z and axial offset from the fission per fine axial bin."""
    ax = np.asarray(ax, float); edges = np.asarray(edges, float)
    dz = np.diff(edges)
    lin = ax / dz
    fz = float(lin.max() / np.average(lin, weights=dz))
    zc = 0.5 * (edges[:-1] + edges[1:])
    top, bot = ax[zc > 0].sum(), ax[zc <= 0].sum()
    return fz, float((top - bot) / (top + bot))


# --------------------------------------------------------------- builder ----
def build_assembly_layered(design, op, geo, spec, n_layers, tr, seed, axial="structure"):
    """Returns (model, info, rows, edges, fine)."""
    import openmc as o
    import hardware3d as hw
    import reactor_model as rm
    import dep_common as dc
    import axial_shape_c9 as ax

    pitch = float(design.get("pitch", 1.26))
    half = geo.lattice * pitch / 2.0
    T = op.mod_T
    water = rm.make_water(op.boron_ppm, T)
    ss = rm.make_ss304(op.clad_T); zr = rm.make_zircaloy(op.clad_T); he = rm.make_helium(op.clad_T)
    inconel = hw.make_inconel(op.clad_T)
    crm = rm.make_cr_materials(op.clad_T)
    lib = {"ss": ss, "zr": zr, "he": he, "inconel": inconel, "water": water,
           "aic": crm["AIC"], "cr_ss": crm["cr_ss"], "cr_he": crm["cr_he"]}
    vf = hw.volume_fractions(geo, pitch, spec)
    slab_cache = {}

    def slab(name):
        if name not in slab_cache:
            slab_cache[name] = hw.mix(name, {lib[k]: v for k, v in vf[name].items()}, T)
        return slab_cache[name]

    # one material set per layer; the plain and the grid lattice of a layer share it
    layer_mats = {k: rm.build_materials(design, op) for k in range(n_layers)}
    lat_cache = {}

    def lattice(layer, grid_kind):
        key = (layer, grid_kind)
        if key not in lat_cache:
            strap = {None: None, "htm": inconel, "htp": zr}[grid_kind]
            with hw._patched(rm, spec, strap_mat=strap, pitch=pitch):
                _u, _cells, lat = rm.build_assembly_universe(design, layer_mats[layer], geo, pitch)
            lat_cache[key] = lat
        return lat_cache[key]

    z = hw.elevations(spec)
    edges, fsegs = fuel_cuts(spec, n_layers)
    segs, fuel_fills = [], []
    for sg in fsegs:
        L = lattice(sg["layer"], sg["grid"])
        segs.append((sg["z0"], sg["z1"], L))
        fuel_fills.append((L, sg["z1"] - sg["z0"]))
    if axial == "structure":
        parked = spec.model_parked_rods

        def plain(base, grid=False):
            return slab(base + ("_parked" if parked else "") + ("_grid" if grid else ""))
        below = [(*z["water_below"], water), (*z["bottom_nozzle"], slab("bottom_nozzle")),
                 (*z["lower_cap"], slab("lower_cap"))]
        above = []
        p0, p1 = z["plenum"]
        zc = p0
        for g0, g1 in [(g0, g1) for g0, g1, _ in z["grids"] if g0 >= p0 and g1 <= p1]:
            if g0 > zc:
                above.append((zc, g0, plain("plenum")))
            above.append((g0, g1, plain("plenum", grid=True)))
            zc = g1
        if zc < p1:
            above.append((zc, p1, plain("plenum")))
        above += [(*z["upper_cap"], plain("upper_cap")), (*z["upper_gap"], plain("upper_gap")),
                  (*z["top_nozzle"], slab("top_nozzle")), (*z["water_above"], water)]
        segs = below + segs + above
        z_bot, z_top = z["water_below"][0], z["water_above"][1]
    elif axial == "bare":
        z_bot, z_top = float(edges[0]), float(edges[-1])
    else:
        raise ValueError(f"unknown axial model {axial!r}")

    zp_bot = o.ZPlane(z0=z_bot, boundary_type="vacuum"); zp_top = o.ZPlane(z0=z_top, boundary_type="vacuum")
    planes = {}

    def zp(v):
        v = round(float(v), 6)
        if abs(v - z_bot) < 1e-6: return zp_bot
        if abs(v - z_top) < 1e-6: return zp_top
        if v not in planes: planes[v] = o.ZPlane(z0=v)
        return planes[v]
    box = o.model.RectangularPrism(2 * half, 2 * half, boundary_type="reflective")
    cells = [o.Cell(fill=f, region=-box & +zp(a) & -zp(b)) for a, b, f in segs]
    geom = o.Geometry(cells)
    materials = o.Materials(set(geom.get_all_materials().values()))
    hz = 0.5 * spec.h_active
    settings = rm._settings(tr["particles"], tr["batches"], tr["inactive"],
                            ((-half, -half, -hz), (half, half, hz)), seed=int(seed))
    model = o.Model(geometry=geom, materials=materials, settings=settings)

    rows = dc.mark_depletable(model, fuel_fills, geo)
    byid = {int(m.id): (layer, role) for layer, mats in layer_mats.items()
            for role, m in mats.items() if role.startswith("fuel")}
    for r in rows:
        r["layer"], r["role"] = byid.get(r["id"], (None, "?"))
    if any(r["layer"] is None for r in rows):
        raise RuntimeError("a depletable material has no layer")
    off = dc.unmark_unused(model, rows)

    fine, _bands, _ = ax.axial_edges(spec, hw)

    def z_tally(name, zgrid):
        m = o.RectilinearMesh(name=name)
        m.x_grid = np.array([-half, half]); m.y_grid = np.array([-half, half]); m.z_grid = np.asarray(zgrid)
        t = o.Tally(name=name); t.filters = [o.MeshFilter(m)]; t.scores = ["fission"]
        return t
    model.tallies = o.Tallies([z_tally("layer_fission", edges), z_tally("axial_fission", fine)])
    info = dict(axial=axial, half_width=half, z_bottom=z_bot, z_top=z_top, n_segments=len(segs),
                n_fuel_segments=len(fsegs), n_layers=n_layers, n_depletable=len(rows),
                layer_edges=np.asarray(edges).tolist(), fine_edges=np.asarray(fine).tolist(),
                materials_off=off)
    return model, info, rows, np.asarray(edges), np.asarray(fine)


def read_states(case, fine):
    """Per-solve k, F_z, axial offset and layer power shares."""
    import openmc as o
    import dep_common as dc
    out = []
    for p in dc.statepoints_in_order(case):
        with o.StatePoint(str(p)) as sp:
            keff, sd = float(sp.keff.nominal_value), float(sp.keff.std_dev)
            lay = sp.get_tally(name="layer_fission").get_values(scores=["fission"]).ravel()
            axf = sp.get_tally(name="axial_fission").get_values(scores=["fission"]).ravel()
        fz, ao = axial_figures(axf, fine)
        out.append(dict(file=str(p), keff=keff, keff_sd=sd, fz=fz, ao=ao,
                        layer_share=(lay / lay.sum()).tolist()))
    return out


# ----------------------------------------------------------------- run ----
def run_design(idx, ckpt, raw, meta, tr, n_layers, out, axial, estimate=False, salt="asm3d"):
    import reactor_model as rm
    import hardware3d as hw
    import dep_common as dc
    from openmc_evaluator import _design_seed
    op, geo, spec = rm.Operating(), rm.Geometry17x17(), hw.HardwareSpec()
    design = design_of(ckpt, idx)
    seed = _design_seed(design, salt=salt)
    model, info, rows, edges, fine = build_assembly_layered(design, op, geo, spec, n_layers, tr, seed, axial)
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
    print(f"[d{idx}] single assembly, axial {axial}, {n_layers} layers, {info['n_fuel_segments']} fuel "
          f"segments, {info['n_depletable']} depletable materials, seed {seed}, transport {tr}, "
          f"end of cycle relative, ratio {ratio:.5f}", flush=True)
    t0 = time.time()
    res = dc.run_adaptive(model, None, spec_power, case=case, verbose=True, k_target_ratio=ratio, **sched)
    wall = time.time() - t0
    errors = {}

    def guarded(name, fn, default):
        try:
            return fn()
        except Exception as exc:              # the transport is on disk: never lose it
            errors[name] = repr(exc)
            print(f"      WARNING {name} failed: {exc!r}", flush=True)
            return default
    times = guarded("solve_times", lambda: dc.solve_times(case), [])
    if estimate:
        pc = dc.project_cost(times, int(rec["n_dep_solves"]), wall)
        pc.update(idx=idx, transport=tr, layers=n_layers, axial=axial, wall_estimate_s=wall,
                  k_bol=res["k_hist"][0])
        return pc
    sdmap = guarded("k_sd_from_case", lambda: dc.k_sd_from_case(case), {})
    ksd = [sdmap.get(v) for v in res["k_hist"]]
    sig, ib, slope = guarded("bracket_sigma_efpd", lambda: dc.bracket_sigma_efpd(
        res["bu_hist"], res["k_hist"], ksd, res["k_target"], spec_power), (None, None, None))
    states = guarded("read_states", lambda: read_states(case, fine), [])
    aligned = len(states) == len(res["k_hist"])
    return dict(
        idx=idx, model="assembly3d", axial=axial, seed=int(seed), salt=salt, transport=tr, layers=n_layers,
        k_eoc=res["k_target"], eoc_mode="relative", k_target_ratio=ratio, spec_power=spec_power,
        info=info, materials=rows, efpd=res["efpd"], bu_eoc=res["bu_eoc"], censored=res["censored"],
        k_hist=res["k_hist"], k_sd=ksd, bu_hist=res["bu_hist"], n_solves=res["n_solves"],
        sigma_efpd=sig, bracket=ib, slope_pcm_per_mwdkg=slope, g_efpd=EFPD_REQ - res["efpd"],
        efpd_archive=rec["cycle_length"], states=states, states_aligned=aligned,
        layer_burnup=layer_burnup(states, res["bu_hist"], edges) if aligned else None,
        solve_times=times, wall_s=wall, post_errors=errors)


def selftest():
    import hardware3d as hw
    spec = hw.HardwareSpec()
    for n in (1, 4, 8, 12):
        edges, segs = fuel_cuts(spec, n)
        assert len(edges) == n + 1 and {s["layer"] for s in segs} == set(range(n)), n
        assert abs(sum(s["z1"] - s["z0"] for s in segs) - spec.h_active) < 1e-6
    # a flat shape has F_z = 1 and no offset, whatever the bin widths
    e = np.array([-60.0, -20.0, 0.0, 60.0])
    fz, ao = axial_figures(np.diff(e), e)
    assert abs(fz - 1.0) < 1e-12 and abs(ao) < 1e-12, (fz, ao)
    # a peak in one thin bin is measured per unit length
    fz, ao = axial_figures(np.array([40.0, 40.0, 60.0]), e)
    assert abs(fz - 2.0 / (140.0 / 120.0)) < 1e-12 and ao < 0, (fz, ao)
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9a/optimization_checkpoint.json")
    ap.add_argument("--designs", type=int, nargs="*", default=[47])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--axial", choices=["structure", "bare"], default="structure")
    ap.add_argument("--particles", type=int, default=DEFAULT_TR["particles"])
    ap.add_argument("--batches", type=int, default=DEFAULT_TR["batches"])
    ap.add_argument("--inactive", type=int, default=DEFAULT_TR["inactive"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="asm3d_dep")
    ap.add_argument("--salt", default="asm3d", help="seed salt; another value gives an independent replica")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ckpt, raw, meta = load_ckpt(a.checkpoint)
    tr = dict(particles=a.particles, batches=a.batches, inactive=a.inactive)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = out / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    print(f"single-assembly depletion: designs {a.designs}, axial {a.axial}, {a.layers} layers "
          f"({12 * a.layers} depletable materials at most), transport {tr}, "
          f"schedule {meta.get('schedule')}, cached {sorted(done)}")
    for idx in a.designs:
        r = raw[idx]
        print(f"  C9-{idx}: campaign assembly cycle {r['cycle_length']:.0f} d, {r['n_dep_solves']} depletion solves")
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
    for idx in a.designs:
        if a.estimate:
            pc = run_design(idx, ckpt, raw, meta, tr, a.layers, out, a.axial, estimate=True, salt=a.salt)
            (out / f"estimate_d{idx}.json").write_text(json.dumps(pc, indent=1))
            if pc.get("ok"):
                print(f"[d{idx}] ESTIMATE fresh solve {pc['fresh_s']:.0f} s, depleted solve {pc['depleted_s']:.0f} s, "
                      f"{pc['n_solves']} solves -> about {pc['projected_h'] * 60:.0f} min (k_BOL {pc['k_bol']:.5f})")
            else:
                print(f"[d{idx}] ESTIMATE failed: no statepoints found")
            continue
        if f"d{idx}" in done:
            print(f"[d{idx}] cached"); continue
        res = run_design(idx, ckpt, raw, meta, tr, a.layers, out, a.axial, salt=a.salt)
        done[f"d{idx}"] = res
        store.write_text(json.dumps(done, indent=1))
        print(f"[d{idx}] EFPD {res['efpd']:.1f} d (campaign assembly {res['efpd_archive']:.1f}) "
              f"[{res['n_solves']} solves, {res['wall_s'] / 60:.1f} min]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
