#!/usr/bin/env python3
r"""
c9_dep_core3d.py -- check 3: deplete the three-dimensional hardware core
(hardware3d.py: grids, plenum, nozzles, barrel, vessel, all rods out) with
one set of fuel materials per axial layer, and read the cycle length at
k_eff = 1 with no leakage factor at all.

WHAT IS DIFFERENT FROM CHECK 2
    layers      the active fuel is cut into --layers equal slabs. Every slab
                has its OWN lattice and its OWN twelve ring materials (the
                plain and the spacer-grid segments inside a slab share
                them), so each layer depletes in its local flux and the
                axial burnup profile is a result, not an assumption
    end of      k_eff(3D) = 1.0 (--k-eoc). The campaign's statement was
    cycle       k_inf(assembly) = k_target = LF_2D x L_ax, that is, the
                three-dimensional core critical with both leakage factors
                frozen at their beginning-of-life values. Here neither is
                frozen
    tallies     the three meshes of axial_shape_c9.py (assembly, pin,
                fine axial) plus a mesh on the layer edges, in every
                per-solve statepoint, so F_dH, F_z, the axial offset and the
                layer power shares are known at every burnup state
    burnup      per layer, from the layer power shares with the predictor
                rule (rates at the beginning of the step act over the step)
    hump, c_max as in check 2, from the measured 3D history and the archived
                boron points

COST    unknown until --estimate. The 3D ARO solve at 150000 x 200 took
        about 250 s fresh in axial_shape_c9; depleted compositions and
        --layers x 12 materials make each solve slower. --estimate runs two
        short steps and projects the full cycle.

USAGE (repository root, openmc-env)
    python c9_dep_core3d.py --selftest
    python c9_dep_core3d.py --dry-run
    python c9_dep_core3d.py --estimate --threads 64
    python c9_dep_core3d.py --threads 64                # C9-47, 8 layers
    python c9_dep_core3d.py --analyse
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EFPD_REQ = 1826.0
DEFAULT_TR = dict(particles=20000, batches=160, inactive=60)
NL = 17


def load_ckpt(path):
    ck = json.loads(Path(path).read_text())
    return ck, ck["all_raw"], ck.get("meta", {}) or {}


def design_of(ckpt, idx):
    from reactor_optimization import campaign9_problem
    spec = campaign9_problem(float(ckpt["meta"]["campaign9"]["efpd_req"]))
    x = [ckpt["all_raw"][idx][v] for v in ckpt["design_variables"]]
    return spec.design_space.as_dict(np.asarray(x, float))


# ---------------------------------------------------------------- layout ----
def fuel_cuts(spec, n_layers):
    """Sorted z cuts inside the active fuel: the layer edges plus the
    spacer-grid band edges, and the layer index of every segment."""
    import hardware3d as hw
    z = hw.elevations(spec)
    lo, hi = z["fuel"]
    edges = np.linspace(lo, hi, n_layers + 1)
    cuts = set(np.round(edges, 6).tolist())
    for g0, g1, _ in z["grids"]:
        if g0 >= lo - 1e-9 and g1 <= hi + 1e-9:
            cuts.update((round(g0, 6), round(g1, 6)))
    cuts = sorted(cuts)
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-6:
            continue
        mid = 0.5 * (a + b)
        layer = int(min(n_layers - 1, max(0, np.searchsorted(edges, mid) - 1)))
        grid = next((kind for g0, g1, kind in z["grids"] if g0 <= mid < g1), None)
        segs.append(dict(z0=a, z1=b, layer=layer, grid=grid))
    return edges, segs


# --------------------------------------------------------------- builder ----
def build_layered(design, op, geo, spec, n_layers, tr, seed, design_map):
    """hardware3d.build_model_3d_hw for the all-rods-out core, with the fuel
    span cut into layers that each carry their own materials. Returns
    (model, info, layer_rows, edges, fine_edges, shape)."""
    import openmc as o
    import core_geometry as cg
    import hardware3d as hw
    import reactor_model as rm
    import dep_common as dc
    import axial_shape_c9 as ax

    pitch = float(design.get("pitch", 1.26)); refl = float(design["refl_thick"])
    core_map = np.asarray(cg.CORE_MAP_32); ny, nx = core_map.shape
    asm_pitch = geo.lattice * pitch
    r_env = cg.core_envelope_radius(pitch, core_map, geo.lattice)
    r_fuel = r_env + cg.FUEL_PAD_CM; r_refl = r_fuel + refl; r_barrel = r_refl + spec.barrel
    if r_barrel > spec.r_vessel_in - 1e-9:
        raise ValueError("barrel outer radius exceeds the vessel inner radius")
    r_vessel_out = spec.r_vessel_in + spec.vessel_wall

    T = op.mod_T
    water = rm.make_water(op.boron_ppm, T)
    ss = rm.make_ss304(op.clad_T); zr = rm.make_zircaloy(op.clad_T); he = rm.make_helium(op.clad_T)
    inconel = hw.make_inconel(op.clad_T)
    crm = rm.make_cr_materials(op.clad_T)
    with hw._patched(rm, spec):
        refl_mat = rm.make_heavy_reflector(op)
    lib = {"ss": ss, "zr": zr, "he": he, "inconel": inconel, "water": water,
           "aic": crm["AIC"], "cr_ss": crm["cr_ss"], "cr_he": crm["cr_he"]}
    vf = hw.volume_fractions(geo, pitch, spec)
    slab_cache = {}

    def slab(name):
        if name not in slab_cache:
            slab_cache[name] = hw.mix(name, {lib[k]: v for k, v in vf[name].items()}, T)
        return slab_cache[name]

    refl_u = o.Universe(cells=[o.Cell(fill=refl_mat)])

    def slab_lattice(mat):
        u0 = o.Universe(cells=[o.Cell(fill=mat)])
        L = o.RectLattice(); L.lower_left = (-nx * asm_pitch / 2, -ny * asm_pitch / 2)
        L.pitch = (asm_pitch, asm_pitch)
        L.universes = [[u0 if core_map[i, j] == 1 else refl_u for j in range(nx)] for i in range(ny)]
        L.outer = refl_u
        return L

    # one materials cache per layer: the plain and the grid lattice of a
    # layer are built on the SAME fuel, water and clad objects
    mat_cache = {}
    orig_build = rm.build_materials

    @contextlib.contextmanager
    def layer_materials(layer):
        def cached(d, op_):
            key = (layer,) + tuple(sorted((k, round(float(v), 9)) for k, v in d.items()
                                          if isinstance(v, (int, float))))
            if key not in mat_cache:
                mat_cache[key] = orig_build(d, op_)
            return mat_cache[key]
        rm.build_materials = cached
        try:
            yield
        finally:
            rm.build_materials = orig_build

    lat_cache = {}

    def lattice(layer, grid_kind):
        key = (layer, grid_kind)
        if key not in lat_cache:
            strap = {None: None, "htm": inconel, "htp": zr}[grid_kind]
            with layer_materials(layer), hw._patched(rm, spec, strap_mat=strap, pitch=pitch):
                lat_cache[key] = hw._lattice_from_2d(rm, design, op, geo, design_map, None)
        return lat_cache[key]

    z = hw.elevations(spec)
    edges, fsegs = fuel_cuts(spec, n_layers)
    segs = [(*z["water_below"], water),
            (*z["bottom_nozzle"], slab_lattice(slab("bottom_nozzle"))),
            (*z["lower_cap"], slab_lattice(slab("lower_cap")))]
    fuel_fills = []
    for sg in fsegs:
        L = lattice(sg["layer"], sg["grid"])
        segs.append((sg["z0"], sg["z1"], L))
        fuel_fills.append((L, sg["z1"] - sg["z0"]))
    parked = spec.model_parked_rods

    def plain(base, grid=False):
        return slab(base + ("_parked" if parked else "") + ("_grid" if grid else ""))
    p0, p1 = z["plenum"]
    in_pl = [(g0, g1) for g0, g1, _ in z["grids"] if g0 >= p0 and g1 <= p1]
    zc = p0
    for g0, g1 in in_pl:
        if g0 > zc:
            segs.append((zc, g0, slab_lattice(plain("plenum"))))
        segs.append((g0, g1, slab_lattice(plain("plenum", grid=True))))
        zc = g1
    if zc < p1:
        segs.append((zc, p1, slab_lattice(plain("plenum"))))
    segs.append((*z["upper_cap"], slab_lattice(plain("upper_cap"))))
    segs.append((*z["upper_gap"], slab_lattice(plain("upper_gap"))))
    segs.append((*z["top_nozzle"], slab_lattice(slab("top_nozzle"))))
    segs.append((*z["water_above"], water))

    z_bot, z_top = z["water_below"][0], z["water_above"][1]
    zp_bot = o.ZPlane(z0=z_bot, boundary_type="vacuum"); zp_top = o.ZPlane(z0=z_top, boundary_type="vacuum")
    cyl_fuel, cyl_refl, cyl_barrel = o.ZCylinder(r=r_fuel), o.ZCylinder(r=r_refl), o.ZCylinder(r=r_barrel)
    cyl_vin = o.ZCylinder(r=spec.r_vessel_in)
    if spec.include_vessel_wall:
        cyl_out = o.ZCylinder(r=r_vessel_out, boundary_type="vacuum")
    else:
        cyl_vin.boundary_type = "vacuum"; cyl_out = cyl_vin
    planes = {}

    def zp(v):
        v = round(v, 6)
        if abs(v - z_bot) < 1e-6: return zp_bot
        if abs(v - z_top) < 1e-6: return zp_top
        if v not in planes: planes[v] = o.ZPlane(z0=v)
        return planes[v]
    cells = [o.Cell(fill=f, region=-cyl_fuel & +zp(a) & -zp(b)) for a, b, f in segs]
    full = +zp_bot & -zp_top
    cells.append(o.Cell(fill=refl_mat, region=+cyl_fuel & -cyl_refl & full))
    cells.append(o.Cell(fill=ss, region=+cyl_refl & -cyl_barrel & full))
    cells.append(o.Cell(fill=water, region=+cyl_barrel & -cyl_vin & full))
    if spec.include_vessel_wall:
        cells.append(o.Cell(fill=ss, region=+cyl_vin & -cyl_out & full))
    geom = o.Geometry(cells)
    materials = o.Materials(set(geom.get_all_materials().values()))
    hz = 0.5 * spec.h_active
    settings = rm._settings(tr["particles"], tr["batches"], tr["inactive"],
                            ((-r_fuel, -r_fuel, -hz), (r_fuel, r_fuel, hz)), seed=int(seed))
    model = o.Model(geometry=geom, materials=materials, settings=settings)

    # depletable materials with their layer-summed volumes, tagged by layer
    rows = dc.mark_depletable(model, fuel_fills, geo)
    byid = {}
    for (layer, *_), mats in mat_cache.items():
        for role, m in mats.items():
            if role.startswith("fuel"):
                byid[int(m.id)] = (layer, role)
    for r in rows:
        r["layer"], r["role"] = byid.get(r["id"], (None, "?"))
    if any(r["layer"] is None for r in rows):
        raise RuntimeError("a depletable material has no layer")
    # same guard as the 2D core: nothing outside the placed layers may deplete
    off = dc.unmark_unused(model, rows)

    # tallies: the three of axial_shape_c9 plus one on the layer edges
    fine, bands, _ = ax.axial_edges(spec, hw)
    tallies, shape = ax.build_tallies(o, geo, pitch, fine, hz)
    half = nx * asm_pitch / 2.0
    m_l = o.RectilinearMesh(name="layers")
    m_l.x_grid = np.array([-half, half]); m_l.y_grid = np.array([-half, half]); m_l.z_grid = np.asarray(edges)
    t_l = o.Tally(name="layer_fission"); t_l.filters = [o.MeshFilter(m_l)]; t_l.scores = ["fission"]
    tallies.append(t_l)
    model.tallies = tallies
    info = dict(r_fuel=r_fuel, r_refl=r_refl, r_barrel=r_barrel, n_segments=len(segs),
                n_fuel_segments=len(fsegs), n_layers=n_layers, n_depletable=len(rows),
                layer_edges=np.asarray(edges).tolist(), fine_edges=np.asarray(fine).tolist())
    info["materials_off"] = off
    return model, info, rows, np.asarray(edges), fine, shape


def read_states(case, shape, fine, edges, inactive, power_mw):
    """Per-solve F_dH, F_z, axial offset, layer shares and entropy convergence."""
    import openmc as o
    import axial_shape_c9 as ax
    import dep_common as dc
    out = []
    for p in dc.statepoints_in_order(case):
        keff, sd, asm, pin, axf, conv = ax.read_statepoint(o, str(p), shape, fine, inactive)
        with o.StatePoint(str(p)) as sp:
            lay = sp.get_tally(name="layer_fission").get_values(scores=["fission"]).ravel()
        d = ax.derive(asm, pin, axf, fine, power_mw)
        share = (lay / lay.sum()).tolist()
        out.append(dict(file=str(p), keff=keff, keff_sd=sd, fdh=d["fdh"], fz=d["fz"], ao=d["ao"],
                        layer_share=share, entropy_conv=conv))
    return out


def layer_burnup(states, bu_hist, edges):
    """Burnup of each layer at every state, predictor rule: the layer share
    at the beginning of a step acts over that step. Equal heavy-metal per
    unit height, so the layer factor is share / (h_layer / H)."""
    H = edges[-1] - edges[0]
    frac = np.diff(edges) / H
    B = [np.zeros(len(frac))]
    for i in range(1, len(bu_hist)):
        s = np.asarray(states[i - 1]["layer_share"]) / frac
        B.append(B[-1] + s * (bu_hist[i] - bu_hist[i - 1]))
    return [b.tolist() for b in B]


# ----------------------------------------------------------------- run ----
def run_design(idx, ckpt, raw, meta, tr, n_layers, k_eoc, out, estimate=False, mode="relative"):
    import reactor_model as rm
    import zoning as zn
    import hardware3d as hw
    import dep_common as dc
    from openmc_evaluator import _design_seed

    op, geo, spec = rm.Operating(), rm.Geometry17x17(), hw.HardwareSpec()
    design = design_of(ckpt, idx)
    seed = _design_seed(design, salt="core3d")
    model, info, rows, edges, fine, shape = build_layered(
        design, op, geo, spec, n_layers, tr, seed, zn.evaluator_design_map(design))
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
    print(f"[d{idx}] 3D hardware core, {n_layers} layers, {info['n_fuel_segments']} fuel segments, "
          f"{info['n_depletable']} depletable materials, seed {seed}, transport {tr}, end of cycle "
          + (f"relative, ratio {ratio:.5f}" if mode == "relative" else f"absolute at k = {k_eoc}"), flush=True)
    t0 = time.time()
    res = dc.run_adaptive(model, k_eoc if mode == "absolute" else None, spec_power, case=case,
                          verbose=True, k_target_ratio=(ratio if mode == "relative" else None), **sched)
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
        pc = dc.project_cost(times, int(raw[idx]["n_dep_solves"]), wall)
        pc.update(idx=idx, transport=tr, layers=n_layers, wall_estimate_s=wall, k_bol=res["k_hist"][0])
        return pc
    sdmap = guarded("k_sd_from_case", lambda: dc.k_sd_from_case(case), {})
    ksd = [sdmap.get(v) for v in res["k_hist"]]
    k_eoc = res["k_target"]
    sig, ib, slope = dc.bracket_sigma_efpd(res["bu_hist"], res["k_hist"], ksd, k_eoc, spec_power)
    c9 = meta["campaign9"]
    obj = dc.hump_and_cmax(res["k_hist"], res["k_hist"][0], raw[idx]["boron_points"],
                           float(c9["hump_noise_pcm"]))
    states = guarded("read_states", lambda: read_states(case, shape, fine, edges, tr["inactive"], op.power_mwth), [])
    aligned = len(states) == len(res["k_hist"])
    lb = layer_burnup(states, res["bu_hist"], edges) if aligned else None
    return dict(
        idx=idx, seed=int(seed), transport=tr, layers=n_layers, k_eoc=k_eoc, eoc_mode=mode,
        k_target_ratio=ratio, spec_power=spec_power,
        info=info, materials=rows, efpd=res["efpd"], bu_eoc=res["bu_eoc"], censored=res["censored"],
        k_hist=res["k_hist"], k_sd=ksd, bu_hist=res["bu_hist"], n_solves=res["n_solves"],
        sigma_efpd=sig, bracket=ib, slope_pcm_per_mwdkg=slope,
        hump_core_pcm=obj["hump_core_pcm"], hump_core_op_pcm=obj["hump_core_op_pcm"],
        k_peak=obj["k_peak"], bu_index=obj["bu_index"], c_max=obj["c_max"],
        c_max_status=obj["c_max_status"], g_efpd=EFPD_REQ - res["efpd"],
        states=states, states_aligned=aligned, layer_burnup=lb, solve_times=times, wall_s=wall,
        post_errors=errors)


# ------------------------------------------------------------- analysis ----
def analyse(done, raw, axial_json=None):
    ax3 = {}
    if axial_json and Path(axial_json).is_file():
        try:
            for k, v in json.loads(Path(axial_json).read_text()).items():
                ax3[k] = v
        except Exception:
            ax3 = {}
    R = {}
    for key, d in done.items():
        idx = d["idx"]; rec = raw[idx]
        row = dict(idx=idx, layers=d["layers"], efpd_3d=d["efpd"], efpd_archive=rec["cycle_length"],
                   d_efpd=d["efpd"] - rec["cycle_length"], sigma_3d=d["sigma_efpd"], censored=d["censored"],
                   n_solves=d["n_solves"], k_bol_3d=d["k_hist"][0],
                   k_bol_sd_pcm=(1e5 * d["k_sd"][0]) if d["k_sd"][0] else None,
                   k_bol_2d_archive=rec["keff_core_bol"],
                   lax_bol=rec["keff_core_bol"] / d["k_hist"][0],
                   hump_3d=d["hump_core_pcm"], hump_3d_op=d["hump_core_op_pcm"],
                   hump_archive=rec["hump_core_pcm"], c_max_3d=d["c_max"], c_max_archive=rec["c_max"],
                   d_c_max=d["c_max"] - rec["c_max"], margin_3d_d=d["efpd"] - EFPD_REQ,
                   margin_archive_d=rec["cycle_length"] - EFPD_REQ, wall_h=d["wall_s"] / 3600.0)
        if d.get("states") and d.get("states_aligned"):
            st = d["states"]
            row["fdh"] = [s["fdh"] for s in st]; row["fz"] = [s["fz"] for s in st]; row["ao"] = [s["ao"] for s in st]
            row["bu"] = d["bu_hist"]
            row["fdh_bol"], row["fdh_eoc"], row["fdh_max"] = row["fdh"][0], row["fdh"][-1], max(row["fdh"])
            row["fz_bol"], row["fz_eoc"], row["fz_max"] = row["fz"][0], row["fz"][-1], max(row["fz"])
            row["ao_bol"], row["ao_eoc"] = row["ao"][0], row["ao"][-1]
            row["entropy_conv_max"] = max((s["entropy_conv"] or 0) for s in st)
            lb = np.asarray(d["layer_burnup"])
            row["layer_burnup_eoc"] = lb[-1].tolist()
            row["layer_burnup_peaking_eoc"] = float(lb[-1].max() / lb[-1].mean()) if lb[-1].mean() > 0 else None
            row["layer_edges"] = d["info"]["layer_edges"]
        R[idx] = row
    return R


def report(R):
    L = ["=== Check 3: three-dimensional depletion, one material set per axial layer"]
    for idx, r in sorted(R.items()):
        L.append(f"C9-{idx}  ({r['layers']} layers)")
        L.append(f"  cycle length   3D {r['efpd_3d']:7.1f} d (sigma {r['sigma_3d'] if r['sigma_3d'] is None else round(r['sigma_3d'], 1)})   "
                 f"archive {r['efpd_archive']:7.1f} d   difference {r['d_efpd']:+.1f} d   "
                 f"margins over {EFPD_REQ:.0f}: 3D {r['margin_3d_d']:+.0f}, archive {r['margin_archive_d']:+.0f}"
                 + ("   CENSORED" if r["censored"] else ""))
        L.append(f"  k_eff BOL      3D {r['k_bol_3d']:.5f} (s.d. {r['k_bol_sd_pcm'] if r['k_bol_sd_pcm'] is None else round(r['k_bol_sd_pcm'])} pcm)   "
                 f"2D archive {r['k_bol_2d_archive']:.5f}   L_ax at BOL {r['lax_bol']:.4f}   (table 1.0289)")
        L.append(f"  hump (3D)      {r['hump_3d']:+.0f} pcm (no floor {r['hump_3d_op']:+.0f})   archive {r['hump_archive']:+.0f}")
        L.append(f"  c_max          3D {r['c_max_3d']:.0f} ppm   archive {r['c_max_archive']:.0f}   difference {r['d_c_max']:+.0f} ppm")
        if "fdh_bol" in r:
            L.append(f"  F_dH           BOL {r['fdh_bol']:.3f}  max {r['fdh_max']:.3f}  EOC {r['fdh_eoc']:.3f}   "
                     f"F_z BOL {r['fz_bol']:.3f}  max {r['fz_max']:.3f}  EOC {r['fz_eoc']:.3f}   "
                     f"AO BOL {r['ao_bol']:+.3f}  EOC {r['ao_eoc']:+.3f}   entropy conv. max {r['entropy_conv_max']}")
            lb = r["layer_burnup_eoc"]
            L.append("  layer burnup at EOC [MWd/kgHM], bottom to top: "
                     + " ".join(f"{b:.1f}" for b in lb)
                     + f"   (max/mean {r['layer_burnup_peaking_eoc']:.3f})")
        L.append(f"  solves {r['n_solves']}, wall {r['wall_h']:.2f} h")
    return "\n".join(L)


def selftest():
    import hardware3d as hw
    spec = hw.HardwareSpec()
    edges, segs = fuel_cuts(spec, 8)
    assert len(edges) == 9 and abs(edges[0] + 60) < 1e-9 and abs(edges[-1] - 60) < 1e-9
    assert sum(s["z1"] - s["z0"] for s in segs) - 120.0 < 1e-6
    assert {s["layer"] for s in segs} == set(range(8))
    hg = sum(s["z1"] - s["z0"] for s in segs if s["grid"])
    assert abs(hg - 3 * spec.grid_height) < 1e-6, hg          # the three bands in the fuel, whole
    # layer burnup with the predictor rule
    st = [dict(layer_share=[0.25, 0.75]), dict(layer_share=[0.5, 0.5]), dict(layer_share=[0.5, 0.5])]
    lb = layer_burnup(st, [0.0, 2.0, 4.0], np.array([0.0, 60.0, 120.0]))
    assert lb[1] == [1.0, 3.0] and lb[2] == [3.0, 5.0], lb
    done = {"d47": dict(idx=47, layers=8, efpd=1800.0, sigma_efpd=30.0, censored=False, n_solves=8,
                        k_hist=[1.0, 0.99], k_sd=[1e-3, 1e-3], bu_hist=[0.0, 0.5],
                        hump_core_pcm=0.0, hump_core_op_pcm=-50.0, c_max=1380.0, wall_s=3600.0,
                        states=[dict(fdh=1.5, fz=1.44, ao=-0.02, layer_share=[0.5, 0.5], entropy_conv=5)] * 2,
                        states_aligned=True, layer_burnup=[[0, 0], [0.5, 0.5]],
                        info=dict(layer_edges=[-60, 0, 60]))}
    raw = {47: dict(cycle_length=1946.7, keff_core_bol=1.02728, hump_core_pcm=0.0, c_max=1405.9)}
    R = analyse(done, raw)
    assert abs(R[47]["lax_bol"] - 1.02728) < 1e-9 and R[47]["layer_burnup_peaking_eoc"] == 1.0
    assert "layer burnup" in report(R)
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--designs", type=int, nargs="*", default=[47])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--particles", type=int, default=DEFAULT_TR["particles"])
    ap.add_argument("--batches", type=int, default=DEFAULT_TR["batches"])
    ap.add_argument("--inactive", type=int, default=DEFAULT_TR["inactive"])
    ap.add_argument("--k-eoc", type=float, default=1.0)
    ap.add_argument("--eoc-mode", choices=["relative", "absolute"], default="relative",
                    help="relative (default): end of cycle when the core has lost the same "
                         "reactivity as the campaign assembly, k = k(BOL) x k_target/k_inf(BOL). "
                         "absolute: end of cycle at the fixed k below")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="c9_dep_core3d")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ckpt, raw, meta = load_ckpt(a.checkpoint)
    tr = dict(particles=a.particles, batches=a.batches, inactive=a.inactive)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = out / "runs.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    if a.analyse:
        R = analyse(done, raw)
        (out / "summary.json").write_text(json.dumps(R, indent=1))
        txt = report(R); (out / "summary.txt").write_text(txt + "\n"); print(txt)
        return 0
    import hardware3d as hw
    edges, segs = fuel_cuts(hw.HardwareSpec(), a.layers)
    print(f"check 3: designs {a.designs}, {a.layers} layers ({len(segs)} fuel segments, "
          f"{12 * a.layers} depletable materials), transport {tr}, end of cycle {a.eoc_mode}, "
          f"schedule {meta.get('schedule')}, cached {sorted(done)}")
    for idx in a.designs:
        r = raw[idx]
        print(f"  C9-{idx}: archive EFPD {r['cycle_length']:.0f} d, {r['n_dep_solves']} depletion solves, c_max {r['c_max']:.0f} ppm")
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
            pc = run_design(idx, ckpt, raw, meta, tr, a.layers, a.k_eoc, out, estimate=True, mode=a.eoc_mode)
            (out / f"estimate_d{idx}.json").write_text(json.dumps(pc, indent=1))
            if pc.get("ok"):
                print(f"[d{idx}] ESTIMATE fresh solve {pc['fresh_s']:.0f} s, depleted solve {pc['depleted_s']:.0f} s, "
                      f"overhead {pc['overhead_s']:.0f} s per solve, {pc['n_solves']} solves -> about "
                      f"{pc['projected_h']:.1f} h (k_BOL 3D {pc['k_bol']:.5f})")
                print(f"PROJECTED_HOURS {pc['projected_h']:.2f}")
            else:
                print(f"[d{idx}] ESTIMATE failed: no statepoints found")
        return 0
    for idx in a.designs:
        if f"d{idx}" in done:
            print(f"[d{idx}] cached"); continue
        res = run_design(idx, ckpt, raw, meta, tr, a.layers, a.k_eoc, out, mode=a.eoc_mode)
        done[f"d{idx}"] = res
        store.write_text(json.dumps(done, indent=1))
        print(f"[d{idx}] EFPD {res['efpd']:.1f} d (archive {raw[idx]['cycle_length']:.1f}), c_max {res['c_max']:.0f} "
              f"(archive {raw[idx]['c_max']:.0f}), hump {res['hump_core_pcm']:+.0f} pcm "
              f"[{res['n_solves']} solves, {res['wall_s'] / 3600:.2f} h]", flush=True)
    print("done; run --analyse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
