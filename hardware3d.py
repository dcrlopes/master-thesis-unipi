#!/usr/bin/env python3
"""
hardware3d.py -- finite-height core with the assembly hardware stack, the
control-rod stack, the core barrel, the downcomer and the vessel wall, for
the three-dimensional confirmation of the Campaign 8 candidates.

EVERY HARDWARE NUMBER BELOW IS READ FROM THE BENCHMARK SCRIPTS of Zenodo
record 15231335 (nuscale/surfaces.py, pins.py, materials.py), Ez Aldeen et
al. 2025, J. Nucl. Eng. 6(4):44, benchmark definition Fridman et al. 2023.
Where this thesis departs from the benchmark the departure is named.

AXIAL STACK, z = 0 at the fuel mid-plane, cm
  water below         --water-below (15)      thesis: benchmark has vacuum here
  bottom nozzle       10.160   SS304L 0.172 + coolant 0.828       (materials.py)
  lower end cap        1.205   Zircaloy plug r=0.4750 per rod, GT walls, water
  ACTIVE FUEL        120.0     thesis active height (benchmark 200), the
                               campaign lattice, with three grid bands inside
  plenum spring       13.490   clad continues, inside: Inconel spring rod
                               r=0.0646 in helium                 (pins.py)
  upper end cap        1.205
  upper coolant gap    8.481   water, GT walls
  top nozzle           9.020   SS304L 0.177 + coolant 0.823
  water above         --water-above (15)      thesis: benchmark has vacuum here

SPACER GRIDS (surfaces.py, pins.py): the strap is EXPLICIT, a square annulus
in every pin cell between the inner box of 1.2242 cm and the pin pitch. Grid
band 4.445 cm. First grid 3.555 cm above the fuel bottom, HMP grid in
Inconel, HTP grids in Zircaloy-4, benchmark span 51.054 cm for five grids
on 200 cm. Thesis: four grids, constant pattern, span 41.4 cm, one grid in
the plenum (wks720 decision). The plenum grid is smeared into the plenum
slab at its exact area fraction, since the plenum has no pin structure.

CONTROL RODS (surfaces.py): the benchmark rod, inserted, from the bottom
nozzle top: 8.386 empty guide tube, 4.859 SS304L end plug, 30.480 AIC,
157.480 B4C (to the fuel top), 23.176 upper plenum. Withdrawn: the end plug
occupies the top 4.859 cm of the fuel and 23.176 cm of AIC sit in the
plenum + end cap + gap region, exactly their combined height. Radii: AIC
0.4267, B4C 0.4229, clad 0.4369 / 0.4839. Thesis: B4C length scaled so the
B4C top coincides with the 120 cm fuel top (77.48 cm), all other rod
dimensions kept. Represented as follows.
  inserted   fuel bottom + 12.04 cm unrodded (empty tube + end plug, the plug
             approximated as water), then 30.48 cm AIC, then B4C to the top,
             then the rod upper plenum (He + SS clad) homogenised into the
             slabs above the fuel
  withdrawn  23.176 cm of AIC + clad homogenised into the plenum, end cap and
             gap slabs of every assembly (all 32 carry a CRA), the end plug
             inside the top 4.859 cm of fuel NOT modelled (stated)

RADIAL, full height: fuel footprint, heavy reflector (SS + coolant, campaign
fraction 0.90 by default, benchmark 0.956 via --refl-steel-vol), SS304L
barrel 5.08 (93.98 to 99.06 in the benchmark), downcomer water to the
vessel inner radius 90.0, SS304 vessel wall 10.0 (Maruyama 2013), vacuum.

KNOWN DEPARTURES OF THE CAMPAIGN MODEL FROM THE BENCHMARK, exposed as flags
so the confirmation can quantify them:
  --refl-steel-vol   0.90 (campaign) against 0.956 (benchmark)
  --cr-abs-radius    0.4331 (campaign, all absorbers) against 0.4229 B4C /
                     0.4267 AIC (benchmark)
  coolant density    0.72 g/cm3 (campaign) against 0.752 implied by the
                     benchmark number densities, not a flag, stated only
"""
from __future__ import annotations

import contextlib
import dataclasses
import math

openmc = None


def _need_openmc():
    global openmc
    if openmc is None:
        import openmc as _o
        openmc = _o
    return openmc


@dataclasses.dataclass
class HardwareSpec:
    h_active: float = 120.0
    bottom_nozzle: float = 10.160
    top_nozzle: float = 9.020
    end_cap: float = 1.205
    plenum: float = 13.490
    upper_gap: float = 8.481
    grid_height: float = 4.445
    grid1_offset: float = 3.555
    grid_span: float = 41.4
    n_grids: int = 4
    grid_box_in: float = 1.2242          # strap inner box, surfaces.py
    nozzle_ss_bottom: float = 0.172
    nozzle_ss_top: float = 0.177
    spring_r: float = 0.0646             # Inconel spring rod radius, pins.py
    cr_empty_gt: float = 8.386           # from the bottom nozzle top
    cr_end_plug: float = 4.859
    cr_aic: float = 30.480
    cr_upper_plenum: float = 23.176
    cr_r_aic: float = 0.4267
    cr_r_b4c: float = 0.4229
    cr_clad_ir: float = 0.4369
    cr_clad_or: float = 0.4839
    n_cr_per_assembly: int = 24          # instrument tube stays empty
    barrel: float = 5.08
    r_vessel_in: float = 90.0
    vessel_wall: float = 10.0
    include_vessel_wall: bool = True
    water_below: float = 15.0
    water_above: float = 15.0
    refl_steel_vol: float | None = None  # None keeps the campaign value
    cr_abs_radius: float | None = None   # None keeps the campaign value
    rod_stack: str = "benchmark"         # or "full-b4c" (axial-study style)
    model_parked_rods: bool = True


# ----------------------------------------------------------------- layout
def elevations(spec: HardwareSpec) -> dict:
    hz = 0.5 * spec.h_active
    z = {"fuel": (-hz, hz)}
    z["lower_cap"] = (-hz - spec.end_cap, -hz)
    z["bottom_nozzle"] = (z["lower_cap"][0] - spec.bottom_nozzle, z["lower_cap"][0])
    z["water_below"] = (z["bottom_nozzle"][0] - spec.water_below, z["bottom_nozzle"][0])
    z["plenum"] = (hz, hz + spec.plenum)
    z["upper_cap"] = (z["plenum"][1], z["plenum"][1] + spec.end_cap)
    z["upper_gap"] = (z["upper_cap"][1], z["upper_cap"][1] + spec.upper_gap)
    z["top_nozzle"] = (z["upper_gap"][1], z["upper_gap"][1] + spec.top_nozzle)
    z["water_above"] = (z["top_nozzle"][1], z["top_nozzle"][1] + spec.water_above)
    z["grids"] = []
    for g in range(spec.n_grids):
        g0 = -hz + spec.grid1_offset + g * spec.grid_span
        z["grids"].append((g0, g0 + spec.grid_height, "htm" if g == 0 else "htp"))
    nz_top = z["bottom_nozzle"][1]
    plug0 = nz_top + spec.cr_empty_gt
    aic0 = plug0 + spec.cr_end_plug
    b4c0 = aic0 + spec.cr_aic
    if spec.rod_stack == "benchmark":
        z["rod_unrodded"] = (-hz, aic0)
        z["rod_aic"] = (aic0, b4c0)
        z["rod_b4c"] = (b4c0, hz)
    else:
        z["rod_unrodded"] = (-hz, -hz)
        z["rod_aic"] = (-hz, -hz)
        z["rod_b4c"] = (-hz, hz)
    z["parked_aic"] = (hz, hz + spec.plenum + spec.end_cap + spec.upper_gap)
    return z


def volume_fractions(geo, pitch: float, spec: HardwareSpec) -> dict:
    """Per-assembly-cell fractions for the homogenised slabs. Water is the
    remainder. Variants with rods parked (AIC in the tubes) or inserted (rod
    upper plenum in the tubes), and with the plenum grid strap."""
    n_rod, n_gt, n_cr = 264, 25, spec.n_cr_per_assembly
    A = (geo.lattice * pitch) ** 2
    a_plug = n_rod * math.pi * geo.clad_or ** 2
    a_clad = n_rod * math.pi * (geo.clad_or ** 2 - geo.clad_ir ** 2)
    a_in = n_rod * math.pi * geo.clad_ir ** 2
    a_gt = n_gt * math.pi * (geo.gt_or ** 2 - geo.gt_ir ** 2)
    a_spring = n_rod * math.pi * spec.spring_r ** 2
    a_strap = (geo.lattice ** 2) * (pitch ** 2 - spec.grid_box_in ** 2)
    a_cr_clad = n_cr * math.pi * (spec.cr_clad_or ** 2 - spec.cr_clad_ir ** 2)
    a_aic = n_cr * math.pi * spec.cr_r_aic ** 2
    a_he_parked = n_cr * math.pi * (spec.cr_clad_ir ** 2 - spec.cr_r_aic ** 2)
    a_he_plenum = n_cr * math.pi * spec.cr_clad_ir ** 2

    def close(d):
        d = dict(d)
        d["water"] = 1.0 - sum(d.values())
        assert d["water"] > 0 and all(v >= 0 for v in d.values()), d
        return d

    vf = {"bottom_nozzle": close({"ss": spec.nozzle_ss_bottom}),
          "top_nozzle": close({"ss": spec.nozzle_ss_top}),
          "lower_cap": close({"zr": (a_plug + a_gt) / A})}
    base = {"plenum": {"zr": (a_clad + a_gt) / A, "inconel": a_spring / A, "he": (a_in - a_spring) / A},
            "upper_cap": {"zr": (a_plug + a_gt) / A},
            "upper_gap": {"zr": a_gt / A}}
    parked = {"aic": a_aic / A, "cr_ss": a_cr_clad / A, "cr_he": a_he_parked / A}
    inserted = {"cr_ss": a_cr_clad / A, "cr_he": a_he_plenum / A}
    for name, b in base.items():
        vf[name] = close(b)
        vf[name + "_parked"] = close({**b, **parked})
        vf[name + "_inserted"] = close({**b, **inserted})
    for key in ("plenum", "plenum_parked", "plenum_inserted"):
        d = {k: v for k, v in vf[key].items() if k != "water"}
        d["zr"] = d["zr"] + a_strap / A
        vf[key + "_grid"] = close(d)
    vf["_areas"] = dict(A=A, strap_frac_of_cell=a_strap / A,
                        spring_frac_of_rod_bore=(spec.spring_r / geo.clad_ir) ** 2)
    return vf


# -------------------------------------------------------------- materials
def mix(name, parts, T):
    o = _need_openmc()
    mats, fr = list(parts.keys()), list(parts.values())
    m = o.Material.mix_materials(mats, fr, "vo")
    m.name, m.temperature = name, T
    if any("water" in (x.name or "") for x in mats):
        if not any("c_H_in_H2O" in str(s) for s in getattr(m, "_sab", [])):
            m.add_s_alpha_beta("c_H_in_H2O")
    return m


def make_inconel(T):
    """Inconel, atom densities verbatim from the benchmark materials.py."""
    o = _need_openmc()
    m = o.Material(name="Inconel")
    for n, d in (("Cr50", 7.82390e-04), ("Cr52", 1.50880e-02), ("Cr53", 1.71080e-03), ("Cr54", 4.25860e-04),
                 ("Fe54", 1.47970e-03), ("Fe56", 2.32290e-02), ("Fe57", 5.36450e-04), ("Fe58", 7.13920e-05),
                 ("Mn55", 7.82010e-04), ("Ni58", 2.93200e-02), ("Ni60", 1.12940e-02), ("Ni61", 4.90940e-04),
                 ("Ni62", 1.56530e-03), ("Ni64", 3.98640e-04)):
        m.add_nuclide(n, d)
    m.set_density("sum"); m.temperature = T
    return m


# ------------------------------------------------------------ monkeypatches
@contextlib.contextmanager
def _patched(rm, spec: HardwareSpec, strap_mat=None, pitch=None):
    """Temporarily (1) wrap the three pin-universe builders to add the
    explicit grid strap, (2) force the reflector steel fraction, (3) force
    the absorber radius. All restored on exit."""
    o = _need_openmc()
    saved = dict(fp=rm._fuel_pin_universe, gt=rm._guide_tube_universe, cr=rm._cr_gt_universe,
                 refl=rm.make_heavy_reflector, rabs=rm.CR_R_ABS)

    def add_strap(u):
        outer = list(u.cells.values())[-1]          # outermost water cell in all three builders
        box_in = o.model.RectangularPrism(spec.grid_box_in, spec.grid_box_in)
        box_out = o.model.RectangularPrism(pitch, pitch)
        outer.region = outer.region & -box_in
        u.add_cell(o.Cell(fill=strap_mat, region=+box_in & -box_out))
        return u

    if strap_mat is not None:
        rm._fuel_pin_universe = lambda *a, **k: add_strap(saved["fp"](*a, **k))
        rm._guide_tube_universe = lambda *a, **k: add_strap(saved["gt"](*a, **k))
        rm._cr_gt_universe = lambda *a, **k: add_strap(saved["cr"](*a, **k))
    if spec.refl_steel_vol is not None:
        rm.make_heavy_reflector = lambda op, steel_vol=None: saved["refl"](op, steel_vol=spec.refl_steel_vol)
    if spec.cr_abs_radius is not None:
        rm.CR_R_ABS = spec.cr_abs_radius
    try:
        yield
    finally:
        rm._fuel_pin_universe, rm._guide_tube_universe, rm._cr_gt_universe = saved["fp"], saved["gt"], saved["cr"]
        rm.make_heavy_reflector, rm.CR_R_ABS = saved["refl"], saved["rabs"]


def _lattice_from_2d(rm, design, op, geo, design_map, rodded_map):
    m = rm.make_core_model(design, op, geo, design_map=design_map, rodded_map=rodded_map, enforce_vessel=True)
    model = m[0] if isinstance(m, tuple) else m
    o = _need_openmc()
    for c in model.geometry.root_universe.cells.values():
        if isinstance(c.fill, o.RectLattice):
            return c.fill
    raise RuntimeError("no lattice found in the 2D core model")


# ------------------------------------------------------------------ builder
def build_model_3d_hw(design, op, geo, *, design_map=None, rodded_map=None, spec=HardwareSpec(),
                      particles=100000, batches=170, inactive=60, seed=1):
    o = _need_openmc()
    import numpy as np
    import reactor_model as rm
    import core_geometry as cg

    positions = set()
    if rodded_map is not None:
        positions = set(rodded_map[0] if isinstance(rodded_map, tuple) else rodded_map)
    pitch = float(design.get("pitch", 1.26)); refl = float(design["refl_thick"])
    core_map = np.asarray(cg.CORE_MAP_32); ny, nx = core_map.shape
    asm_pitch = geo.lattice * pitch
    r_env = cg.core_envelope_radius(pitch, core_map, geo.lattice)
    r_fuel = r_env + cg.FUEL_PAD_CM; r_refl = r_fuel + refl; r_barrel = r_refl + spec.barrel
    if r_barrel > spec.r_vessel_in - 1e-9:
        raise ValueError(f"barrel outer radius {r_barrel:.3f} exceeds the vessel inner radius")
    r_vessel_out = spec.r_vessel_in + spec.vessel_wall

    T = op.mod_T
    water = rm.make_water(op.boron_ppm, T)
    ss = rm.make_ss304(op.clad_T); zr = rm.make_zircaloy(op.clad_T); he = rm.make_helium(op.clad_T)
    inconel = make_inconel(op.clad_T)
    crm = rm.make_cr_materials(op.clad_T)
    with _patched(rm, spec):
        refl_mat = rm.make_heavy_reflector(op)
    lib = {"ss": ss, "zr": zr, "he": he, "inconel": inconel, "water": water,
           "aic": crm["AIC"], "cr_ss": crm["cr_ss"], "cr_he": crm["cr_he"]}
    vf = volume_fractions(geo, pitch, spec)
    slab_cache = {}
    def slab(name):
        if name not in slab_cache:
            slab_cache[name] = mix(name, {lib[k]: v for k, v in vf[name].items()}, T)
        return slab_cache[name]

    lat_cache = {}
    def lattice(rod_kind, grid_kind):
        key = (rod_kind, grid_kind)
        if key not in lat_cache:
            rmap = None if (rod_kind is None or not positions) else (positions, rod_kind)
            strap = {None: None, "htm": inconel, "htp": zr}[grid_kind]
            with _patched(rm, spec, strap_mat=strap, pitch=pitch):
                lat_cache[key] = _lattice_from_2d(rm, design, op, geo, design_map, rmap)
        return lat_cache[key]

    refl_u = o.Universe(cells=[o.Cell(fill=refl_mat)])
    def slab_lattice(mat_plain, mat_rodded=None):
        u0 = o.Universe(cells=[o.Cell(fill=mat_plain)])
        u1 = o.Universe(cells=[o.Cell(fill=mat_rodded)]) if mat_rodded is not None else u0
        L = o.RectLattice(); L.lower_left = (-nx * asm_pitch / 2, -ny * asm_pitch / 2)
        L.pitch = (asm_pitch, asm_pitch)
        L.universes = [[(u1 if (i, j) in positions else u0) if core_map[i, j] == 1 else refl_u
                        for j in range(nx)] for i in range(ny)]
        L.outer = refl_u
        return L

    z = elevations(spec)
    def rod_kind_at(zz):
        if not positions:
            return None
        if z["rod_b4c"][0] <= zz < z["rod_b4c"][1]: return "B4C"
        if z["rod_aic"][0] <= zz < z["rod_aic"][1]: return "AIC"
        return None
    def grid_kind_at(zz):
        for g0, g1, kind in z["grids"]:
            if g0 <= zz < g1 and g1 <= z["fuel"][1]: return kind
        return None
    cuts = {z["fuel"][0], z["fuel"][1]}
    for g0, g1, _ in z["grids"]:
        if g1 <= z["fuel"][1]: cuts |= {g0, g1}
    if positions and spec.rod_stack == "benchmark":
        cuts |= {z["rod_aic"][0], z["rod_b4c"][0]}
    cuts = sorted(c for c in cuts if z["fuel"][0] - 1e-9 <= c <= z["fuel"][1] + 1e-9)

    segs = [(*z["water_below"], water),
            (*z["bottom_nozzle"], slab_lattice(slab("bottom_nozzle"))),
            (*z["lower_cap"], slab_lattice(slab("lower_cap")))]
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (a + b)
        segs.append((a, b, lattice(rod_kind_at(mid), grid_kind_at(mid))))
    parked = spec.model_parked_rods
    def pair(base, grid=False):
        sfx = "_grid" if grid else ""
        plain = slab(base + ("_parked" if parked else "") + sfx)
        rodded = slab(base + "_inserted" + sfx) if positions else None
        return plain, rodded
    p0, p1 = z["plenum"]
    in_pl = [(g0, g1) for g0, g1, _ in z["grids"] if g0 >= p0 and g1 <= p1]
    zc = p0
    for g0, g1 in in_pl:
        if g0 > zc: segs.append((zc, g0, slab_lattice(*pair("plenum"))))
        segs.append((g0, g1, slab_lattice(*pair("plenum", grid=True))))
        zc = g1
    if zc < p1: segs.append((zc, p1, slab_lattice(*pair("plenum"))))
    segs.append((*z["upper_cap"], slab_lattice(*pair("upper_cap"))))
    segs.append((*z["upper_gap"], slab_lattice(*pair("upper_gap"))))
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
    settings = rm._settings(particles, batches, inactive,
                            ((-r_fuel, -r_fuel, -hz), (r_fuel, r_fuel, hz)), seed=seed)
    model = o.Model(geometry=geom, materials=materials, settings=settings)
    info = dict(r_env=r_env, r_fuel=r_fuel, r_refl=r_refl, r_barrel=r_barrel,
                downcomer=spec.r_vessel_in - r_barrel, r_vessel_out=r_vessel_out,
                z_bottom=z_bot, z_top=z_top, n_segments=len(segs), n_lattices=len(lat_cache),
                rod_positions=len(positions), vf=vf)
    return model, info


def describe(design, geo, spec: HardwareSpec) -> str:
    import core_geometry as cg
    pitch = float(design.get("pitch", 1.26)); refl = float(design["refl_thick"])
    r_env = cg.core_envelope_radius(pitch); r_fuel = r_env + cg.FUEL_PAD_CM
    r_refl = r_fuel + refl; r_bar = r_refl + spec.barrel
    z = elevations(spec); vf = volume_fractions(geo, pitch, spec)
    L = [f"radial [cm]: R_env {r_env:.3f}  fuel cyl {r_fuel:.3f}  reflector -> {r_refl:.3f}  "
         f"barrel -> {r_bar:.3f}  downcomer {spec.r_vessel_in - r_bar:.3f}  vessel {spec.r_vessel_in} -> "
         f"{spec.r_vessel_in + (spec.vessel_wall if spec.include_vessel_wall else 0):.1f}"]
    if r_bar > spec.r_vessel_in: L.append("  !! barrel does not fit")
    L.append("axial [cm from the fuel mid-plane]:")
    for k in ("water_below", "bottom_nozzle", "lower_cap", "fuel", "plenum", "upper_cap", "upper_gap",
              "top_nozzle", "water_above"):
        L.append(f"  {k:14s} {z[k][0]:8.3f} to {z[k][1]:8.3f}")
    for i, (g0, g1, kind) in enumerate(z["grids"], 1):
        where = "in fuel" if g1 <= z["fuel"][1] else "in plenum" if g0 >= z["plenum"][0] else "STRADDLES"
        L.append(f"  grid {i} {kind}  {g0:8.3f} to {g1:8.3f}  {where}")
    L.append(f"inserted rod ({spec.rod_stack}): unrodded {z['rod_unrodded'][0]:.3f} to "
             f"{z['rod_unrodded'][1]:.3f}, AIC to {z['rod_aic'][1]:.3f}, B4C to {z['rod_b4c'][1]:.3f} "
             f"(B4C length {z['rod_b4c'][1] - z['rod_b4c'][0]:.2f} cm)")
    L.append(f"parked rod AIC: {z['parked_aic'][0]:.3f} to {z['parked_aic'][1]:.3f} "
             f"(plenum + cap + gap, {'on' if spec.model_parked_rods else 'off'})")
    a = vf["_areas"]
    L.append(f"grid strap area fraction of the cell {a['strap_frac_of_cell']:.4f} (explicit box "
             f"{spec.grid_box_in} -> pitch), spring fraction of the rod bore {a['spring_frac_of_rod_bore']:.4f}")
    L.append("homogenised slabs (volume fractions per assembly cell):")
    for k, d in vf.items():
        if not k.startswith("_"):
            L.append(f"  {k:22s} " + "  ".join(f"{m} {v:.4f}" for m, v in d.items()))
    L.append(f"departures exposed: refl_steel_vol {spec.refl_steel_vol or 'campaign 0.90'}, "
             f"cr_abs_radius {spec.cr_abs_radius or 'campaign 0.4331'}")
    return "\n".join(L)
