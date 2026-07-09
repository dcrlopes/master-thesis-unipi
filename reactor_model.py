"""
reactor_model.py
================

Parametric OpenMC model of a LABGENE-like small PWR core, anchored to the
Brazilian-Navy open-source data and using NuScale-style fuel-pin/assembly
geometry as the design analog (standard 17x17 LEU UO2, the same family your
existing NuScale scripts already use).

The model is PARAMETRIC: every quantity the optimizer is allowed to vary is read
from a `design` dictionary, so the SAME builder serves (a) the teaching notebook
and (b) the OpenMCEvaluator in reactor_optimization.py. There is one source of
truth for the physics.

Three model fidelities, cheapest first:
    make_pincell_model(design)     -> single pin, infinite lattice  (k_inf, ~seconds)
    make_assembly_model(design)    -> one 17x17 assembly             (k_inf + pin power)
    make_core_model(design)        -> small 2D multi-assembly core   (k_eff, leakage)

NOTE ON OPEN DATA
-----------------
Several LABGENE/SNCA parameters are NOT public. Values below marked (ASSUMED)
are engineering placeholders chosen to be consistent with the open data
(48 MWth, PWR, UO2, LEU, ~31 assemblies, ~1.2 m active height, vessel ID ~1.8 m
i.e. 0.9 m inner radius) and the NuScale analog. Quantities marked
(EST. FROM IMAGE) were measured from the CTMSP vessel drawing and cross-checked
against the notebook heavy-metal mass. Document every assumption in your thesis.

Requires: openmc (with cross-section data) — see README.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np
import openmc


# =============================================================================
# Fixed reference geometry (NuScale-style 17x17 pin) and operating conditions
# =============================================================================
@dataclass
class Geometry17x17:
    # radial pin dimensions [cm] (NuScale-style; see your surfaces.py)
    fuel_or:   float = 0.4058       # fuel pellet outer radius
    clad_ir:   float = 0.4140       # cladding inner radius (gap between)
    clad_or:   float = 0.4750       # cladding outer radius
    gt_ir:     float = 0.5715       # guide-tube inner radius
    gt_or:     float = 0.6121       # guide-tube outer radius
    lattice:   int   = 17           # 17x17 assembly
    # (EST. FROM IMAGE) active fuel height ~1.20 m, measured from the CTMSP vessel
    # drawing (fuel top sits at nozzle level) and cross-checked against the fixed
    # HM = 4733 kg with a 17x17 fuel fraction (~0.30). The previous 180 cm was an
    # early placeholder that is INCONSISTENT with the drawing. Height enters ONLY
    # the HM-mass / specific-power bookkeeping -- the 2D transport is infinite in z
    # -- so this edit changes specific power, NOT any k value.
    active_height: float = 120.0    # active fuel height [cm] ~1.20 m


@dataclass
class Operating:
    fuel_T:    float = 900.0        # K
    clad_T:    float = 600.0        # K
    mod_T:     float = 580.0        # K  (~307 C inlet-ish; adjust if you wish)
    boron_ppm: float = 1000.0       # soluble boron in coolant
    power_mwth: float = 48.0        # core thermal power
    # (EST. FROM IMAGE) at H~1.2 m the 17x17 fuel fraction + HM 4733 kg imply
    # ~31.5 assemblies -> a compact 6x6-minus-corners core (32). This drives the
    # specific power fed to depletion: 32 -> ~9.98 W/gHM. The old value 21 was
    # tied to the superseded 1.8 m height. To reproduce EXACTLY the 10.14 W/gHM /
    # 4733 kg of your earlier runs and deck, use n_assemblies = 31 with
    # active_height = 122.0 (both sit inside the drawing's 1.15-1.25 m envelope).
    n_assemblies: int = 32          # image-consistent compact core


# Standard Westinghouse 17x17 guide-tube map: 24 guide tubes + 1 central
# instrument tube. (row, col), 0-indexed. Matches your NuScale nonfuel array.
GUIDE_TUBE_POSITIONS = [
    (2, 5), (2, 8), (2, 11),
    (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13),
    (14, 5), (14, 8), (14, 11),
]

# Where Gd burnable-poison pins go (a typical symmetric pattern). Used only
# when design['gd_wt'] > 0. Keep modest; too many Gd pins over-flatten and
# waste neutrons (you will study this trade-off).
GD_PIN_POSITIONS = [
    (2, 2), (2, 14), (14, 2), (14, 14),
    (6, 6), (6, 10), (10, 6), (10, 10),
    (3, 8), (8, 3), (8, 13), (13, 8),
]


# =============================================================================
# Materials (parametric)
# =============================================================================
def make_water(boron_ppm: float, T: float) -> openmc.Material:
    """Borated light water. add_s_alpha_beta gives proper thermal scattering."""
    w = openmc.Material(name=f"water_{boron_ppm:.0f}ppmB")
    # number of H2O molecules; boron added as natural B at the given ppm (mass).
    w.add_element("H", 2.0, "ao")
    w.add_element("O", 1.0, "ao")
    if boron_ppm > 0:
        # ppm by mass of natural boron relative to water
        w.add_element("B", boron_ppm * 1e-6 * (18.015 / 10.81), "ao")
    w.set_density("g/cm3", 0.72)     # ~ hot pressurized water density
    w.temperature = T
    w.add_s_alpha_beta("c_H_in_H2O")
    return w


def make_zircaloy(T: float) -> openmc.Material:
    z = openmc.Material(name="Zircaloy-4")
    z.add_element("Zr", 0.982, "wo")
    z.add_element("Sn", 0.0145, "wo")
    z.add_element("Fe", 0.002, "wo")
    z.add_element("Cr", 0.001, "wo")
    z.add_element("O", 0.0005, "wo")
    z.set_density("g/cm3", 6.55)
    z.temperature = T
    return z


def make_helium(T: float) -> openmc.Material:
    he = openmc.Material(name="He gap")
    he.add_element("He", 1.0)
    he.set_density("g/cm3", 0.0015)
    he.temperature = T
    return he


# Heavy (steel) reflector volume fraction. LABGENE's reflector is a steel ring,
# not a water pond (the NuScale analog is SS + water at ~95.6/4.4 vol). 0.90 is a
# slightly conservative (more-water) ENGINEERING ESTIMATE; raise toward 0.956 for
# a closer NuScale match. Document this choice in the thesis.
HEAVY_REFL_STEEL_VOL = 0.90


def make_ss304(T: float) -> openmc.Material:
    """Type-304 stainless steel: the heavy-reflector / structural analog."""
    ss = openmc.Material(name="SS304")
    ss.add_element("Fe", 0.695, "wo")
    ss.add_element("Cr", 0.190, "wo")
    ss.add_element("Ni", 0.095, "wo")
    ss.add_element("Mn", 0.020, "wo")
    ss.set_density("g/cm3", 7.90)
    ss.temperature = T
    return ss


def make_heavy_reflector(op: "Operating",
                         steel_vol: float = HEAVY_REFL_STEEL_VOL) -> openmc.Material:
    """Homogenised heavy reflector = SS304 + borated water by VOLUME.

    This is the physically-correct reflector for a LABGENE-class core and REPLACES
    the old plain-borated-water reflector. Its albedo (< 1) and finite thickness
    are what let the radial reflector actually return neutrons -- the mechanism the
    reflected-assembly model (make_assembly_model reflector=True) relies on."""
    steel = make_ss304(op.clad_T)
    water = make_water(op.boron_ppm, op.mod_T)

    # openmc.Material.mix_materials() refuses to mix any material that
    # already carries an S(a,b) thermal scattering table (see OpenMC's
    # material.py, mix_materials(), which raises NotImplementedError
    # unconditionally if any input material has one). make_water() attaches
    # the light-water S(a,b) table, so we save it, strip it before mixing,
    # then re-attach it to the homogenized mixture afterward.
    water_sab = list(water._sab)   # e.g. [('c_H_in_H2O', 1.0)]
    water._sab = []

    refl = openmc.Material.mix_materials(
        [steel, water], [steel_vol, 1.0 - steel_vol], "vo",
        name="heavy_reflector")

    # Re-attach the thermal scattering law to the homogenized reflector —
    # still correct physics since the hydrogen fraction is still bound water.
    for name, fraction in water_sab:
        refl.add_s_alpha_beta(name, fraction)
        
    refl.temperature = op.clad_T
    return refl


def make_uo2(enrichment_wt: float, T: float, density: float = 10.4) -> openmc.Material:
    """UO2 at a given U-235 enrichment [wt%]. Fully parametric for the optimizer."""
    f = openmc.Material(name=f"UO2_{enrichment_wt:.2f}")
    f.add_element("U", 1.0, "ao", enrichment=enrichment_wt)
    f.add_element("O", 2.0, "ao")
    f.set_density("g/cm3", density)
    f.temperature = T
    f.volume = None       # set later for depletion
    return f


def make_uo2_gd(enrichment_wt: float, gd2o3_wt: float, T: float,
                density: float = 10.2) -> openmc.Material:
    """(U,Gd)O2 burnable-poison fuel: UO2 with gd2o3_wt% Gd2O3 by weight."""
    base = make_uo2(enrichment_wt, T, density)
    gd = openmc.Material(name="Gd2O3")
    gd.add_element("Gd", 2.0, "ao")
    gd.add_element("O", 3.0, "ao")
    gd.set_density("g/cm3", 7.41)
    frac = gd2o3_wt / 100.0
    mix = openmc.Material.mix_materials([base, gd], [1 - frac, frac], "wo",
                                        name=f"UGd_{enrichment_wt:.1f}_{gd2o3_wt:.1f}")
    mix.temperature = T
    return mix


def build_materials(design: dict, op: Operating):
    """Return a dict of all materials for a given design."""
    e_in = design["enrich_inner"]
    e_out = design["enrich_outer"]
    gd = design.get("gd_wt", 0.0)
    mats = {
        "water":   make_water(op.boron_ppm, op.mod_T),
        "clad":    make_zircaloy(op.clad_T),
        "he":      make_helium(op.clad_T),
        "fuel_in":  make_uo2(e_in, op.fuel_T),
        "fuel_out": make_uo2(e_out, op.fuel_T),
    }
    if gd > 0:
        # Gd pins use the inner enrichment by convention (edit if you prefer)
        mats["fuel_gd"] = make_uo2_gd(e_in, gd, op.fuel_T)
    return mats


# =============================================================================
# Pin and assembly universes (2D: infinite cylinders, no z bounds)
# =============================================================================
def _fuel_pin_universe(fuel_mat, mats, geo: Geometry17x17):
    r_f = openmc.ZCylinder(r=geo.fuel_or)
    r_ci = openmc.ZCylinder(r=geo.clad_ir)
    r_co = openmc.ZCylinder(r=geo.clad_or)
    fuel = openmc.Cell(fill=fuel_mat, region=-r_f)
    gap = openmc.Cell(fill=mats["he"], region=+r_f & -r_ci)
    clad = openmc.Cell(fill=mats["clad"], region=+r_ci & -r_co)
    water = openmc.Cell(fill=mats["water"], region=+r_co)
    u = openmc.Universe(cells=[fuel, gap, clad, water])
    u._fuel_cell = fuel        # stash for tallies/depletion
    return u


def _guide_tube_universe(mats, geo: Geometry17x17):
    r_i = openmc.ZCylinder(r=geo.gt_ir)
    r_o = openmc.ZCylinder(r=geo.gt_or)
    inner = openmc.Cell(fill=mats["water"], region=-r_i)
    tube = openmc.Cell(fill=mats["clad"], region=+r_i & -r_o)
    outer = openmc.Cell(fill=mats["water"], region=+r_o)
    return openmc.Universe(cells=[inner, tube, outer])


def build_assembly_universe(design, mats, geo: Geometry17x17, pitch: float):
    """A 17x17 lattice. Inner ring of fuel uses 'fuel_in', outer uses
    'fuel_out'; Gd pins (if any) replace selected positions. Returns the
    lattice-filled universe and the list of distinct fuel cells (for tallies)."""
    N = geo.lattice
    gt = _guide_tube_universe(mats, geo)

    # which lattice positions count as "inner" (a centered block) vs "outer"
    inner_lo, inner_hi = N // 2 - 4, N // 2 + 4    # central 9x9 block = inner
    fuel_cells = []

    universes = np.empty((N, N), dtype=openmc.Universe)
    for i in range(N):
        for j in range(N):
            if (i, j) in GUIDE_TUBE_POSITIONS:
                universes[i, j] = gt
                continue
            if "fuel_gd" in mats and (i, j) in GD_PIN_POSITIONS:
                u = _fuel_pin_universe(mats["fuel_gd"], mats, geo)
            elif inner_lo <= i <= inner_hi and inner_lo <= j <= inner_hi:
                u = _fuel_pin_universe(mats["fuel_in"], mats, geo)
            else:
                u = _fuel_pin_universe(mats["fuel_out"], mats, geo)
            universes[i, j] = u
            fuel_cells.append(u._fuel_cell)

    lat = openmc.RectLattice(name="assembly")
    lat.lower_left = (-N * pitch / 2.0, -N * pitch / 2.0)
    lat.pitch = (pitch, pitch)
    lat.universes = universes
    outer = openmc.Cell(fill=mats["water"])
    lat.outer = openmc.Universe(cells=[outer])

    box_cell = openmc.Cell(fill=lat)
    return openmc.Universe(cells=[box_cell]), fuel_cells, lat


# =============================================================================
# Model factories
# =============================================================================
def _settings(particles=20000, batches=120, inactive=30, bb=None, seed=1):
    s = openmc.Settings()
    s.particles = particles
    s.batches = batches
    s.inactive = inactive
    s.run_mode = "eigenvalue"
    s.temperature = {"method": "interpolation",
                     "range": (294.0, 1500.0), "default": 900.0}
    if bb is not None:
        s.source = openmc.IndependentSource(
            space=openmc.stats.Box(bb[0], bb[1], only_fissionable=True))
    s.seed = seed
    return s


def make_pincell_model(design: dict, op: Operating = Operating(),
                       geo: Geometry17x17 = Geometry17x17(),
                       particles=10000, batches=100, inactive=20) -> openmc.Model:
    """Cheapest fidelity: a single (inner-enrichment) pin in an infinite
    lattice -> k_inf. Good for quick checks and moderation studies."""
    pitch = design.get("pitch", 1.26)
    mats = build_materials(design, op)
    u = _fuel_pin_universe(mats["fuel_in"], mats, geo)
    # reflective square box around the pin -> infinite lattice
    box = openmc.model.RectangularPrism(pitch, pitch, boundary_type="reflective")
    root = openmc.Cell(fill=u, region=-box)
    geom = openmc.Geometry([root])
    materials = openmc.Materials([m for m in mats.values()])
    bb = ((-pitch/2, -pitch/2, -1e9), (pitch/2, pitch/2, 1e9))
    model = openmc.Model(geometry=geom, materials=materials,
                         settings=_settings(particles, batches, inactive, bb))
    return model


def make_assembly_model(design: dict, op: Operating = Operating(),
                        geo: Geometry17x17 = Geometry17x17(),
                        bc: str = "reflective", reflector: bool = False,
                        particles=20000, batches=150, inactive=40):
    """One 17x17 assembly, infinite in z (2D). Returns (model, fuel_cells, lattice).

    reflector=False  (default, UNCHANGED behaviour)
        The assembly is wrapped in a single box with boundary_type=bc. With
        bc='reflective' this is the classic INFINITE-MEDIUM k_inf: albedo = 1 on
        every face, ZERO leakage, and the radial reflector is invisible. This is
        why `refl_thick` had no effect in the old evaluator.

    reflector=True   (NEW: the reflector actually influences the 2D result)
        The assembly is surrounded by a HEAVY-REFLECTOR frame of thickness
        design['refl_thick'] (cm) and the OUTER boundary is VACUUM. Physically the
        pure reflective box (albedo 1) is replaced by a finite reflector whose
        albedo is < 1 and RISES WITH THICKNESS: more steel returns more neutrons
        before they leak out the vacuum edge. So k now (a) drops below k_inf
        because leakage is real, and (b) responds directly to refl_thick -- this
        is how the design variable is allowed to change the physics.

        CAVEAT to state in the thesis: a single assembly wrapped on ALL four sides
        sees more reflector per unit fuel than a core-average assembly, so this is
        a reflector-SAVINGS SCREENING model -- it captures the direction and the
        saturation of the reflector benefit (what the MOO trades off), not the
        absolute core leakage. Pin the absolute cycle length against make_core_model
        (or the core-based K_TARGET(refl_thick) route) before quoting final EFPD."""
    pitch = design.get("pitch", 1.26)
    mats = build_materials(design, op)
    asm_u, fuel_cells, lat = build_assembly_universe(design, mats, geo, pitch)
    half = geo.lattice * pitch / 2.0
    materials = openmc.Materials([m for m in mats.values()])

    if not reflector:
        box = openmc.model.RectangularPrism(2 * half, 2 * half, boundary_type=bc)
        geom = openmc.Geometry([openmc.Cell(fill=asm_u, region=-box)])
    else:
        t = float(design.get("refl_thick", 15.0))          # cm  (DESIGN VARIABLE)
        refl_mat = make_heavy_reflector(op)
        materials.append(refl_mat)
        inner = openmc.model.RectangularPrism(2 * half, 2 * half)  # transmission
        outer = openmc.model.RectangularPrism(
            2 * (half + t), 2 * (half + t), boundary_type="vacuum")
        asm_cell  = openmc.Cell(fill=asm_u,   region=-inner)
        refl_cell = openmc.Cell(fill=refl_mat, region=+inner & -outer)
        geom = openmc.Geometry([asm_cell, refl_cell])

    bb = ((-half, -half, -1e9), (half, half, 1e9))         # source: fuel region
    model = openmc.Model(geometry=geom, materials=materials,
                         settings=_settings(particles, batches, inactive, bb))
    return model, fuel_cells, lat


def make_core_model(design: dict, op: Operating = Operating(),
                    geo: Geometry17x17 = Geometry17x17(),
                    core_map=None, refl_thick=None, r_fuel=None,
                    particles=40000, batches=200, inactive=50):
    """A small 2D multi-assembly core with a HEAVY (steel) reflector and vacuum BC.

    `core_map` is a 2D array of 1 (assembly) / 0 (reflector). Defaults to a
    ~32-assembly compact layout consistent with the image-estimated core size.

    refl_thick : float | None
        Radial thickness [cm] of the heavy-reflector annulus around the fuel.
        This is now a REAL geometric dimension: the fuel is bounded by a cylinder
        of radius r_fuel and the reflector is an annulus of this thickness with a
        vacuum edge at r_fuel + refl_thick. Sweeping it moves the measured core
        leakage, so this is the model to build K_TARGET(refl_thick) from. Falls
        back to design['refl_thick'], then to the drawing nominal (~11.5 cm).
    r_fuel : float | None
        Radius [cm] of the cylinder bounding the fuel. Defaults to the
        equivalent-area radius sqrt(N_assemblies/pi) * assembly_pitch, which for
        the 32-assembly layout is ~0.68 m and tracks the CTMSP drawing's R_fuel
        (~0.675 m); it scales automatically with pitch and assembly count.
        The outer corners of the edge assemblies are clipped by this cylinder,
        which makes the square lattice rounder and closer to the real core.
    """
    pitch = design.get("pitch", 1.26)
    assembly_pitch = geo.lattice * pitch
    mats = build_materials(design, op)
    asm_u, fuel_cells, _ = build_assembly_universe(design, mats, geo, pitch)
    # Reflector is now the homogenised steel reflector, not borated water -- this
    # is what LABGENE actually has, and it changes the measured assembly->core
    # leakage (so re-run measure_leakage_target.py after this edit).
    refl_mat = make_heavy_reflector(op)
    refl_u = openmc.Universe(cells=[openmc.Cell(fill=refl_mat)])

    if core_map is None:
        # (EST. FROM IMAGE) 6x6 grid with corners removed -> 32 assemblies,
        # replacing the old 21-assembly 5x5 layout tied to the 1.8 m height.
        core_map = np.array([
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ])
    ny, nx = core_map.shape
    # build the object array explicitly (np.where on object dtype is finicky)
    universes = np.empty(core_map.shape, dtype=openmc.Universe)
    for i in range(ny):
        for j in range(nx):
            universes[i, j] = asm_u if core_map[i, j] == 1 else refl_u

    lat = openmc.RectLattice(name="core")
    lat.lower_left = (-nx * assembly_pitch / 2.0, -ny * assembly_pitch / 2.0)
    lat.pitch = (assembly_pitch, assembly_pitch)
    lat.universes = universes
    lat.outer = refl_u

    # ---- explicit reflector thickness ------------------------------------ #
    # refl_thick is a REAL radial dimension here (unlike the old fixed 90 cm
    # vacuum). Fuel is bounded by r_fuel; the heavy reflector is an annulus of
    # thickness refl_thick; the vacuum edge floats at r_fuel + refl_thick. So a
    # thicker reflector returns more neutrons and the measured k_eff rises and
    # saturates -> sweep refl_thick to tabulate K_TARGET(refl_thick).
    if refl_thick is None:
        refl_thick = design.get("refl_thick", 11.5)   # cm, drawing nominal
    n_fuel_asm = int(np.count_nonzero(core_map == 1))
    if r_fuel is None:
        # equivalent-area fuel radius ~ drawing R_fuel (~0.675 m for 32 asm);
        # scales with pitch and assembly count automatically.
        r_fuel = math.sqrt(n_fuel_asm / math.pi) * assembly_pitch

    r_fuel_cyl = openmc.ZCylinder(r=r_fuel)
    r_refl_cyl = openmc.ZCylinder(r=r_fuel + refl_thick, boundary_type="vacuum")
    fuel_cell = openmc.Cell(fill=lat, region=-r_fuel_cyl)             # fuel + gaps
    refl_cell = openmc.Cell(fill=refl_mat, region=+r_fuel_cyl & -r_refl_cyl)
    geom = openmc.Geometry([fuel_cell, refl_cell])
    materials = openmc.Materials([m for m in mats.values()] + [refl_mat])

    # seed the initial fission source inside the fuel cylinder
    bb = ((-r_fuel, -r_fuel, -1e9), (r_fuel, r_fuel, 1e9))
    model = openmc.Model(geometry=geom, materials=materials,
                         settings=_settings(particles, batches, inactive, bb))
    return model, fuel_cells


# =============================================================================
# Helpers: heavy-metal mass and specific power
# =============================================================================
def pin_hm_mass_g(geo: Geometry17x17, density=10.4) -> float:
    """Heavy-metal (U) mass in ONE fuel pin over the active height [g].
    UO2 -> U mass fraction ~ 0.8815 (238/270)."""
    area = math.pi * geo.fuel_or ** 2          # cm^2
    vol = area * geo.active_height             # cm^3
    uo2_mass = vol * density                   # g
    return uo2_mass * 0.8815                    # g of U


def core_specific_power_w_per_g(op: Operating, geo: Geometry17x17,
                                n_fuel_pins_per_assembly: int = None) -> float:
    """Specific power [W/gHM] = total thermal power / total core HM mass.
    This is the number you feed to OpenMC depletion as `power_density`."""
    if n_fuel_pins_per_assembly is None:
        n_fuel_pins_per_assembly = geo.lattice**2 - len(GUIDE_TUBE_POSITIONS)
    pin_u = pin_hm_mass_g(geo)
    total_hm = pin_u * n_fuel_pins_per_assembly * op.n_assemblies
    return op.power_mwth * 1e6 / total_hm


def build_model(design: dict):
    """Default builder used by reactor_optimization.OpenMCEvaluator.
    Returns just the openmc.Model (assembly fidelity). For depletion you also
    need the fuel cells/volumes -- see the notebook."""
    model, _fuel_cells, _lat = make_assembly_model(design)
    return model
