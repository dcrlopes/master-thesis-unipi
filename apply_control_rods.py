#!/usr/bin/env python3
"""
apply_control_rods.py -- add fully inserted control rods to the model.

Adds to reactor_model.py:
  * AIC (80-15-5), B4C, SS 304L and He materials, nuclide densities copied
    verbatim from the validated NuScale-like notebook (Zenodo 15231335),
  * _cr_gt_universe(), a guide tube whose water column is replaced by an
    absorber pin: absorber, helium gap, 304L clad, water, zircaloy tube,
  * a `rodded` argument on build_assembly_universe / make_assembly_model
    (None, "B4C" or "AIC") filling all 24 guide tube positions,
  * a `rodded_map` argument on make_core_model: a set of (row, col) core
    positions whose assembly is built rodded.
Adds to zoning.py:
  * `rodded_map` passthrough on core_bol_solve.

Rod pin radii are the Westinghouse 17x17 RCCA values used by the benchmark
(VERIFY against notebook cell 1): absorber 0.4331, gap 0.43688, clad
0.48387 cm. The clad must fit inside gt_ir = 0.5715, which is asserted.
2D is valid for FULLY inserted rods only, since those are axially uniform.
"""
import py_compile
import shutil
from pathlib import Path

MATS = '''
CR_R_ABS, CR_R_GAP, CR_R_CLAD = 0.4331, 0.43688, 0.48387   # cm, VERIFY nb cell 1


def make_cr_materials(T: float = 600.0):
    """Absorber-pin materials, atom densities verbatim from the validated
    NuScale-like notebook (Zenodo 15231335). set_density('sum')."""
    aic = openmc.Material(name="AIC (80-15-5)")
    for n, d in (("Ag107", 2.35230e-02), ("Ag109", 2.18540e-02),
                 ("Cd106", 3.38820e-05), ("Cd108", 2.41660e-05),
                 ("Cd110", 3.39360e-04), ("Cd111", 3.48210e-04),
                 ("Cd112", 6.56110e-04), ("Cd113", 3.32750e-04),
                 ("Cd114", 7.82520e-04), ("Cd116", 2.04430e-04),
                 ("In113", 3.42190e-04), ("In115", 7.65110e-03)):
        aic.add_nuclide(n, d)
    b4c = openmc.Material(name="B4C")
    for n, d in (("B10", 1.52060e-02), ("B11", 6.15140e-02),
                 ("C0", 1.89720e-02 + 2.12520e-04)):
        b4c.add_nuclide(n, d)
    ss = openmc.Material(name="SS 304L")
    for n, d in (("Cr50", 7.67780e-04), ("Cr52", 1.48060e-02),
                 ("Cr53", 1.67890e-03), ("Cr54", 4.17910e-04),
                 ("Fe54", 3.46200e-03), ("Fe56", 5.43450e-02),
                 ("Fe57", 1.25510e-03), ("Fe58", 1.67030e-04),
                 ("Mn55", 1.76040e-03), ("Ni58", 5.60890e-03),
                 ("Ni60", 2.16050e-03), ("Ni61", 9.39170e-05),
                 ("Ni62", 2.99450e-04), ("Ni64", 7.62610e-05),
                 ("Si28", 9.52810e-04), ("Si29", 4.83810e-05),
                 ("Si30", 3.18930e-05)):
        ss.add_nuclide(n, d)
    he = openmc.Material(name="He")
    he.add_nuclide("He3", 4.80890e-10)
    he.add_nuclide("He4", 2.40440e-04)
    for m in (aic, b4c, ss, he):
        m.set_density("sum")
        m.temperature = T
    return {"AIC": aic, "B4C": b4c, "cr_ss": ss, "cr_he": he}


def _cr_gt_universe(mats, geo: "Geometry17x17", absorber: str):
    """Guide tube with a fully inserted rod: absorber, He gap, 304L clad,
    water annulus, zircaloy tube, water. 2D-valid for FULL insertion only."""
    assert CR_R_CLAD < geo.gt_ir, "rod clad does not fit this guide tube"
    crm = make_cr_materials(600.0)
    r1 = openmc.ZCylinder(r=CR_R_ABS)
    r2 = openmc.ZCylinder(r=CR_R_GAP)
    r3 = openmc.ZCylinder(r=CR_R_CLAD)
    r4 = openmc.ZCylinder(r=geo.gt_ir)
    r5 = openmc.ZCylinder(r=geo.gt_or)
    cs = [openmc.Cell(fill=crm[absorber], region=-r1),
          openmc.Cell(fill=crm["cr_he"], region=+r1 & -r2),
          openmc.Cell(fill=crm["cr_ss"], region=+r2 & -r3),
          openmc.Cell(fill=mats["water"], region=+r3 & -r4),
          openmc.Cell(fill=mats["zirc"], region=+r4 & -r5),
          openmc.Cell(fill=mats["water"], region=+r5)]
    return openmc.Universe(name=f"gt_cr_{absorber}", cells=cs)

'''

P = Path("reactor_model.py")
s = P.read_text()
if "make_cr_materials" in s:
    raise SystemExit("reactor_model.py already patched.")

a1 = 'def build_assembly_universe(design, mats, geo: Geometry17x17, pitch: float):'
assert s.count(a1) == 1
s = s.replace(a1, MATS + '\n'
              'def build_assembly_universe(design, mats, geo: Geometry17x17,'
              ' pitch: float,\n                            rodded=None):')

a2 = '''    N = geo.lattice
    gt = _guide_tube_universe(mats, geo)'''
assert s.count(a2) == 1
s = s.replace(a2, '''    N = geo.lattice
    gt = (_cr_gt_universe(mats, geo, rodded) if rodded
          else _guide_tube_universe(mats, geo))''')

# thread `rodded` through make_assembly_model -> build_assembly_universe
a3 = 'def make_assembly_model(design: dict, op: Operating = Operating(),'
assert s.count(a3) == 1
i = s.index(a3)
j = s.index('):', i)
sig = s[i:j]
assert 'rodded' not in sig
s = s[:j] + ',\n                        rodded=None' + s[j:]
a4 = 'build_assembly_universe(design, mats, geo,'
first = s.index(a4, i)
k = s.index(')', first)
s = s[:k] + ', rodded=rodded' + s[k:]

# make_core_model: accept rodded_map and build rodded variants per position
a5 = '''                    design_map=None,
                    enforce_vessel=True,'''
assert s.count(a5) == 1
s = s.replace(a5, '''                    design_map=None, rodded_map=None,
                    enforce_vessel=True,''')

a6 = '''    def _variant_u(d):
        key = tuple(sorted((k, round(float(v), 9)) for k, v in d.items()
                           if isinstance(v, (int, float))))'''
assert s.count(a6) == 1
s = s.replace(a6, '''    def _variant_u(d, rodded=None):
        key = (rodded,) + tuple(sorted((k, round(float(v), 9))
                                       for k, v in d.items()
                                       if isinstance(v, (int, float))))''')
a7 = 'uv, _fc, _ = build_assembly_universe(d, mv, geo, pitch)'
assert s.count(a7) == 1
s = s.replace(a7, 'uv, _fc, _ = build_assembly_universe(d, mv, geo, pitch,\n'
                  '                                     rodded=rodded)')
# route the per-position lookup at the exact core-lattice call
oc = "_variant_u(design_map[(i, j)])"
nc = ("_variant_u(design_map[(i, j)], rodded=(_rodded_kind if "
      "(rodded_map is not None and (i, j) in rodded_map) else None))")
assert s.count(oc) == 1, "core lattice call not found verbatim"
s = s.replace(oc, nc)
# _rodded_kind: rodded_map may be a dict position->absorber or a set + kind
a8 = '''    variant_mats = []'''
assert s.count(a8) == 1
s = s.replace(a8, '''    _rodded_kind = "B4C"
    if isinstance(rodded_map, tuple):          # (positions, "AIC"/"B4C")
        rodded_map, _rodded_kind = rodded_map
    variant_mats = []''')

shutil.copy(P, "reactor_model.py.cr.bak")
P.write_text(s)
py_compile.compile(str(P), doraise=True)
print("patched reactor_model.py (backup .cr.bak)")

Z = Path("zoning.py")
z = Z.read_text()
if "rodded_map" not in z:
    b1 = '''def core_bol_solve(base_design: dict, design_map, op, geo, *,
                   particles: int, batches: int, inactive: int,
                   seed: int, case: Path) -> dict:'''
    assert z.count(b1) == 1
    z = z.replace(b1, '''def core_bol_solve(base_design: dict, design_map, op, geo, *,
                   particles: int, batches: int, inactive: int,
                   seed: int, case: Path, rodded_map=None) -> dict:''')
    c1 = 'design_map=design_map,'
    assert z.count(c1) >= 1
    z = z.replace(c1, 'design_map=design_map, rodded_map=rodded_map,', 1)
    shutil.copy(Z, "zoning.py.cr.bak")
    Z.write_text(z)
    py_compile.compile(str(Z), doraise=True)
    print("patched zoning.py (backup .cr.bak)")
else:
    print("zoning.py already patched.")
