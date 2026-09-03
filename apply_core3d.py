#!/usr/bin/env python3
"""apply_core3d.py -- give the core model a FINITE ACTIVE HEIGHT, so axial
leakage can be measured on the same cores the 2D campaigns evaluated.

WHAT THE 2D MODEL OMITS
-----------------------
make_core_model builds the 32-assembly lattice inside a Z cylinder with no
axial bounds: an infinitely tall core. Every eigenvalue the campaigns have
used, and every rod worth from rod_bank_worth.py, therefore excludes axial
leakage. That affects two places at once:

  1. the Route B end-of-cycle target k_target = k_inf / k_eff(core) was
     calibrated on the 2D core, so it carries radial leakage only, and the
     reported cycle lengths are optimistic by the axial part;
  2. the controllability constraint g_ctrl compares a 2D k_ALLRE with the
     margin, so the derived k_max_ctrl is also a 2D quantity.

WHAT THIS ADDS (off by default, 2D behaviour bit for bit unchanged)
--------------------------------------------------------------------
make_core_model(..., h_active=None, axial_refl_cm=0.0)
    h_active      active fuel height in cm; when given, the fuel lattice and
                  the radial steel reflector are bounded by z = +/- h/2
    axial_refl_cm thickness of a borated-water axial reflector above and
                  below the active zone (same water as the moderator), with
                  a vacuum boundary beyond it; 0 puts the vacuum boundary
                  directly on the fuel ends

zoning.core_bol_solve(..., h_active=None, axial_refl_cm=0.0)
    passthrough, so the same solve path, seeds, case tree and peaking tally
    serve the 3D core. The F_dH tally is a 2D mesh with no z bounds, which
    OpenMC integrates over z, so in 3D it is the axially integrated
    enthalpy-rise factor by construction: the correct definition of F_dH,
    which the 2D model could only approximate.

The initial fission source and the Shannon-entropy mesh are confined to
the active height when it is finite.

Anchors verified on campaign7 head b8ffad4. Backups <file>.bak.core3d.
Refuses to run twice (CORE3D marker).
"""
from __future__ import annotations
import py_compile
import shutil
import sys
from pathlib import Path

MARKER = "CORE3D"

FILES = {
    "reactor_model.py": [
(
'''                    design_map=None, rodded_map=None,
                    enforce_vessel=True,
                    particles=40000, batches=200, inactive=50):
''',
'''                    design_map=None, rodded_map=None,
                    enforce_vessel=True,
                    particles=40000, batches=200, inactive=50,
                    h_active=None, axial_refl_cm=0.0):
''',
),
(
'''    r_fuel_cyl = openmc.ZCylinder(r=r_fuel)
    r_refl_cyl = openmc.ZCylinder(r=r_outer, boundary_type="vacuum")
    fuel_cell = openmc.Cell(fill=lat, region=-r_fuel_cyl)             # fuel + gaps
    refl_cell = openmc.Cell(fill=refl_mat, region=+r_fuel_cyl & -r_refl_cyl)
    geom = openmc.Geometry([fuel_cell, refl_cell])
''',
'''    r_fuel_cyl = openmc.ZCylinder(r=r_fuel)
    r_refl_cyl = openmc.ZCylinder(r=r_outer, boundary_type="vacuum")
    if h_active is None:
        # 2D: infinitely tall core, exactly as every campaign evaluated it
        fuel_cell = openmc.Cell(fill=lat, region=-r_fuel_cyl)         # fuel + gaps
        refl_cell = openmc.Cell(fill=refl_mat, region=+r_fuel_cyl & -r_refl_cyl)
        geom = openmc.Geometry([fuel_cell, refl_cell])
    else:
        # CORE3D: finite active height, optional borated-water axial
        # reflectors, vacuum beyond them. The radial steel reflector spans
        # the full height, including the axial-reflector zones.
        hz = 0.5 * float(h_active)
        ax = max(0.0, float(axial_refl_cm))
        z_lo = openmc.ZPlane(z0=-hz)
        z_hi = openmc.ZPlane(z0=+hz)
        if ax > 0.0:
            z_bot = openmc.ZPlane(z0=-hz - ax, boundary_type="vacuum")
            z_top = openmc.ZPlane(z0=+hz + ax, boundary_type="vacuum")
        else:
            z_lo.boundary_type = "vacuum"
            z_hi.boundary_type = "vacuum"
            z_bot, z_top = z_lo, z_hi
        fuel_cell = openmc.Cell(fill=lat,
                                region=-r_fuel_cyl & +z_lo & -z_hi)
        refl_cell = openmc.Cell(fill=refl_mat,
                                region=+r_fuel_cyl & -r_refl_cyl
                                & +z_bot & -z_top)
        cells = [fuel_cell, refl_cell]
        if ax > 0.0:
            cells += [openmc.Cell(fill=mats["water"],
                                  region=-r_fuel_cyl & +z_bot & -z_lo),
                      openmc.Cell(fill=mats["water"],
                                  region=-r_fuel_cyl & +z_hi & -z_top)]
        geom = openmc.Geometry(cells)
''',
),
(
'''    # seed the initial fission source inside the fuel cylinder
    bb = ((-r_fuel, -r_fuel, -1e9), (r_fuel, r_fuel, 1e9))
''',
'''    # seed the initial fission source inside the fuel cylinder, and within
    # the active height when the core is finite (CORE3D)
    zz = 1e9 if h_active is None else 0.5 * float(h_active)
    bb = ((-r_fuel, -r_fuel, -zz), (r_fuel, r_fuel, zz))
''',
),
    ],

    "zoning.py": [
(
'''                   seed: int, case: Path, rodded_map=None) -> dict:
''',
'''                   seed: int, case: Path, rodded_map=None,
                   h_active=None, axial_refl_cm=0.0) -> dict:
''',
),
(
'''    m = rm.make_core_model(base_design, op, geo, design_map=design_map, rodded_map=rodded_map,
                           particles=particles, batches=batches,
                           inactive=inactive)
''',
'''    m = rm.make_core_model(base_design, op, geo, design_map=design_map, rodded_map=rodded_map,
                           particles=particles, batches=batches,
                           inactive=inactive,
                           h_active=h_active,            # CORE3D passthrough
                           axial_refl_cm=axial_refl_cm)
''',
),
    ],
}


def main() -> None:
    root = Path(".")
    for fname, edits in FILES.items():
        p = root / fname
        if not p.is_file():
            sys.exit(f"ABORT: {p} not found. Run from the repository root.")
        text = p.read_text()
        if MARKER in text:
            sys.exit(f"REFUSED: {fname} already carries the {MARKER} marker.")
        for i, (anchor, _) in enumerate(edits, 1):
            n = text.count(anchor)
            if n != 1:
                sys.exit(f"ABORT: anchor {i} for {fname} found {n} times "
                         f"(need exactly 1). Nothing modified.\\n"
                         f"Anchor begins: {anchor.splitlines()[0]!r}")
    for fname, edits in FILES.items():
        p = root / fname
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak.core3d"))
        text = p.read_text()
        for anchor, repl in edits:
            text = text.replace(anchor, repl)
        p.write_text(text)
        py_compile.compile(str(p), doraise=True)
        print(f"[ok] {fname}: {len(edits)} edit(s), backup "
              f"{p.name}.bak.core3d, py_compile passed")
    print("[done] make_core_model and core_bol_solve accept h_active and "
          "axial_refl_cm. With both absent, the 2D model is unchanged.")


if __name__ == "__main__":
    main()
