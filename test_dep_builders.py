#!/usr/bin/env python3
"""
test_dep_builders.py -- the model builders of checks 2 and 3 on the REAL
reactor_model / zoning / hardware3d code, with openmc replaced by
fake_openmc (geometry only, no transport). Verifies, for C9-47:

  2D  twelve depletable ring materials, 8448 pin instances, each tagged to
      its ring, volumes adding up to the core heavy metal
  3D  --layers x 12 depletable materials, every layer holding 8448 pin
      instances over its own height, the plain and grid segments of a
      layer sharing one material set, the four tallies attached

    python test_dep_builders.py
"""
import json
import math
import sys

sys.path.insert(0, ".")
import fake_openmc                                   # noqa: E402
fake_openmc.install()
import reactor_model as rm                           # noqa: E402
import zoning as zn                                  # noqa: E402
import hardware3d as hw                              # noqa: E402
import c9_dep_core2d as C2                           # noqa: E402
import c9_dep_core3d as C3                           # noqa: E402

N_FUEL = 32 * (17 * 17 - 25)                         # 8448 pin instances in the core


def main():
    ck = json.load(open("out_c9/optimization_checkpoint.json"))
    design = C2.design_of(ck, 47)
    op, geo = rm.Operating(), rm.Geometry17x17()
    area = math.pi * geo.fuel_or ** 2

    model, rows, shape = C2.build(design, dict(particles=10, batches=5, inactive=2), 7, geo, op)
    assert len(rows) == 12 and sum(r["pins"] for r in rows) == N_FUEL
    assert sorted({r["zone"] for r in rows}) == ["C", "M", "P"] and "?" not in {r["zone"] for r in rows}
    per_ring = {z: sum(r["pins"] for r in rows if r["zone"] == z) for z in "CMP"}
    assert per_ring == {"C": 4 * 264, "M": 12 * 264, "P": 16 * 264}, per_ring
    assert abs(sum(r["volume_cm3"] for r in rows) - N_FUEL * area * geo.active_height) < 1e-6
    assert sum(getattr(m, "depletable", False) for m in model.materials) == 12
    assert [t.name for t in model.tallies] == ["core_pin_fission"]
    print(f"2D: {len(rows)} depletable materials, rings {per_ring}, volume OK")

    for n_layers in (4, 8):
        spec = hw.HardwareSpec()
        model, info, rows, edges, fine, shp = C3.build_layered(
            design, op, geo, spec, n_layers, dict(particles=10, batches=5, inactive=2), 7,
            zn.evaluator_design_map(design))
        assert len(rows) == 12 * n_layers, len(rows)
        h = spec.h_active / n_layers
        for L in range(n_layers):
            rs = [r for r in rows if r["layer"] == L]
            assert len(rs) == 12 and sum(r["pins"] for r in rs) == N_FUEL
            assert abs(sum(r["volume_cm3"] for r in rs) - N_FUEL * area * h) < 1e-6
        assert sum(getattr(m, "depletable", False) for m in model.materials) == 12 * n_layers
        assert [t.name for t in model.tallies] == ["asm_fission", "pin_fission", "axial_fission", "layer_fission"]
        assert len(edges) == n_layers + 1 and info["n_fuel_segments"] >= n_layers
        print(f"3D: {n_layers} layers, {info['n_fuel_segments']} fuel segments, {len(rows)} depletable materials, "
              f"{len(model.materials)} materials in the model, volumes OK")
    print("test_dep_builders OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
