#!/usr/bin/env python3
"""
apply_campaign5.py -- the Campaign-5 change set, applied with verified anchors.

THE ONE STRUCTURAL CHANGE
  gd_pins: 6th design variable, gadolinia-bearing ROD COUNT, range 12-40,
  DISCRETE on the Strategy-A nested ladder (interior-first, guide-tube-
  ringed, adopted from the placement-practice literature review):

      12 -> 16 -> 20 -> 24 -> 32 -> 40

  Rules at every level: octant symmetry; outer two rod rows gadolinia-free;
  no face-adjacent Gd-Gd pairs; every rod face- or diagonally-adjacent to a
  guide tube; nested (monotone authority). The optimizer sees a continuous
  variable; the model SNAPS to the nearest ladder count (rm.snap_gd_pins;
  exact midpoints snap down), and the snapped value is recorded per
  evaluation as gd_pins_used.

THE GD-ROD MATERIAL MODEL (bundled, inseparable from the pin variable)
  Zone-matched enrichment: a gadolinia rod takes the enrichment of the zone
  it sits in (inner 9x9 -> e_in, otherwise e_out), replacing the previous
  single-material e_in convention.
  Industrial reduction rule: that enrichment is reduced by 5% RELATIVE per
  wt% Gd2O3 (e_gd = e_zone * (1 - 0.05*gd), floored at natural 0.2 wt%) --
  the documented Westinghouse/ENUSA low-concentration-gadolinia practice
  (INIS FR0200561; same rule in INL LWRS uprate assessment), motivated by
  the degraded thermal conductivity of the urania-gadolinia mixture.

INSTRUMENTATION (no physics change)
  A pin-resolved fission tally is attached to the DEPLETION assembly model,
  so every depletion statepoint records the pin power map: assembly
  F_dH(t) through burnup and at EOL becomes extractable for every future
  evaluation.

PLUMBING
  --workdir flag on run_optimization.py (Campaign 4 overwrote Campaign 3's
  scratch because openmc_runs was hardcoded); checkpoint meta records the
  gd model version. Campaign-5 checkpoints are incompatible with Campaign-4
  ones BY CONSTRUCTION (the design_variables list differs and
  load_checkpoint refuses the mismatch) -- a fresh out_c5 is mandatory.

USAGE
    cd ~/master-thesis-unipi        # on branch campaign5
    python3 apply_campaign5.py --check
    python3 apply_campaign5.py
"""
import argparse
import shutil
import sys
from pathlib import Path

MARKER = "GD_PATTERNS"

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--model", default="reactor_model.py")
ap.add_argument("--evaluator", default="openmc_evaluator.py")
ap.add_argument("--optimization", default="reactor_optimization.py")
ap.add_argument("--runner", default="run_optimization.py")
args = ap.parse_args()

rm_p = Path(args.model); ev_p = Path(args.evaluator)
op_p = Path(args.optimization); ru_p = Path(args.runner)
for p in (rm_p, ev_p, op_p, ru_p):
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Run from the repository root.")
rm, ev, op, ru = (p.read_text() for p in (rm_p, ev_p, op_p, ru_p))

if MARKER in rm:
    sys.exit("Already applied (GD_PATTERNS present). Nothing to do.")

# =========================================================================== #
# anchors                                                                     #
# =========================================================================== #
A_GDPOS = '''# Where Gd burnable-poison pins go (a typical symmetric pattern). Used only
# when design['gd_wt'] > 0. Keep modest; too many Gd pins over-flatten and
# waste neutrons (you will study this trade-off).
GD_PIN_POSITIONS = [
    (2, 2), (2, 14), (14, 2), (14, 14),
    (6, 6), (6, 10), (10, 6), (10, 10),
    (3, 8), (8, 3), (8, 13), (13, 8),
]'''
A_MATGD = '''    if gd > 0:
        # Gd pins use the inner enrichment by convention (edit if you prefer)
        mats["fuel_gd"] = make_uo2_gd(e_in, gd, op.fuel_T)
    return mats'''
A_LATGD = '''            if "fuel_gd" in mats and (i, j) in GD_PIN_POSITIONS:
                u = _fuel_pin_universe(mats["fuel_gd"], mats, geo)
            elif inner_lo <= i <= inner_hi and inner_lo <= j <= inner_hi:'''
A_ASMSIG = '''def make_assembly_model(design: dict, op: Operating = Operating(),
                        geo: Geometry17x17 = Geometry17x17(),
                        bc: str = "reflective", reflector: bool = False,
                        particles=20000, batches=150, inactive=40):'''
A_ASMRET = '''    bb = ((-half, -half, -1e9), (half, half, 1e9))         # source: fuel region
    model = openmc.Model(geometry=geom, materials=materials,
                         settings=_settings(particles, batches, inactive, bb))
    return model, fuel_cells, lat'''
A_EVCALL = '''        model, fuel_cells, _lat = rm.make_assembly_model(
            design, self.op, self.geo, bc="reflective", **self.transport)'''
A_EVRES = '''            "peaking_asm": peaking,'''
A_DS = '''    ds = DesignSpace([
        DesignVariable("enrich_inner", 2.0, 19.75, "%"),
        DesignVariable("enrich_outer", 2.0, 19.75, "%"),
        DesignVariable("gd_wt",        0.0,  8.0,  "wt% Gd2O3"),
        DesignVariable("pitch",        1.15, 1.43, "cm"),
        DesignVariable("refl_thick",   2.0,  19.5, "cm"),
    ])'''
A_CTOR = '''                         workdir="openmc_runs", **schedule)'''
A_OUT = '''ap.add_argument("--out"'''
A_META = '''                           "objective_def":
                               "peaking = core BOL F_dh (Campaign 4)",'''
A_FINALMETA = '''    ckpt = opt.save_checkpoint(ckpt_out,
                               meta={"k_target": k_target_arg,
                                     "smoke": bool(args.smoke),
                                     "transport": dict(transport),
                                     "schedule": dict(schedule),
                                     "geometry": "v2-envelope",
                                     "omp_threads": n_threads})'''

checks = [("GD_PIN_POSITIONS block", A_GDPOS, rm, "reactor_model"),
          ("materials gd block", A_MATGD, rm, "reactor_model"),
          ("lattice gd branch", A_LATGD, rm, "reactor_model"),
          ("make_assembly_model signature", A_ASMSIG, rm, "reactor_model"),
          ("make_assembly_model return", A_ASMRET, rm, "reactor_model"),
          ("_cycle_length model call", A_EVCALL, ev, "openmc_evaluator"),
          ("result peaking_asm", A_EVRES, ev, "openmc_evaluator"),
          ("DesignSpace block", A_DS, op, "reactor_optimization"),
          ("evaluator ctor workdir", A_CTOR, ru, "run_optimization"),
          ("--out argument", A_OUT, ru, "run_optimization"),
          ("meta objective_def", A_META, ru, "run_optimization"),
          ("final save_checkpoint meta", A_FINALMETA, ru, "run_optimization")]
bad = False
for name, anchor, text, fn in checks:
    n = text.count(anchor)
    print(f"  {'OK ' if n == 1 else 'BAD'} [{fn}] {name}: found {n}x")
    bad |= (n != 1)
if bad:
    sys.exit("\nERROR: anchors do not match this branch's files. Stop.")
print("all anchors matched.")
if args.check:
    sys.exit(0)

# =========================================================================== #
# 1. reactor_model.py -- pattern ladder                                       #
# =========================================================================== #
rm = rm.replace(A_GDPOS, '''# ---------------------------------------------------------------------------
# CAMPAIGN 5: gadolinia-bearing ROD COUNT is a design variable ("gd_pins").
#
# STRATEGY A (interior-first, guide-tube-ringed), adopted after a literature
# review of vendor and licensing practice (ORNL/TM-2023/3098 optimized
# 16/20/24-pin maps; U.S. EPR FSAR "central zone" placement; NuScale
# NuFuel-HTP2 benchmark patterns). The nested ladder is
#     12 -> 16 -> 20 -> 24 -> 32 -> 40
# (steps of 4 then 8: past 24 pins no compliant 4-orbit remains). Rules
# enforced at EVERY level: octant symmetry; no rod in the outer two rows;
# no face-adjacent Gd-Gd pairs (self-shadowing); every rod face- or
# diagonally-adjacent to a guide tube; no guide-tube collisions; nesting
# (pattern(n+) contains pattern(n)), so absorber authority is monotone.
# ---------------------------------------------------------------------------
def _orbit(i, j):
    return sorted({(i, j), (j, i), (16 - i, j), (i, 16 - j),
                   (16 - i, 16 - j), (16 - j, i), (j, 16 - i),
                   (16 - j, 16 - i)})


_L12 = sorted(set(_orbit(2, 2) + _orbit(6, 6) + _orbit(3, 8)))   # heritage
GD_PATTERNS = {12: _L12}
GD_PATTERNS[16] = sorted(set(GD_PATTERNS[12] + _orbit(4, 4)))
GD_PATTERNS[20] = sorted(set(GD_PATTERNS[16] + _orbit(6, 8)))
GD_PATTERNS[24] = sorted(set(GD_PATTERNS[20] + _orbit(7, 7)))
GD_PATTERNS[32] = sorted(set(GD_PATTERNS[24] + _orbit(3, 5)))
GD_PATTERNS[40] = sorted(set(GD_PATTERNS[32] + _orbit(4, 6)))
GD_PIN_COUNTS = sorted(GD_PATTERNS)          # [12, 16, 20, 24, 32, 40]


def snap_gd_pins(x) -> int:
    """Nearest available pattern count (the DISCRETE ladder the continuous
    optimizer variable maps onto)."""
    x = float(x)
    return min(GD_PIN_COUNTS, key=lambda n: (abs(n - x), n))


def gd_pattern(n_or_x):
    return GD_PATTERNS[snap_gd_pins(n_or_x)]


# back-compatibility: the heritage 12-pin pattern under the historical name
GD_PIN_POSITIONS = GD_PATTERNS[12]''')

# =========================================================================== #
# 2. reactor_model.py -- zone-matched, industrially reduced Gd materials      #
# =========================================================================== #
rm = rm.replace(A_MATGD, '''    if gd > 0:
        # CAMPAIGN 5: (a) a gadolinia rod takes the enrichment of the ZONE it
        # sits in; (b) that enrichment is reduced by 5% RELATIVE per wt% of
        # Gd2O3 -- the documented low-concentration-gadolinia practice
        # (Westinghouse/ENUSA, INIS FR0200561; same rule in the INL LWRS
        # uprate assessment), motivated by the degraded thermal conductivity
        # of the urania-gadolinia mixture. Floored at natural uranium.
        red = max(0.0, 1.0 - 0.05 * gd)
        mats["fuel_gd_in"] = make_uo2_gd(max(0.2, e_in * red), gd, op.fuel_T)
        mats["fuel_gd_out"] = make_uo2_gd(max(0.2, e_out * red), gd,
                                          op.fuel_T)
    return mats''')

rm = rm.replace(A_LATGD, '''            in_zone = inner_lo <= i <= inner_hi and inner_lo <= j <= inner_hi
            if "fuel_gd_in" in mats and (i, j) in _gd_set:
                u = _fuel_pin_universe(
                    mats["fuel_gd_in" if in_zone else "fuel_gd_out"],
                    mats, geo)
            elif in_zone:''')

# the lattice loop needs the per-design pattern; inject its computation right
# after the function's docstring line "N = geo.lattice"
A_LATHEAD = '''    N = geo.lattice
    gt = _guide_tube_universe(mats, geo)'''
if rm.count(A_LATHEAD) != 1:
    sys.exit("ERROR: lattice head anchor not found after earlier edits.")
rm = rm.replace(A_LATHEAD, '''    N = geo.lattice
    gt = _guide_tube_universe(mats, geo)
    _gd_set = set(gd_pattern(design.get("gd_pins", 12)))   # CAMPAIGN 5''')

# =========================================================================== #
# 3. reactor_model.py -- pin tally for the depletion model                    #
# =========================================================================== #
rm = rm.replace(A_ASMSIG, '''def make_assembly_model(design: dict, op: Operating = Operating(),
                        geo: Geometry17x17 = Geometry17x17(),
                        bc: str = "reflective", reflector: bool = False,
                        particles=20000, batches=150, inactive=40,
                        pin_tally: bool = False):''')

rm = rm.replace(A_ASMRET, '''    bb = ((-half, -half, -1e9), (half, half, 1e9))         # source: fuel region
    model = openmc.Model(geometry=geom, materials=materials,
                         settings=_settings(particles, batches, inactive, bb))
    if pin_tally:
        # CAMPAIGN 5: pin-resolved fission map on every transport solve of the
        # DEPLETION sequence, so assembly F_dH(t) through burnup and at EOL is
        # recorded in each statepoint (openmc_simulation_n*.h5). Cost: one
        # NxN mesh tally per solve -- negligible against transport.
        _mesh = openmc.RegularMesh()
        _mesh.dimension = (geo.lattice, geo.lattice)
        _mesh.lower_left = (-half, -half)
        _mesh.upper_right = (half, half)
        _t = openmc.Tally(name="asm_pin_fission")
        _t.filters = [openmc.MeshFilter(_mesh)]
        _t.scores = ["fission"]
        model.tallies = openmc.Tallies([_t])
    return model, fuel_cells, lat''')

# =========================================================================== #
# 4. openmc_evaluator.py -- request the tally; record the snapped count       #
# =========================================================================== #
ev = ev.replace(A_EVCALL, '''        model, fuel_cells, _lat = rm.make_assembly_model(
            design, self.op, self.geo, bc="reflective", pin_tally=True,
            **self.transport)''')
ev = ev.replace(A_EVRES, '''            "peaking_asm": peaking,
            "gd_pins_used": rm.snap_gd_pins(design.get("gd_pins", 12)),''')

# =========================================================================== #
# 5. reactor_optimization.py -- the 6th variable                              #
# =========================================================================== #
op = op.replace(A_DS, '''    ds = DesignSpace([
        DesignVariable("enrich_inner", 2.0, 19.75, "%"),
        DesignVariable("enrich_outer", 2.0, 19.75, "%"),
        DesignVariable("gd_wt",        0.0,  8.0,  "wt% Gd2O3"),
        DesignVariable("pitch",        1.15, 1.43, "cm"),
        DesignVariable("refl_thick",   2.0,  19.5, "cm"),
        # CAMPAIGN 5: gadolinia-bearing rod count. Continuous for the
        # surrogate and NSGA-II; the model snaps to the nested symmetric
        # ladder {12,16,...,40} (reactor_model.snap_gd_pins) and the snapped
        # value is recorded per evaluation as "gd_pins_used".
        DesignVariable("gd_pins",      12.0, 40.0, "rods"),
    ])''')

# =========================================================================== #
# 6. run_optimization.py -- --workdir + meta                                  #
# =========================================================================== #
i = ru.index(A_OUT)
line_start = ru.rindex("\n", 0, i) + 1
indent = ru[line_start:i]
ru = (ru[:line_start]
      + f'{indent}ap.add_argument("--workdir", default="openmc_runs",\n'
      + f'{indent}                help="scratch directory for OpenMC cases '
        f'(one per campaign!)")\n'
      + ru[line_start:])
ru = ru.replace(A_CTOR,
                '''                         workdir=args.workdir, **schedule)''')
ru = ru.replace(A_META, '''                           "gd_model":
                               "zone-matched enrichment, 5 percent/wt% "
                               "relative reduction, Strategy-A ladder "
                               "{12,16,20,24,32,40}",
                           "objective_def":
                               "peaking = core BOL F_dh (Campaign 5)",''')
# align the FINAL checkpoint write with the per-iteration one (they diverged
# in Campaign 4: the final write lacked core_transport and objective_def)
ru = ru.replace(A_FINALMETA,
                '''    ckpt = opt.save_checkpoint(ckpt_out, meta=opt.checkpoint_meta)''')

# =========================================================================== #
for p, s in ((rm_p, rm), (ev_p, ev), (op_p, op), (ru_p, ru)):
    shutil.copy(p, p.with_suffix(".py.bak"))
    p.write_text(s)
print("\nwritten:", ", ".join(str(p) for p in (rm_p, ev_p, op_p, ru_p)),
      "  (backups: *.py.bak)")
print("""
verify:
  python3 - << 'EOF'
import reactor_model as rm
for n in rm.GD_PIN_COUNTS:
    P = rm.GD_PATTERNS[n]
    assert len(P) == n and len(set(P)) == n
    assert {(16-i,16-j) for i,j in P} == set(P), n      # 180-deg symmetric
    assert {(j,i) for i,j in P} == set(P), n            # diagonal symmetric
print("ladder OK:", rm.GD_PIN_COUNTS)
print("snap 13.4 ->", rm.snap_gd_pins(13.4), "| 26.1 ->", rm.snap_gd_pins(26.1),
      "| 36.5 ->", rm.snap_gd_pins(36.5), "| 40 ->", rm.snap_gd_pins(40))
EOF
  python3 -c "import ast; [ast.parse(open(f).read()) for f in
     ['reactor_model.py','openmc_evaluator.py','reactor_optimization.py',
      'run_optimization.py']]; print('all parse OK')"

then a smoke run (fresh out_smoke_c5, tiny settings) BEFORE the campaign,
checking the checkpoint records gd_pins, gd_pins_used and the gd_model meta.
""")
