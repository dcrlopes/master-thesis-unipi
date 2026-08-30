#!/usr/bin/env python3
"""apply_ctrl_constraint.py -- suggestion 1: make the OPTIMIZER see
controllability, so Campaign 7 searches only among reactors the regulating
banks can hold down.

WHAT IT ADDS
------------
One extra zoned core solve per evaluation, with the sixteen regulating-bank
CRAs (RE1..RE4, the complete inner C and M rings) fully inserted and the SH
banks out, because SH3, SH4 and SH5 are reserved for scram. The new
constraint is

    g_ctrl = k_ALLRE - (1 - margin)  <=  0

so a design is feasible only if the RE banks alone hold it subcritical by
the operating margin. Cost: one core solve of about 73 s on a 1370 s
evaluation, roughly 5 percent.

Rod worth is measured, not proxied: this is the direct version of the
controllability screen, and it is what the rod_bank_worth.py measurements
motivate after design 71 proved that a Pareto design can be physically
un-shutdownable.

OFF BY DEFAULT. The solve runs and the constraint exists only when
run_optimization.py is given --ctrl-margin, so every earlier campaign and
checkpoint resumes bit for bit. A Campaign 6 checkpoint cannot be resumed
INTO a --ctrl-margin run: the constraint set differs and the resume guard
raises, which is deliberate.

EDITS (anchor-verified against campaign6 head bf407cc)
------------------------------------------------------
  zoning.py            RE_BANK_POSITIONS, the single source of the sixteen
                       regulating positions, next to the map machinery
  openmc_evaluator.py  the g_ctrl entry in the results dict, plus the
                       module-level _ctrl_solve helper
  run_optimization.py  --ctrl-margin and --ctrl-absorber flags, evaluator
                       configuration, constraint and scale registration,
                       checkpoint metadata

Backups: <file>.bak.ctrl. Refuses to run twice (CTRL-SCREEN marker).
"""
from __future__ import annotations
import py_compile
import shutil
import sys
from pathlib import Path

MARKER = "CTRL-SCREEN"

FILES = {
    "zoning.py": [
(
'''    rmap, m_c, m_m, m_p = evaluator_multipliers()
    return design_map_for(rmap, zone_designs(design, m_c, m_m, m_p))
''',
'''    rmap, m_c, m_m, m_p = evaluator_multipliers()
    return design_map_for(rmap, zone_designs(design, m_c, m_m, m_p))


# --------------------------------------------------------------------------- #
# CTRL-SCREEN: the sixteen regulating-bank positions (RE1..RE4)               #
# --------------------------------------------------------------------------- #
# The complete inner sixteen assemblies: the C ring (RE1), the M-ring
# diagonals (RE2) and both M-edge orbits (RE3, RE4). The SH banks occupy the
# outer sixteen and are reserved for scram, so operational controllability
# means subcritical under ALL-RE. Single source shared by the evaluator's
# g_ctrl constraint and by rod_bank_worth.py.
RE_BANK_POSITIONS = frozenset([
    (2, 2), (2, 3), (3, 2), (3, 3),          # RE1  inner ring
    (1, 1), (1, 4), (4, 1), (4, 4),          # RE2  M diagonals
    (1, 2), (2, 4), (4, 3), (3, 1),          # RE3  M edges, orbit A
    (1, 3), (3, 4), (4, 2), (2, 1),          # RE4  M edges, orbit B
])
''',
),
    ],

    "openmc_evaluator.py": [
(
'''            "g_peak":  core["fdh_core"] - self.f_max,     # CORE peaking
''',
'''            "g_peak":  core["fdh_core"] - self.f_max,     # CORE peaking
            # CTRL-SCREEN (Campaign 7): operational controllability. One
            # extra zoned core solve with the sixteen regulating-bank CRAs
            # (zn.RE_BANK_POSITIONS) fully inserted; SH banks stay out,
            # reserved for scram. Feasible iff subcritical by the operating
            # margin. Enabled only when run_optimization passes
            # --ctrl-margin, so earlier campaigns are bit-for-bit unchanged.
            **({"k_allre": (_c := _ctrl_solve(self, design))["keff"],
                "F_allre": _c["fdh_core"],
                "g_ctrl":  _c["keff"] - (1.0 - self.ctrl_margin_dk)}
               if getattr(self, "ctrl_margin_dk", None) is not None
               else {}),
''',
),
    ],

    "run_optimization.py": [
(
'''    ap.add_argument("--enr-max", type=float, default=19.75,
                    help="LEU (Low Enriched Uranium) enrichment cap in "
                         "wt%% U-235")
''',
'''    ap.add_argument("--enr-max", type=float, default=19.75,
                    help="LEU (Low Enriched Uranium) enrichment cap in "
                         "wt%% U-235")
    ap.add_argument("--ctrl-margin", type=float, default=None,
                    help="CTRL-SCREEN: required subcriticality under the "
                         "fully inserted regulating banks (RE1..RE4), in "
                         "pcm of dk, e.g. 1000. Adds constraint g_ctrl and "
                         "one extra core solve per evaluation. Off when "
                         "absent, reproducing Campaign 6 behaviour.")
    ap.add_argument("--ctrl-absorber", choices=["B4C", "AIC"],
                    default="B4C",
                    help="CTRL-SCREEN: absorber of the regulating CRAs")
''',
),
(
'''        "g_geom": _cg.R_VESSEL_INNER - _cg.VESSEL_CLEARANCE_CM,
    })
''',
'''        "g_geom": _cg.R_VESSEL_INNER - _cg.VESSEL_CLEARANCE_CM,
    })
    # CTRL-SCREEN (Campaign 7): configure the evaluator and register the
    # constraint only when the flag is given, so the constraint set of every
    # earlier campaign is untouched and their checkpoints resume unchanged.
    if args.ctrl_margin is not None:
        ev.ctrl_margin_dk = float(args.ctrl_margin) * 1.0e-5
        ev.ctrl_absorber = args.ctrl_absorber
        spec.constraint_names.append("g_ctrl")
        spec.constraint_scales["g_ctrl"] = 1.0     # k-units: limit is 1.0
''',
),
(
'''                               "ring_counts": list(
                                   _zn.ring_counts(_zn.ring_map()))},
''',
'''                               "ring_counts": list(
                                   _zn.ring_counts(_zn.ring_map()))},
                           "ctrl_screen": {
                               "enabled": args.ctrl_margin is not None,
                               "margin_pcm": args.ctrl_margin,
                               "absorber": args.ctrl_absorber,
                               "re_positions": sorted(
                                   _zn.RE_BANK_POSITIONS)},
''',
),
    ],
}

CTRL_SOLVE = '''

# --------------------------------------------------------------------------- #
# CTRL-SCREEN helper (appended by apply_ctrl_constraint.py)                    #
# --------------------------------------------------------------------------- #
def _ctrl_solve(ev, design):
    """One zoned core solve with the sixteen regulating-bank CRAs inserted
    (zn.RE_BANK_POSITIONS), SH banks out. Same fidelity, zoning path and
    deterministic seeding as every other core solve; the case directory is
    keyed by the design hash so re-evaluations reuse it."""
    tag = _design_seed(design, salt="ctrl") & 0xFFFFFFFF
    return zn.core_bol_solve(
        design, zn.evaluator_design_map(design), ev.op, ev.geo,
        particles=ev.core_particles, batches=ev.core_batches,
        inactive=ev.core_inactive,
        seed=_design_seed(design, salt="ctrl"),
        case=ev.workdir / f"ctrl_{tag:08x}",
        rodded_map=(set(zn.RE_BANK_POSITIONS),
                    getattr(ev, "ctrl_absorber", "B4C")))
'''


def main() -> None:
    root = Path(".")
    for fname, edits in FILES.items():
        p = root / fname
        if not p.is_file():
            sys.exit(f"ABORT: {p} not found. Run from the repository root.")
        text = p.read_text()
        if MARKER in text:
            sys.exit(f"REFUSED: {fname} already contains the {MARKER} "
                     f"marker. Nothing was changed.")
        for i, (anchor, _) in enumerate(edits, 1):
            n = text.count(anchor)
            if n != 1:
                sys.exit(f"ABORT: anchor {i} for {fname} found {n} times "
                         f"(need exactly 1). No file was modified.\\n"
                         f"Anchor begins: {anchor.splitlines()[0]!r}")
    for fname, edits in FILES.items():
        p = root / fname
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak.ctrl"))
        text = p.read_text()
        for anchor, repl in edits:
            text = text.replace(anchor, repl)
        if fname == "openmc_evaluator.py":
            text += CTRL_SOLVE
        p.write_text(text)
        py_compile.compile(str(p), doraise=True)
        print(f"[ok] {fname}: {len(edits)} edit(s)"
              + (" + _ctrl_solve appended" if fname == "openmc_evaluator.py"
                 else "")
              + f", backup {p.name}.bak.ctrl, py_compile passed")
    print("[done] g_ctrl is available. Enable it per run with "
          "--ctrl-margin <pcm>.")


if __name__ == "__main__":
    main()
