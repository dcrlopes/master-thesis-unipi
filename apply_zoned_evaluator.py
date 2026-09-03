#!/usr/bin/env python3
"""apply_zoned_evaluator.py -- make the OPTIMIZER evaluate the ZONED core.

WHY
---
Campaign 6 shrinks the enrichment search box to LEU_CAP_WTPC / M_P_DESIGN
(leu_policy.py) because the as-built peripheral ring carries the multiplier
m_P. Until now the multiplier existed only in post-processing: the truth
evaluator built an UNZONED core, so the peaking objective F_dH and the
g_peak constraint were measured on a core the design basis does not use,
while the search box paid the enrichment cost of a zoning it never saw.

The zoned study measured what that costs: zoning brought F_core from ~2.0
down to 1.4495 (idx58) and 1.4863 (idx54). An optimizer blind to that
benefit is optimizing the wrong core.

WHAT THIS DOES (three files, four anchored edits)
-------------------------------------------------
zoning.py            gains the FROZEN Campaign 6 map next to the zoning
                     machinery it uses:
                       M_C_DESIGN = 0.720
                       m_P        = leu_policy.M_P_DESIGN  (single source,
                                    the same value that sized the box)
                       m_M        = balanced_multipliers(...), derived so
                                    the core-average enrichment multiplier
                                    is exactly 1
                     plus evaluator_multipliers() and
                     evaluator_design_map(design).

openmc_evaluator.py  imports zoning, and _bol_core_peaking builds the core
                     with design_map=zn.evaluator_design_map(design). The
                     peaking objective, g_peak, and (when k_basis="core")
                     the reactivity window now all read the ZONED core.

run_optimization.py  records the zoning policy in the checkpoint metadata
                     next to the enrichment policy, so every archive states
                     which loading its numbers describe.

WHAT IT DOES NOT TOUCH
----------------------
  * The assembly depletion path (cycle length). The balanced map preserves
    the core-average enrichment multiplier at exactly 1, so the
    base-assembly cycle-length proxy stays consistent. Documented choice.
  * Geometry. zone_designs scales enrichments only, never pitch.
  * Seeds and caches. The seed is hashed from the BASE design dict, which
    is unchanged; the map is derived deterministically from it.
  * leu_policy.py. m_P has exactly one home.

RUN ORDER (wks720, repo root, conda env openmc-env)
---------------------------------------------------
    python -c "import numpy, openmc; print('env ok')" \
        && python apply_zoned_evaluator.py \
        && python verify_zoned_evaluator.py

Backups: <file>.bak.zoned_ev. Refuses to run twice (ZONED-EVALUATOR marker).
Requires leu_policy.py to exist (generate with make_leu_policy.py if absent).
"""
from __future__ import annotations
import py_compile
import shutil
import sys
from pathlib import Path

MARKER = "ZONED-EVALUATOR"

FILES = {
    "zoning.py": [

# -- Z1: frozen policy, appended after max_zoned_enrichment ---------------
(
'''def max_zoned_enrichment(base: dict, m_p: float) -> float:
    """Highest enrichment anywhere in the zoned core. The periphery carries
    the largest multiplier in every map this study uses."""
    return max(float(base["enrich_inner"]), float(base["enrich_outer"])) * m_p
''',
'''def max_zoned_enrichment(base: dict, m_p: float) -> float:
    """Highest enrichment anywhere in the zoned core. The periphery carries
    the largest multiplier in every map this study uses."""
    return max(float(base["enrich_inner"]), float(base["enrich_outer"])) * m_p


# --------------------------------------------------------------------------- #
# ZONED-EVALUATOR: the FROZEN Campaign 6 loading map the optimizer builds     #
# --------------------------------------------------------------------------- #
# Chosen from the Campaign 5 zoned-loading study, whose optimum over all four
# champions was m_C / m_M / m_P = 0.720 / 0.893 / 1.150. m_P is imported from
# leu_policy so the search box (E_SEARCH_MAX = LEU_CAP_WTPC / M_P_DESIGN) and
# the as-built periphery are sized by the SAME number. m_M is not stored: it
# is re-derived from the fissile balance, so the core-average enrichment
# multiplier is exactly 1 whatever m_C and m_P say.
M_C_DESIGN = 0.720


def evaluator_multipliers():
    """(rmap, m_C, m_M, m_P) of the frozen map used by the truth evaluator."""
    import leu_policy as _leu
    rmap = ring_map()
    m_c, m_m, m_p = balanced_multipliers(M_C_DESIGN, _leu.M_P_DESIGN,
                                         ring_counts(rmap))
    return rmap, m_c, m_m, m_p


def evaluator_design_map(design: dict) -> dict:
    """{(row, col): design} of the frozen zoned loading for one base design.

    Consumed by reactor_model.make_core_model(design_map=...). Enrichments
    of each ring are the base values scaled by that ring's multiplier; every
    other design variable is copied unchanged (zone_designs), so the pin
    layout, the gadolinia pattern and the zone-enrichment derate of the
    gadolinia rods stay single-sourced in the builders."""
    rmap, m_c, m_m, m_p = evaluator_multipliers()
    return design_map_for(rmap, zone_designs(design, m_c, m_m, m_p))
''',
),
    ],

    "openmc_evaluator.py": [

# -- Z2: import the zoning module -----------------------------------------
(
'''import leu_policy as _leu
import reactor_model as rm
''',
'''import leu_policy as _leu
import reactor_model as rm
import zoning as zn          # ZONED-EVALUATOR: frozen Campaign 6 loading map
''',
),

# -- Z3: build the zoned core in the BOL core solve ------------------------
(
'''        m = rm.make_core_model(design, self.op, self.geo,
                               particles=self.core_particles,
                               batches=self.core_batches,
                               inactive=self.core_inactive)
''',
'''        # ZONED-EVALUATOR: build the core the design basis assumes. The
        # frozen map (m_C = zoning.M_C_DESIGN, balanced m_M, m_P from
        # leu_policy) is applied on every core solve, so the peaking
        # objective, g_peak, and, under k_basis="core", the reactivity
        # window all describe the SAME as-built loading that sized the
        # enrichment search box. The assembly depletion path is untouched:
        # the balanced map keeps the core-average enrichment multiplier at
        # exactly 1, so the base-assembly cycle-length proxy remains
        # consistent.
        m = rm.make_core_model(design, self.op, self.geo,
                               design_map=zn.evaluator_design_map(design),
                               particles=self.core_particles,
                               batches=self.core_batches,
                               inactive=self.core_inactive)
''',
),
    ],

    "run_optimization.py": [

# -- Z4a: import zoning where leu_policy is imported -----------------------
(
'''    import leu_policy as _leu
    spec = example_reactor_problem()
''',
'''    import leu_policy as _leu
    import zoning as _zn     # ZONED-EVALUATOR: record the map in metadata
    spec = example_reactor_problem()
''',
),

# -- Z4b: record the zoning policy in the checkpoint metadata --------------
(
'''                           "enrichment_policy": {
                               "leu_cap_wtpc": _leu.LEU_CAP_WTPC,
                               "m_p_design": _leu.M_P_DESIGN,
                               "e_search_max_wtpc": _leu.E_SEARCH_MAX},
''',
'''                           "enrichment_policy": {
                               "leu_cap_wtpc": _leu.LEU_CAP_WTPC,
                               "m_p_design": _leu.M_P_DESIGN,
                               "e_search_max_wtpc": _leu.E_SEARCH_MAX},
                           "zoning_policy": {
                               "evaluator_zoned": True,
                               "m_c_design": _zn.M_C_DESIGN,
                               "m_m_balanced": _zn.evaluator_multipliers()[2],
                               "m_p_design": _leu.M_P_DESIGN,
                               "ring_counts": list(
                                   _zn.ring_counts(_zn.ring_map()))},
''',
),
    ],
}


def main() -> None:
    root = Path(".")
    # verify EVERYTHING before touching ANYTHING
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
                         f"(need exactly 1). No file was modified.\n"
                         f"Anchor begins: {anchor.splitlines()[0]!r}")
    if not (root / "leu_policy.py").is_file():
        sys.exit("ABORT: leu_policy.py is missing. Generate it first "
                 "(make_leu_policy.py) so m_P has its single source.")
    # apply
    for fname, edits in FILES.items():
        p = root / fname
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak.zoned_ev"))
        text = p.read_text()
        for anchor, repl in edits:
            text = text.replace(anchor, repl)
        p.write_text(text)
        py_compile.compile(str(p), doraise=True)
        print(f"[ok] {fname}: {len(edits)} edit(s), backup "
              f"{p.name}.bak.zoned_ev, py_compile passed")
    print("[done] the truth evaluator now builds the zoned core on every "
          "core solve.")
    print("       Run verify_zoned_evaluator.py before launching "
          "Campaign 6.")


if __name__ == "__main__":
    main()
