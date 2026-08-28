"""
leu_policy.py
=============
Single source of truth for the enrichment policy of the optimization.

Two numbers govern every enrichment decision in this pipeline. They are
defined here so that reactor_optimization.py (the search box),
openmc_evaluator.py (the audit constraint) and run_optimization.py (the
campaign provenance) can never disagree.

    LEU_CAP_WTPC : float
        Maximum permitted U-235 enrichment anywhere in the as-built core, in
        weight per cent. 19.75 wt% is the conventional LEU (Low Enriched
        Uranium) ceiling, set below the 20 wt% boundary that defines HEU
        (High Enriched Uranium) so manufacturing tolerance cannot cross it.

    M_P_DESIGN : float
        Peripheral zoning multiplier of the loading map the candidate cores
        will use, dimensionless. The zoned core's highest enrichment is
        max(enrich_inner, enrich_outer) * M_P_DESIGN, because
        zoning.assign_zone_designs scales both intra-assembly enrichments of
        a ring by one multiplier.

    E_SEARCH_MAX : float
        Upper bound on BOTH enrichment design variables, in weight per cent.
        Derived, not chosen:

            E_SEARCH_MAX = LEU_CAP_WTPC / M_P_DESIGN

        so no design the optimizer can propose exceeds the LEU cap once the
        peripheral multiplier is applied.

At M_P_DESIGN = 1.0 the search box and the LEU cap coincide and every
formula below reduces to the unzoned behaviour exactly.
"""
from __future__ import annotations

LEU_CAP_WTPC = 19.75
M_P_DESIGN = 1.15
E_SEARCH_MAX = LEU_CAP_WTPC / M_P_DESIGN


def max_zoned_enrichment_wtpc(e_inner: float, e_outer: float) -> float:
    """Highest U-235 enrichment anywhere in the zoned core, in weight per cent.

    Mirrors zoning.max_zoned_enrichment, duplicated here so the evaluator can
    audit the enrichment without importing the zoning module.
    """
    return max(float(e_inner), float(e_outer)) * M_P_DESIGN
