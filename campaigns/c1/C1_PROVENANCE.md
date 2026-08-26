# Campaign 1 (C1) provenance

Identified 26 August 2026 by forensic reconstruction from the files in
this directory. All statements below are measured from these files.

## Execution

- Date: 10 July 2026, 10:09:43 to 19:54:09 UTC, single continuous run
  (no resume line in the log, 1008 solver headers, one per state)
- Wall clock: 9.74 h. OpenMC solver time 8.33 h (86% of wall),
  mean 29.7 s per state solve, about 312 core-hours
- Host: AWS EC2 c7a.8xlarge, AMD EPYC 9R14, 32 physical cores, no SMT,
  32 OpenMP threads, 1 MPI process
- Environment: labgene-openmc Docker image, conda env openmc-env,
  OpenMC 0.15.3 (commit 27e38e89), ENDF/B-VII.1,
  chain_endfb71_pwr.xml

## Code state

- 10 July 2026 working tree. Candidate commits of that day:
  80bf508, 94ef3ee, a4cf458, 7956958. Exact commit not yet pinned.
- k-target: Route-B interpolant in reflector thickness
  (ktarget_vs_refl.json), commit a4cf458
- Constraint set: 4 constraints (g_kmin, g_kmax, g_enr, g_peak).
  Predates g_geom, added in 0f571cd on 11 July
- Depletion: evaluator default, fixed 14 transport states per
  evaluation, no censored/n_dep_solves/bu_eoc fields in the records

## Campaign content

- 72 real evaluations, objectives (cycle_length max, peaking min)
- Constraint limits inverted from the stored g values:
  k_BOL in [1.02, 1.35], enrichment cap 19.75 wt%, F_dH <= 2.0
- Realized variable ranges: e_in 2.000 to 19.605 wt%,
  e_out 2.000 to 19.529 wt%, gd 0 to 8.0 wt%,
  pitch 1.1500 to 1.4424 cm, refl 2.656 to 25.000 cm
- Feasibility: 46 of 72 (25 g_kmax violations, 1 g_peak violation)
- Right-censoring: 39 of 72 (54.2%) at exactly 3555.9192 EFPD,
  the implied evaluator-default horizon of 35.5 MWd/kgHM at
  P_spec = 9.983e-3 MW/kgHM
- Pareto front: 4 designs, F_dH 1.1362 to 1.1513, cycle 2007 to
  3556 EFPD, one at the horizon. All four have refl 23.99 to
  24.99 cm and violate the vessel envelope (max 19.5 cm), the
  finding that introduced g_geom
- Hypervolume: 1132.9 final, reference point (-1787.53, 1.7847),
  9 recorded iterations

## Files

- optimization_checkpoint.json: full 72-record archive (all_raw)
- optimization_results.json: Pareto front and hypervolume history
- optimization_openmc.png: contemporaneous two-panel summary figure
- c1_run.log.gz: complete OpenMC output of the run (gzipped)
