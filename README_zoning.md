# Zoned-loading study, to run AFTER Campaign 5 Block 2

Four new files plus this guide. Order of operations, expected cost, and
what each result feeds in the thesis.

The study answers the methodological objection raised on 25 August 2026:
the campaigns optimise under uniform loading, so is the optimum still an
optimum once the core is zoned? Stage T measures the rank transfer, Stage 2
refines the zoning of the champions, Stage 3 puts a confirmed cycle length
on the zoned champion. The methodology mirrors the Campaign 3 core rescore
one level up, which is the argument to make in the text.

## 0. Prerequisites (once, minutes)

Block 2 finished and its checkpoint written, for example at
`openmc_runs_c5/out/optimization_checkpoint.json`. Then, on wks720, inside
the repo with the campaign5 code as run:

```
conda activate openmc-env
python apply_zoned_core.py --check
```

Flag `--check` builds one zoned 32-assembly model in memory after patching
(imports OpenMC, runs zero particles) to prove the geometry assembles. The
patch adds ONE optional argument `design_map` to
`reactor_model.make_core_model`. With `design_map=None` (every existing
caller) the builder is unchanged line for line, so nothing about the
finished campaigns is touched. A backup `reactor_model.py.zoned.bak` is
written. The script refuses to run twice.

Commit the patched state before running the stages, so the as-run zoning
code is versioned like the campaigns were.

## 1. Stage T, the transfer test (about 1.5 h for a 42-design block)

```
setsid nohup python -u rescore_zoned_core.py \
    --checkpoint openmc_runs_c5/out/optimization_checkpoint.json \
    --threads 64 --out zoned_rescore > zoned_rescore.log 2>&1 < /dev/null &
```

`setsid` detaches the process from the ssh session, `nohup` ignores the
hangup signal, `-u` keeps the log unbuffered, `&` backgrounds it. One
fixed map (centre 0.85, middle solved to 0.95, periphery 1.075, the
core-average enrichment preserved exactly) is applied to EVERY archive
design, one core Beginning of Life solve each, about 2 minutes per design
at the campaign settings (100000 particles, 170 batches, 60 inactive).
All designs are rescored by default, not only the feasible ones, because
Campaign 4 showed the feasible set can be empty while the near-feasible
designs carry the story. The run is resumable: a wks720 reboot costs only
the interrupted design.

Outputs in `zoned_rescore/`: `zoned_rescore.csv`, `transfer_summary.json`,
`fig_zoned_transfer.png`.

Decision rule, stated as guidance to argue in the text, not as law:

1. Spearman rho of the best 15 designs at or above about 0.8 AND the best
   3 to 5 designs preserved: the uniform campaign is a validated proxy.
   Zoning is applied at candidate selection (Stages 2 and 3) and the
   methodology chapter reports rho as the proxy-fidelity number.
2. A scrambled top: that is a finding, reported exactly like the Campaign
   3 rank inversion, and joint lattice-plus-loading optimisation is scoped
   as future work with the measured evidence.

Check `rankshift_vs_refl_thick` in the summary: the working hypothesis is
that any rank movement concentrates in the reflector and pitch pricing,
because those are the only uniform-loading variables that also act on the
tilt.

## 2. Stage 2, refine the champions (under 1 h per champion)

Browse the archive and pick champions first:

```
python refine_zoning.py --checkpoint openmc_runs_c5/out/optimization_checkpoint.json --list
```

Then, per champion (example index 44):

```
setsid nohup python -u refine_zoning.py \
    --checkpoint openmc_runs_c5/out/optimization_checkpoint.json \
    --idx 44 --case-dir openmc_runs_c5/case_0044 \
    --threads 64 --out zoned_refine > zoned_refine.log 2>&1 < /dev/null &
```

`--idx` is repeatable for several champions in one call. `--case-dir`
points at the champion's campaign case directory so the script extracts
the late-cycle reactivity slope from its own depletion history and prices
every grid point in Effective Full Power Days. Without it, pass
`--slope-pcm-per-mwdkg` (magnitude, pcm per MWd/kgHM) or the price column
is omitted. The default grid is 5 centre multipliers times 5 periphery
multipliers, 25 core solves, with the uniform anchor (1, 1) always run by
the same script so the comparison is internally consistent. Points whose
periphery enrichment crosses the 19.75 wt% screen are run but flagged.

Outputs per champion: `zoning_grid_idxNN.csv`,
`fig_zoning_grid_idxNN.png`, and a printed best point with its measured F,
its measured k change, and its estimated cycle price.

## 3. Stage 3, confirm the zoned champion (about 3 depletions, hours)

Take the Stage 2 winning multipliers, for example 0.85 and 1.075:

```
setsid nohup python -u confirm_zoned_champion.py \
    --checkpoint openmc_runs_c5/out/optimization_checkpoint.json \
    --idx 44 --m-center 0.85 --m-periphery 1.075 \
    --ktarget-table ktarget_table.json \
    --threads 64 --out zoned_confirm > zoned_confirm.log 2>&1 < /dev/null &
```

The three ring variants are depleted with the UNMODIFIED campaign
machinery (`OpenMCEvaluator._cycle_length`, Route B target from
`--ktarget-table`, transport settings read from the checkpoint meta unless
overridden with `--particles`, `--batches`, `--inactive`). The zoned cycle
length is then the power-weighted linear-reactivity combination of the
three ring histories against the champion's own Route B target, with the
ring power shares measured by one zoned core solve. The script prints its
wall-time expectation at start (about three times the champion's archived
depletion time) and reuses any finished ring after a reboot.

Outputs: `confirm_idxNN.json`, `fig_zoned_confirm_idxNN.png`.

Two assumptions are printed into the summary and belong VERBATIM next to
the result in the thesis: the ring power shares are frozen at their
beginning-of-life values, and each ring's reactivity history is
parameterised by its own burnup. The gold standard, a full-core zoned
depletion, is future work.

## 4. What feeds the thesis where

1. Methodology chapter: the two-stage hierarchy (lattice optimisation,
   then loading-pattern refinement), positioned against the literature
   that treats loading-pattern design as its own discipline, and the
   mirror-image relation to the KSMR paper (they freeze the lattice and
   search loadings, this work does the opposite). Suggested
   `\openpoint{}`: confirm the FORMOSA-P and Parks 1996 references before
   submission, both currently VERIFY.
2. Results chapter: `fig_zoned_transfer.png` with the Spearman numbers as
   the proxy-fidelity result, then the champion's zoning grid, then the
   confirmed zoned point (F at beginning of life, k, EFPD) against the
   uniform archive value. The count of designs moving under F = 2.0 from
   the uniform to the zoned column is the headline of the feasibility
   discussion.
3. Discussion: zoning buys peaking at a small cycle-length price through
   increased peripheral power and leakage, so the uniform-loading front
   brackets one end of a real trade-off. Quote the measured dk and dEFPD.

## 5. Numbers to sanity-check on arrival

1. Ring counts printed at start must read (4, 12, 16) and the multipliers
   must satisfy 4 m_C + 12 m_M + 16 m_P = 32 exactly.
2. The Stage T anchor statistics: with sigma_F about 0.018 per solve at
   campaign settings (Campaign 3 measurement), rank swaps between designs
   closer than about 0.04 in F are ties, not signal.
3. In Stage 3, the ring C depletion should show a LOWER k at beginning of
   life than the uniform champion and ring P a higher one, with the
   power-weighted mix starting close to the champion's archived
   `keff_core_bol` mapped through the Route B relation. A large mismatch
   means the shares file or the target table does not belong to this
   design.
