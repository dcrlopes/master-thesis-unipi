#!/usr/bin/env python
"""
campaign_timing.py -- reconstruct the wall-clock budget of a finished campaign
from the OpenMC statepoint files that every evaluation already wrote.

Nothing in the optimisation loop recorded timings (reactor_optimization.py only
prints "Done in ...s" at the very end of a block). The statepoints do, however,
carry two things that make a full reconstruction possible after the fact:

  * the root attribute  date_and_time   (wall-clock instant the file was written)
  * the group           /runtime/       (seconds spent in initialisation,
                                         simulation, transport, ...)

Both are documented in the OpenMC statepoint format (docs.openmc.org,
io_formats/statepoint). Depletion statepoints (openmc_simulation_n*.h5, one per
transport solve, written by CoupledOperator.write_bos_data) carry valid
/runtime/ values since OpenMC PR #2302 (0.13.3+), so OpenMC 0.15.3 is fine.

Because OpenMCEvaluator.evaluate() is strictly serial (one design after the
other, OpenMP inside each solve), the gaps between consecutive cases are the
optimiser's own cost: Gaussian-process fit, NSGA-II on the surrogate,
acquisition, hypervolume, checkpoint. Gaps longer than --session-gap-min are
treated as session breaks (machine idle between blocks) and reported apart.

Layout assumed per evaluation (see openmc_evaluator.py):

    <workdir>/case_NNNN/bol/statepoint.*.h5                 assembly BOL peaking
    <workdir>/case_NNNN/core_bol/statepoint.*.h5            core BOL peaking (C4+)
    <workdir>/case_NNNN/dep_MM/openmc_simulation_n*.h5      one per depletion solve
    <workdir>/case_NNNN/dep_MM/depletion_results.h5

Usage (inside the openmc-env conda environment, on the machine holding the runs):

    python campaign_timing.py --workdir openmc_runs_c3 --campaign C3 \
        --n-init 36 --n-infill 6 --checkpoint out_c3/optimization_checkpoint.json \
        --threads 64 --host wks720 --out timing_c3

    python campaign_timing.py --mode tree --workdir rescore_runs --campaign "C3 core rescoring" \
        --out timing_rescore

Flags
    --workdir          directory with the case_NNNN folders (or any tree in --mode tree)
    --campaign         label used in the tables
    --n-init           size of the DOE (first n-init cases), default 36
    --n-infill         real evaluations per active-learning iteration, default 6
    --checkpoint       optimisation checkpoint JSON, used for omp_threads, transport
                       settings and a sanity check on the number of evaluations
    --threads          OpenMP thread count, if you know it and no checkpoint is given
    --host             hostname of the machine that RAN the campaign (the script
                       cannot know it, statepoints do not store it)
    --skip-cases       comma-separated case indices to drop entirely (files
                       overwritten by a later job). Reported in the summary
    --after / --before restrict to statepoints written inside a time window,
                       'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'. Use when a directory
                       holds more than one job
    --session-gap-min  gaps longer than this (minutes) count as idle time, default 30
    --mode             'campaign' (default, needs case_NNNN) or 'tree' (sum every
                       statepoint under --workdir, for rescoring / confirmation runs)
    --out              output prefix, writes <out>_cases.csv, <out>_summary.json,
                       <out>_table.tex

Outputs are written next to where you run the script. Copy the .tex fragment
into the dissertation repository only after reading the numbers yourself.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

# --------------------------------------------------------------------------- #
# statepoint reading                                                          #
# --------------------------------------------------------------------------- #
_SP_DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _decode(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    if isinstance(x, np.ndarray):
        return _decode(x.tolist())
    if isinstance(x, list) and len(x) == 1:
        return _decode(x[0])
    return str(x)


def read_statepoint(path: Path) -> dict:
    """Return timing metadata of one statepoint file.

    Keys: end (datetime, when the file was written), runtime (dict of seconds),
    init_s, sim_s, transport_s, wall_s (= init + sim), n_particles, n_batches,
    n_inactive, path_attr, mtime (datetime, fallback), runtime_ok (bool).
    """
    rec = dict(file=str(path), end=None, runtime={}, init_s=0.0, sim_s=0.0,
               transport_s=0.0, wall_s=0.0, n_particles=None, n_batches=None,
               n_inactive=None, path_attr=None, runtime_ok=False,
               mtime=datetime.fromtimestamp(path.stat().st_mtime))
    with h5py.File(path, "r") as f:
        dt_raw = f.attrs.get("date_and_time")
        if dt_raw is not None:
            s = _decode(dt_raw).strip()
            for fmt in _SP_DATE_FORMATS:
                try:
                    rec["end"] = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
        if "path" in f.attrs:
            rec["path_attr"] = _decode(f.attrs["path"])
        for key in ("n_particles", "n_batches", "n_inactive"):
            if key in f:
                rec[key] = int(np.asarray(f[key][()]).ravel()[0])
        if "runtime" in f:
            rt = {}
            for k in f["runtime"]:
                try:
                    rt[k] = float(np.asarray(f["runtime"][k][()]).ravel()[0])
                except Exception:
                    pass
            rec["runtime"] = rt
            rec["init_s"] = rt.get("total initialization", 0.0)
            rec["sim_s"] = rt.get("simulation", 0.0)
            rec["transport_s"] = rt.get("transport", 0.0)
            rec["wall_s"] = rec["init_s"] + rec["sim_s"]
            rec["runtime_ok"] = rec["sim_s"] > 0.0
    if rec["end"] is None:            # very old format or stripped attrs
        rec["end"] = rec["mtime"]
    return rec


def _sp_files(d: Path, pattern: str) -> list[Path]:
    return sorted(d.glob(pattern)) if d.is_dir() else []


# --------------------------------------------------------------------------- #
# one evaluation directory -> phase breakdown                                 #
# --------------------------------------------------------------------------- #
PHASES = ("asm_bol", "core_bol", "dep_transport")


def _parse_when(text):
    """Accept 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM[:SS]'."""
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise SystemExit(f"cannot parse date/time {text!r}")


def analyse_case(case: Path, gap_thr_s: float = 1800.0,
                 after=None, before=None) -> dict:
    """Phase breakdown of one case directory.

    A case directory can hold files from TWO runs: an evaluation that was lost
    (mid-loop crash, reboot) and the re-run that reused the same case number
    after a resume, because the evaluator numbers cases from the checkpoint
    length and OpenMC overwrites statepoint.<batches>.h5 in place. Leftover
    chunks of the lost run survive. The statepoints are therefore clustered
    by time. The LAST contiguous cluster is the evaluation in the archive.
    Earlier clusters are reported as stale (discarded) compute.
    """
    idx = int(re.search(r"(\d+)$", case.name).group(1))
    out = dict(case=case.name, idx=idx, n_solves=0, n_dep_solves=0,
               n_dep_chunks=0, runtime_missing=0, n_stale_solves=0,
               stale_solve_s=0.0, mixed=False,
               t_first_end=None, t_last_end=None, t_start=None,
               particles_asm=None, batches_asm=None,
               particles_core=None, batches_core=None, inactive_core=None,
               path_attr=None)
    for ph in PHASES:
        out[f"{ph}_s"] = 0.0
        out[f"{ph}_transport_s"] = 0.0
    sps: list[tuple[str, dict]] = []

    for p in _sp_files(case / "bol", "statepoint.*.h5"):
        sps.append(("asm_bol", read_statepoint(p)))
    for p in _sp_files(case / "core_bol", "statepoint.*.h5"):
        sps.append(("core_bol", read_statepoint(p)))
    dep_dirs = sorted(d for d in case.glob("dep_*") if d.is_dir())
    out["n_dep_chunks"] = len(dep_dirs)
    for d in dep_dirs:
        for p in _sp_files(d, "openmc_simulation_n*.h5"):
            sps.append(("dep_transport", read_statepoint(p)))

    if after is not None:
        sps = [(ph, r) for ph, r in sps if r["end"] >= after]
    if before is not None:
        sps = [(ph, r) for ph, r in sps if r["end"] <= before]
    if not sps:
        return out

    # ---- split stale leftovers from the evaluation actually archived -------
    sps.sort(key=lambda t: t[1]["end"])
    clusters: list[list[tuple[str, dict]]] = [[sps[0]]]
    for item in sps[1:]:
        if (item[1]["end"] - clusters[-1][-1][1]["end"]).total_seconds() > gap_thr_s:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    if len(clusters) > 1:
        out["mixed"] = True
        for cl in clusters[:-1]:
            out["n_stale_solves"] += len(cl)
            out["stale_solve_s"] += sum(r["wall_s"] for _, r in cl)
        sps = clusters[-1]

    for ph, r in sps:
        out[f"{ph}_s"] += r["wall_s"]
        out[f"{ph}_transport_s"] += r["transport_s"]
        out["n_solves"] += 1
        if ph == "dep_transport":
            out["n_dep_solves"] += 1
        if not r["runtime_ok"]:
            out["runtime_missing"] += 1
        if ph == "asm_bol":
            out["particles_asm"], out["batches_asm"] = r["n_particles"], r["n_batches"]
        if ph == "core_bol":
            out["particles_core"], out["batches_core"] = r["n_particles"], r["n_batches"]
            out["inactive_core"] = r["n_inactive"]
        if out["path_attr"] is None and r["path_attr"]:
            out["path_attr"] = r["path_attr"]

    ends = [r["end"] for _, r in sps]
    first = min(sps, key=lambda t: t[1]["end"])[1]
    out["t_first_end"] = min(ends)
    out["t_last_end"] = max(ends)
    # start of the evaluation = end of the first solve minus its own duration
    out["t_start"] = first["end"] - timedelta(seconds=first["wall_s"])
    out["solve_s"] = sum(out[f"{ph}_s"] for ph in PHASES)
    out["wall_s"] = (out["t_last_end"] - out["t_start"]).total_seconds()
    # everything that is not inside an OpenMC run: Bateman solves, material
    # updates, XML export, Python bookkeeping, statepoint reads
    out["overhead_s"] = max(out["wall_s"] - out["solve_s"], 0.0)
    return out


# --------------------------------------------------------------------------- #
# campaign assembly                                                           #
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        ck = json.load(f)
    meta = ck.get("meta", {}) or {}
    return dict(n_real=ck.get("n_real_evaluations"),
                hv_len=len(ck.get("hv_history", []) or []),
                omp_threads=meta.get("omp_threads"),
                transport=meta.get("transport"),
                schedule=meta.get("schedule"),
                geometry=meta.get("geometry"),
                n_dep_solves=[r.get("n_dep_solves") for r in ck.get("all_raw", [])])


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h, rem = divmod(int(round(seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def fmt_min(seconds: float) -> str:
    return f"{seconds / 60.0:.1f}"


def run_campaign(args) -> None:
    wd = Path(args.workdir)
    cases = sorted((d for d in wd.glob("case_*") if d.is_dir()),
                   key=lambda d: int(re.search(r"(\d+)$", d.name).group(1)))
    if not cases:
        sys.exit(f"no case_* directories under {wd}")
    ck = load_checkpoint(args.checkpoint)

    gap_thr = args.session_gap_min * 60.0
    after, before = _parse_when(args.after), _parse_when(args.before)
    skip = set()
    if args.skip_cases:
        skip = {int(x) for x in args.skip_cases.replace(" ", "").split(",") if x}
    rows = [analyse_case(c, gap_thr, after, before) for c in cases]
    n_before_skip = len(rows)
    if skip:
        rows = [r for r in rows if r["idx"] not in skip]
        print(f"dropped {n_before_skip - len(rows)} case(s) by --skip-cases "
              f"{sorted(skip)}: their files were overwritten by a later job, so "
              f"their wall time is not measurable.")
    rows = [r for r in rows if r["t_start"] is not None]
    if not rows:
        sys.exit("no statepoints found in any case directory")

    # ---- iteration labels --------------------------------------------------
    for r in rows:
        if r["idx"] < args.n_init:
            r["stage"], r["iteration"] = "DOE", 0
        else:
            r["stage"] = "infill"
            r["iteration"] = (r["idx"] - args.n_init) // args.n_infill + 1

    # ---- gaps between consecutive evaluations --------------------------------
    # Gaps are computed in TIME order, never in case-index order. A campaign
    # normally runs its cases in index order, but a directory can be revisited
    # later (a core-proxy validation, a rescoring, a re-run after a crash), and
    # then index order and time order disagree. Using index order there would
    # subtract a later timestamp from an earlier one and yield negative time.
    chrono = sorted(rows, key=lambda r: r["t_start"])
    out_of_order = [r["case"] for a, r in zip(rows, rows[1:])
                    if r["t_start"] < a["t_last_end"]]
    for prev, cur in zip(chrono, chrono[1:]):
        g = (cur["t_start"] - prev["t_last_end"]).total_seconds()
        cur["gap_before_s"] = g
        boundary = cur["iteration"] != prev["iteration"]
        cur["gap_kind"] = ("session_break" if g > gap_thr else
                           "optimiser" if boundary else "intra")
    chrono[0]["gap_before_s"], chrono[0]["gap_kind"] = 0.0, "start"

    # ---- aggregates -----------------------------------------------------------
    def mean(xs):
        xs = list(xs)
        return float(np.mean(xs)) if xs else float("nan")

    doe = [r for r in rows if r["stage"] == "DOE"]
    inf = [r for r in rows if r["stage"] == "infill"]
    opt_gaps = [r["gap_before_s"] for r in rows if r["gap_kind"] == "optimiser"]
    intra_gaps = [r["gap_before_s"] for r in rows if r["gap_kind"] == "intra"]
    breaks = [r["gap_before_s"] for r in rows if r["gap_kind"] == "session_break"]

    eval_wall = sum(r["wall_s"] for r in rows)
    active = eval_wall + sum(opt_gaps) + sum(intra_gaps)
    t_first = min(r["t_start"] for r in rows)
    t_last = max(r["t_last_end"] for r in rows)
    calendar = (t_last - t_first).total_seconds()

    summary = dict(
        campaign=args.campaign,
        workdir=str(wd.resolve()),
        host_reported=args.host or "UNKNOWN (pass --host)",
        host_running_this_script=platform.node(),
        omp_threads=ck.get("omp_threads") or args.threads or "UNKNOWN",
        transport_assembly=ck.get("transport") or dict(
            particles=rows[0]["particles_asm"], batches=rows[0]["batches_asm"]),
        transport_core=(dict(particles=rows[0]["particles_core"],
                             batches=rows[0]["batches_core"],
                             inactive=rows[0]["inactive_core"])
                        if rows[0]["particles_core"] else None),
        run_path_in_statepoints=rows[0]["path_attr"],
        n_cases_measured=len(rows),
        n_real_in_checkpoint=ck.get("n_real"),
        n_doe=len(doe), n_infill=len(inf),
        n_iterations=len({r["iteration"] for r in inf}),
        n_solves_total=int(sum(r["n_solves"] for r in rows)),
        n_dep_solves_total=int(sum(r["n_dep_solves"] for r in rows)),
        mean_dep_solves_per_eval=mean(r["n_dep_solves"] for r in rows),
        statepoints_without_runtime=int(sum(r["runtime_missing"] for r in rows)),
        first_start=t_first.isoformat(sep=" "),
        last_end=t_last.isoformat(sep=" "),
        out_of_order_cases=list(out_of_order),
        skipped_cases=sorted(skip),
        n_cases_on_disk=int(n_before_skip),
        # --- per evaluation ---
        mean_eval_wall_s=mean(r["wall_s"] for r in rows),
        sd_eval_wall_s=float(np.std([r["wall_s"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        min_eval_wall_s=min(r["wall_s"] for r in rows),
        max_eval_wall_s=max(r["wall_s"] for r in rows),
        mean_eval_wall_doe_s=mean(r["wall_s"] for r in doe),
        mean_eval_wall_infill_s=mean(r["wall_s"] for r in inf),
        # --- per phase inside an evaluation (means over evaluations) ---
        mean_asm_bol_s=mean(r["asm_bol_s"] for r in rows),
        mean_core_bol_s=mean(r["core_bol_s"] for r in rows
                             if r["core_bol_s"] > 0),
        n_cases_with_core=int(sum(1 for r in rows if r["core_bol_s"] > 0)),
        mean_dep_transport_s=mean(r["dep_transport_s"] for r in rows),
        mean_overhead_s=mean(r["overhead_s"] for r in rows),
        mean_solve_s_per_dep_solve=(sum(r["dep_transport_s"] for r in rows)
                                    / max(sum(r["n_dep_solves"] for r in rows), 1)),
        # --- optimiser phases (between iterations) ---
        mean_optimiser_gap_s=mean(opt_gaps),
        n_optimiser_gaps=len(opt_gaps),
        mean_intra_gap_s=mean(intra_gaps),
        # --- totals ---
        total_eval_wall_s=eval_wall,
        total_optimiser_s=sum(opt_gaps),
        total_intra_gap_s=sum(intra_gaps),
        total_active_wall_s=active,
        total_session_breaks_s=sum(breaks),
        n_session_breaks=len(breaks),
        calendar_span_s=calendar,
        total_doe_wall_s=sum(r["wall_s"] for r in doe),
        total_infill_wall_s=sum(r["wall_s"] for r in inf),
        core_hours=((active / 3600.0) * float(ck.get("omp_threads") or args.threads)
                    if (ck.get("omp_threads") or args.threads) else None),
        # --- compute that was spent and then discarded (lower bound) ---
        n_mixed_cases=int(sum(1 for r in rows if r["mixed"])),
        stale_solves=int(sum(r["n_stale_solves"] for r in rows)),
        stale_solve_s=float(sum(r["stale_solve_s"] for r in rows)),
    )

    # ---- write CSV ------------------------------------------------------------
    fields = ["case", "idx", "stage", "iteration", "t_start", "t_last_end",
              "wall_s", "solve_s", "overhead_s", "asm_bol_s", "core_bol_s",
              "dep_transport_s", "n_dep_solves", "n_dep_chunks", "n_solves",
              "gap_before_s", "gap_kind", "runtime_missing", "mixed",
              "n_stale_solves", "stale_solve_s"]
    with open(f"{args.out}_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["t_start"] = r["t_start"].isoformat(sep=" ")
            rr["t_last_end"] = r["t_last_end"].isoformat(sep=" ")
            w.writerow(rr)

    with open(f"{args.out}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    write_campaign_tex(summary, f"{args.out}_table.tex")
    print_summary(summary)

    # ---- sanity warnings ------------------------------------------------------
    if ck.get("n_real") and ck["n_real"] != len(rows):
        print(f"\nWARNING: checkpoint says {ck['n_real']} evaluations, "
              f"{len(rows)} case directories found. Partial tree, or a block "
              f"ran in another workdir.")
    if summary["statepoints_without_runtime"]:
        print(f"\nWARNING: {summary['statepoints_without_runtime']} statepoints "
              f"have no /runtime/ data. Their solve time counts as zero and "
              f"shows up in 'overhead' instead.")
    if summary["out_of_order_cases"]:
        n = len(summary["out_of_order_cases"])
        print(f"\nWARNING: {n} case(s) start BEFORE the previous case index "
              f"finished, so this directory is not a single chronological run. "
              f"Files from a later job (core validation, rescoring, a re-run) "
              f"were written into it. Gaps and the active wall time were "
              f"computed in time order, but the per-phase means still mix both "
              f"jobs. Restrict the window with --after / --before to separate "
              f"them. First few: {summary['out_of_order_cases'][:5]}")
    if summary["n_mixed_cases"]:
        print(f"\nNOTE: {summary['n_mixed_cases']} case directories hold leftovers "
              f"of an earlier, discarded run ({summary['stale_solves']} stale solves, "
              f"{fmt_hms(summary['stale_solve_s'])} of solve time, a LOWER bound on the "
              f"discarded compute because overwritten files are invisible).")
    if breaks:
        print(f"\nNOTE: {len(breaks)} gap(s) longer than {args.session_gap_min} min "
              f"were treated as idle time (blocks run in separate sessions). "
              f"They are excluded from the active wall time.")


def print_summary(s: dict) -> None:
    print(f"\n=== {s['campaign']} ===")
    print(f"host (as reported by you): {s['host_reported']}   threads: {s['omp_threads']}")
    print(f"assembly transport: {s['transport_assembly']}   core: {s['transport_core']}")
    if s.get("skipped_cases"):
        print(f"cases on disk: {s['n_cases_on_disk']}, "
              f"dropped {len(s['skipped_cases'])} as unmeasurable "
              f"{s['skipped_cases']}")
    print(f"evaluations measured: {s['n_cases_measured']} "
          f"(DOE {s['n_doe']}, infill {s['n_infill']} in {s['n_iterations']} iterations)")
    print(f"transport solves: {s['n_solves_total']} "
          f"({s['mean_dep_solves_per_eval']:.1f} depletion solves per evaluation)")
    print(f"mean evaluation wall time : {fmt_min(s['mean_eval_wall_s'])} min "
          f"(sd {fmt_min(s['sd_eval_wall_s'])}, "
          f"min {fmt_min(s['min_eval_wall_s'])}, max {fmt_min(s['max_eval_wall_s'])})")
    print(f"   assembly BOL solve     : {s['mean_asm_bol_s']:.0f} s")
    ncore = s.get("n_cases_with_core", 0)
    if ncore:
        print(f"   core BOL solve         : {s['mean_core_bol_s']:.0f} s "
              f"(over the {ncore} case(s) that have one)")
    else:
        print("   core BOL solve         : none in this tree")
    print(f"   depletion transport    : {fmt_min(s['mean_dep_transport_s'])} min "
          f"({s['mean_solve_s_per_dep_solve']:.0f} s per solve)")
    print(f"   non-transport overhead : {fmt_min(s['mean_overhead_s'])} min")
    print(f"mean optimiser time per iteration (GP + NSGA-II + acquisition + HV + "
          f"checkpoint): {fmt_min(s['mean_optimiser_gap_s'])} min over "
          f"{s['n_optimiser_gaps']} boundaries")
    print(f"DOE phase total    : {fmt_hms(s['total_doe_wall_s'])}")
    print(f"infill phase total : {fmt_hms(s['total_infill_wall_s'])}")
    ch = (f"{s['core_hours']:.0f} core-hours" if s['core_hours'] is not None
          else "core-hours unknown (pass --threads or --checkpoint)")
    print(f"ACTIVE wall time   : {fmt_hms(s['total_active_wall_s'])}  "
          f"(= {s['total_active_wall_s']/3600:.1f} h, {ch})")
    print(f"calendar span      : {fmt_hms(s['calendar_span_s'])} "
          f"incl. {s['n_session_breaks']} idle gap(s) totalling "
          f"{fmt_hms(s['total_session_breaks_s'])}")
    print(f"first start {s['first_start']}   last end {s['last_end']}")
    if s.get('stale_solves'):
        print(f"discarded compute  : >= {fmt_hms(s['stale_solve_s'])} in {s['stale_solves']} stale solves")


# --------------------------------------------------------------------------- #
# LaTeX fragment                                                              #
# --------------------------------------------------------------------------- #
def write_campaign_tex(s: dict, path: str) -> None:
    core_row = ""
    if s["transport_core"]:
        core_row = (f"    core BOL solve & {s['mean_core_bol_s']:.0f}\\,s "
                    f"& per evaluation \\\\\n")
    ta = s["transport_assembly"] or {}
    tex = f"""% auto-generated by campaign_timing.py -- {s['campaign']}
% host: {s['host_reported']} | threads: {s['omp_threads']} | workdir: {s['workdir']}
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Computational budget of {s['campaign']}. Wall times are
  reconstructed from the statepoint timestamps of the serial evaluation
  chain. Idle time between blocks is excluded.}}
  \\label{{tab:compute-{s['campaign'].lower().replace(' ', '-')}}}
  \\begin{{tabular}}{{lrl}}
    \\toprule
    Quantity & Value & Basis \\\\
    \\midrule
    Host & {s['host_reported']} & {s['omp_threads']} OpenMP threads \\\\
    Assembly fidelity & ${ta.get('particles', '?')}\\times{ta.get('batches', '?')}$ & particles $\\times$ batches \\\\
    Evaluations measured & {s['n_cases_measured']} of {s['n_cases_on_disk']} & {s['n_doe']} DOE + {s['n_infill']} infill ({s['n_iterations']} iterations) \\\\
    Transport solves & {s['n_solves_total']} & {s['mean_dep_solves_per_eval']:.1f} depletion solves per evaluation \\\\
    \\midrule
    Mean evaluation & {fmt_min(s['mean_eval_wall_s'])}\\,min & s.d.\\ {fmt_min(s['sd_eval_wall_s'])}\\,min \\\\
    assembly BOL solve & {s['mean_asm_bol_s']:.0f}\\,s & per evaluation \\\\
{core_row}    depletion transport & {fmt_min(s['mean_dep_transport_s'])}\\,min & {s['mean_solve_s_per_dep_solve']:.0f}\\,s per solve \\\\
    non-transport overhead & {fmt_min(s['mean_overhead_s'])}\\,min & Bateman solves, material updates, I/O \\\\
    Optimiser per iteration & {fmt_min(s['mean_optimiser_gap_s'])}\\,min & GP fit, NSGA-II, acquisition, HV, checkpoint \\\\
    \\midrule
    DOE phase & {fmt_hms(s['total_doe_wall_s'])} & h:mm:ss \\\\
    Infill phase & {fmt_hms(s['total_infill_wall_s'])} & h:mm:ss \\\\
    Total active wall time & {fmt_hms(s['total_active_wall_s'])} & {(f"{s['core_hours']:.0f} core-hours" if s['core_hours'] is not None else "core-hours: threads unknown")} \\\\
    Calendar span & {fmt_hms(s['calendar_span_s'])} & {s['n_session_breaks']} idle gaps excluded \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w") as f:
        f.write(tex)


# --------------------------------------------------------------------------- #
# tree mode: any directory of statepoints (rescoring, confirmation, sweeps)   #
# --------------------------------------------------------------------------- #
def run_tree(args) -> None:
    wd = Path(args.workdir)
    files = sorted(list(wd.rglob("statepoint.*.h5")) +
                   list(wd.rglob("openmc_simulation_n*.h5")))
    if not files:
        sys.exit(f"no statepoints under {wd}")
    recs = [read_statepoint(p) for p in files]
    # honour the same time window as campaign mode, so a directory shared by
    # two jobs can be split into its parts
    after, before = _parse_when(args.after), _parse_when(args.before)
    n_all = len(recs)
    if after is not None:
        recs = [r for r in recs if r["end"] >= after]
    if before is not None:
        recs = [r for r in recs if r["end"] <= before]
    if not recs:
        sys.exit(f"no statepoints under {wd} inside the requested time window")
    if len(recs) != n_all:
        print(f"time window keeps {len(recs)} of {n_all} statepoints "
              f"(after={args.after}, before={args.before})")
    recs.sort(key=lambda r: r["end"])
    by_fid: dict[tuple, dict] = {}
    for r in recs:
        key = (r["n_particles"], r["n_batches"], r["n_inactive"])
        d = by_fid.setdefault(key, dict(n=0, sim_s=0.0, init_s=0.0, transport_s=0.0))
        d["n"] += 1
        d["sim_s"] += r["sim_s"]
        d["init_s"] += r["init_s"]
        d["transport_s"] += r["transport_s"]
    starts = [r["end"] - timedelta(seconds=r["wall_s"]) for r in recs]
    gaps = [(b - a["end"]).total_seconds() for a, b in zip(recs, starts[1:])]
    thr = args.session_gap_min * 60.0
    idle = sum(g for g in gaps if g > thr)
    calendar = (recs[-1]["end"] - starts[0]).total_seconds()
    active = calendar - idle
    summary = dict(
        campaign=args.campaign, workdir=str(wd.resolve()),
        window_after=args.after, window_before=args.before,
        n_statepoints_in_tree=int(n_all),
        host_reported=args.host or "UNKNOWN (pass --host)",
        omp_threads=args.threads or "UNKNOWN",
        n_statepoints=len(recs),
        by_fidelity={f"{k[0]}x{k[1]}(i{k[2]})": v for k, v in by_fid.items()},
        total_in_solve_s=sum(r["wall_s"] for r in recs),
        total_active_wall_s=active, calendar_span_s=calendar,
        idle_s=idle, n_idle_gaps=sum(1 for g in gaps if g > thr),
        first_start=starts[0].isoformat(sep=" "),
        last_end=recs[-1]["end"].isoformat(sep=" "),
        statepoints_without_runtime=sum(1 for r in recs if not r["runtime_ok"]),
    )
    with open(f"{args.out}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(f"{args.out}_statepoints.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "end", "n_particles", "n_batches", "n_inactive",
                    "init_s", "sim_s", "transport_s"])
        for r in recs:
            w.writerow([r["file"], r["end"].isoformat(sep=" "), r["n_particles"],
                        r["n_batches"], r["n_inactive"], f"{r['init_s']:.1f}",
                        f"{r['sim_s']:.1f}", f"{r['transport_s']:.1f}"])
    print(f"\n=== {summary['campaign']} (tree mode) ===")
    print(f"statepoints: {summary['n_statepoints']}")
    for k, v in summary["by_fidelity"].items():
        print(f"  {k:>22s}: {v['n']:4d} solves, {v['sim_s']/3600:.2f} h in simulation, "
              f"{v['sim_s']/max(v['n'],1):.0f} s per solve")
    print(f"in-solve total : {fmt_hms(summary['total_in_solve_s'])}")
    print(f"active wall    : {fmt_hms(summary['total_active_wall_s'])}  "
          f"(calendar {fmt_hms(summary['calendar_span_s'])}, "
          f"{summary['n_idle_gaps']} idle gaps = {fmt_hms(summary['idle_s'])})")
    if summary["statepoints_without_runtime"]:
        print(f"WARNING: {summary['statepoints_without_runtime']} statepoints without /runtime/")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--n-init", type=int, default=36)
    ap.add_argument("--n-infill", type=int, default=6)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--skip-cases", default=None,
                    help="comma-separated case INDICES to drop entirely, e.g. "
                         "'0,1'. Use for cases whose files a later job "
                         "overwrote, so the campaign mean is taken over the "
                         "cases that survived instead of over corrupted ones. "
                         "The count of dropped cases is reported.")
    ap.add_argument("--after", default=None,
                    help="ignore statepoints written BEFORE this instant, "
                         "'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'. Excludes an "
                         "earlier job sharing the directory.")
    ap.add_argument("--before", default=None,
                    help="ignore statepoints written AFTER this instant, same "
                         "format. Excludes a later job, e.g. a core-proxy "
                         "validation written into an old campaign tree.")
    ap.add_argument("--session-gap-min", type=float, default=30.0)
    ap.add_argument("--mode", choices=("campaign", "tree"), default="campaign")
    ap.add_argument("--out", default="timing")
    args = ap.parse_args()
    if args.mode == "tree":
        run_tree(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
