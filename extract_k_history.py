#!/usr/bin/env python3
# =====================================================================
#  extract_k_history.py   (revision 2)
#
#  Recovers the multiplication-factor trajectory k_inf(B) of a Campaign 6
#  evaluation from the depletion result files the campaign already wrote.
#  NO transport is run. Nothing is re-simulated.
#
#  WHY REVISION 2 EXISTS
#  ---------------------
#  Revision 1 tried to infer, from each chunk file, how many of its states
#  were newly computed. That is NOT recoverable information. A restarted
#  OpenMC chunk writes a file of CUMULATIVE length whose leading entries
#  are unpopulated slots rather than the previous k values, so neither a
#  length heuristic nor a prefix comparison can separate old from new.
#
#  The production code never had this problem, because it takes
#
#      k_hist.extend(kvals[-len(steps_mwd_kg):])
#
#  knowing len(steps) in advance. This revision does the same: it rebuilds
#  the chunk schedule deterministically from the checkpoint metadata and
#  the archived n_dep_solves, then takes the last len(steps) entries of
#  each chunk. Nothing is inferred from file lengths.
#
#  WHAT IS RECOVERED, AND WHAT IS NOT
#  ----------------------------------
#  Recovered exactly : k_inf(B) of the REFLECTIVE SINGLE ASSEMBLY, the
#                      quantity the Route B end-of-cycle criterion acts on.
#  NOT recovered     : k_eff(B) of the 32-assembly core. The core was
#                      solved once per evaluation, at beginning of life
#                      only (_bol_core_peaking -> "keff_core_bol"). No
#                      depleted core solve exists anywhere on disk.
#  Estimated         : a core proxy k_eff_core(B) ~ k_inf(B) * r, with
#                      r = keff_core_bol / k_bol measured for that same
#                      design at beginning of life. This assumes a
#                      burnup-independent leakage factor, which is the
#                      SAME assumption Route B already makes and which
#                      validate_ktarget_burnup.py is designed to test.
#                      It is labelled as a proxy in every output file.
#
#  SELF-CHECKS PER DESIGN (a design is skipped if any fails)
#  --------------------------------------------------------
#    1. recovered k[0] equals the archived k_bol            (case alignment)
#    2. number of dep_* directories equals the planned count (schedule)
#    3. every recovered k lies in a physical range           (no empty slots)
#    4. for a censored design, the recovered final burnup equals the
#       archived bu_eoc_mwd_kg                               (burnup axis)
#
#  USAGE
#  -----
#    lab python extract_k_history.py \
#        --checkpoint out_c6/optimization_checkpoint.json --all --out kh_c6
#
#    lab python extract_k_history.py \
#        --checkpoint out_c6/optimization_checkpoint.json --designs 81 --diagnose
# =====================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------- #
# 0. ENVIRONMENT CHECK. Nothing below runs until this passes.            #
#    No transport is executed, so OPENMC_CROSS_SECTIONS and              #
#    OPENMC_CHAIN_FILE are NOT required. Only openmc.deplete and numpy.  #
# --------------------------------------------------------------------- #
def environment_check(verbose: bool = True) -> None:
    problems: list[str] = []

    try:
        import numpy  # noqa: F401
    except Exception as exc:
        problems.append(f"numpy is not importable: {exc}")

    version = None
    try:
        import openmc
        import openmc.deplete  # noqa: F401
        version = getattr(openmc, "__version__", "unknown")
    except Exception as exc:
        problems.append(
            f"openmc is not importable: {exc}. "
            "You are probably in the (base) conda environment. "
            "Run `conda activate openmc-env` on wks720, or prefix with "
            "`lab python` on AWS so the labgene-openmc container provides it.")

    if problems:
        print("ENVIRONMENT CHECK FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    if verbose:
        print(f"env ok | openmc {version} | python {sys.version.split()[0]}")
        print("        no transport is run, cross sections and chain "
              "are not needed")
        xs = os.environ.get("OPENMC_CROSS_SECTIONS")
        if xs:
            print(f"        OPENMC_CROSS_SECTIONS is set ({xs}), ignored here")


# --------------------------------------------------------------------- #
# 1. the chunk schedule, rebuilt with the EXACT rule of the production   #
#    loop in OpenMCEvaluator._cycle_length                               #
# --------------------------------------------------------------------- #
def next_steps(n_wanted: int, cur_bu: float,
               dep_step: float, max_burnup: float) -> list[float]:
    """Reproduce the step generator of OpenMCEvaluator._cycle_length.

        steps = []
        for _ in range(self.chunk_steps):
            s = min(self.dep_step, remaining - sum(steps))
            if s <= 1e-9:
                break
            steps.append(s)

    with remaining = self.max_burnup - bu_hist[-1].
    """
    steps: list[float] = []
    remaining = max_burnup - cur_bu
    for _ in range(int(n_wanted)):
        s = min(dep_step, remaining - sum(steps))
        if s <= 1.0e-9:
            break
        steps.append(float(s))
    return steps


def plan_chunks(n_total: int, bol_steps: list[float], dep_step: float,
                chunk_steps: int, max_burnup: float) -> list[list[float]]:
    """Rebuild the per-chunk step lists of an evaluation that ended with
    `n_total` transport states, which the archive records as n_dep_solves.

    The production loop always runs a FULL chunk of `chunk_steps` marching
    steps, shortened only by the burnup cap, and then tests the crossing.
    The state count is therefore determined and nothing here is a guess.
    """
    chunks = [list(bol_steps)]
    n_states = 1 + len(bol_steps)          # the BOL state plus its block
    bu = float(sum(bol_steps))

    if n_total < n_states:
        raise RuntimeError(
            f"the archive records {n_total} transport states but the "
            f"beginning-of-life block alone has {n_states}. The checkpoint "
            f"schedule does not describe this evaluation.")

    while n_states < n_total:
        steps = next_steps(chunk_steps, bu, dep_step, max_burnup)
        if not steps:
            raise RuntimeError(
                f"the archive records {n_total} states but the schedule is "
                f"exhausted at {n_states} (burnup {bu:g} of a "
                f"{max_burnup:g} MWd/kgHM cap).")
        if n_states + len(steps) > n_total:
            raise RuntimeError(
                f"chunk {len(chunks)} would bring the count to "
                f"{n_states + len(steps)} against an archived {n_total}. The "
                f"schedule in the checkpoint meta (bol_steps={bol_steps}, "
                f"dep_step={dep_step}, chunk_steps={chunk_steps}, "
                f"max_burnup={max_burnup}) is not the one this evaluation "
                f"ran with.")
        chunks.append(steps)
        n_states += len(steps)
        bu += sum(steps)

    return chunks


def read_chunk_keff(h5_path: Path) -> tuple[list[float], list[float]]:
    """Return (k, sigma_k) of one depletion_results.h5, in step order.

    Same call as the production code, openmc_evaluator.py:
        _t, karr = results.get_keff()
        kvals = [float(v) for v in karr[:, 0]]
    Column 1 of the same array is the one-sigma uncertainty.
    """
    import openmc.deplete
    results = openmc.deplete.Results(str(h5_path))
    _t, karr = results.get_keff()
    return ([float(v) for v in karr[:, 0]],
            [float(v) for v in karr[:, 1]])


K_LOW, K_HIGH = 0.05, 3.0          # physical band, catches unpopulated slots


def recover_history(case_dir: Path, plan: list[list[float]],
                    diagnose: bool = False) -> dict:
    """Stitch every dep_* chunk of one case into (bu_hist, k_hist).

    For each chunk the LAST len(steps) entries are the newly computed
    end-of-step states, which is exactly what the production code takes.
    A restarted chunk file may be longer than that, with unpopulated
    leading slots. Those slots are never read.
    """
    chunks = sorted(case_dir.glob("dep_*/depletion_results.h5"))
    if not chunks:
        raise FileNotFoundError(
            f"{case_dir}: no dep_*/depletion_results.h5 found. The run "
            f"directory of this evaluation was deleted or never written.")

    if len(chunks) != len(plan):
        raise RuntimeError(
            f"{case_dir}: {len(chunks)} chunk directories on disk against "
            f"{len(plan)} planned from the archived state count. The run "
            f"directory and the archive entry do not describe the same "
            f"evaluation.")

    bu_hist: list[float] = [0.0]
    k_hist: list[float] = []
    s_hist: list[float] = []
    diag: list[dict] = []

    for i, (h5, steps) in enumerate(zip(chunks, plan)):
        kvals, svals = read_chunk_keff(h5)

        if i == 0:
            if len(kvals) != len(steps) + 1:
                raise RuntimeError(
                    f"{h5}: first chunk holds {len(kvals)} states for "
                    f"{len(steps)} steps, expected {len(steps) + 1} "
                    f"including the beginning-of-life state.")
            take_k, take_s = kvals, svals
        else:
            if len(kvals) < len(steps):
                raise RuntimeError(
                    f"{h5}: {len(kvals)} states in the file but {len(steps)} "
                    f"new steps expected. File truncated.")
            take_k, take_s = kvals[-len(steps):], svals[-len(steps):]

        for v in take_k:
            if not (K_LOW < v < K_HIGH):
                raise RuntimeError(
                    f"{h5}: recovered a non-physical k = {v!r}. An "
                    f"unpopulated slot was read, so the stitching rule is "
                    f"wrong for this file.")

        if diagnose:
            diag.append({"chunk": i, "file": str(h5),
                         "states_in_file": len(kvals),
                         "new_steps_taken": len(steps),
                         "leading_slots_ignored": len(kvals) - len(take_k),
                         "steps_mwd_kg": steps})

        k_hist.extend(take_k)
        s_hist.extend(take_s)
        for s in steps:
            bu_hist.append(bu_hist[-1] + s)

    if len(k_hist) != len(bu_hist):
        raise RuntimeError(
            f"{case_dir}: {len(k_hist)} k values against {len(bu_hist)} "
            f"burnup points. Bookkeeping out of sync.")

    return {"bu": bu_hist, "k": k_hist, "k_sigma": s_hist,
            "n_chunks": len(chunks), "diagnostics": diag}


# --------------------------------------------------------------------- #
# 2. selection of designs from the checkpoint                            #
# --------------------------------------------------------------------- #
def is_feasible(rec: dict, cnames: list[str], tol: float = 0.0) -> bool:
    return all(float(rec.get(c, 1.0)) <= tol for c in cnames)


def pareto_front(records: list[dict], idx: list[int]) -> list[int]:
    """Non-dominated set on (maximise cycle_length, minimise peaking)."""
    front = []
    for i in idx:
        a = records[i]
        dominated = False
        for j in idx:
            if i == j:
                continue
            b = records[j]
            better_eq = (float(b["cycle_length"]) >= float(a["cycle_length"])
                         and float(b["peaking"]) <= float(a["peaking"]))
            strictly = (float(b["cycle_length"]) > float(a["cycle_length"])
                        or float(b["peaking"]) < float(a["peaking"]))
            if better_eq and strictly:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda i: float(records[i]["cycle_length"]))


# --------------------------------------------------------------------- #
# 3. main                                                                #
# --------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recover k_inf(B) trajectories from the stored "
                    "Campaign 6 depletion files. Runs no transport.")
    ap.add_argument("--checkpoint", required=True,
                    help="optimization_checkpoint.json of the campaign")
    ap.add_argument("--workdir", default=None,
                    help="the --workdir the campaign used, holding case_NNNN "
                         "(default: the value recorded in the checkpoint meta)")
    ap.add_argument("--designs", type=int, nargs="+", default=None,
                    help="archive positions to extract, e.g. --designs 81 101")
    ap.add_argument("--front", action="store_true",
                    help="extract every feasible non-dominated design")
    ap.add_argument("--all", action="store_true",
                    help="extract every evaluation in the archive")
    ap.add_argument("--out", default="kh_out",
                    help="output directory (default: kh_out)")
    ap.add_argument("--k-max", type=float, default=None,
                    help="reactivity ceiling to draw and to test the peak "
                         "against (default: the archived k_max_used)")
    ap.add_argument("--no-plot", action="store_true",
                    help="write the data only, skip the figure")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first failure instead of skipping it")
    ap.add_argument("--diagnose", action="store_true",
                    help="print the per-chunk file layout of each design, "
                         "for auditing the stitching rule")
    args = ap.parse_args()

    environment_check()

    import numpy as np

    ckpt = json.loads(Path(args.checkpoint).read_text())
    records = ckpt["all_raw"]
    cnames = list(ckpt["constraint_names"])
    meta = ckpt.get("meta", {})
    sched = meta.get("schedule", {})

    missing = [k for k in ("bol_steps", "dep_step", "chunk_steps", "max_burnup")
               if k not in sched]
    if missing:
        print(f"!! the checkpoint meta carries no schedule keys {missing}.",
              file=sys.stderr)
        sys.exit(3)

    if args.workdir is None:
        args.workdir = str(meta.get("workdir", "openmc_runs"))
        print(f"workdir        : {args.workdir} (from the checkpoint meta)")

    bol_steps = [float(s) for s in sched["bol_steps"]]
    dep_step = float(sched["dep_step"])
    chunk_steps = int(sched["chunk_steps"])
    max_burnup = float(sched["max_burnup"])

    print(f"archive        : {len(records)} evaluations")
    print(f"constraints    : {cnames}")
    print(f"schedule       : BOL {bol_steps} then {dep_step} MWd/kgHM "
          f"({chunk_steps}/chunk), cap {max_burnup} MWd/kgHM")

    feas = [i for i, r in enumerate(records) if is_feasible(r, cnames)]
    if args.all:
        sel = list(range(len(records)))
    elif args.front:
        sel = pareto_front(records, feas)
    elif args.designs:
        sel = list(args.designs)
    else:
        ap.error("choose --designs, --front or --all")
    print(f"feasible       : {len(feas)}")
    print(f"selected       : {len(sel)} design(s)\n")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)

    summary: list[dict] = []
    trajectories: dict[str, dict] = {}
    failures: list[tuple[int, str]] = []

    for pos in sel:
        if pos >= len(records):
            failures.append((pos, "position beyond the archive"))
            continue
        rec = records[pos]
        case_dir = workdir / f"case_{pos:04d}"
        n_arch = int(rec["n_dep_solves"])

        try:
            plan = plan_chunks(n_arch, bol_steps, dep_step,
                               chunk_steps, max_burnup)
            h = recover_history(case_dir, plan, diagnose=args.diagnose)
        except Exception as exc:
            failures.append((pos, str(exc)))
            if args.strict:
                raise
            print(f"  [{pos:>3}] SKIPPED: {exc}")
            continue

        bu = np.array(h["bu"])
        k = np.array(h["k"])
        sig = np.array(h["k_sigma"])

        # ---- self-checks --------------------------------------------- #
        k_bol_arch = float(rec["k_bol"])
        d_k = abs(float(k[0]) - k_bol_arch)
        if d_k >= 1.0e-6:
            msg = (f"case alignment failed: recovered k[0]={k[0]:.6f} "
                   f"against archived k_bol={k_bol_arch:.6f} "
                   f"(d={d_k:.2e}). case_{pos:04d} is not this entry.")
            failures.append((pos, msg))
            if args.strict:
                raise RuntimeError(msg)
            print(f"  [{pos:>3}] SKIPPED: {msg}")
            continue

        bu_eoc = float(rec["bu_eoc_mwd_kg"])
        censored = bool(rec.get("censored", False))
        if censored and abs(float(bu[-1]) - bu_eoc) > 1.0e-6:
            msg = (f"burnup axis failed: this design is censored, so the "
                   f"archived bu_eoc={bu_eoc:g} should equal the recovered "
                   f"final burnup {bu[-1]:g} MWd/kgHM.")
            failures.append((pos, msg))
            if args.strict:
                raise RuntimeError(msg)
            print(f"  [{pos:>3}] SKIPPED: {msg}")
            continue

        if args.diagnose:
            print(f"  [{pos:>3}] chunk layout, {n_arch} archived states:")
            for d in h["diagnostics"]:
                print(f"        dep_{d['chunk']:02d}  file holds "
                      f"{d['states_in_file']:>3} states, took the last "
                      f"{d['new_steps_taken']}, ignored "
                      f"{d['leading_slots_ignored']} leading slot(s), "
                      f"steps {d['steps_mwd_kg']}")

        # ---- quantities ---------------------------------------------- #
        k_core_bol = float(rec["keff_core_bol"])
        ratio = k_core_bol / k_bol_arch          # BOL leakage/loading factor
        k_core_proxy = k * ratio

        # peak on the OPERATIONAL trajectory, which excludes the xenon-free
        # beginning-of-life point, exactly as the production peak detector
        # does (openmc_evaluator.py: k_op = k_hist[1:])
        j = int(np.argmax(k[1:])) + 1 if len(k) > 1 else 0
        k_peak = float(k[j])
        bu_peak = float(bu[j])
        hump = k_peak - float(k[0])

        k_max = args.k_max if args.k_max is not None \
            else float(rec.get("k_max_used", 1.35))

        row = {
            "pos": pos,
            "enrich_inner": float(rec["enrich_inner"]),
            "enrich_outer": float(rec["enrich_outer"]),
            "gd_wt": float(rec["gd_wt"]),
            "gd_pins_used": int(rec.get("gd_pins_used", rec.get("gd_pins", 0))),
            "pitch": float(rec["pitch"]),
            "refl_thick": float(rec["refl_thick"]),
            "feasible": is_feasible(rec, cnames),
            "efpd": float(rec["cycle_length"]),
            "fdh_core": float(rec["peaking"]),
            "censored": censored,
            "bu_eoc": bu_eoc,
            "k_target": float(rec["k_target"]),
            "k_bol_asm": float(k[0]),
            "k_core_bol": k_core_bol,
            "bol_ratio_core_over_asm": ratio,
            "k_peak_asm": k_peak,
            "bu_at_peak": bu_peak,
            "hump_dk_asm": hump,
            "hump_pcm_asm": 1.0e5 * hump,
            "k_core_peak_proxy": k_peak * ratio,
            "k_max_used": k_max,
            "core_peak_exceeds_kmax": bool(k_peak * ratio > k_max),
            "core_peak_margin_pcm": 1.0e5 * (k_peak * ratio - k_max),
            "n_states": int(len(k)),
            "n_chunks": h["n_chunks"],
        }
        summary.append(row)
        trajectories[str(pos)] = {
            "bu_mwd_kghm": [float(v) for v in bu],
            "k_inf_assembly": [float(v) for v in k],
            "k_inf_sigma": [float(v) for v in sig],
            "k_core_proxy": [float(v) for v in k_core_proxy],
            "k_target": float(rec["k_target"]),
            "bu_eoc": bu_eoc,
            "feasible": is_feasible(rec, cnames),
            "proxy_note": ("k_core_proxy = k_inf(B) * keff_core_bol/k_bol, "
                           "a BOL-calibrated estimate assuming a "
                           "burnup-independent leakage factor. It is not a "
                           "measured core eigenvalue."),
        }
        flag = "ABOVE" if k_peak * ratio > k_max else "below"
        print(f"  [{pos:>3}] ok  {len(k):>3} states  "
              f"gd={row['gd_wt']:5.2f}wt%x{row['gd_pins_used']:>2}  "
              f"k_bol={k[0]:.5f}  k_peak={k_peak:.5f} at "
              f"{bu_peak:5.1f} MWd/kgHM  hump={1e5 * hump:+7.0f} pcm  "
              f"core proxy peak={k_peak * ratio:.5f} {flag} {k_max:.2f}")

    # ---- write ------------------------------------------------------- #
    (outdir / "k_histories.json").write_text(
        json.dumps({"trajectories": trajectories,
                    "summary": summary,
                    "failures": [{"pos": p, "reason": r} for p, r in failures],
                    "source_checkpoint": str(args.checkpoint),
                    "workdir": str(args.workdir),
                    "schedule": {"bol_steps": bol_steps,
                                 "dep_step": dep_step,
                                 "chunk_steps": chunk_steps,
                                 "max_burnup": max_burnup}},
                   indent=2))

    if summary:
        cols = ["pos", "feasible", "enrich_inner", "enrich_outer", "gd_wt",
                "gd_pins_used", "pitch", "refl_thick", "efpd", "fdh_core",
                "k_bol_asm", "k_core_bol", "k_peak_asm", "bu_at_peak",
                "hump_pcm_asm", "k_core_peak_proxy", "core_peak_margin_pcm",
                "censored"]
        with (outdir / "k_hump_summary.csv").open("w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in summary:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")

    n_ok = len(summary)
    print(f"\n{n_ok} recovered, {len(failures)} failed")
    print(f"wrote {outdir / 'k_histories.json'}")
    if summary:
        print(f"wrote {outdir / 'k_hump_summary.csv'}")

    if summary:
        exceed = [r for r in summary if r["core_peak_exceeds_kmax"]]
        exceed_feas = [r for r in exceed if r["feasible"]]
        print(f"\ncore proxy peak above k_max: {len(exceed)} of {n_ok} "
              f"({len(exceed_feas)} of them recorded FEASIBLE by the "
              f"beginning-of-life screen)")
        if exceed_feas:
            worst = sorted(exceed_feas,
                           key=lambda r: -r["core_peak_margin_pcm"])[:10]
            print("  worst feasible exceedances (position, margin in pcm, "
                  "gadolinia):")
            for r in worst:
                print(f"    pos {r['pos']:>3}  "
                      f"{r['core_peak_margin_pcm']:+8.0f} pcm  "
                      f"{r['gd_wt']:.2f} wt% x {r['gd_pins_used']} pins")

    # ---- figure ------------------------------------------------------ #
    if args.no_plot or not summary:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}), figure skipped")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    order = sorted(summary, key=lambda r: r["efpd"])
    cmap = plt.get_cmap("viridis")
    label_all = len(order) <= 16

    for n, r in enumerate(order):
        tr = trajectories[str(r["pos"])]
        c = cmap(n / max(1, len(order) - 1))
        lab = f"{r['pos']} ({r['efpd']:.0f} EFPD)" if label_all else None
        axes[0].plot(tr["bu_mwd_kghm"], tr["k_inf_assembly"],
                     color=c, lw=1.0, label=lab)
        axes[1].plot(tr["bu_mwd_kghm"], tr["k_core_proxy"], color=c, lw=1.0)

    axes[0].set_xlabel(r"Burnup $B$ [MWd/kgHM]")
    axes[0].set_ylabel(r"$k_\infty$, reflective assembly")
    axes[0].set_title("Recovered depletion trajectories")
    axes[0].grid(alpha=0.3)
    if label_all:
        axes[0].legend(fontsize=6, ncol=2, loc="best")

    k_max = order[0]["k_max_used"]
    axes[1].axhline(k_max, color="crimson", lw=1.2, ls="--",
                    label=rf"$k_{{\max}}={k_max:.2f}$")
    axes[1].set_xlabel(r"Burnup $B$ [MWd/kgHM]")
    axes[1].set_ylabel(r"$k_\mathrm{eff}$ core, BOL-calibrated proxy")
    axes[1].set_title("Core proxy against the reactivity screen")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "k_histories.pdf")
    fig.savefig(outdir / "k_histories.png", dpi=180)
    print(f"wrote {outdir / 'k_histories.pdf'} and .png")


if __name__ == "__main__":
    main()
