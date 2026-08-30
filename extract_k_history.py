#!/usr/bin/env python3
# =====================================================================
#  extract_k_history.py
#
#  Recovers the multiplication-factor trajectory k_inf(B) of a Campaign 6
#  evaluation from the depletion result files that the campaign already
#  wrote to disk. NO transport is run. Nothing is re-simulated.
#
#  WHY THIS SCRIPT EXISTS
#  ----------------------
#  OpenMCEvaluator._cycle_length (openmc_evaluator.py, branch campaign6)
#  builds the full history internally
#
#      k_hist : list[float]      # k_inf of the reflective assembly
#      bu_hist: list[float]      # cumulative burnup [MWd/kgHM]
#
#  but returns only
#
#      efpd, k_bol, k_target, censored, bu_eoc, len(k_hist)
#
#  so the archive keeps k_hist[0] (as "k_bol") and len(k_hist) (as
#  "n_dep_solves") and discards the rest. The trajectory is nevertheless
#  fully recoverable, because every chunk wrote its own
#
#      <workdir>/case_<pos>/dep_<nn>/depletion_results.h5
#
#  and no cleanup step ever deletes them.
#
#  WHAT IS RECOVERED, AND WHAT IS NOT
#  ----------------------------------
#  Recovered exactly : k_inf(B) of the REFLECTIVE SINGLE ASSEMBLY, which
#                      is the quantity the end-of-cycle criterion acts on
#                      (Route B, Equation eq:eoc-routeB).
#  NOT recovered     : k_eff(B) of the 32-assembly core. The core was
#                      solved once per evaluation, at beginning of life
#                      only (_bol_core_peaking -> "keff_core_bol"). No
#                      depleted core solve exists anywhere on disk.
#  Estimated         : a core proxy k_eff_core(B) ~ k_inf(B) * r, with
#                      r = keff_core_bol / k_bol measured for that same
#                      design at beginning of life. This assumes the
#                      leakage factor is burnup independent, which is the
#                      SAME assumption Route B already makes and which
#                      validate_ktarget_burnup.py is designed to test.
#                      It is a proxy, and it is labelled as one in every
#                      output file.
#
#  ALIGNMENT SELF-CHECK
#  --------------------
#  case_NNNN is numbered by OpenMCEvaluator.n_calls, which load_checkpoint
#  restores as len(ckpt["all_raw"]). Case index therefore equals archive
#  position, provided every block of the campaign used the same --workdir.
#  The script does not trust that. For each design it compares the
#  recovered k_hist[0] with the archived "k_bol" and the recovered number
#  of states with the archived "n_dep_solves", and refuses the design if
#  either disagrees.
#
#  USAGE
#  -----
#    python extract_k_history.py --checkpoint out_c6/optimization_checkpoint.json \
#                                --workdir openmc_runs --front --out kh_c6
#    python extract_k_history.py --checkpoint out_c6/optimization_checkpoint.json \
#                                --workdir openmc_runs --designs 81 101 71
#
#  On the AWS container the `lab` alias already mounts the repository at
#  /work, so run it exactly like any other script of the repository.
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
#    OPENMC_CHAIN_FILE are NOT required. Only the openmc.deplete reader  #
#    and numpy are.                                                      #
# --------------------------------------------------------------------- #
def environment_check(verbose: bool = True) -> None:
    problems: list[str] = []

    try:
        import numpy  # noqa: F401
    except Exception as exc:
        problems.append(f"numpy is not importable: {exc}")

    try:
        import openmc
        import openmc.deplete  # noqa: F401
        version = getattr(openmc, "__version__", "unknown")
    except Exception as exc:
        version = None
        problems.append(
            f"openmc is not importable: {exc}. "
            "You are probably in the (base) conda environment. "
            "Run `conda activate openmc-env` on wks720, or use the `lab` "
            "alias on AWS so the labgene-openmc container provides it.")

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
# 1. burnup axis, rebuilt with the EXACT rule of the production code     #
# --------------------------------------------------------------------- #
def next_steps(n_wanted: int, cur_bu: float,
               dep_step: float, max_burnup: float) -> list[float]:
    """Reproduce the step generator of OpenMCEvaluator._cycle_length.

    Production code (openmc_evaluator.py):

        steps = []
        for _ in range(self.chunk_steps):
            s = min(self.dep_step, remaining - sum(steps))
            if s <= 1e-9:
                break
            steps.append(s)

    with `remaining = self.max_burnup - bu_hist[-1]`. Here `n_wanted` is
    the number of new states actually found in the chunk file rather than
    `chunk_steps`, so a chunk truncated by the burnup cap is reproduced
    correctly.
    """
    steps: list[float] = []
    remaining = max_burnup - cur_bu
    for _ in range(int(n_wanted)):
        s = min(dep_step, remaining - sum(steps))
        if s <= 1e-9:
            break
        steps.append(float(s))
    return steps


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


def recover_history(case_dir: Path, bol_steps: list[float],
                    dep_step: float, chunk_steps: int,
                    max_burnup: float) -> dict:
    """Stitch every dep_* chunk of one case into (bu_hist, k_hist).

    The production code stitches with
        first chunk : take everything, including the beginning-of-life state
        later chunks: take the LAST len(steps) entries
    which is agnostic to whether a restarted OpenMC writes chunk-local or
    cumulative results. The same rule is applied here, with the number of
    new states inferred from the file rather than assumed.
    """
    chunks = sorted(case_dir.glob("dep_*/depletion_results.h5"))
    if not chunks:
        raise FileNotFoundError(
            f"{case_dir}: no dep_*/depletion_results.h5 found. The run "
            f"directory of this evaluation was deleted or never written.")

    bu_hist: list[float] = [0.0]
    k_hist: list[float] = []
    s_hist: list[float] = []
    cumulative: bool | None = None

    for i, h5 in enumerate(chunks):
        kvals, svals = read_chunk_keff(h5)

        if i == 0:
            n_new = len(kvals) - 1
            if n_new != len(bol_steps):
                raise RuntimeError(
                    f"{h5}: first chunk holds {n_new} steps, the checkpoint "
                    f"schedule declares bol_steps={bol_steps} "
                    f"({len(bol_steps)} steps). The checkpoint and the run "
                    f"directory do not describe the same campaign.")
            k_hist.extend(kvals)
            s_hist.extend(svals)
            for s in bol_steps:
                bu_hist.append(bu_hist[-1] + float(s))
            continue

        # Decide the restart semantics once, on the second chunk, by an
        # EXACT test rather than a length heuristic: a cumulative results
        # file repeats the whole previous history as its prefix.
        if cumulative is None:
            n_prev = len(k_hist)
            cumulative = (
                len(kvals) > n_prev
                and all(abs(a - b) < 1.0e-12
                        for a, b in zip(kvals[:n_prev], k_hist)))

        n_new = (len(kvals) - len(k_hist)) if cumulative else (len(kvals) - 1)
        if n_new <= 0:
            raise RuntimeError(
                f"{h5}: inferred {n_new} new states. Chunk stitching failed, "
                f"do not trust this design.")

        steps = next_steps(n_new, bu_hist[-1], dep_step, max_burnup)
        if len(steps) != n_new:
            raise RuntimeError(
                f"{h5}: {n_new} new states but the schedule allows only "
                f"{len(steps)} more steps before the {max_burnup} MWd/kgHM "
                f"cap. Schedule mismatch.")

        k_hist.extend(kvals[-n_new:])
        s_hist.extend(svals[-n_new:])
        for s in steps:
            bu_hist.append(bu_hist[-1] + s)

    if len(k_hist) != len(bu_hist):
        raise RuntimeError(
            f"{case_dir}: {len(k_hist)} k values against {len(bu_hist)} "
            f"burnup points. Bookkeeping out of sync.")

    return {"bu": bu_hist, "k": k_hist, "k_sigma": s_hist,
            "n_chunks": len(chunks),
            "restart_semantics": ("cumulative" if cumulative else
                                  "chunk-local" if cumulative is not None
                                  else "single-chunk")}


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
                         "(default: the value recorded in the checkpoint "
                         "meta, else openmc_runs)")
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
                         "against (default: the checkpoint's k_max_used)")
    ap.add_argument("--no-plot", action="store_true",
                    help="write the data only, skip the figure")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first design that fails the alignment "
                         "check instead of skipping it")
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
        print(f"!! the checkpoint meta carries no schedule keys {missing}. "
              f"This checkpoint predates the schedule metadata. Pass the "
              f"campaign values by editing the defaults below before "
              f"trusting the burnup axis.", file=sys.stderr)
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

    # ---- select ------------------------------------------------------ #
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
    print(f"selected       : {len(sel)} -> {sel}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)

    summary = []
    trajectories = {}
    failures = []

    for pos in sel:
        if pos >= len(records):
            failures.append((pos, "position beyond the archive"))
            continue
        rec = records[pos]
        case_dir = workdir / f"case_{pos:04d}"

        try:
            h = recover_history(case_dir, bol_steps, dep_step,
                                chunk_steps, max_burnup)
        except Exception as exc:
            failures.append((pos, str(exc)))
            if args.strict:
                raise
            print(f"  [{pos:>3}] SKIPPED: {exc}")
            continue

        bu = np.array(h["bu"])
        k = np.array(h["k"])
        sig = np.array(h["k_sigma"])

        # ---- alignment self-check ------------------------------------ #
        k_bol_arch = float(rec["k_bol"])
        n_arch = int(rec["n_dep_solves"])
        d_k = abs(float(k[0]) - k_bol_arch)
        ok_k = d_k < 1.0e-6
        ok_n = (len(k) == n_arch)
        if not (ok_k and ok_n):
            msg = (f"alignment check failed: recovered k[0]={k[0]:.6f} "
                   f"against archived k_bol={k_bol_arch:.6f} (d={d_k:.2e}), "
                   f"recovered {len(k)} states against archived {n_arch}. "
                   f"case_{pos:04d} is NOT this archive entry.")
            failures.append((pos, msg))
            if args.strict:
                raise RuntimeError(msg)
            print(f"  [{pos:>3}] SKIPPED: {msg}")
            continue

        # ---- quantities ---------------------------------------------- #
        k_core_bol = float(rec["keff_core_bol"])
        ratio = k_core_bol / k_bol_arch          # BOL leakage/loading factor
        k_core_proxy = k * ratio

        # peak on the OPERATIONAL trajectory, which excludes the
        # xenon-free beginning-of-life point, exactly as the production
        # peak detector does (openmc_evaluator.py: k_op = k_hist[1:])
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
            "gd_pins_used": int(rec.get("gd_pins_used",
                                        rec.get("gd_pins", 0))),
            "pitch": float(rec["pitch"]),
            "refl_thick": float(rec["refl_thick"]),
            "efpd": float(rec["cycle_length"]),
            "fdh_core": float(rec["peaking"]),
            "censored": bool(rec.get("censored", False)),
            "bu_eoc": float(rec["bu_eoc_mwd_kg"]),
            "k_target": float(rec["k_target"]),
            "k_bol_asm": float(k[0]),
            "k_core_bol": k_core_bol,
            "bol_ratio_core_over_asm": ratio,
            "k_peak_asm": k_peak,
            "bu_at_peak": bu_peak,
            "hump_dk_asm": hump,
            "hump_pcm_asm": 1.0e5 * hump,
            "k_core_bol_proxy_check": float(k[0]) * ratio,
            "k_core_peak_proxy": k_peak * ratio,
            "k_max_used": k_max,
            "core_peak_exceeds_kmax": bool(k_peak * ratio > k_max),
            "core_peak_margin_pcm": 1.0e5 * (k_peak * ratio - k_max),
            "n_states": int(len(k)),
            "n_chunks": h["n_chunks"],
            "restart_semantics": h["restart_semantics"],
        }
        summary.append(row)
        trajectories[str(pos)] = {
            "bu_mwd_kghm": [float(v) for v in bu],
            "k_inf_assembly": [float(v) for v in k],
            "k_inf_sigma": [float(v) for v in sig],
            "k_core_proxy": [float(v) for v in k_core_proxy],
            "k_target": float(rec["k_target"]),
            "bu_eoc": float(rec["bu_eoc_mwd_kg"]),
            "proxy_note": ("k_core_proxy = k_inf(B) * keff_core_bol/k_bol, "
                           "a BOL-calibrated estimate that assumes a "
                           "burnup-independent leakage factor. It is not a "
                           "measured core eigenvalue."),
        }
        print(f"  [{pos:>3}] ok  {len(k):>3} states  "
              f"k_bol={k[0]:.5f}  k_peak={k_peak:.5f} at "
              f"{bu_peak:5.1f} MWd/kgHM  hump={1e5*hump:+7.0f} pcm  "
              f"core peak proxy={k_peak*ratio:.5f} "
              f"({'ABOVE' if k_peak*ratio > k_max else 'below'} "
              f"k_max={k_max:.2f})")

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
        cols = ["pos", "enrich_inner", "enrich_outer", "gd_wt",
                "gd_pins_used", "pitch", "refl_thick", "efpd", "fdh_core",
                "k_bol_asm", "k_core_bol", "k_peak_asm", "bu_at_peak",
                "hump_pcm_asm", "k_core_peak_proxy", "core_peak_margin_pcm",
                "censored"]
        with (outdir / "k_hump_summary.csv").open("w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in summary:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")

    print(f"\nwrote {outdir/'k_histories.json'}")
    if summary:
        print(f"wrote {outdir/'k_hump_summary.csv'}")
    if failures:
        print(f"\n{len(failures)} design(s) failed, see the failures block "
              f"of the JSON")

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

    for n, r in enumerate(order):
        tr = trajectories[str(r["pos"])]
        c = cmap(n / max(1, len(order) - 1))
        axes[0].plot(tr["bu_mwd_kghm"], tr["k_inf_assembly"],
                     color=c, lw=1.2,
                     label=f"{r['pos']} ({r['efpd']:.0f} EFPD)")
        axes[0].axhline(tr["k_target"], color=c, lw=0.5, ls=":", alpha=0.4)
        axes[1].plot(tr["bu_mwd_kghm"], tr["k_core_proxy"], color=c, lw=1.2)

    axes[0].set_xlabel(r"Burnup $B$ [MWd/kgHM]")
    axes[0].set_ylabel(r"$k_\infty$, reflective assembly")
    axes[0].set_title("Recovered depletion trajectory")
    axes[0].grid(alpha=0.3)

    k_max = order[0]["k_max_used"]
    axes[1].axhline(k_max, color="crimson", lw=1.0, ls="--",
                    label=rf"$k_{{\max}}={k_max:.2f}$")
    axes[1].set_xlabel(r"Burnup $B$ [MWd/kgHM]")
    axes[1].set_ylabel(r"$k_\mathrm{eff}$ core, BOL-calibrated proxy")
    axes[1].set_title("Core proxy against the reactivity screen")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")

    if len(order) <= 16:
        axes[0].legend(fontsize=6, ncol=2, loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "k_histories.pdf")
    fig.savefig(outdir / "k_histories.png", dpi=180)
    print(f"wrote {outdir/'k_histories.pdf'} and .png")


if __name__ == "__main__":
    main()
