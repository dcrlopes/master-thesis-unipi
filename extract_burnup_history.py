#!/usr/bin/env python3
"""
extract_burnup_history.py -- pulls the reactivity history that ALREADY EXISTS
on disk for any completed case: no new transport is run.

WHAT IT READS
  Every evaluation stored its adaptive-depletion chunks in
     openmc_runs/case_XXXX/dep_00/depletion_results.h5
     openmc_runs/case_XXXX/dep_01/depletion_results.h5   ...
  Each file holds k_inf at every depletion step (mean +/- MC sd). This script
  concatenates the chunks in order, converts time to burnup and EFPD, and
  writes a CSV + a two-panel figure per case:

    panel (a): assembly k_inf vs EFPD, with the leakage-corrected TARGET
               k_target(pitch, refl) as the EOC line (Route B)
    panel (b): the ESTIMATED core k_eff(t) = k_inf(t) / k_target -- the
               Route-B construction applied along the whole cycle, i.e. the
               "decrease of k_core" to the accuracy of the k-target table
               (validated to -91 ... -429 pcm on the finalists at BOL)

WHAT IT CANNOT GIVE (and why)
  F_dH during burnup: the depletion model carries no pin-power tally, so the
  per-step statepoints contain k only. That needs a one-line tally patch and
  a re-run -- see the discussion in the conversation / methodology.
  Measured core k_eff(t): would need a full-core depletion (~1 h per design);
  worth doing only for the final champion(s).

USAGE
  conda activate openmc-env
  python3 extract_burnup_history.py --case openmc_runs/case_0044 \
      --checkpoint out_c4/optimization_checkpoint.json --design-idx 44 \
      --ktarget-table ktarget_table.json --out burnup_history
  # several cases: repeat --case, or use --all-cases openmc_runs
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

P_SPEC_W_PER_G = 9.9834          # W/gHM -> EFPD = 1000 * BU[MWd/kg] / P_spec


def k_target_at(table, pitch, refl):
    """Bilinear interpolation on the 2-D k-target table."""
    P = np.asarray(table["pitch_cm"], float)
    R = np.asarray(table["refl_thick_cm"], float)
    K = np.asarray(table["k_target"], float)          # [pitch][refl]
    p = float(np.clip(pitch, P.min(), P.max()))
    r = float(np.clip(refl, R.min(), R.max()))
    i = int(np.clip(np.searchsorted(P, p) - 1, 0, len(P) - 2))
    j = int(np.clip(np.searchsorted(R, r) - 1, 0, len(R) - 2))
    tp = (p - P[i]) / (P[i + 1] - P[i]) if P[i + 1] > P[i] else 0.0
    tr = (r - R[j]) / (R[j + 1] - R[j]) if R[j + 1] > R[j] else 0.0
    return float((1 - tp) * (1 - tr) * K[i][j] + tp * (1 - tr) * K[i + 1][j]
                 + (1 - tp) * tr * K[i][j + 1] + tp * tr * K[i + 1][j + 1])


def read_case(case: Path):
    """Concatenate k_inf(t) over the dep_XX chunks of one case."""
    import openmc.deplete as dep
    t_all, k_all, ks_all = [], [], []
    t_off = 0.0
    chunks = sorted(case.glob("dep_*/depletion_results.h5"))
    if not chunks:
        raise SystemExit(f"no depletion_results.h5 under {case}")
    for ch in chunks:
        res = dep.Results(str(ch))
        t, k = res.get_keff()                 # t in s; k -> (n, 2) mean, sd
        k = np.atleast_2d(k)
        start = 1 if (t_all and abs(t[0]) < 1e-9) else 0   # drop dup t=0
        t_all += list(t[start:] + t_off)
        k_all += list(k[start:, 0])
        ks_all += list(k[start:, 1])
        t_off = t_all[-1]
    t = np.asarray(t_all)
    days = t / 86400.0
    bu = days * P_SPEC_W_PER_G / 1000.0 * 1.0   # placeholder, fixed below
    # burnup from specific power: BU[MWd/kgHM] = P_spec[W/g] * t[d] / 1000
    bu = P_SPEC_W_PER_G * days / 1000.0
    return days, bu, np.asarray(k_all), np.asarray(ks_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", default=[],
                    help="case directory (repeatable)")
    ap.add_argument("--all-cases", default=None,
                    help="scan every case_* under this directory")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--design-idx", type=int, action="append", default=[],
                    help="archive index matching each --case (repeatable)")
    ap.add_argument("--ktarget-table", required=True)
    ap.add_argument("--out", default="burnup_history")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10})

    ck = json.loads(Path(args.checkpoint).read_text())
    table = json.loads(Path(args.ktarget_table).read_text())
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    cases = [Path(c) for c in args.case]
    idxs = list(args.design_idx)
    if args.all_cases:
        cases = sorted(Path(args.all_cases).glob("case_*"))
        idxs = [int(c.name.split("_")[1]) for c in cases]
    if len(idxs) != len(cases):
        raise SystemExit("give one --design-idx per --case")

    for case, idx in zip(cases, idxs):
        rec = ck["all_raw"][idx]
        pitch, refl = float(rec["pitch"]), float(rec["refl_thick"])
        kt = k_target_at(table, pitch, refl)
        days, bu, k, ksd = read_case(case)
        efpd = 1000.0 * bu / P_SPEC_W_PER_G
        kcore = k / kt

        csv = outdir / f"history_idx{idx}.csv"
        with open(csv, "w") as f:
            f.write("days,burnup_MWd_kg,EFPD,k_inf,k_inf_sd,"
                    "k_core_est\n")
            for row in zip(days, bu, efpd, k, ksd, kcore):
                f.write(",".join(f"{x:.6g}" for x in row) + "\n")

        fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.4))
        a.errorbar(efpd, k, yerr=1.96 * ksd, fmt="o-", ms=4, lw=1.4,
                   color="#173a5e", ecolor="#9fb2c4", capsize=2)
        a.axhline(kt, color="#B23A48", lw=1.4)
        a.text(efpd.max() * 0.02, kt, f"  $k_\\mathrm{{target}}$ = {kt:.4f} "
               "(EOC, Route B)", color="#B23A48", fontsize=8.6, va="bottom")
        a.set_xlabel("EFPD"); a.set_ylabel(r"assembly $k_\infty$")
        a.set_title(f"(a) idx{idx}: measured $k_\\infty$ vs burnup",
                    fontsize=10)
        a.grid(alpha=.25, lw=.5)
        b.plot(efpd, kcore, "s-", ms=4, lw=1.4, color="#2E6F4E")
        b.axhline(1.0, color="#B23A48", lw=1.4)
        b.text(efpd.max() * 0.02, 1.0, "  criticality", color="#B23A48",
               fontsize=8.6, va="bottom")
        b.set_xlabel("EFPD")
        b.set_ylabel(r"estimated core $k_\mathrm{eff}$"
                     r"$\;=k_\infty/k_\mathrm{target}$")
        b.set_title("(b) Route-B estimate of the core decrease", fontsize=10)
        b.grid(alpha=.25, lw=.5)
        top = f"{case.name} | pitch={pitch:.3f} refl={refl:.2f} " \
              f"gd={float(rec['gd_wt']):.2f} " \
              f"e=({float(rec['enrich_inner']):.2f}/" \
              f"{float(rec['enrich_outer']):.2f})"
        fig.suptitle(top, fontsize=10)
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"history_idx{idx}.{ext}",
                        bbox_inches="tight",
                        dpi=200 if ext == "png" else None)
        plt.close(fig)
        print(f"idx{idx}: {len(k)} steps, BOL k_inf={k[0]:.5f}, "
              f"EOL k_inf={k[-1]:.5f} at {efpd[-1]:.0f} EFPD "
              f"-> {csv}")


if __name__ == "__main__":
    main()
