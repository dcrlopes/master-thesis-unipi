#!/usr/bin/env python3
"""
salvage_k_histories.py -- correct the archived depletion histories.

Every campaign was executed under OpenMC 0.15.3 with the write_rates trap
active, so in every restarted block the first step depleted by decay only
while the burnup label advanced by the full step. The physical states in the
archives are nevertheless correct. This script recovers the true histories
without rerunning any depletion:

  1. read every dep_NN/depletion_results.h5 of a case,
  2. keep the entries that carry a real eigenvalue (k > 0),
  3. inside every restarted block, drop the duplicated restart entry and the
     dead-step entry (the xenon ghost), keeping the entries that burned,
  4. relabel the kept entries with the burnup the fuel actually reached,
  5. recompute the late reactivity slope, and the end of cycle using the
     evaluator's own per-design target lookup and crossing routine.

Outputs, under --out:
  case_NNNN_corrected.csv   bu_label, bu_true, k for the kept entries
  salvage_summary.csv       one row per case with the corrected quantities

Usage (one campaign root per invocation):
  python salvage_k_histories.py \
      --checkpoint out_c5/optimization_checkpoint.json \
      --case-root openmc_runs_c5 \
      --ktarget-table ktarget_table.json \
      --out salvage_c5

Flags
  --checkpoint      campaign checkpoint, supplies the design variables per
                    archive index so the target lookup matches the campaign
  --case-root       directory holding the case_NNNN folders
  --ktarget-table   the Route B target table used by the campaign. When
                    given, end of cycle is recomputed with the evaluator's
                    own lookup. When omitted, --k-target may supply a
                    constant, otherwise only burnups and slopes are reported
  --k-target        constant target, used only when no table is given
  --q-spec          specific power in W per gram of heavy metal, converting
                    burnup to Effective Full Power Days (default 9.9827)
  --out             output directory (default salvage)
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np


def read_chunks(case_dir):
    """Return one list of (burnup_label, k) per depletion block, in order,
    keeping only entries with a real eigenvalue. Burnup is in days here and
    converted by the caller."""
    files = sorted(glob.glob(str(Path(case_dir) / "dep_*" /
                                 "depletion_results.h5")))
    out = []
    import openmc.deplete
    for f in files:
        res = openmc.deplete.Results(f)
        try:
            t, k = res.get_keff(time_units="d")
        except TypeError:
            ts, k = res.get_keff()
            t = np.asarray(ts) / 86400.0
        t = np.asarray(t, dtype=float)
        kv = np.asarray(k, dtype=float)[:, 0]
        real = kv > 0.0
        out.append(list(zip(t[real], kv[real])))
    return out


def relabel(chunks, q_spec):
    """Apply the correction. Returns (rows, ghost_jumps_pcm) where rows are
    (bu_label, bu_true, k) for every kept entry and ghost_jumps_pcm are the
    xenon rises measured on the discarded dead-step entries."""
    rows, ghosts = [], []
    true = 0.0
    for ci, ch in enumerate(chunks):
        if not ch:
            continue
        bu = [t * q_spec / 1000.0 for t, _ in ch]
        kk = [k for _, k in ch]
        if ci == 0:
            for b, k in zip(bu, kk):
                rows.append((b, b, k))
            true = bu[-1]
            continue
        if len(ch) < 2:
            continue
        ghosts.append((kk[1] - kk[0]) * 1e5)
        for j in range(2, len(ch)):
            true += bu[j] - bu[j - 1]
            rows.append((bu[j], true, kk[j]))
    return rows, ghosts


def late_slope(bu, k):
    n = max(3, int(len(bu) * 0.3))
    if len(bu) < 3:
        return float("nan")
    x, y = np.asarray(bu[-n:]), np.asarray(k[-n:])
    return float(np.polyfit(x, y, 1)[0]) * 1e5


def crossing(bu, k, kt):
    """Last downward crossing of k through kt after the global maximum,
    matching the evaluator's definition. Returns the burnup or None."""
    bu, k = np.asarray(bu), np.asarray(k)
    i0 = int(np.argmax(k))
    hit = None
    for i in range(max(i0, 1), len(k)):
        if k[i - 1] > kt >= k[i]:
            f = (k[i - 1] - kt) / (k[i - 1] - k[i])
            hit = bu[i - 1] + f * (bu[i] - bu[i - 1])
    return hit


def main():
    ap = argparse.ArgumentParser(
        description="correct archived depletion histories")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--case-root", required=True)
    ap.add_argument("--ktarget-table", default=None)
    ap.add_argument("--k-target", type=float, default=None)
    ap.add_argument("--q-spec", type=float, default=9.9827)
    ap.add_argument("--cap-label", type=float, default=75.0,
                    help="burnup ceiling in LABEL units, used to tell a run "
                         "stopped at the cap from one that stopped because "
                         "its peak never reached the target")
    ap.add_argument("--out", default="salvage")
    args = ap.parse_args()

    ck = json.loads(Path(args.checkpoint).read_text())
    dv = ck["design_variables"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = None
    if args.ktarget_table:
        from reactor_optimization import example_reactor_problem
        from openmc_evaluator import OpenMCEvaluator
        ev = OpenMCEvaluator(example_reactor_problem(),
                             k_target=args.ktarget_table,
                             workdir=str(out / "evtmp"))

    cases = sorted(glob.glob(str(Path(args.case_root) / "case_[0-9]*")))
    rows_out = []
    for cdir in cases:
        m = re.search(r"case_(\d+)$", cdir)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            chunks = read_chunks(cdir)
        except Exception as e:
            print(f"case {idx:4d}: unreadable ({e}), skipped")
            continue
        if not chunks or not chunks[0]:
            print(f"case {idx:4d}: no depletion output, skipped")
            continue
        rows, ghosts = relabel(chunks, args.q_spec)
        if not rows:
            print(f"case {idx:4d}: nothing to keep, skipped")
            continue

        with open(out / f"case_{idx:04d}_corrected.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["bu_label_MWdkg", "bu_true_MWdkg", "k_inf"])
            w.writerows([[f"{a:.4f}", f"{b:.4f}", f"{c:.6f}"]
                         for a, b, c in rows])

        bu_t = [r[1] for r in rows]
        kk = [r[2] for r in rows]
        kt = None
        if ev is not None and idx < len(ck["all_raw"]):
            design = {k: float(ck["all_raw"][idx][k]) for k in dv}
            kt = float(ev._k_target_for(design))
        elif args.k_target is not None:
            kt = args.k_target

        eoc = crossing(bu_t, kk, kt) if kt is not None else None
        # round-trip through days loses the last digits, so compare
        # at the scale of a depletion step, not at machine precision
        at_cap = rows[-1][0] >= args.cap_label - 0.5
        if eoc is not None:
            status = "crossed"
        elif at_cap:
            status = "censored_at_cap"
        else:
            status = "never_critical"
        censored = status == "censored_at_cap"
        rec = dict(
            idx=idx,
            n_blocks=len(chunks),
            n_ghosts=len(ghosts),
            ghost_jump_pcm=round(float(np.mean(ghosts)), 1) if ghosts
            else "",
            bu_label_final=round(rows[-1][0], 3),
            bu_true_final=round(rows[-1][1], 3),
            efpd_label=round(rows[-1][0] * 1000 / args.q_spec, 1),
            efpd_true=round(rows[-1][1] * 1000 / args.q_spec, 1),
            overstatement=round(rows[-1][0] / rows[-1][1], 3)
            if rows[-1][1] > 0 else "",
            slope_true_pcm=round(abs(late_slope(bu_t, kk)), 1),
            k_target=round(kt, 5) if kt is not None else "",
            status=status,
            censored=censored,
            eoc_true_bu=round(eoc, 3) if eoc is not None else "",
            eoc_true_efpd=round(eoc * 1000 / args.q_spec, 1)
            if eoc is not None else "",
        )
        rows_out.append(rec)
        tail = {"crossed": f"EOC true {rec['eoc_true_bu']} MWd/kg = "
                           f"{rec['eoc_true_efpd']} EFPD",
                "censored_at_cap": f"censored at true "
                                   f"{rec['bu_true_final']}",
                "never_critical": "never reached its target, no cycle",
                }[status]
        print(f"case {idx:4d}: label {rec['bu_label_final']:6.1f} -> "
              f"true {rec['bu_true_final']:6.1f} MWd/kg  "
              f"(x{rec['overstatement']}), {tail}")

    if rows_out:
        with open(out / "salvage_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        ratios = [r["overstatement"] for r in rows_out
                  if isinstance(r["overstatement"], float)]
        print(f"\n{len(rows_out)} cases corrected, overstatement "
              f"mean {np.mean(ratios):.2f}x, "
              f"range {min(ratios):.2f}-{max(ratios):.2f}x")
        print(f"summary: {out/'salvage_summary.csv'}")


if __name__ == "__main__":
    main()
