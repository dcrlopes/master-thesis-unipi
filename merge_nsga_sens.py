#!/usr/bin/env python3
"""Merge per-seed nsga_sensitivity runs into one aggregated result.

Concatenates the per-seed records, re-aggregates over seeds, and rewrites the
CSV, the booktabs table and the figure using the original script's own writers
so the merged output cannot drift from the single-process format.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import nsga_sensitivity as NS

AGG = ["cv_min_pop", "cv_mean_pop", "hv_surrogate", "n_feasible_pop",
       "t_search_s", "t_acquisition_s", "agreement", "gd_pins_agreement",
       "d_cycle_EFPD", "d_peaking", "violation_selected_mean",
       "n_candidates_used"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="nsga_sens_c6_s*/nsga_sensitivity.json")
    ap.add_argument("--out", default="nsga_sens_c6")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files matched {args.pattern!r}")

    parts = [json.loads(Path(p).read_text()) for p in paths]
    base = parts[0]

    # Every part must describe the same study, otherwise the merge is invalid.
    for p, d in zip(paths, parts):
        for key in ("baseline", "n_infill", "top_k", "match_tol",
                    "design_variables", "constraint_names"):
            if d[key] != base[key]:
                raise SystemExit(f"{p}: {key} differs from {paths[0]}")
        if d["n_feasible_in_archive"] != base["n_feasible_in_archive"]:
            raise SystemExit(f"{p}: archive feasibility differs")

    records = [r for d in parts for r in d["records"]]
    seeds = sorted({r["seed"] for r in records})
    settings = []
    for r in records:
        if (r["pop"], r["gen"]) not in settings:
            settings.append((r["pop"], r["gen"]))

    n_expected = len(settings) * len(seeds)
    if len(records) != n_expected:
        raise SystemExit(
            f"{len(records)} records but {len(settings)} settings x "
            f"{len(seeds)} seeds = {n_expected}. A process failed or a seed "
            f"was duplicated.")

    baseline = tuple(int(v) for v in base["baseline"].lower().split("x"))

    rows = []
    for pop, gen in settings:
        sub = [r for r in records if r["pop"] == pop and r["gen"] == gen]
        row = {"pop": pop, "gen": gen, "n_surrogate_evals": pop * gen,
               "n_seeds": len(sub), "is_baseline": (pop, gen) == baseline}
        for f in AGG:
            vals = np.array([r[f] for r in sub], dtype=float)
            with np.errstate(invalid="ignore"):
                row[f + "_mean"] = float(np.nanmean(vals))
                row[f + "_std"] = float(np.nanstd(vals))
        rows.append(row)

    infeasible_regime = not any(r["had_feasible"] for r in records)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "nsga_sensitivity.json").write_text(json.dumps({
        "archive": base["archive"],
        "n_feasible_in_archive": base["n_feasible_in_archive"],
        "baseline": base["baseline"],
        "seeds": seeds,
        "n_infill": base["n_infill"],
        "top_k": base["top_k"],
        "match_tol": base["match_tol"],
        "hv_ref": base["hv_ref"],
        "infeasible_regime": infeasible_regime,
        "design_variables": base["design_variables"],
        "constraint_names": base["constraint_names"],
        "merged_from": paths,
        "aggregate": rows,
        "records": records,
    }, indent=2, default=float))

    fields = (["pop", "gen", "n_surrogate_evals", "n_seeds", "is_baseline"] +
              [f + s for f in AGG for s in ("_mean", "_std")])
    NS.write_csv(out / "nsga_sensitivity.csv", rows, fields)
    NS.write_tex(out / "nsga_sensitivity.tex", rows, base["baseline"],
                 infeasible_regime, len(seeds))
    NS.make_figure(out / "nsga_sensitivity.pdf", rows, infeasible_regime,
                   base["baseline"])

    print(f"merged {len(paths)} runs, {len(records)} records, "
          f"{len(settings)} settings, {len(seeds)} seeds")
    print(f"archive feasible: {base['n_feasible_in_archive']}")
    print(f"infeasible_regime: {infeasible_regime}")
    print(f"{'pop':>5} {'gen':>5} {'CVmin':>9} {'HV':>10} {'agree':>7} {'t[s]':>7}")
    for r in rows:
        hv = r["hv_surrogate_mean"]
        hv_s = "n/a" if not np.isfinite(hv) else f"{hv:.4g}"
        print(f"{r['pop']:5d} {r['gen']:5d} {r['cv_min_pop_mean']:9.4f} "
              f"{hv_s:>10} {r['agreement_mean']:7.2f} "
              f"{r['t_search_s_mean']:7.1f}")
    print(f"written -> {out}/")


if __name__ == "__main__":
    main()
