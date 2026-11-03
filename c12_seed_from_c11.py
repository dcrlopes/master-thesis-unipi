#!/usr/bin/env python3
"""
c12_seed_from_c11.py -- the design of experiments of Campaign 12: the 99
evaluations of Campaign 11 with the mission-burnup margin added as a constraint
quantity, and the designs confirmed at full fidelity replaced by their
confirmed values.

Campaign 12 constrains g_margin = k_eoc - k_3D(B_mission), with B_mission =
1826 d x P_spec / 1000 = 18.23 MWd/kgHM, instead of g_efpd (paper_finds.md
Sections 13 and 15). Every Campaign 11 record carries its three-dimensional k
history and the corrected criterion (k_target_ratio_3d x k_3D(0)), so g_margin
is read from the stored history: interpolated when the history reaches
B_mission, extrapolated from the last two states when the depletion ended at an
earlier crossing (flagged margin_extrapolated). The designs depleted twice at
20 000 x 160 (c10_dep_core3d_seed1/2 on 19 Oct 2026, c11_dep_core3d_seed1/2 on
2 Nov 2026) take the mean of the two seeds for the cycle length and the
margin, under the corrected criterion of their record, with the proxy values
kept under the suffix _proxy, as the loop's own confirmation does.

Also computes kappa of the margin rule for g_margin and g_kmin from the
leave-one-out error of the campaign kernel (Matern 5/2 ARD) on the seed,
kappa = RMSE(LOO) / mean(sigma_GP), written to c12_kappa.json.

usage:  python c12_seed_from_c11.py [--no-kappa]   (needs out_c11, c11_seed.json, the four runs.json)
writes c12_seed.json (evaluator cache format, for --doe-from-cache), c12_kappa.json
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import core_geometry as cg
import dep_common as dc
from c11_seed_from_c10 import kappa_from_loo

CKPT = "out_c11/optimization_checkpoint.json"
CACHES = ("openmc_runs_c11/eval_cache.json", "c11_seed.json")   # the 42 loop designs, the 57 seed designs
CONFIRM = {"out_c11/optimization_checkpoint.json": ("c11_dep_core3d_seed1", "c11_dep_core3d_seed2"),
           "out_c10/optimization_checkpoint.json": ("c10_dep_core3d_seed1", "c10_dep_core3d_seed2")}
EFPD_REQ = 1826.0
SPEC_POWER = 9.983354024602374      # W/gHM, the campaign value (c9_dep_core3d runs.json)
B_MISSION = EFPD_REQ * SPEC_POWER / 1000.0
VARS = ["enrich", "gd_wt", "refl_thick", "gd_pins"]


def key_of(d):
    return tuple(round(float(d[v]), 6) for v in VARS)


class KeyIndex:
    """Cache key of a design by its variables, matched within 1e-6 (a rounded
    tuple fails at a rounding boundary: C11-57's pin count did)."""
    def __init__(self, keys):
        self.keys = list(keys)
        self.X = np.array([[json.loads(k)[v] for v in VARS] for k in self.keys])

    def get(self, d):
        x = np.array([float(d[v]) for v in VARS])
        dist = np.abs(self.X - x).max(axis=1)
        j = int(dist.argmin())
        return self.keys[j] if dist[j] < 1e-6 else None


def margin_of(rec):
    """g_margin of a record from its stored history and corrected criterion."""
    k, bu = rec["k_hist_3d"], rec["bu_hist_3d"]
    k_eoc = float(rec["k_target_ratio_3d"]) * float(k[0])
    g, km, ex = dc.margin_at_mission(bu, k, k_eoc, B_MISSION)
    return dict(g_margin=g, k_mission_3d=km, b_mission=B_MISSION, margin_extrapolated=ex)


def confirmed_values(rec, runs):
    """Mean of the seeds; the criterion is k_3D(0) x the record's corrected ratio,
    so a confirmation made before the k_bol correction is corrected here."""
    ratio = float(rec["k_target_ratio_3d"])
    efpds, margins, kms, seeds = [], [], [], []
    for r in runs:
        bu, k = r["bu_hist"], r["k_hist"]
        k_eoc = ratio * float(k[0])
        b = cg.eoc_crossing_burnup(bu, k, k_eoc)
        if b is None:
            b = bu[-1] if k[-1] > k_eoc else 0.0
        efpds.append(b * 1000.0 / float(r.get("spec_power", SPEC_POWER)))
        g, km, _ = dc.margin_at_mission(bu, k, k_eoc, B_MISSION)
        margins.append(g)
        kms.append(km)
        seeds.append(int(r["seed"]))
    out = {k + "_proxy": rec[k] for k in ("efpd_3d", "cycle_length", "g_efpd", "g_margin", "k_mission_3d",
                                           "censored_3d", "sigma_efpd_3d") if k in rec}
    e = float(np.mean(efpds))
    out.update(efpd_3d=e, cycle_length=e, g_efpd=EFPD_REQ - e, g_margin=float(np.mean(margins)),
               k_mission_3d=float(np.mean(kms)), confirmed_3d=True, confirm_transport=runs[0]["transport"],
               confirm_seeds=seeds, efpd_3d_seeds=efpds, g_margin_seeds=margins, censored_3d=False,
               t_confirm_s=float(sum(float(r["wall_s"]) for r in runs)),
               confirm_source="c12_seed_from_c11.py: two full-fidelity seeds, criterion of the corrected record")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-kappa", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    R = json.load(open(CKPT))["all_raw"]
    # the cache key is the design dict the evaluator hashes: the 42 loop designs
    # are in the Campaign 11 cache, the 57 seed designs in c11_seed.json
    by_design = KeyIndex(k for c in CACHES for k in json.load(open(c)))
    seed, n_margin, n_ex = {}, 0, 0
    for r in R:                           # the checkpoint record is the complete evaluation
        key = by_design.get(r)
        assert key is not None, f"no cache key for the design {key_of(r)}"
        rec = dict(r)
        if rec.get("k_hist_3d") and rec.get("k_target_ratio_3d") is not None:
            rec.update(margin_of(rec))
            n_margin += 1
            n_ex += int(rec["margin_extrapolated"])
        seed[key] = rec
    print(f"{len(seed)} Campaign 11 records, {n_margin} with g_margin ({n_ex} extrapolated past the end of the history)")
    # the confirmed designs, matched by their design variables
    rows = []
    for ckpt, dirs in CONFIRM.items():
        raw = json.load(open(ckpt))["all_raw"]
        runs = [json.load(open(Path(d) / "runs.json")) for d in dirs]
        for name in sorted(set(runs[0]) & set(runs[1]), key=lambda s: int(s[1:])):
            idx = int(name[1:])
            key = by_design.get(raw[idx])
            if key is None:
                print(f"  {ckpt}:{name} is not in the Campaign 11 seed, skipped")
                continue
            rec = seed[key]
            upd = confirmed_values(rec, [runs[0][name], runs[1][name]])
            rec.update(upd)
            rows.append((Path(ckpt).parent.name.replace("out_c", "C") + "-" + str(idx), rec))
    print(f"{len(rows)} designs confirmed at full fidelity (two seeds):")
    print("  design    EFPD proxy  EFPD conf   margin proxy  margin conf  [pcm]")
    for label, rec in rows:
        print(f"  {label:8s} {rec['efpd_3d_proxy']:9.0f} {rec['efpd_3d']:10.1f}   {1e5 * rec['g_margin_proxy']:+10.0f}"
              f"   {1e5 * rec['g_margin']:+10.0f}")
    g_m = np.array([v["g_margin"] for v in seed.values() if "g_margin" in v])
    g_e = np.array([v["g_efpd"] for v in seed.values() if "g_margin" in v])
    agree = int(((g_m <= 0) == (g_e <= 0)).sum())
    print(f"feasible by the margin {int((g_m <= 0).sum())}, by the cycle length {int((g_e <= 0).sum())}, "
          f"sign agreement {agree} of {len(g_m)}")
    Path("c12_seed.json").write_text(json.dumps(seed))
    print(f"wrote c12_seed.json ({len(seed)} records)")
    if not a.no_kappa:
        use = [v for v in seed.values() if "g_margin" in v]
        X = np.array([[v[n] for n in VARS] for v in use])
        G = {"g_margin": np.array([v["g_margin"] for v in use]), "g_kmin": np.array([v["g_kmin"] for v in use])}
        kap = kappa_from_loo(X, G, ["g_margin", "g_kmin"], nu=2.5)
        for n, k in kap.items():
            print(f"  {n}: leave-one-out RMSE {k['rmse']:.4g}, mean GP sigma {k['mean_sigma']:.4g} -> kappa {k['kappa']:.2f}")
        Path("c12_kappa.json").write_text(json.dumps(kap, indent=1))
        print("  suggested: --feas-kappa-per " + " ".join(f"{n}={k['kappa']:.2f}" for n, k in kap.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
