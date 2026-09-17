#!/usr/bin/env python3
"""c9_axial_checks.py -- statistical checks on the Campaign 9 axial study.

No transport. Reads
  axial_c9/runs.json          keys  idx|state|seed|PxBxI|boron   rec keff, sd, entropy_conv
  axial_c9/d<idx>/<st>_s<seed>.npz                               asm, pin, ax, edges
  confirm3d_c9_all/runs.json  keys  idx|state|mode|s|...|P|B|I|boron   rec keff, sd, fdh

CHECK A, seed agreement of k.
  For two independent seeds z = (k1 - k2) / sqrt(sd1^2 + sd2^2) should be a
  standard normal variable. One pair at 3 sigma means little on its own. The
  test is over all pairs: the mean of z^2 should be 1. A value clearly above 1
  says the OpenMC standard deviation underestimates the real seed-to-seed
  scatter, which is what inter-cycle correlation of the fission source does.
  The factor sqrt(mean z^2) is then the correction to apply to quoted sd.

CHECK B, the F_dH estimator.
  per-seed   F = max(map_s / mean(map_s)) for each seed, then averaged.
             This is what confirm3d.py reports.
  pooled     F = max(mean_s(map_s) / mean), the maximum of the averaged map.
             This is what axial_shape_c9.py reports in summary.json.
  The maximum of a noisy map is biased upward, so per-seed >= pooled.
  The two scripts use different seeds (1, 2 against hashed seeds), so their
  per-seed values are four independent solves of the same problem.

CHECK C, source convergence, entropy_conv against the inactive batches.
CHECK D, boron and fidelity identical in both studies.

USAGE (repository root)
  python c9_axial_checks.py
  python c9_axial_checks.py --axial axial_c9 --confirm confirm3d_c9_all --out c9_post
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def fdh_of(pin):
    f = np.ma.masked_equal(np.asarray(pin, dtype=float), 0.0)
    return float((f / f.mean()).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axial", default="axial_c9")
    ap.add_argument("--confirm", default="confirm3d_c9_all")
    ap.add_argument("--out", default="c9_post")
    ap.add_argument("--inactive", type=int, default=80)
    a = ap.parse_args()

    ax_dir, cf_dir = Path(a.axial), Path(a.confirm)
    for p in (ax_dir / "runs.json", cf_dir / "runs.json"):
        if not p.exists():
            print(f"ABORT: {p} not found"); return 1
    L = []
    P = lambda s="": (L.append(s), print(s))  # noqa: E731

    # ------------------------------------------------------------ axial runs
    ax_runs = json.loads((ax_dir / "runs.json").read_text())
    ax = defaultdict(dict)                # (idx, state) -> {seed: rec}
    fids, borons = set(), set()
    for key, rec in ax_runs.items():
        idx, st, seed, fid, boron = key.split("|")
        if int(fid.split("x")[0]) < 50000:
            continue                      # smoke runs
        fids.add(fid); borons.add(float(boron))
        rec = dict(rec)
        npz = ax_dir / f"d{idx}" / f"{st}_s{seed}.npz"
        rec["fdh"] = fdh_of(np.load(npz)["pin"]) if npz.exists() else float("nan")
        rec["pin"] = np.load(npz)["pin"] if npz.exists() else None
        ax[(int(idx), st)][int(seed)] = rec

    # --------------------------------------------------------- confirm3d runs
    cf_runs = json.loads((cf_dir / "runs.json").read_text())
    cf = defaultdict(list)                # (idx, state) -> [rec] for 3Dhw
    cf_fid, cf_boron = set(), set()
    for key, rec in cf_runs.items():
        p = key.split("|")
        if p[2] != "3Dhw" or p[4:7] != ["None", "None", "None"]:
            continue                      # 2D solves and sensitivity variants
        cf_fid.add("x".join(p[-4:-1])); cf_boron.add(float(p[-1]))
        cf[(int(p[0]), p[1])].append(rec)

    designs = sorted({i for i, _ in ax})
    P("=" * 78)
    P("CHECK D, identical conditions")
    P(f"  axial      fidelity {sorted(fids)}  boron {sorted(borons)} ppm")
    P(f"  confirm3d  fidelity {sorted(cf_fid)}  boron {sorted(cf_boron)} ppm")
    same = fids == cf_fid and borons == cf_boron
    P(f"  {'same conditions, the solves can be pooled' if same else 'CONDITIONS DIFFER, do not pool'}")

    # --------------------------------------------------------------- check A
    P("=" * 78)
    P("CHECK A, seed agreement of k, z = dk / sqrt(sd1^2 + sd2^2)")
    P("  source     design state   k1        k2        dk(pcm)  z")
    zs = {"axial": [], "confirm3d": []}
    for (i, st), seeds in sorted(ax.items()):
        if len(seeds) < 2:
            continue
        r1, r2 = seeds[min(seeds)], seeds[max(seeds)]
        dk = r1["keff"] - r2["keff"]; z = dk / math.hypot(r1["sd"], r2["sd"])
        zs["axial"].append(z)
        P(f"  axial      C9-{i:<3d} {st:5s} {r1['keff']:.5f}  {r2['keff']:.5f}  {dk*1e5:+6.0f}  {z:+5.2f}")
    for (i, st), recs in sorted(cf.items()):
        if len(recs) != 2:
            continue
        dk = recs[0]["keff"] - recs[1]["keff"]; z = dk / math.hypot(recs[0]["sd"], recs[1]["sd"])
        zs["confirm3d"].append(z)
        if i in designs:
            P(f"  confirm3d  C9-{i:<3d} {st:5s} {recs[0]['keff']:.5f}  {recs[1]['keff']:.5f}  {dk*1e5:+6.0f}  {z:+5.2f}")
    P("")
    for src, v in list(zs.items()) + [("both", zs["axial"] + zs["confirm3d"])]:
        v = np.array(v)
        if v.size == 0:
            continue
        m2 = float(np.mean(v ** 2)); n = v.size
        # mean z^2 of n standard normals: mean 1, sd sqrt(2/n)
        P(f"  {src:10s} pairs {n:3d}  mean z^2 {m2:5.2f} +/- {math.sqrt(2 / n):.2f}  "
          f"sd scale sqrt(mean z^2) {math.sqrt(m2):.2f}  |z|>3: {int(np.sum(np.abs(v) > 3))}"
          f"  (expected {n * 0.0027:.2f})")

    # --------------------------------------------------------------- check B
    P("=" * 78)
    P("CHECK B, ARO F_dH on the hardware model, four independent solves per design")
    P("  design  confirm3d s0 s1   axial s1 s2    per-seed mean  sd     sem    pooled(axial)")
    rows = []
    for i in designs:
        c = [r["fdh"] for r in cf.get((i, "ARO"), [])]
        s = ax.get((i, "ARO"), {})
        x = [s[k]["fdh"] for k in sorted(s)]
        pins = [s[k]["pin"] for k in sorted(s) if s[k]["pin"] is not None]
        pooled = fdh_of(np.mean(pins, axis=0)) if pins else float("nan")
        v = np.array(c + x)
        sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        rows.append(dict(idx=i, confirm3d=c, axial=x, mean=float(v.mean()), sd=sd,
                         sem=sd / math.sqrt(v.size), n=int(v.size), pooled_axial=pooled))
        P(f"  C9-{i:<3d}  {' '.join(f'{t:.3f}' for t in c):13s}  {' '.join(f'{t:.3f}' for t in x):13s}"
          f"  {v.mean():.3f}          {sd:.3f}  {sd / math.sqrt(v.size):.3f}  {pooled:.3f}")
    P("  per-seed is the estimator of confirm3d.py and of the Campaign 9 3D front.")

    # --------------------------------------------------------------- check C
    P("=" * 78)
    P(f"CHECK C, source entropy convergence batch (inactive = {a.inactive})")
    worst = sorted(((r.get("entropy_conv") or 0, i, st, sd) for (i, st), ss in ax.items()
                    for sd, r in ss.items()), reverse=True)[:5]
    for conv, i, st, sd in worst:
        flag = "  CLOSE TO THE INACTIVE LIMIT" if conv > 0.75 * a.inactive else ""
        P(f"  C9-{i:<3d} {st:5s} seed {sd}: converged at batch {conv}{flag}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "c9_axial_checks.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "c9_axial_checks.json").write_text(json.dumps(
        dict(z=zs, fdh=rows), indent=1), encoding="utf-8")
    print(f"\nwrote {out}/c9_axial_checks.txt and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
