#!/usr/bin/env python3
"""
boron_worth.py -- how much of the beginning-of-life hold-down is carried by
the soluble boron, and whether the regulating banks alone would control the
core if the boron were absent.

WHY
----
Every eigenvalue of Campaigns 1 to 8 is computed at the benchmark
concentration of 1000 ppm. The control screen g_ctrl therefore asks: with
1000 ppm in the coolant AND the four regulating banks inserted, is the
core subcritical by 1000 pcm? The boron's contribution is hidden inside k.
This script makes it explicit for a chosen set of designs by re-solving the
same core at several concentrations, unrodded and rodded.

WHAT IT MEASURES, per design
-----------------------------
  k_ARO(c)      unrodded core eigenvalue at boron concentration c
  k_ARI(c)      the same with the four regulating banks inserted
  k_RE12(c)     the same with the first two banks inserted
for c in --ppm (default 0, 500, 1000, 1500).

From them, in reactivity rho = (k - 1)/k:
  boron worth        W_B(c1 -> c2) = rho(k_ARO, c1) - rho(k_ARO, c2)      [pcm]
  differential worth W_B / (c2 - c1)                                       [pcm/ppm]
  bank worth         W_bank(c) = rho(k_ARO, c) - rho(k_ARI, c)             [pcm]
  rods-only margin   M(c) = -rho(k_ARI, c)  (positive = subcritical)       [pcm]
  boron share        W_B(0 -> 1000) / [W_B(0 -> 1000) + W_bank(1000)]      [-]

The margins at c = 0 answer the question "does the regulating system alone
hold down the fresh core", which is the soluble-boron-free question and is
NOT the campaign's design basis (the core is borated). The margins at
1000 ppm reproduce the campaign screen and serve as the closure check.

REQUIRES  openmc-env with the campaign8 branch (zoning.RE12_POSITIONS).
COST      one core solve per (state, concentration, seed): 3 states x 4
          concentrations x 1 seed = 12 solves, about 20 min per design at
          100000 x 170 on 32 threads. Resumable: finished solves are read
          back from <out>/runs.json.

USAGE
  python boron_worth.py --selftest
  python boron_worth.py --checkpoint out_c8/optimization_checkpoint.json \\
      --designs 47 23 29 21 44 1 --dry-run
  setsid nohup python -u boron_worth.py --checkpoint out_c8/optimization_checkpoint.json \\
      --designs 47 23 29 21 44 1 --threads 32 --out boron_c8 > boron_c8.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------- algebra
def rho_pcm(k: float) -> float:
    """Reactivity of an eigenvalue, in pcm."""
    return 1.0e5 * (k - 1.0) / k


def analyse(sol: dict, ppm: list, ref: float = 1000.0) -> dict:
    """sol[(state, c)] -> {'keff': .., 'sd': ..}. Returns the derived table."""
    out = {"ppm": ppm, "states": {}}
    for st in ("ARO", "ARI", "RE12"):
        out["states"][st] = {str(c): sol[(st, c)]["keff"] for c in ppm if (st, c) in sol}
    r = {c: rho_pcm(sol[("ARO", c)]["keff"]) for c in ppm if ("ARO", c) in sol}
    out["rho_ARO_pcm"] = {str(c): round(v, 1) for c, v in r.items()}
    if 0 in r and ref in r:
        wb = r[0] - r[ref]
        out["boron_worth_0_to_ref_pcm"] = round(wb, 1)
        out["differential_worth_pcm_per_ppm"] = round(wb / ref, 3)
    cs = sorted(r)
    out["differential_by_interval_pcm_per_ppm"] = {
        f"{a}-{b}": round((r[a] - r[b]) / (b - a), 3) for a, b in zip(cs[:-1], cs[1:])}
    for st in ("ARI", "RE12"):
        out[f"margin_{st}_pcm"] = {str(c): round(-rho_pcm(sol[(st, c)]["keff"]), 1)
                                  for c in ppm if (st, c) in sol}
        out[f"worth_{st}_pcm"] = {str(c): round(r[c] - rho_pcm(sol[(st, c)]["keff"]), 1)
                                 for c in ppm if (st, c) in sol and c in r}
    if 0 in r and ref in r and ("ARI", ref) in sol:
        wb = r[0] - r[ref]
        wbank = r[ref] - rho_pcm(sol[("ARI", ref)]["keff"])
        out["boron_share_of_holddown"] = round(wb / (wb + wbank), 3)
    return out


def selftest() -> int:
    print("selftest (no OpenMC):")
    # synthetic core: boron worth -8 pcm/ppm, bank worth 16000 pcm at 1000 ppm
    def k_of_rho(p):
        return 1.0 / (1.0 - p / 1.0e5)
    sol = {}
    for c in (0, 500, 1000, 1500):
        r_aro = 12000.0 + 8.0 * (1000 - c)          # 12000 pcm excess at 1000 ppm
        sol[("ARO", c)] = {"keff": k_of_rho(r_aro), "sd": 0.0}
        sol[("ARI", c)] = {"keff": k_of_rho(r_aro - 16000.0), "sd": 0.0}
        sol[("RE12", c)] = {"keff": k_of_rho(r_aro - 8000.0), "sd": 0.0}
    a = analyse(sol, [0, 500, 1000, 1500])
    assert abs(a["differential_worth_pcm_per_ppm"] - 8.0) < 1e-6, a   # magnitude, boron removal ADDS reactivity
    assert abs(a["worth_ARI_pcm"]["1000"] - 16000.0) < 1e-6, a
    assert abs(a["margin_ARI_pcm"]["1000"] - 4000.0) < 1e-6, a
    assert abs(a["margin_ARI_pcm"]["0"] + 4000.0) < 1e-6, a   # supercritical without boron
    assert abs(a["boron_share_of_holddown"] - round(8000.0 / 24000.0, 3)) < 1e-6, a
    print("  reactivity algebra ok: 8 pcm/ppm magnitude, bank 16000 pcm, margin +4000 at 1000 ppm, -4000 at 0 ppm")
    print("selftest OK")
    return 0


# ---------------------------------------------------------------------- solves
def design_from_ckpt(ckpt: dict, idx: int) -> dict:
    r = ckpt["all_raw"][idx]
    keys = ("enrich_inner", "enrich_outer", "gd_wt", "pitch", "refl_thick", "gd_pins")
    missing = [k for k in keys if k not in r]
    if missing:
        raise KeyError(f"design {idx} lacks {missing}")
    return {k: float(r[k]) for k in keys}


def seed_for(design: dict, salt: str) -> int:
    key = json.dumps(design, sort_keys=True) + salt
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def run_design(idx: int, design: dict, ppm: list, states: list, fid: dict,
               out: Path, threads: int, seeds: int, cache: dict) -> dict:
    import reactor_model as rm
    import zoning as zn
    geo = rm.Geometry17x17()
    op0 = rm.Operating()
    dmap = zn.evaluator_design_map(design)
    rodded = {"ARO": None,
              "ARI": (set(zn.RE_BANK_POSITIONS), "B4C"),
              "RE12": (set(zn.RE12_POSITIONS), "B4C")}
    sol = {}
    for st in states:
        for c in ppm:
            vals = []
            for s in range(seeds):
                key = f"{idx}|{st}|{c}|{s}"
                if key in cache:
                    vals.append(cache[key]); continue
                op = dataclasses.replace(op0, boron_ppm=float(c))
                case = out / f"d{idx}" / f"{st}_{int(c)}ppm_s{s}"
                case.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                res = zn.core_bol_solve(design, dmap, op, geo,
                                        particles=fid["particles"], batches=fid["batches"],
                                        inactive=fid["inactive"],
                                        seed=seed_for(design, f"{st}{c}s{s}"),
                                        case=case, rodded_map=rodded[st])
                rec = {"keff": float(res["keff"]), "sd": float(res.get("keff_sd", 0.0)),
                       "fdh": float(res.get("fdh_core", float("nan"))),
                       "wall_s": round(time.time() - t0, 1)}
                cache[key] = rec
                (out / "runs.json").write_text(json.dumps(cache, indent=1))
                print(f"  d{idx} {st:4s} {c:5.0f} ppm seed {s}: k = {rec['keff']:.5f} "
                      f"+/- {rec['sd']:.5f}  F = {rec['fdh']:.3f}  ({rec['wall_s']:.0f} s)",
                      flush=True)
                vals.append(rec)
            k = sum(v["keff"] for v in vals) / len(vals)
            sd = (sum(v["sd"] ** 2 for v in vals) ** 0.5) / len(vals)
            sol[(st, c)] = {"keff": k, "sd": sd}
    return sol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--designs", type=int, nargs="*", default=[])
    ap.add_argument("--ppm", type=float, nargs="*", default=[0, 500, 1000, 1500])
    ap.add_argument("--states", nargs="*", default=["ARO", "ARI", "RE12"])
    ap.add_argument("--particles", type=int, default=100000)
    ap.add_argument("--batches", type=int, default=170)
    ap.add_argument("--inactive", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default="boron_c8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    print(f"python : {sys.version.split()[0]}   cwd: {os.getcwd()}")
    try:
        import openmc, numpy  # noqa: F401
        print(f"openmc : {openmc.__version__}   XS: {os.environ.get('OPENMC_CROSS_SECTIONS')}")
    except ImportError as e:
        print(f"FAIL: {e}. Activate openmc-env.")
        return 2
    try:
        import zoning as zn
        _ = zn.RE12_POSITIONS
    except (ImportError, AttributeError):
        print("FAIL: zoning.RE12_POSITIONS not found, this needs the campaign8 branch.")
        return 2
    if not a.checkpoint or not a.designs:
        print("FAIL: --checkpoint and --designs are required.")
        return 2
    os.environ["OMP_NUM_THREADS"] = str(a.threads)

    ckpt = json.loads(Path(a.checkpoint).read_text())
    ppm = [float(c) for c in a.ppm]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else {}
    fid = dict(particles=a.particles, batches=a.batches, inactive=a.inactive)

    n = len(a.designs) * len(a.states) * len(ppm) * a.seeds
    print(f"plan: {len(a.designs)} designs x {len(a.states)} states x {len(ppm)} ppm "
          f"x {a.seeds} seeds = {n} core solves at {fid['particles']} x {fid['batches']}")
    for idx in a.designs:
        d = design_from_ckpt(ckpt, idx); r = ckpt["all_raw"][idx]
        print(f"  design {idx}: e={d['enrich_inner']:.2f} gd={d['gd_wt']:.2f} refl={d['refl_thick']:.2f} "
              f"pins={r.get('gd_pins_used')}  archive k_core={r['keff_core_bol']:.4f} k_ARI={r['k_allre']:.4f}")
    if a.dry_run:
        return 0

    summary = {}
    for idx in a.designs:
        d = design_from_ckpt(ckpt, idx)
        sol = run_design(idx, d, ppm, a.states, fid, out, a.threads, a.seeds, cache)
        res = analyse(sol, ppm)
        res["archive"] = {k: ckpt["all_raw"][idx][k] for k in
                          ("keff_core_bol", "k_allre", "k_re12", "cycle_length", "peaking")}
        summary[str(idx)] = res
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        m = res["margin_ARI_pcm"]
        print(f"design {idx}: boron {res.get('differential_worth_pcm_per_ppm')} pcm/ppm, "
              f"four-bank margin at 1000 ppm {m.get('1000.0', m.get('1000'))} pcm, "
              f"at 0 ppm {m.get('0.0', m.get('0'))} pcm, boron share "
              f"{res.get('boron_share_of_holddown')}")

    # LaTeX table
    rows = []
    for idx, res in summary.items():
        m16, m8 = res["margin_ARI_pcm"], res["margin_RE12_pcm"]
        g = lambda dct, c: dct.get(str(float(c)), dct.get(str(c), float("nan")))
        rows.append(f"    {idx} & {res.get('differential_worth_pcm_per_ppm', float('nan')):.2f} "
                    f"& {res.get('boron_worth_0_to_ref_pcm', float('nan')):.0f} "
                    f"& {g(m16, 1000):.0f} & {g(m16, 0):.0f} & {g(m8, 1000):.0f} & {g(m8, 0):.0f} "
                    f"& {res.get('boron_share_of_holddown', float('nan')):.2f} \\\\")
    tex = ("\\begin{tabular}{cccccccc}\n\\toprule\n"
           "Design & $\\partial\\rho/\\partial c$ [pcm/ppm] & $W_B$ [pcm] & "
           "$M_{16}(1000)$ & $M_{16}(0)$ & $M_{2}(1000)$ & $M_{2}(0)$ & boron share \\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "boron_table.tex").write_text(tex)
    print(f"wrote {out}/summary.json and {out}/boron_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
