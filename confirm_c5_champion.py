#!/usr/bin/env python3
"""
confirm_c5_champion.py -- settle whether Campaign 5's best design is feasible.

THE QUESTION
  Archive idx 54 (40 Gd pins, gd 1.4232 wt%, e_in/e_out 13.0719/15.0019,
  pitch 1.1502 cm, refl 19.4145 cm) satisfies the reactivity limit
  (k_BOL = 1.3449 <= 1.35) and reaches the burnup ceiling (7513 EFPD), but
  its single-seed core peaking came out at F_dH = 2.0012 against a limit of
  2.0 -- a violation of 0.0012.

  The two-seed spread measured in the pin-count test at the same fidelity
  (100k particles) was 0.016 to 0.031. The violation is therefore about
  1/13 of the run-to-run scatter and cannot be distinguished from zero at
  that fidelity. This script measures it properly.

WHAT IT DOES
  Re-solves the CORE at beginning of life with independent seeds at raised
  fidelity (default 8 seeds, 200k particles, 260 batches, 80 inactive), and
  the ASSEMBLY (default 4 seeds) to confirm k_BOL keeps its margin. Reports
  mean, standard deviation, standard error and a 95% confidence interval,
  and states the verdict against both limits.

  No depletion is run: the peaking constraint is a beginning-of-life
  quantity and the cycle length is already fixed by the burnup ceiling.

  Every solve is cached in <out>/runs.json, so the script is resumable and
  can be re-run to ADD seeds (raise --seeds and run again; completed seeds
  are reused).

USAGE
  conda activate openmc-env
  setsid nohup python -u confirm_c5_champion.py \
      --checkpoint out_c5/optimization_checkpoint.json --design-idx 54 \
      --seeds 8 --asm-seeds 4 --threads 64 --out c5_confirm \
      > c5_confirm.log 2>&1 < /dev/null &
"""
import argparse
import json
import math
import statistics as st
import time
import zlib
from pathlib import Path

F_LIMIT = 2.0
K_LIMIT = 1.35


def design_seed(design, salt=""):
    key = json.dumps({k: round(float(v), 10)
                      for k, v in sorted(design.items())}) + salt
    return 1 + zlib.crc32(key.encode()) % 2_000_000_000


def stats(xs):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    # normal approximation is adequate at n >= 8; t(0.975, 7) = 2.365
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    half = tcrit * sem if n > 1 else float("nan")
    return m, sd, sem, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--design-idx", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=8,
                    help="independent core solves")
    ap.add_argument("--asm-seeds", type=int, default=4,
                    help="independent assembly solves (k_BOL check)")
    ap.add_argument("--core-particles", type=int, default=200000)
    ap.add_argument("--core-batches", type=int, default=260)
    ap.add_argument("--core-inactive", type=int, default=80)
    ap.add_argument("--asm-particles", type=int, default=64000)
    ap.add_argument("--asm-batches", type=int, default=140)
    ap.add_argument("--asm-inactive", type=int, default=40)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="c5_confirm")
    args = ap.parse_args()

    if args.threads:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    import numpy as np
    import openmc
    import reactor_model as rm
    try:
        import core_geometry as cg
        nx = ny = cg.CORE_MAP_32.shape[0]
    except Exception:
        nx = ny = 6

    ck = json.loads(Path(args.checkpoint).read_text())
    rec = ck["all_raw"][args.design_idx]
    dv = ck["design_variables"]
    design = {k: float(rec[k]) for k in dv}

    print(f"design idx{args.design_idx}")
    for k in dv:
        print(f"   {k:14s} = {design[k]:.4f}")
    pins = rm.snap_gd_pins(design.get("gd_pins", 12))
    print(f"   gd_pins snapped -> {pins} "
          f"(archive recorded {rec.get('gd_pins_used')})")
    print(f"archive single-seed: F_core={rec['peaking']:.4f} "
          f"k_bol={rec['k_bol']:.4f} EFPD={rec['cycle_length']:.0f}")
    print(f"fidelity: core {args.core_particles} x {args.core_batches} "
          f"({args.core_inactive} inactive) vs campaign 100000 x 170 (60)\n")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    store_p = outdir / "runs.json"
    store = json.loads(store_p.read_text()) if store_p.exists() else {}
    op, geo = rm.Operating(), rm.Geometry17x17()
    N = geo.lattice

    def run_core(seed, case):
        m = rm.make_core_model(design, op, geo,
                               particles=args.core_particles,
                               batches=args.core_batches,
                               inactive=args.core_inactive)
        model = m[0] if isinstance(m, tuple) else m
        model.settings.seed = seed
        pitch = design["pitch"]
        half = nx * N * pitch / 2.0
        mesh = openmc.RegularMesh()
        mesh.dimension = (nx * N, ny * N)
        mesh.lower_left = (-half, -half); mesh.upper_right = (half, half)
        t = openmc.Tally(name="core_pin")
        t.filters = [openmc.MeshFilter(mesh)]; t.scores = ["fission"]
        model.tallies = openmc.Tallies([t])
        sp_path = model.run(cwd=str(case), output=False)
        with openmc.StatePoint(sp_path) as sp:
            keff = float(sp.keff.nominal_value)
            v = sp.get_tally(name="core_pin").get_values(
                scores=["fission"]).reshape(ny * N, nx * N)
            H = np.asarray(getattr(sp, "entropy", []), dtype=float)
        fm = np.ma.masked_equal(v, 0.0)
        conv = None
        if H.size:
            tail = H[args.core_inactive + (len(H) - args.core_inactive) // 2:]
            mu, sd = float(tail.mean()), float(tail.std(ddof=1))
            Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
            Hs[0], Hs[-1] = H[0], H[-1]
            bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
            conv = int(bad[-1]) + 2 if len(bad) else 1
        return dict(fdh=float((fm / fm.mean()).max()), keff=keff, conv=conv)

    def run_asm(seed, case):
        m = rm.make_assembly_model(design, op, geo, bc="reflective",
                                   particles=args.asm_particles,
                                   batches=args.asm_batches,
                                   inactive=args.asm_inactive)
        model = m[0] if isinstance(m, tuple) else m
        model.settings.seed = seed
        sp_path = model.run(cwd=str(case), output=False)
        with openmc.StatePoint(sp_path) as sp:
            return dict(k=float(sp.keff.nominal_value),
                        ksd=float(sp.keff.std_dev))

    # ---- core seeds --------------------------------------------------------
    print("=== CORE (beginning of life) ===", flush=True)
    fdh, kcore = [], []
    for s in range(1, args.seeds + 1):
        key = f"core_s{s}"
        if key not in store:
            case = outdir / f"core_s{s}"; case.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            store[key] = run_core(design_seed(design, f"conf_core{s}"), case)
            store[key]["wall"] = time.time() - t0
            store_p.write_text(json.dumps(store, indent=1))
        r = store[key]
        flag = ("  !! entropy converged at batch %s > %d inactive"
                % (r["conv"], args.core_inactive)
                if r["conv"] and r["conv"] > args.core_inactive else "")
        print(f"  seed {s}: F_dH={r['fdh']:.4f}  k_eff_core={r['keff']:.5f}"
              f"  ({r['wall']:.0f}s){flag}", flush=True)
        fdh.append(r["fdh"]); kcore.append(r["keff"])

    # ---- assembly seeds ----------------------------------------------------
    print("\n=== ASSEMBLY (k_BOL) ===", flush=True)
    kb = []
    for s in range(1, args.asm_seeds + 1):
        key = f"asm_s{s}"
        if key not in store:
            case = outdir / f"asm_s{s}"; case.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            store[key] = run_asm(design_seed(design, f"conf_asm{s}"), case)
            store[key]["wall"] = time.time() - t0
            store_p.write_text(json.dumps(store, indent=1))
        r = store[key]
        print(f"  seed {s}: k_BOL={r['k']:.5f} +/- {r['ksd']:.5f}"
              f"  ({r['wall']:.0f}s)", flush=True)
        kb.append(r["k"])

    # ---- verdict -----------------------------------------------------------
    fm, fsd, fsem, fhalf = stats(fdh)
    km, ksd_, ksem, khalf = stats(kb)
    print("\n" + "=" * 72)
    print(f"core F_dH   : {fm:.4f}  sd {fsd:.4f}  sem {fsem:.4f}  "
          f"95% CI [{fm-fhalf:.4f}, {fm+fhalf:.4f}]   (n={len(fdh)})")
    print(f"              limit {F_LIMIT}  ->  margin {fm - F_LIMIT:+.4f}")
    print(f"assembly k  : {km:.5f} sd {ksd_:.5f} sem {ksem:.5f} "
          f"95% CI [{km-khalf:.5f}, {km+khalf:.5f}]   (n={len(kb)})")
    print(f"              limit {K_LIMIT} ->  margin {km - K_LIMIT:+.5f}")
    print(f"core k_eff  : {st.mean(kcore):.5f} (recorded, not constrained)")
    print("-" * 72)
    fs = "PASS" if fm + fhalf <= F_LIMIT else (
        "INCONCLUSIVE (CI straddles the limit)" if fm - fhalf <= F_LIMIT
        else "FAIL")
    ks = "PASS" if km + khalf <= K_LIMIT else (
        "INCONCLUSIVE (CI straddles the limit)" if km - khalf <= K_LIMIT
        else "FAIL")
    print(f"  g_peak  ({F_LIMIT}) : {fs}")
    print(f"  g_kmax  ({K_LIMIT}): {ks}")
    print("=" * 72)
    if "INCONCLUSIVE" in fs + ks:
        print("to sharpen: re-run with a larger --seeds (completed seeds are "
              "reused from runs.json, so only the new ones cost time)")
    (outdir / "summary.json").write_text(json.dumps(
        dict(design=design, gd_pins=pins, n_core=len(fdh), n_asm=len(kb),
             fdh=fdh, fdh_mean=fm, fdh_sd=fsd, fdh_sem=fsem, fdh_ci95=fhalf,
             k_bol=kb, k_mean=km, k_sd=ksd_, k_ci95=khalf,
             keff_core_mean=st.mean(kcore),
             verdict_peak=fs, verdict_k=ks), indent=1))
    print(f"\nsummary -> {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
