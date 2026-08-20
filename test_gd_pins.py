#!/usr/bin/env python3
"""
test_gd_pins.py -- the two-hour decisive test after Campaign 4.

QUESTION
  Campaign 4 converged with zero feasible designs: the iteration-2 champion
  (archive idx 44) reaches core F_dH = 1.947 but misses k_BOL <= 1.35 by
  +0.0064, with gadolinium CONCENTRATION pinned at its 8 wt% bound. Linear
  scaling of the fitted authority (0.0525 dk over 12 pins) says ~14 pins
  close the gap. Does the direct measurement agree -- and does core peaking
  stay under 2.0 when the pins are added?

WHAT IT DOES
  For each pin count in --pins (default 12 14 16 20):
    1. swaps rm.GD_PIN_POSITIONS for an extended pattern (validated: no
       guide-tube collisions, no duplicates; 16- and 20-pin patterns are
       fully 4-fold symmetric, the 14-pin pattern is 2-fold -- documented)
    2. assembly BOL transport (seed-replicated)  -> k_BOL   vs 1.35
    3. full-core BOL transport (seed-replicated) -> F_dH    vs 2.0
                                                    k_eff_core, entropy conv
  Everything cached in <out>/runs.json; fully resumable. Ends with a verdict
  table and the measured dk-per-pin against the linear prediction.

PATTERNS (base 12 heritage + additions on free fuel cells)
  14: + (4,8), (12,8)                      [2-fold: vertical mirror + 180]
  16: + (4,4), (4,12), (12,4), (12,12)     [4-fold]
  20: 16-pattern + (4,8), (8,4), (8,12), (12,8)   [4-fold]

USAGE
  conda activate openmc-env
  setsid nohup python -u test_gd_pins.py \
      --checkpoint out_c4/optimization_checkpoint.json --design-idx 44 \
      --pins 12 14 16 --seeds 2 --threads 64 --out gd_pin_test \
      > gd_pin_test.log 2>&1 < /dev/null &
"""
import argparse
import json
import math
import statistics as st
import time
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# patterns (module-level so they are testable without OpenMC)
# ---------------------------------------------------------------------------
BASE12 = [(2, 2), (2, 14), (14, 2), (14, 14),
          (6, 6), (6, 10), (10, 6), (10, 10),
          (3, 8), (8, 3), (8, 13), (13, 8)]
ADD = {12: [],
       14: [(4, 8), (12, 8)],
       16: [(4, 4), (4, 12), (12, 4), (12, 12)],
       20: [(4, 4), (4, 12), (12, 4), (12, 12),
            (4, 8), (8, 4), (8, 12), (12, 8)]}
GUIDE_TUBES = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13),
               (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
               (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
               (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
               (13, 3), (13, 13), (14, 5), (14, 8), (14, 11)]


def pattern(n):
    """The n-pin gadolinia pattern, validated."""
    if n not in ADD:
        raise SystemExit(f"no pattern defined for {n} pins "
                         f"(available: {sorted(ADD)})")
    pat = BASE12 + ADD[n]
    assert len(pat) == n, f"pattern for {n} has {len(pat)} entries"
    assert len(set(pat)) == n, f"pattern for {n} has duplicates"
    clash = set(pat) & set(GUIDE_TUBES)
    assert not clash, f"pattern for {n} collides with guide tubes: {clash}"
    assert all(0 <= i <= 16 and 0 <= j <= 16 for i, j in pat)
    return pat


def design_seed(design, salt=""):
    key = json.dumps({k: round(float(v), 10)
                      for k, v in sorted(design.items())}) + salt
    return 1 + zlib.crc32(key.encode()) % 2_000_000_000


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--design-idx", type=int, default=44)
    ap.add_argument("--pins", type=int, nargs="+", default=[12, 14, 16])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--asm-particles", type=int, default=64000)
    ap.add_argument("--asm-batches", type=int, default=140)
    ap.add_argument("--asm-inactive", type=int, default=40)
    ap.add_argument("--core-particles", type=int, default=100000)
    ap.add_argument("--core-batches", type=int, default=170)
    ap.add_argument("--core-inactive", type=int, default=60)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="gd_pin_test")
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
    print(f"design idx{args.design_idx}:",
          " ".join(f"{k}={design[k]:.4f}" for k in dv))
    print(f"archive values: F_core={rec['peaking']:.4f} "
          f"k_bol={rec['k_bol']:.4f} EFPD={rec['cycle_length']:.0f}\n")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    store_p = outdir / "runs.json"
    store = json.loads(store_p.read_text()) if store_p.exists() else {}
    op, geo = rm.Operating(), rm.Geometry17x17()
    N = geo.lattice

    def run_assembly(seed, case):
        m = rm.make_assembly_model(design, op, geo,
                                   particles=args.asm_particles,
                                   batches=args.asm_batches,
                                   inactive=args.asm_inactive)
        model = m[0] if isinstance(m, tuple) else m
        model.settings.seed = seed
        sp_path = model.run(cwd=str(case), output=False)
        with openmc.StatePoint(sp_path) as sp:
            return dict(k=float(sp.keff.nominal_value),
                        ksd=float(sp.keff.std_dev))

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
        fmask = np.ma.masked_equal(v, 0.0)
        fdh = float((fmask / fmask.mean()).max())
        conv = None
        if H.size:
            tail = H[args.core_inactive
                     + (len(H) - args.core_inactive) // 2:]
            mu, sd = float(tail.mean()), float(tail.std(ddof=1))
            Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
            Hs[0], Hs[-1] = H[0], H[-1]
            bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
            conv = int(bad[-1]) + 2 if len(bad) else 1
        return dict(fdh=fdh, keff=keff, conv=conv)

    results = {}
    for n in args.pins:
        pat = pattern(n)
        rm.GD_PIN_POSITIONS = pat        # the swap both builders read
        print(f"=== {n} Gd pins ({len(pat)} positions) ===", flush=True)
        ks, fs, kcs = [], [], []
        for s in range(1, args.seeds + 1):
            seed = design_seed(design, salt=f"gdpins{n}s{s}")
            ka = f"p{n}_asm_s{s}"
            if ka not in store:
                case = outdir / f"pins{n}" / f"asm_s{s}"
                case.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                store[ka] = run_assembly(seed, case)
                store[ka]["wall"] = time.time() - t0
                store_p.write_text(json.dumps(store, indent=1))
            kc = f"p{n}_core_s{s}"
            if kc not in store:
                case = outdir / f"pins{n}" / f"core_s{s}"
                case.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                store[kc] = run_core(seed + 1, case)
                store[kc]["wall"] = time.time() - t0
                store_p.write_text(json.dumps(store, indent=1))
            a, c = store[ka], store[kc]
            flag = ("  !! entropy b%s>i%d" % (c["conv"], args.core_inactive)
                    if c["conv"] and c["conv"] > args.core_inactive else "")
            print(f"  s{s}: k_bol={a['k']:.5f}+/-{a['ksd']:.5f}  "
                  f"F_core={c['fdh']:.4f}  keff_core={c['keff']:.5f}"
                  f"  ({a['wall']:.0f}+{c['wall']:.0f}s){flag}", flush=True)
            ks.append(a["k"]); fs.append(c["fdh"]); kcs.append(c["keff"])
        results[n] = dict(
            k=st.mean(ks), ksem=(st.stdev(ks) / math.sqrt(len(ks))
                                 if len(ks) > 1 else float("nan")),
            f=st.mean(fs), fsem=(st.stdev(fs) / math.sqrt(len(fs))
                                 if len(fs) > 1 else float("nan")),
            keff=st.mean(kcs))

    # ---------------- verdict ------------------------------------------------
    print("\n" + "=" * 74)
    print(f"{'pins':>5} {'k_BOL':>9} {'g_kmax':>8} {'F_core':>9} "
          f"{'g_peak':>8} {'keff_core':>10}  verdict")
    print("-" * 74)
    for n in sorted(results):
        r = results[n]
        gk, gp = r["k"] - 1.35, r["f"] - 2.0
        ok = gk <= 0 and gp <= 0
        v = "FEASIBLE" if ok else ("k over" if gk > 0 else "peaking over")
        print(f"{n:>5} {r['k']:>9.5f} {gk:>+8.4f} {r['f']:>9.4f} "
              f"{gp:>+8.4f} {r['keff']:>10.5f}  {v}")
    print("=" * 74)
    ns = sorted(results)
    if len(ns) >= 2:
        n0, n1 = ns[0], ns[-1]
        dk_pin = (results[n0]["k"] - results[n1]["k"]) / (n1 - n0)
        print(f"measured authority: {dk_pin*1000:.2f} pcm... "
              f"dk per added pin = {dk_pin:.5f}")
        print(f"linear prediction from the 12-pin fit: "
              f"{0.0525/12:.5f} per pin")
    (outdir / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"\nsummary: {outdir/'summary.json'}   raw: {store_p}")


if __name__ == "__main__":
    main()
