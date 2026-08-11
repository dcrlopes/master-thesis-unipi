#!/usr/bin/env python3
"""
validate_core_proxy.py -- the study that closes the biggest open point of the
thesis: how much does the SINGLE-ASSEMBLY optimisation proxy miss at CORE level?

For each selected finalist it builds the full 32-assembly core (the same
make_core_model used for the k-target calibration), runs BOL (Beginning of
Life) transport only -- no depletion -- and measures three things the
assembly model cannot see:

  1. CORE PEAKING vs ASSEMBLY PEAKING. A pin-resolved mesh over the whole
     core gives the true core-wide F_dH; an assembly-resolved mesh gives the
     radial assembly peaking F_asm. The ratio (core pin F_dH) / (assembly
     F_dH stored in the checkpoint) is the proxy gap -- the number the
     examiners will ask for. Radial enrichment zoning exists to flatten CORE
     power, which an infinite lattice cannot price; this measurement prices it.

  2. ROUTE-B CLOSURE. The optimiser's end-of-cycle criterion assumes
     k_target(refl_thick) = k_inf(assembly)/k_eff(core). Here both sides are
     measured for the SAME design: the checkpoint's stored k_bol and the
     core run's k_eff give an implied k_target, compared against the
     calibration table's value at this design's reflector thickness. Small
     residuals validate the whole Route-B construction on the final designs.

  3. SOURCE CONVERGENCE AT CORE SCALE. The single-assembly entropy study
     showed a flat H(batch) because the converged source there is uniform;
     the finite core has a REAL spatial transient. The entropy trace of each
     run is checked against the inactive cutoff (needs the entropy-mesh patch
     in reactor_model).

Runs are seed-replicated (default 3) so every reported number carries a
spread, and every completed run is checkpointed to <out>/runs.json --
interrupting and relaunching resumes.

USAGE
  conda activate openmc-env
  # front designs, 3 seeds each (default):
  nohup python -u validate_core_proxy.py \
      --checkpoint out_c3_atf75/optimization_checkpoint.json \
      --front --threads 64 --out core_check > core_check.log 2>&1 &
  # or explicit archive indices:  --designs 12 47 63
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import openmc

import core_geometry as cg
import reactor_model as rm

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--front", action="store_true",
                help="validate every design on the feasible Pareto front")
ap.add_argument("--designs", type=int, nargs="*", default=None,
                help="explicit indices into all_raw (overrides --front)")
ap.add_argument("--particles", type=int, default=100000,
                help="per-batch particles for the core run (default 100000; "
                     "the core is ~30x the assembly, so per-pin statistics "
                     "at 100k are comparable to the assembly at 16k)")
ap.add_argument("--batches", type=int, default=150)
ap.add_argument("--inactive", type=int, default=40,
                help="the finite core has a REAL source transient; 40 is a "
                     "prudent default, and the entropy check verifies it")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--ktarget-table", default="ktarget_table.json",
                help="Route-B calibration table for the closure check")
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--out", default="core_check")
args = ap.parse_args()

if args.threads:
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

ck = json.loads(Path(args.checkpoint).read_text())
raw, cn, dv = ck["all_raw"], ck.get("constraint_names", []), ck["design_variables"]


def feasible(r, tol=1e-9):
    return all(float(r.get(c, 0.0)) <= tol for c in cn)


if args.designs:
    picks = [(i, raw[i]) for i in args.designs]
else:
    feas = [(i, r) for i, r in enumerate(raw) if feasible(r)]
    picks = [(i, a) for i, a in feas
             if not any((b["cycle_length"] >= a["cycle_length"]
                         and b["peaking"] <= a["peaking"]
                         and (b["cycle_length"] > a["cycle_length"]
                              or b["peaking"] < a["peaking"]))
                        for _, b in feas)]
    picks.sort(key=lambda t: -t[1]["cycle_length"])

# Route-B table: k_target(pitch, refl_thick), bilinearly interpolated.
# Real schema (schema 2): {"pitch_cm":[...], "refl_thick_cm":[...],
#                          "k_target":[[...per refl...] per pitch]}
ktab = None
tp = Path(args.ktarget_table)
if tp.exists():
    tj = json.loads(tp.read_text())
    if isinstance(tj, dict) and "k_target" in tj and "pitch_cm" in tj:
        kp = np.asarray(tj["pitch_cm"], float)
        kr = np.asarray(tj["refl_thick_cm"], float)
        kv = np.asarray(tj["k_target"], float)          # shape (n_pitch, n_refl)
        ktab = (kp, kr, kv)
    else:
        print("!! WARNING: unrecognised k_target table schema; "
              "Route-B closure check skipped.")


def k_target_at(pitch, refl):
    """Bilinear interpolation on the (pitch, refl_thick) grid, clamped to it."""
    kp, kr, kv = ktab
    p = float(np.clip(pitch, kp[0], kp[-1]))
    t = float(np.clip(refl, kr[0], kr[-1]))
    i = int(np.clip(np.searchsorted(kp, p) - 1, 0, len(kp) - 2))
    j = int(np.clip(np.searchsorted(kr, t) - 1, 0, len(kr) - 2))
    fp = 0.0 if kp[i + 1] == kp[i] else (p - kp[i]) / (kp[i + 1] - kp[i])
    ft = 0.0 if kr[j + 1] == kr[j] else (t - kr[j]) / (kr[j + 1] - kr[j])
    return float((1 - fp) * ((1 - ft) * kv[i, j] + ft * kv[i, j + 1])
                 + fp * ((1 - ft) * kv[i + 1, j] + ft * kv[i + 1, j + 1]))

outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)
store = outdir / "runs.json"
done = json.loads(store.read_text()) if store.exists() else {}
geo, op = rm.Geometry17x17(), rm.Operating()
NL = geo.lattice                      # 17
cmap = cg.CORE_MAP_32
ny, nx = cmap.shape                   # 6 x 6


def one_core_run(design, seed, case):
    m = rm.make_core_model(design, op, geo,
                           particles=args.particles, batches=args.batches,
                           inactive=args.inactive)
    model = m[0] if isinstance(m, tuple) else m
    model.settings.seed = seed

    pitch = design.get("pitch", 1.26)
    A = NL * pitch
    half = nx * A / 2.0
    pin = openmc.RegularMesh(); pin.dimension = (nx * NL, ny * NL)
    pin.lower_left = (-half, -half); pin.upper_right = (half, half)
    asm = openmc.RegularMesh(); asm.dimension = (nx, ny)
    asm.lower_left = (-half, -half); asm.upper_right = (half, half)
    t1 = openmc.Tally(name="core_pin");  t1.filters = [openmc.MeshFilter(pin)]
    t1.scores = ["fission"]
    t2 = openmc.Tally(name="core_asm");  t2.filters = [openmc.MeshFilter(asm)]
    t2.scores = ["fission"]
    model.tallies = openmc.Tallies([t1, t2])

    t0 = time.time()
    sp_path = model.run(cwd=str(case), output=False)
    wall = time.time() - t0
    with openmc.StatePoint(sp_path) as sp:
        keff = float(sp.keff.nominal_value)
        kerr = float(sp.keff.std_dev)
        vp = sp.get_tally(name="core_pin").get_values(
            scores=["fission"]).reshape(ny * NL, nx * NL)
        sdp = sp.get_tally(name="core_pin").get_values(
            scores=["fission"], value="std_dev").reshape(ny * NL, nx * NL)
        va = sp.get_tally(name="core_asm").get_values(
            scores=["fission"]).reshape(ny, nx)
        H = np.asarray(getattr(sp, "entropy", []), dtype=float)

    fp = np.ma.masked_equal(vp, 0.0)          # reflector / GT / corner cells
    npin = fp / fp.mean()
    fdh_core = float(npin.max())
    hi = np.unravel_index(np.ma.argmax(npin), npin.shape)
    rel = float(sdp[hi] / vp[hi])
    fa = np.ma.masked_equal(va, 0.0)
    fasm = float((fa / fa.mean()).max())
    hot_asm = tuple(int(x) for x in
                    np.unravel_index(np.ma.argmax(fa / fa.mean()), fa.shape))

    # entropy convergence: smoothed +-3 sigma of the active tail
    conv = None
    if H.size:
        tail = H[args.inactive + (len(H) - args.inactive) // 2:]
        mu, sd = tail.mean(), tail.std(ddof=1)
        Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
        Hs[0], Hs[-1] = H[0], H[-1]
        out_i = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
        conv = int(out_i[-1]) + 2 if len(out_i) else 1
    return dict(keff=keff, kerr=kerr, fdh_core=fdh_core, f_asm=fasm,
                hot_pin=[int(hi[0]), int(hi[1])], hot_asm=list(hot_asm),
                hotpin_rel_err=rel, entropy_conv_batch=conv, wall_s=wall)


rows = []
print("=" * 84)
print(f"core validation: {args.particles} particles x {args.batches} batches "
      f"({args.inactive} inactive), {args.seeds} seeds")
print(f"designs: {[i for i, _ in picks]}")
print("=" * 84)
for idx, rec in picks:
    design = {k: float(rec[k]) for k in dv}
    print(f"\n[design {idx}] assembly F_dH={rec['peaking']:.4f}  "
          f"k_bol={rec.get('k_bol', float('nan')):.4f}  "
          f"EFPD={rec['cycle_length']:.0f}  "
          + "  ".join(f"{k}={design[k]:.3f}" for k in dv))
    vals = {"keff": [], "fdh": [], "fasm": []}
    for s in range(1, args.seeds + 1):
        key = f"d{idx}_s{s}"
        if key not in done:
            case = outdir / f"design{idx}" / f"seed{s}"
            case.mkdir(parents=True, exist_ok=True)
            done[key] = one_core_run(design, s, case)
            store.write_text(json.dumps(done, indent=1))
        r = done[key]
        vals["keff"].append(r["keff"]); vals["fdh"].append(r["fdh_core"])
        vals["fasm"].append(r["f_asm"])
        print(f"   seed {s}: k_eff={r['keff']:.5f}({r['kerr']:.5f})  "
              f"core F_dH={r['fdh_core']:.4f} (rel.err {r['hotpin_rel_err']*100:.1f}%)"
              f"  F_asm={r['f_asm']:.4f}  hot asm {tuple(r['hot_asm'])}  "
              f"entropy conv b{r['entropy_conv_batch']}  ({r['wall_s']:.0f}s)")

    mk, mf, ma = (st.mean(vals[k]) for k in ("keff", "fdh", "fasm"))
    sf = st.stdev(vals["fdh"]) if len(vals["fdh"]) > 1 else 0.0
    ratio = mf / rec["peaking"]
    row = dict(idx=idx, k_bol=rec.get("k_bol"), keff=mk,
               fdh_asm=rec["peaking"], fdh_core=mf, fdh_core_sd=sf,
               f_asm=ma, proxy_ratio=ratio)
    # Route-B closure
    if ktab is not None and rec.get("k_bol"):
        kt_imp = rec["k_bol"] / mk
        kt_tab = k_target_at(design["pitch"], design["refl_thick"])
        row.update(kt_implied=kt_imp, kt_table=kt_tab,
                   kt_resid_pcm=1e5 * (kt_imp - kt_tab) / kt_tab)
    rows.append(row)
    print(f"   -> core F_dH {mf:.4f} +/- {sf:.4f}   assembly proxy "
          f"{rec['peaking']:.4f}   RATIO core/assembly = {ratio:.3f}")
    print(f"   -> radial assembly peaking F_asm = {ma:.4f}")
    if "kt_implied" in row:
        print(f"   -> Route-B closure: implied k_target={row['kt_implied']:.4f}"
              f"  table={row['kt_table']:.4f}"
              f"  residual={row['kt_resid_pcm']:+.0f} pcm")

print("\n" + "=" * 84)
print("SUMMARY -- the single-assembly proxy gap")
print("=" * 84)
print(f"{'idx':>4s} {'F_dH asm':>9s} {'F_dH core':>10s} {'ratio':>6s} "
      f"{'F_asm':>6s} {'k_eff':>8s} {'kt resid':>9s}")
for r in rows:
    kt = f"{r.get('kt_resid_pcm', float('nan')):+8.0f}" \
         if "kt_resid_pcm" in r else "     n/a"
    print(f"{r['idx']:>4d} {r['fdh_asm']:9.4f} {r['fdh_core']:10.4f} "
          f"{r['proxy_ratio']:6.3f} {r['f_asm']:6.3f} {r['keff']:8.5f} {kt}")
csv = outdir / "core_validation.csv"
cols = list(rows[0].keys())
csv.write_text(",".join(cols) + "\n" + "\n".join(
    ",".join(str(r.get(c, "")) for c in cols) for r in rows) + "\n")
print(f"\nCSV: {csv}   raw (resumable): {store}")
print("\nReading the ratio: ~1.0x means intra-assembly structure dominates and "
      "the proxy holds;\nsubstantially above 1 means core-level radial shape "
      "adds peaking the optimiser never saw --\nquote the ratio as the proxy "
      "gap and revisit the near-uniform-zoning conclusion in that light.")
