#!/usr/bin/env python3
r"""
axial_shape_c9.py -- axial power shape, assembly and pin power maps, and axial
rod worth on the hardware model, for the Campaign 9 candidates.

Nothing in the repository tallied an axial shape before this script. The
radial mesh of confirm3d.py is axially integrated, and axial_leakage_study.py
measures leakage worth and L_ax, not a shape. This script reuses exactly the
model that confirm3d.py solves, hardware3d.build_model_3d_hw with the same
zoning, spec, rod maps and fidelity, and adds three mesh tallies:

  assembly mesh   6 x 6 assemblies over the active fuel height
  pin mesh        (6*17) x (6*17) pins over the active fuel height
  axial mesh      bins along z that follow the spacer-grid bands, so the
                  grid depressions are resolved as in the NuScale benchmark

From them, per design, state and seed:

  k, sd           eigenvalue and its Monte Carlo uncertainty
  F_dH            radial hot channel factor, same rule as confirm3d.py
  F_z             axial peaking factor, max over bins of (p_i / p_mean)
                  computed per unit length, so a thin grid bin is not
                  penalised for being thin
  AO              axial offset, (P_top - P_bottom) / (P_top + P_bottom)
  P_i             power per axial bin, normalised to op.power_mwth, MW
  rho(state)      reactivity of each rodded state relative to ARO, pcm

This script runs transport and saves the tally arrays. It draws nothing.
Figures come from axial_figures_c9.py, which reads <out>/ and needs only
numpy and matplotlib, so it can run on the laptop.

Resumable: finished solves are cached in <out>/runs.json and the tally
arrays in <out>/d<idx>/<state>_s<seed>.npz. Re-running skips finished solves.

USAGE (wks720, conda env openmc-env, from the repository root)
    python -c "import numpy, openmc; print('env ok')" && \
    python axial_shape_c9.py --selftest && \
    python axial_shape_c9.py --checkpoint out_c9/optimization_checkpoint.json \
        --designs 47 44 34 40 35 58 12 --states ARO RE12 --seeds 2 \
        --threads 64 --out axial_c9 --dry-run
then drop --dry-run and launch detached:
    setsid nohup python -u axial_shape_c9.py --checkpoint out_c9/optimization_checkpoint.json \
        --designs 47 44 34 40 35 58 12 --states ARO RE12 --seeds 2 \
        --threads 64 --out axial_c9 > axial_c9.log 2>&1 < /dev/null &
then, anywhere with the axial_c9/ directory:
    python axial_figures_c9.py --out axial_c9 --champion 47

Cost: one hardware solve at 150000 x 200 is about 215 s, so 7 designs x 2
states x 2 seeds is 28 solves, about 1.7 h. Add ARI to --states for the
four-bank shape at a further 14 solves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

NL = 17                     # pins per assembly side
AXIAL_TARGET_CM = 5.0       # bin length between grid bands
STATE_ORDER = ["ARO", "RE12", "ARI"]


# ---------------------------------------------------------------- helpers --
def design_from(ckpt, idx):
    """Same six physical keys confirm3d.py reads, so the lattice is identical."""
    r = ckpt["all_raw"][idx]
    return {k: float(r[k]) for k in ("enrich_inner", "enrich_outer", "gd_wt",
                                      "pitch", "refl_thick", "gd_pins")}


def axial_edges(spec, hw):
    """Bin edges along z inside the active fuel that follow the grid bands."""
    z = hw.elevations(spec)
    lo, hi = z["fuel"]
    cuts = {lo, hi}
    bands = []
    for g0, g1, _ in z["grids"]:
        if g0 >= lo and g1 <= hi:
            cuts.update((g0, g1)); bands.append((g0, g1))
    cuts = sorted(cuts)
    edges = [cuts[0]]
    for a, b in zip(cuts[:-1], cuts[1:]):
        is_band = any(abs(a - g0) < 1e-9 and abs(b - g1) < 1e-9 for g0, g1 in bands)
        n = 1 if is_band else max(1, int(round((b - a) / AXIAL_TARGET_CM)))
        edges.extend(np.linspace(a, b, n + 1)[1:].tolist())
    return np.array(edges), bands, (lo, hi)


def build_tallies(o, geo, pitch, edges, hz):
    """Three mesh tallies over the active fuel. Returns an openmc.Tallies."""
    import core_geometry as cg
    core_map = np.asarray(cg.CORE_MAP_32); ny, nx = core_map.shape
    asm = NL * pitch; half = nx * asm / 2.0

    m_asm = o.RectilinearMesh(name="assembly")
    m_asm.x_grid = np.linspace(-half, half, nx + 1)
    m_asm.y_grid = np.linspace(-half, half, ny + 1)
    m_asm.z_grid = np.array([-hz, hz])
    t_asm = o.Tally(name="asm_fission"); t_asm.filters = [o.MeshFilter(m_asm)]
    t_asm.scores = ["fission"]

    m_pin = o.RectilinearMesh(name="pin")
    m_pin.x_grid = np.linspace(-half, half, nx * NL + 1)
    m_pin.y_grid = np.linspace(-half, half, ny * NL + 1)
    m_pin.z_grid = np.array([-hz, hz])
    t_pin = o.Tally(name="pin_fission"); t_pin.filters = [o.MeshFilter(m_pin)]
    t_pin.scores = ["fission"]

    m_ax = o.RectilinearMesh(name="axial")
    m_ax.x_grid = np.array([-half, half]); m_ax.y_grid = np.array([-half, half])
    m_ax.z_grid = edges
    t_ax = o.Tally(name="axial_fission"); t_ax.filters = [o.MeshFilter(m_ax)]
    t_ax.scores = ["fission"]

    return o.Tallies([t_asm, t_pin, t_ax]), (ny, nx)


def read_statepoint(o, sp_path, shape_asm, edges, inactive):
    ny, nx = shape_asm
    with o.StatePoint(sp_path) as sp:
        keff = float(sp.keff.nominal_value); sd = float(sp.keff.std_dev)
        asm = sp.get_tally(name="asm_fission").get_values(scores=["fission"])
        pin = sp.get_tally(name="pin_fission").get_values(scores=["fission"])
        ax = sp.get_tally(name="axial_fission").get_values(scores=["fission"]).ravel()
        H = np.asarray(getattr(sp, "entropy", []), dtype=float)
    # OpenMC orders mesh bins with x fastest, then y, then z. One z bin here.
    asm = asm.ravel().reshape(ny, nx)
    pin = pin.ravel().reshape(ny * NL, nx * NL)
    conv = None
    if H.size:
        tail = H[inactive + (len(H) - inactive) // 2:]
        mu, s = float(tail.mean()), float(tail.std(ddof=1))
        Hs = np.convolve(H, np.ones(3) / 3.0, mode="same"); Hs[0], Hs[-1] = H[0], H[-1]
        bad = np.where(~((Hs >= mu - 3 * s) & (Hs <= mu + 3 * s)))[0]
        conv = int(bad[-1]) + 2 if len(bad) else 1
    return keff, sd, asm, pin, ax, conv


def derive(asm, pin, ax, edges, power_mw):
    """Normalised maps and the axial figures of merit."""
    f = np.ma.masked_equal(pin, 0.0)
    fdh = float((f / f.mean()).max())
    a = np.ma.masked_equal(asm, 0.0)
    asm_norm = np.where(asm > 0, asm / a.mean(), np.nan)
    dz = np.diff(edges)
    lin = ax / dz                                   # fission per unit length
    fz = float(lin.max() / np.average(lin, weights=dz))
    zc = 0.5 * (edges[:-1] + edges[1:])
    top = ax[zc > 0].sum(); bot = ax[zc <= 0].sum()
    ao = float((top - bot) / (top + bot))
    p_mw = ax / ax.sum() * power_mw
    return dict(fdh=fdh, fz=fz, ao=ao, asm_norm=asm_norm, p_mw=p_mw,
                shape=lin / np.average(lin, weights=dz))


# ------------------------------------------------------------------ selftest --
def selftest():
    import hardware3d as hw
    spec = hw.HardwareSpec()
    edges, bands, fuel = axial_edges(spec, hw)
    assert abs(edges[0] - fuel[0]) < 1e-9 and abs(edges[-1] - fuel[1]) < 1e-9
    assert np.all(np.diff(edges) > 0)
    for g0, g1 in bands:
        assert any(abs(e - g0) < 1e-9 for e in edges) and any(abs(e - g1) < 1e-9 for e in edges)
    dz = np.diff(edges)
    print(f"selftest: {len(edges) - 1} axial bins over {fuel[0]:.1f} to {fuel[1]:.1f} cm, "
          f"{len(bands)} grid bands resolved, bin length {dz.min():.2f} to {dz.max():.2f} cm")
    # derive() on a synthetic cosine shape with grid dips
    zc = 0.5 * (edges[:-1] + edges[1:])
    lin = np.cos(np.pi * zc / (fuel[1] - fuel[0]) * 0.9) + 0.05
    for g0, g1 in bands:
        lin[(zc > g0) & (zc < g1)] *= 0.9
    ax = lin * dz
    pin = np.random.default_rng(0).uniform(0.8, 1.2, (6 * NL, 6 * NL)); pin[:NL, :NL] = 0
    asm = np.ones((6, 6)); asm[0, 0] = asm[0, -1] = asm[-1, 0] = asm[-1, -1] = 0
    d = derive(asm, pin, ax, edges, 48.0)
    assert abs(d["p_mw"].sum() - 48.0) < 1e-9 and 1.0 < d["fz"] < 2.0 and abs(d["ao"]) < 0.05
    print(f"selftest: F_z {d['fz']:.3f}, AO {d['ao']:+.4f}, power sums to {d['p_mw'].sum():.1f} MW")
    print("selftest OK")
    return 0


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint"); ap.add_argument("--designs", type=int, nargs="*", default=[])
    ap.add_argument("--states", nargs="*", default=["ARO", "RE12"])
    ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--particles", type=int, default=150000); ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--inactive", type=int, default=80)
    ap.add_argument("--m-center", type=float, default=0.72); ap.add_argument("--m-periphery", type=float, default=1.15)
    ap.add_argument("--boron-ppm", type=float, default=1000.0)
    ap.add_argument("--out", default="axial_c9")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.checkpoint or not a.designs:
        print("FAIL: --checkpoint and --designs required"); return 2
    for s in a.states:
        if s not in STATE_ORDER:
            print(f"FAIL: unknown state {s}, use {STATE_ORDER}"); return 2

    print(f"python : {sys.version.split()[0]}   cwd: {os.getcwd()}   host: {os.uname().nodename}")
    try:
        import openmc as o, hardware3d as hw, reactor_model as rm, zoning as zn
        print(f"openmc : {o.__version__}   XS: {os.environ.get('OPENMC_CROSS_SECTIONS')}")
    except Exception as e:
        print(f"FAIL: {e}. Activate openmc-env."); return 2
    os.environ["OMP_NUM_THREADS"] = str(a.threads)

    ckpt = json.loads(Path(a.checkpoint).read_text())
    spec = hw.HardwareSpec()
    geo, op = rm.Geometry17x17(), rm.Operating()
    op.boron_ppm = float(a.boron_ppm)
    rmap = zn.ring_map(); nC, nM, nP = zn.ring_counts(rmap)
    m_m = (32 - nC * a.m_center - nP * a.m_periphery) / nM
    rodded = {"ARO": None, "ARI": (set(zn.RE_BANK_POSITIONS), "B4C"),
              "RE12": (set(zn.RE12_POSITIONS), "B4C")}
    fid = dict(particles=a.particles, batches=a.batches, inactive=a.inactive)
    if a.smoke:
        fid = dict(particles=5000, batches=40, inactive=15); a.seeds = 1
        print("SMOKE: 5000 x 40 (15 inactive), one seed")
    edges, bands, fuel = axial_edges(spec, hw)
    hz = 0.5 * spec.h_active
    print(f"axial  : {len(edges) - 1} bins, {len(bands)} grid bands resolved, "
          f"boron {op.boron_ppm:.0f} ppm, {a.seeds} seeds, states {a.states}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache_p = out / "runs.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    results = {}
    t_all = time.time()
    for idx in a.designs:
        d = design_from(ckpt, idx)
        print(f"\n=== design C9-{idx}: e {d['enrich_outer']:.3f} wt%, Gd {d['gd_wt']:.2f} wt%, "
              f"pins {d['gd_pins']:.0f}, refl {d['refl_thick']:.2f} cm")
        if a.dry_run:
            continue
        zdes = zn.zone_designs(d, a.m_center, m_m, a.m_periphery)
        dmap = zn.design_map_for(rmap, zdes)
        ddir = out / f"d{idx}"; ddir.mkdir(exist_ok=True)
        per_state = {}
        for st in a.states:
            recs, arrays = [], []
            for seed in range(1, a.seeds + 1):
                key = f"{idx}|{st}|{seed}|{fid['particles']}x{fid['batches']}x{fid['inactive']}|{op.boron_ppm:g}"
                npz = ddir / f"{st}_s{seed}.npz"
                if key in cache and npz.exists():
                    rec = cache[key]; z = np.load(npz)
                    asm, pin, ax = z["asm"], z["pin"], z["ax"]
                else:
                    model, info = hw.build_model_3d_hw(d, op, geo, design_map=dmap,
                                                       rodded_map=rodded[st], spec=spec,
                                                       seed=seed, **fid)
                    tallies, shape_asm = build_tallies(o, geo, d["pitch"], edges, hz)
                    model.tallies = tallies
                    case = ddir / f"case_{st}_s{seed}"; case.mkdir(exist_ok=True)
                    t0 = time.time()
                    sp_path = model.run(cwd=str(case), output=False)
                    wall = time.time() - t0
                    keff, sd, asm, pin, ax, conv = read_statepoint(o, sp_path, shape_asm, edges,
                                                                    fid["inactive"])
                    np.savez_compressed(npz, asm=asm, pin=pin, ax=ax, edges=edges)
                    rec = dict(keff=keff, sd=sd, entropy_conv=conv, wall_s=round(wall, 1))
                    cache[key] = rec
                    cache_p.write_text(json.dumps(cache, indent=1))
                dv = derive(asm, pin, ax, edges, op.power_mwth)
                print(f"  C9-{idx} {st:4s} seed {seed}: k = {rec.get('keff', float('nan')):.5f} "
                      f"+/- {rec.get('sd', float('nan')):.5f}  F_dH {dv['fdh']:.3f}  "
                      f"F_z {dv['fz']:.3f}  AO {dv['ao']:+.3f}  "
                      f"entropy_conv {rec.get('entropy_conv')} ({rec.get('wall_s')} s)", flush=True)
                recs.append(rec); arrays.append((asm, pin, ax))
            if not recs:
                continue
            asm = np.mean([x[0] for x in arrays], axis=0); pin = np.mean([x[1] for x in arrays], axis=0)
            ax = np.mean([x[2] for x in arrays], axis=0)
            dv = derive(asm, pin, ax, edges, op.power_mwth)
            ks = np.array([r["keff"] for r in recs]); sds = np.array([r["sd"] for r in recs])
            per_state[st] = dict(keff=float(ks.mean()), sd=float(np.sqrt((sds ** 2).sum()) / len(sds)),
                                 fdh=dv["fdh"], fz=dv["fz"], ao=dv["ao"], n_seeds=len(recs))
        if "ARO" in per_state:
            k0 = per_state["ARO"]["keff"]
            for st in per_state:
                per_state[st]["rho_vs_ARO_pcm"] = float((1.0 / k0 - 1.0 / per_state[st]["keff"]) * 1e5)
        results[idx] = per_state

    if a.dry_run:
        print("\ndry run, nothing solved"); return 0

    # summary
    summ = {str(i): {s: {k: v for k, v in r.items()
                         if k in ("keff", "sd", "fdh", "fz", "ao", "n_seeds", "rho_vs_ARO_pcm")}
                     for s, r in ps.items()} for i, ps in results.items()}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    L = ["\\begin{table}[htbp]", "  \\centering",
         "  \\caption{Axial peaking factor $F_z$, axial offset AO and radial hot channel factor "
         "$F_{\\Delta H}$ on the hardware model at \\SI{1000}{ppm}, with the reactivity of each "
         "rodded state relative to all rods out.}", "  \\label{tab:c9-axial}",
         "  \\begin{tabular}{llrrrrr}", "    \\toprule",
         "    design & state & $k$ & $F_{\\Delta H}$ & $F_z$ & AO & $\\rho$ vs ARO (pcm) \\\\", "    \\midrule"]
    for i, ps in sorted(results.items()):
        for s in STATE_ORDER:
            if s in ps:
                r = ps[s]
                L.append(f"    C9-{i} & {s} & {r['keff']:.5f} & {r['fdh']:.3f} & {r['fz']:.3f} & "
                         f"{r['ao']:+.3f} & {r.get('rho_vs_ARO_pcm', 0):+.0f} \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    (out / "summary_table.tex").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out}/summary.json and {out}/summary_table.tex")
    print(f"total {time.time() - t_all:.0f} s")
    print(f"figures: python axial_figures_c9.py --out {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
