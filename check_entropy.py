#!/usr/bin/env python3
"""
check_entropy.py -- Shannon-entropy source-convergence study, run BEFORE
Campaign 3 to justify the (particles, batches, inactive) choice with data.

WHAT IT ANSWERS
---------------
An eigenvalue Monte Carlo (MC) run must discard enough INACTIVE batches for
the fission source to converge before tallies start; otherwise k and every
tally inherit source-convergence bias. The standard diagnostic is the Shannon
entropy H of the fission-site distribution: H(batch) rises/falls while the
source redistributes and goes flat (statistical noise only) once converged.
The inactive count is justified when H is flat WELL BEFORE the inactive ->
active transition.

Campaigns 1-2 never recorded H (no entropy_mesh was set). Apply entropy.patch
first -- this script refuses to run without it.

WHAT IT DOES
------------
Picks three PHYSICALLY EXTREME feasible designs from the Campaign 2 archive
(thinnest reflector, thickest reflector, highest gadolinium load -- the three
drivers of source-distribution character), runs the BOL (Beginning of Life)
transport ONLY (no depletion, ~20 s per run at 16000x120) at each requested
fidelity, and produces:

  * H(batch) figure, one panel per fidelity, inactive cutoff marked
  * a convergence table: batch at which H enters and stays inside the
    +/- 3 sigma band of the converged tail, the margin to the inactive
    cutoff, and PASS/FAIL
  * CSV of every H(batch) series for the thesis appendix

Physics expectation: a single 2D assembly is small and tightly coupled (low
dominance ratio), so H should converge within ~10-15 batches and inactive=30
should PASS with about 2x margin. If a case converges LATER than the cutoff,
raise --inactive for the campaign before launching it.

USAGE
  conda activate openmc-env
  python check_entropy.py \
      --checkpoint results_campaign2/block2/out/optimization_checkpoint.json \
      --fidelity 4000 60 20 --fidelity 16000 120 30 \
      --threads 64 --out entropy_study
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless: no display needed over SSH
import matplotlib.pyplot as plt

import openmc
import reactor_model as rm

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True,
                help="Campaign 2 checkpoint; probe designs are picked from it")
ap.add_argument("--fidelity", type=int, nargs=3, action="append",
                metavar=("PARTICLES", "BATCHES", "INACTIVE"),
                help="one fidelity to test; repeat the flag for a ladder "
                     "(default: 4000 60 20 and 16000 120 30)")
ap.add_argument("--threads", type=int, default=None,
                help="OpenMP (Open Multi-Processing) threads; 64 on wks720")
ap.add_argument("--sigma-band", type=float, default=3.0,
                help="convergence band half-width in units of the converged "
                     "tail's standard deviation (default 3)")
ap.add_argument("--out", default="entropy_study")
args = ap.parse_args()

if args.threads:
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

fidelities = args.fidelity or [[4000, 60, 20], [16000, 120, 30]]

ck = json.loads(Path(args.checkpoint).read_text())
raw, con, dv = ck["all_raw"], ck.get("constraint_names", []), ck["design_variables"]
feas = [r for r in raw
        if all(float(r.get(c, 0.0)) <= 1e-9 for c in con)]

# three physically extreme probes: source shape is driven by how much the
# reflector pulls the flux outward and how hard gadolinium depresses it inside
probes = {
    "thin_reflector":  min(feas, key=lambda r: float(r["refl_thick"])),
    "thick_reflector": max(feas, key=lambda r: float(r["refl_thick"])),
    "high_gadolinium": max(feas, key=lambda r: float(r["gd_wt"])),
}
geo, op = rm.Geometry17x17(), rm.Operating()
outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("probe designs (from the Campaign 2 feasible archive):")
for tag, r in probes.items():
    print(f"  {tag:15s}: " + "  ".join(f"{k}={float(r[k]):.3f}" for k in dv))
print("=" * 78)


def entropy_series(design, particles, batches, inactive, case):
    model, _c, _l = rm.make_assembly_model(
        design, op, geo, bc="reflective",
        particles=particles, batches=batches, inactive=inactive)
    if model.settings.entropy_mesh is None:
        raise SystemExit("ERROR: no entropy mesh in settings -- apply "
                         "entropy.patch to reactor_model.py first.")
    sp_path = model.run(cwd=str(case), output=False)
    with openmc.StatePoint(sp_path) as sp:
        H = np.asarray(sp.entropy, dtype=float)
        keff = float(sp.keff.nominal_value)
    return H, keff


def converged_at(H, inactive, band_sigma):
    """First batch after which H STAYS inside +/- band of the converged tail.

    The tail (reference for 'converged') is the last half of the ACTIVE
    batches -- far from any initial transient by construction.
    """
    tail = H[inactive + (len(H) - inactive) // 2:]
    mu, sd = tail.mean(), tail.std(ddof=1)
    # 3-point centered moving average: a single-batch noise outlier must not
    # masquerade as a late transient (validated: flat traces report batch 1,
    # a tau=25 pathological transient still FAILs against inactive=30)
    Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
    Hs[0], Hs[-1] = H[0], H[-1]
    inside = (Hs >= mu - band_sigma * sd) & (Hs <= mu + band_sigma * sd)
    out_idx = np.where(~inside)[0]
    conv = int(out_idx[-1]) + 2 if len(out_idx) else 1   # 1-based batch number
    return conv, mu, sd


rows = []
fig, axes = plt.subplots(1, len(fidelities),
                         figsize=(6.2 * len(fidelities), 4.6), squeeze=False)
colors = {"thin_reflector": "#4878a8", "thick_reflector": "#d1495b",
          "high_gadolinium": "#3d8f5f"}

for ax, (P, B, I) in zip(axes[0], fidelities):
    for tag, rec in probes.items():
        design = {k: float(rec[k]) for k in dv}
        case = outdir / f"p{P}_b{B}_i{I}" / tag
        case.mkdir(parents=True, exist_ok=True)
        H, keff = entropy_series(design, P, B, I, case)
        conv, mu, sd = converged_at(H, I, args.sigma_band)
        margin = I / conv if conv else float("inf")
        ok = conv <= I
        rows.append(dict(particles=P, batches=B, inactive=I, probe=tag,
                         converged_batch=conv, margin=round(margin, 2),
                         tail_mean=mu, tail_sd=sd, keff=keff,
                         verdict="PASS" if ok else "FAIL"))
        b = np.arange(1, len(H) + 1)
        ax.plot(b, H, lw=1.1, color=colors[tag],
                label=f"{tag} (conv. batch {conv})")
        ax.axhspan(mu - args.sigma_band * sd, mu + args.sigma_band * sd,
                   color=colors[tag], alpha=0.06)
        print(f"[{P}x{B}, inactive {I}] {tag:15s}: converged at batch "
              f"{conv:3d}  (cutoff {I}, margin x{margin:.1f})  "
              f"{'PASS' if ok else '** FAIL **'}   k={keff:.5f}")
    ax.axvline(I, color="black", ls="--", lw=1.3)
    ax.text(I, ax.get_ylim()[1], f"  inactive = {I}", va="top", fontsize=8.5)
    ax.set_title(f"{P} particles x {B} batches", fontsize=10.5)
    ax.set_xlabel("batch")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, loc="lower right")
axes[0][0].set_ylabel("Shannon entropy  H  [bits]")
fig.suptitle("Fission-source convergence: Shannon entropy vs batch "
             "(single-assembly BOL transport)", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(outdir / "entropy_convergence.png", dpi=200)
fig.savefig(outdir / "entropy_convergence.pdf")

csv = outdir / "entropy_convergence.csv"
cols = list(rows[0].keys())
csv.write_text(",".join(cols) + "\n"
               + "\n".join(",".join(str(r[c]) for c in cols) for r in rows)
               + "\n")

worst = max(rows, key=lambda r: r["converged_batch"] / r["inactive"])
print("-" * 78)
print(f"worst case: {worst['probe']} at {worst['particles']}x{worst['batches']}"
      f" -> converged batch {worst['converged_batch']} vs inactive "
      f"{worst['inactive']} ({worst['verdict']})")
print(f"figure: {outdir}/entropy_convergence.pdf   table: {csv}")
if any(r["verdict"] == "FAIL" for r in rows):
    print("\n** At least one case converges AFTER the inactive cutoff. **\n"
          "Raise --inactive for the campaign before launching it.")
