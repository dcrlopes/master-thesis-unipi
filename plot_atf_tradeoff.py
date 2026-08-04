#!/usr/bin/env python3
"""
plot_atf_tradeoff.py -- objective-space figure showing what an ATF (Accident
Tolerant Fuel) discharge burnup limit does to the Pareto front.

Reads an optimization checkpoint (the file written by save_checkpoint(), which
holds ALL real evaluations in `all_raw`) and produces a two-panel figure:

  LEFT  : uncapped objective space. All 84 evaluations, feasible/infeasible
          separated, censored designs flagged, the uncapped Pareto front
          joined by a step line, and horizontal lines at the EFPD (Effective
          Full Power Days) equivalent of each candidate burnup limit.
  RIGHT : the same archive after applying each limit, showing the front
          collapsing onto the cap.

Needs only numpy + matplotlib. No OpenMC (Open source Monte Carlo particle
transport code), so it runs in the base environment.

USAGE
  python3 plot_atf_tradeoff.py \
      --checkpoint block2/out/optimization_checkpoint.json \
      --limits 62 75 --out figs/atf_tradeoff.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend: no display needed over SSH
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--limits", type=float, nargs="+", default=[62.0, 75.0],
                help="burnup limits to overlay [MWd/kgHM]; 62 = current LWR "
                     "licensing, 75 = typical ATF programme target")
ap.add_argument("--out", default="atf_tradeoff.png")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--peak-limit", type=float, default=None,
                help="draw the F_dH constraint line (e.g. 2.0). Off by "
                     "default because it is far outside the data range.")
args = ap.parse_args()

ck = json.loads(Path(args.checkpoint).read_text())
raw = ck["all_raw"]
con = ck.get("constraint_names", [])

# --- specific power, recovered exactly as apply_atf_limit.py does ---------- #
ratios = sorted(r["bu_eoc_mwd_kg"] * 1000.0 / r["cycle_length"]
                for r in raw
                if r.get("cycle_length", 0) > 0 and r.get("bu_eoc_mwd_kg", 0) > 0)
spec_power = ratios[len(ratios) // 2]


def feasible(r, tol=1e-9):
    """All constraints use the <= 0 convention."""
    return all(float(r.get(c, 0.0)) <= tol for c in con)


def pareto(pts):
    """pts = list of (efpd, peaking, record). Maximise efpd, minimise peaking."""
    out = []
    for e, p, r in pts:
        if not any((e2 >= e and p2 <= p and (e2 > e or p2 < p))
                   for e2, p2, _ in pts):
            out.append((e, p, r))
    return sorted(out, key=lambda t: t[0])


feas = [r for r in raw if feasible(r)]
infe = [r for r in raw if not feasible(r)]

F = np.array([[r["cycle_length"], r["peaking"]] for r in feas])
I = (np.array([[r["cycle_length"], r["peaking"]] for r in infe])
     if infe else np.empty((0, 2)))
cen = np.array([bool(r.get("censored")) for r in feas])

front = pareto([(r["cycle_length"], r["peaking"], r) for r in feas])
FX = np.array([[e, p] for e, p, _ in front])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
colors = plt.cm.viridis(np.linspace(0.15, 0.78, len(args.limits)))

# =========================== LEFT: uncapped ================================ #
if len(I):
    axL.scatter(I[:, 0], I[:, 1], s=26, marker="x", c="0.72",
                linewidths=1.0, label=f"infeasible ({len(I)})", zorder=2)
axL.scatter(F[~cen, 0], F[~cen, 1], s=34, c="#4878a8", edgecolors="white",
            linewidths=0.4, label=f"feasible, exact ({int((~cen).sum())})", zorder=3)
if cen.any():
    axL.scatter(F[cen, 0], F[cen, 1], s=44, marker="^", c="#d1495b",
                edgecolors="white", linewidths=0.4,
                label=f"censored at ceiling ({int(cen.sum())})", zorder=4)
axL.step(FX[:, 0], FX[:, 1], where="post", color="#111111", lw=1.4,
         alpha=0.85, zorder=5)
axL.scatter(FX[:, 0], FX[:, 1], s=95, facecolors="none", edgecolors="#111111",
            linewidths=1.5, label=f"Pareto front ({len(front)})", zorder=6)

for c, L in zip(colors, args.limits):
    cap = L * 1000.0 / spec_power
    axL.axvline(cap, color=c, ls="--", lw=1.5,
                label=f"{L:g} MWd/kgHM = {cap:.0f} EFPD")

axL.set_title("Uncapped objective space", fontsize=11)
axL.set_xlabel("cycle length [EFPD]")
axL.set_ylabel(r"radial peaking factor  $F_{\Delta H}$")
axL.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
axL.grid(alpha=0.25, lw=0.6)

# =========================== RIGHT: capped ================================= #
axR.scatter(F[:, 0], F[:, 1], s=22, c="0.82", edgecolors="none",
            label="uncapped feasible", zorder=2)

for c, L in zip(colors, args.limits):
    cap = L * 1000.0 / spec_power
    pts = [(min(r["cycle_length"], cap), r["peaking"], r) for r in feas]
    fr = pareto(pts)
    A = np.array([[e, p] for e, p, _ in pts])
    axR.scatter(A[:, 0], A[:, 1], s=30, color=c, alpha=0.55,
                edgecolors="none", zorder=3)
    axR.axvline(cap, color=c, ls="--", lw=1.4, alpha=0.8)
    B = np.array([[e, p] for e, p, _ in fr])
    axR.scatter(B[:, 0], B[:, 1], s=150, marker="*", color=c,
                edgecolors="black", linewidths=0.7, zorder=6,
                label=f"{L:g} MWd/kgHM front ({len(fr)} design"
                      f"{'s' if len(fr) != 1 else ''})")

if args.peak_limit is not None:
    axR.axhline(args.peak_limit, color="#8c2f39", ls=":", lw=1.4,
                label=rf"$F_{{\Delta H}} \leq$ {args.peak_limit:g}")

axR.set_title("After applying the ATF discharge burnup limit", fontsize=11)
axR.set_xlabel("cycle length [EFPD]")
axR.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
axR.grid(alpha=0.25, lw=0.6)

fig.suptitle(
    f"LABGENE-class SMR core: ATF burnup limit collapses the trade-off   "
    f"({len(raw)} evaluations, specific power {spec_power:.3f} W/gHM)",
    fontsize=11.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955))

dest = Path(args.out)
dest.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dest, dpi=args.dpi)
fig.savefig(dest.with_suffix(".pdf"))       # vector copy for LaTeX
print(f"written: {dest}  and  {dest.with_suffix('.pdf')}")

# --- console summary, so the figure never has to be trusted blind --------- #
print(f"\nspecific power        : {spec_power:.4f} W/gHM")
print(f"evaluations           : {len(raw)}  (feasible {len(feas)})")
print(f"uncapped front size   : {len(front)}")
print(f"  EFPD span           : {FX[:, 0].min():.0f} -> {FX[:, 0].max():.0f}"
      f"  ({100 * (FX[:, 0].max() / FX[:, 0].min() - 1):.1f} %)")
print(f"  F_dH span           : {FX[:, 1].min():.4f} -> {FX[:, 1].max():.4f}"
      f"  ({100 * (FX[:, 1].max() / FX[:, 1].min() - 1):.1f} %)")
for L in args.limits:
    cap = L * 1000.0 / spec_power
    fr = pareto([(min(r["cycle_length"], cap), r["peaking"], r) for r in feas])
    at_cap = sum(1 for r in feas if r["bu_eoc_mwd_kg"] > L)
    print(f"limit {L:6.1f} MWd/kgHM : front {len(fr):2d} design(s), "
          f"{at_cap}/{len(feas)} feasible designs truncated")
