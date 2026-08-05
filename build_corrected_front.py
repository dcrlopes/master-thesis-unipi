#!/usr/bin/env python3
"""
build_corrected_front.py -- rebuild the ATF (Accident Tolerant Fuel) capped
Pareto front using the high-fidelity F_dH values from rank_front_peaking.py,
and produce the bias-convergence figure.

WHY THE ORIGINAL FRONT WAS WRONG
--------------------------------
F_dH is a MAXIMUM over a 17x17 pin-fission mesh tally. At campaign fidelity
(4000 particles) that estimator is biased UPWARD by ~0.07 -- LARGER than the
entire F_dH spread of the uncapped Pareto front (0.0599). The bias is also
design-dependent: it shrinks faster for designs whose true power distribution
is flat. So the campaign's ranking was not merely noisy, it was systematically
distorted, and the apparent winner was a noise artifact.

MIXED FIDELITY IS HANDLED EXPLICITLY
------------------------------------
Stage A measured every candidate at the screening fidelity; Stage B measured
only the survivors at top fidelity. Comparing a top-fidelity value against a
screening-fidelity value would be unfair to the screening one, since the
latter is still biased upward. This script therefore:

  1. measures the screening -> top shift on the candidates that have BOTH,
  2. applies that mean shift as a bias adjustment to screening-only
     candidates, carrying its spread as an uncertainty, and
  3. FLAGS every front member whose value is adjusted rather than measured,
     so nothing is presented as more certain than it is.

A candidate is only reported as beaten if it loses even after adjustment.

OUTPUT
  * capped Pareto fronts at each limit, with provenance per point
  * bias-convergence figure (F_dH vs particle count)
  * corrected objective-space figure, original front overlaid
  * CSV of every candidate with stored, screened, resolved and adjusted values

USAGE
  python3 build_corrected_front.py \
      --checkpoint results_campaign2/block2/out/optimization_checkpoint.json \
      --runs rank_front/runs.json --limits 62 75 --out figs
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless: no display needed over SSH
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--runs", required=True, help="rank_front/runs.json")
ap.add_argument("--limits", type=float, nargs="+", default=[62.0, 75.0, 100.0])
ap.add_argument("--prior-sd", type=float, default=0.0182)
ap.add_argument("--margin-k", type=float, default=4.0)
ap.add_argument("--max-candidates", type=int, default=60)
ap.add_argument("--out", default="figs")
args = ap.parse_args()

ck = json.loads(Path(args.checkpoint).read_text())
runs = json.loads(Path(args.runs).read_text())
raw, con, dv = ck["all_raw"], ck.get("constraint_names", []), ck["design_variables"]


def feasible(r, tol=1e-9):
    return all(float(r.get(c, 0.0)) <= tol for c in con)


feas = [r for r in raw if feasible(r)]
lo = min(r["peaking"] for r in feas)
thresh = lo + args.margin_k * args.prior_sd
cands = sorted((r for r in feas if r["peaking"] <= thresh),
               key=lambda r: r["peaking"])[:args.max_candidates]

# specific power, recovered exactly as apply_atf_limit.py does
ratios = sorted(r["bu_eoc_mwd_kg"] * 1000.0 / r["cycle_length"]
                for r in raw
                if r.get("cycle_length", 0) > 0 and r.get("bu_eoc_mwd_kg", 0) > 0)
spec_power = ratios[len(ratios) // 2]

# --------------------------------------------------------------------------- #
# collect measurements per candidate, per fidelity                             #
# --------------------------------------------------------------------------- #
meas = {}                       # meas[i][particles] = (mean, sd, sem, n)
for key, v in runs.items():
    if not key.startswith("c") or "_p" not in key:
        continue
    i = int(key[1:].split("_p")[0])
    P = int(key.split("_p")[1].split("_s")[0])
    meas.setdefault(i, {}).setdefault(P, []).append(v["fdh"])
for i in meas:
    for P, vals in list(meas[i].items()):
        m = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        meas[i][P] = (m, sd, sd / len(vals) ** 0.5 if vals else 0.0, len(vals))

fids = sorted({P for i in meas for P in meas[i]})
LOW, TOP = fids[0], fids[-1]

# --- bias adjustment, measured on candidates that have BOTH fidelities ------ #
both = [i for i in meas if LOW in meas[i] and TOP in meas[i]]
shifts = [meas[i][TOP][0] - meas[i][LOW][0] for i in both]
bias = st.mean(shifts) if shifts else 0.0
bias_sd = st.stdev(shifts) if len(shifts) > 1 else 0.0

print("=" * 78)
print(f"fidelities present      : {fids}")
print(f"candidates              : {len(cands)}   measured: {len(meas)}")
print(f"bias {LOW} -> {TOP}   : {bias:+.4f} +/- {bias_sd:.4f}  "
      f"(from {len(both)} candidate(s) measured at both)")
print(f"campaign F_dH spread across the uncapped front was 0.0599 -- the bias "
      f"is {abs(bias) / 0.0599:.1f}x that" if bias else "")
print("=" * 78)

# --- best estimate per candidate ------------------------------------------- #
est = {}
for i, rec in enumerate(cands):
    if i not in meas:
        continue
    if TOP in meas[i]:
        m, _, sem, n = meas[i][TOP]
        est[i] = dict(f=m, u=sem, src="measured", fid=TOP, n=n)
    else:
        m, _, sem, n = meas[i][LOW]
        est[i] = dict(f=m + bias, u=(sem ** 2 + bias_sd ** 2) ** 0.5,
                      src="adjusted", fid=LOW, n=n)
    est[i].update(bu=rec["bu_eoc_mwd_kg"], efpd=rec["cycle_length"],
                  stored=rec["peaking"],
                  design={k: float(rec[k]) for k in dv})

print(f"\n{'cand':>5s} {'stored':>8s} {'F_dH':>8s} {'+/-':>7s} {'src':>9s} "
      f"{'bu':>7s} {'EFPD':>7s}")
for i in sorted(est, key=lambda j: est[j]["f"]):
    e = est[i]
    print(f"{i:>5d} {e['stored']:8.4f} {e['f']:8.4f} {e['u']:7.4f} "
          f"{e['src']:>9s} {e['bu']:7.2f} {e['efpd']:7.0f}")


# --------------------------------------------------------------------------- #
def capped_front(limit):
    """Non-dominated set after truncating cycle length at the burnup limit."""
    cap = limit * 1000.0 / spec_power
    pts = [(i, min(est[i]["efpd"], cap), est[i]["f"]) for i in est]
    out = []
    for i, e, p in pts:
        if not any((e2 >= e and p2 <= p and (e2 > e or p2 < p))
                   for _, e2, p2 in pts):
            out.append((i, e, p))
    return cap, sorted(out, key=lambda t: -t[1])


print("\n" + "=" * 78)
print("CORRECTED CAPPED PARETO FRONTS")
print("=" * 78)
fronts = {}
for L in args.limits:
    cap, fr = capped_front(L)
    fronts[L] = (cap, fr)
    print(f"\nlimit {L:g} MWd/kgHM  (cap = {cap:.0f} EFPD)  ->  "
          f"{len(fr)} design(s) on the front")
    for i, e, p in fr:
        trunc = "capped" if est[i]["bu"] > L else "own cycle"
        print(f"   cand{i:<3d} EFPD={e:7.0f} ({trunc:9s})  "
              f"F_dH={p:.4f} +/- {est[i]['u']:.4f} [{est[i]['src']}]  "
              f"bu={est[i]['bu']:.2f}")
    adj = [i for i, _, _ in fr if est[i]["src"] == "adjusted"]
    if adj:
        print(f"   NOTE: cand{adj} on this front rest on a bias-ADJUSTED "
              f"value.\n   Re-run rank_front_peaking.py at {TOP} particles for "
              f"these before quoting them as final.")

# --------------------------------------------------------------------------- #
outdir = Path(args.out)
outdir.mkdir(parents=True, exist_ok=True)

# --- FIGURE 1: bias convergence -------------------------------------------- #
fig, ax = plt.subplots(figsize=(7.2, 5.0))
cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(meas)))
for c, i in zip(cmap, sorted(meas)):
    xs = [4000] + [P for P in fids if P in meas[i]]
    ys = [cands[i]["peaking"]] + [meas[i][P][0] for P in fids if P in meas[i]]
    es = [args.prior_sd] + [meas[i][P][2] for P in fids if P in meas[i]]
    lw = 2.0 if TOP in meas[i] else 0.9
    ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=lw, color=c,
                alpha=0.95 if TOP in meas[i] else 0.5,
                label=f"cand{i}" if TOP in meas[i] else None)
ax.set_xscale("log")
ax.set_xlabel("particles per batch (log scale)")
ax.set_ylabel(r"$F_{\Delta H}$")
ax.set_title("Max-over-cells estimator converges downward with fidelity\n"
             f"(bias {bias:+.4f} from {LOW} to {TOP} particles)", fontsize=10.5)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(outdir / "bias_convergence.png", dpi=200)
fig.savefig(outdir / "bias_convergence.pdf")

# --- FIGURE 2: corrected capped fronts ------------------------------------- #
fig, ax = plt.subplots(figsize=(7.8, 5.2))
X = np.array([[r["cycle_length"], r["peaking"]] for r in feas])
ax.scatter(X[:, 0], X[:, 1], s=20, c="0.84", edgecolors="none",
           label=f"campaign values, all feasible ({len(feas)})", zorder=2)
E = np.array([[est[i]["efpd"], est[i]["f"]] for i in est])
ax.scatter(E[:, 0], E[:, 1], s=34, c="#4878a8", edgecolors="white",
           linewidths=0.4, label=f"corrected candidates ({len(est)})", zorder=3)
for i in est:
    ax.plot([est[i]["efpd"]] * 2, [est[i]["stored"], est[i]["f"]],
            color="#b0b0b0", lw=0.7, zorder=1)

cols = plt.cm.plasma(np.linspace(0.15, 0.72, len(args.limits)))
for c, L in zip(cols, args.limits):
    cap, fr = fronts[L]
    ax.axvline(cap, color=c, ls="--", lw=1.3, alpha=0.85)
    F = np.array([[e, p] for _, e, p in fr])
    ax.step(F[:, 0], F[:, 1], where="post", color=c, lw=1.6, alpha=0.9)
    ax.scatter(F[:, 0], F[:, 1], s=170, marker="*", color=c,
               edgecolors="black", linewidths=0.7, zorder=6,
               label=f"{L:g} MWd/kgHM front ({len(fr)} pt)")

ax.set_xlabel("cycle length [EFPD]")
ax.set_ylabel(r"$F_{\Delta H}$")
ax.set_title("Corrected ATF-capped Pareto fronts\n"
             "(grey lines: campaign value -> high-fidelity value)", fontsize=10.5)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
fig.tight_layout()
fig.savefig(outdir / "corrected_front.png", dpi=200)
fig.savefig(outdir / "corrected_front.pdf")

# --- CSV ------------------------------------------------------------------- #
csv = outdir / "corrected_candidates.csv"
cols_ = (["cand", "stored", "f_corrected", "uncertainty", "source", "fidelity",
          "n_seeds", "bu_eoc", "efpd"] + list(dv))
with open(csv, "w") as fh:
    fh.write(",".join(cols_) + "\n")
    for i in sorted(est, key=lambda j: est[j]["f"]):
        e = est[i]
        fh.write(f"{i},{e['stored']},{e['f']},{e['u']},{e['src']},{e['fid']},"
                 f"{e['n']},{e['bu']},{e['efpd']},"
                 + ",".join(str(e['design'][k]) for k in dv) + "\n")

print(f"\nwritten: {outdir}/bias_convergence.png|.pdf")
print(f"         {outdir}/corrected_front.png|.pdf")
print(f"         {csv}")
