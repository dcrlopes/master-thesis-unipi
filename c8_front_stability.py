#!/usr/bin/env python3
"""
c8_front_stability.py -- how much of the Campaign 8 Pareto front depends on
the last evaluations, on any single evaluation, and on Monte Carlo noise.
No OpenMC: numpy, pymoo (hypervolume) and matplotlib.

FOUR ANALYSES ON THE ARCHIVE
  chronological  the feasible non-dominated front after every evaluation in
                 the order the campaign ran them, its hypervolume at the
                 campaign's own frozen reference point (checkpoint hv_ref),
                 the evaluation at which each final front member entered,
                 and every design that sat on the front and was later
                 displaced. The hypervolume at the end of each block must
                 reproduce the checkpoint's hv_history: the script asserts
                 it, so the bookkeeping is verified against the campaign.
  contributions  the exclusive hypervolume contribution of each front
                 member, HV(front) - HV(front without it), as a fraction of
                 HV(front). A front carried by one member is fragile.
  leave-k-out    EXACT (no sampling): the hypervolume after removing k
                 evaluations chosen uniformly at random from the archive,
                 its mean, its standard deviation and its worst case, for
                 k = 1 to --k-max, and the mean hypervolume of a uniformly
                 random subset of size n for n = 24, 30, ..., N. Removing
                 designs cannot create new front members, so the outcome
                 depends only on which front members survive, and the
                 expectation is a sum over the 2^m subsets of the m front
                 members with hypergeometric weights.
  noise          the objectives of every feasible design are perturbed B
                 times by Gaussian Monte Carlo noise, the front is recomputed,
                 and the probability that each design sits on the front is
                 reported. Feasibility is held at the archived verdict.
                 Cycle-length noise per design follows analyze_results.py:
                     sigma_cycle = sqrt(2) sigma_k / ((k_bol - k_target) / cycle)
                 with sigma_k from --sigma-k [pcm]. Peaking noise is the
                 flat --sigma-f. REPLACE THE DEFAULTS by the archive-fidelity
                 sigmas from c8_post_numbers.json once run_c8_night.sh has
                 produced them.

OUTPUT (--out, default figs_c8_stab/)
    c8_stab_chrono.pdf/.png     HV vs evaluations, front membership timeline
    c8_stab_robust.pdf/.png     exclusive contributions, leave-k-out, noise
    c8_front_stability.json     every number
    c8_stab_table.tex           booktabs table of the front members

    python c8_front_stability.py --check
    python c8_front_stability.py --sigma-k 44 --sigma-f 0.010
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV


# ----------------------------------------------------------------------------
def load(path):
    ck = json.loads(Path(path).read_text())
    raw = ck["all_raw"]
    cons = ck["constraint_names"]
    F = np.array([[-float(r["cycle_length"]), float(r["peaking"])] for r in raw])
    feas = np.array([all(float(r[c]) <= 1e-9 for c in cons) for r in raw])
    ref = np.array(ck["hv_ref"], dtype=float)
    return ck, raw, F, feas, ref


def nondominated(F):
    n = len(F)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dom[i] = False
        if dom.any():
            nd[i] = False
    return nd


def hv_of(F, ref):
    """Hypervolume of the non-dominated subset of F (minimise space) with
    respect to ref, counting only points that dominate ref, exactly as
    ActiveLearningMOO._hv does."""
    if len(F) == 0:
        return 0.0
    nd = nondominated(F)
    front = F[nd]
    keep = np.all(front < ref, axis=1)
    if not keep.any():
        return 0.0
    return float(HV(ref_point=ref)(front[keep]))


def log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


# ----------------------------------------------------------------------------
def chronological(F, feas, ref, ck):
    N = len(F)
    hv, size, fronts = [], [], []
    for n in range(1, N + 1):
        idx = np.flatnonzero(feas[:n])
        if len(idx) == 0:
            hv.append(0.0); size.append(0); fronts.append(set()); continue
        nd = nondominated(F[idx])
        members = set(int(i) for i in idx[nd])
        fronts.append(members)
        size.append(len(members))
        hv.append(hv_of(F[idx], ref))
    final = sorted(fronts[-1])
    entered = {m: next(n for n in range(1, N + 1) if m in fronts[n - 1]) for m in final}
    ever = set().union(*fronts)
    displaced = {}
    for d in sorted(ever - set(final)):
        on = [n for n in range(1, N + 1) if d in fronts[n - 1]]
        # who removed it: the first later design that dominates it
        later = [j for j in range(on[-1], N) if feas[j]
                 and np.all(F[j] <= F[d]) and np.any(F[j] < F[d])]
        displaced[d] = dict(first=on[0], last=on[-1], displaced_by=later[:1])
    # verify against the campaign's own hv_history at the block ends
    pl = ck.get("phase_log", [])
    ends, s = [], 0
    for p in pl:
        s += int(p.get("n_eval", 0)); ends.append(s)
    hist = ck.get("hv_history", [])
    check = []
    for e, h in zip(ends, hist):
        if 1 <= e <= N:
            check.append(dict(n=e, checkpoint=float(h), recomputed=hv[e - 1],
                              agree=bool(abs(float(h) - hv[e - 1]) <= 1e-6 * max(1.0, abs(h)))))
    return dict(hv=hv, size=size, final=final, entered=entered,
                displaced=displaced, block_ends=ends, hv_check=check)


def contributions(F, front_idx, ref):
    full = hv_of(F[front_idx], ref)
    out = {}
    for m in front_idx:
        rest = [i for i in front_idx if i != m]
        h = hv_of(F[rest], ref) if rest else 0.0
        out[m] = dict(exclusive=full - h, fraction=(full - h) / full if full > 0 else float("nan"))
    return full, out


def leave_k_out_exact(F, front_idx, ref, N, k_max, sizes):
    """Exact expectations over uniformly random removals, using the fact
    that the front after removal is the surviving subset of the front."""
    m = len(front_idx)
    subsets = {}
    for r in range(m + 1):
        for T in itertools.combinations(front_idx, r):      # survivors
            subsets[T] = hv_of(F[list(T)], ref) if T else 0.0
    full = subsets[tuple(front_idx)]

    def stats(n_keep):
        # survivors T of size t among the front, with n_keep - t of the
        # other N - m designs: weight C(N-m, n_keep-t) / C(N, n_keep)
        mean = var = 0.0
        worst = float("inf")
        best_prob_full = 0.0
        for T, h in subsets.items():
            t = len(T)
            if n_keep - t < 0 or n_keep - t > N - m:
                continue
            w = math.exp(log_comb(N - m, n_keep - t) - log_comb(N, n_keep))
            mean += w * h
            var += w * h * h
            if w > 0:
                worst = min(worst, h)
            if t == m:
                best_prob_full += w
        var = max(var - mean * mean, 0.0)
        return dict(n_keep=int(n_keep), mean=mean, std=math.sqrt(var),
                    worst=worst, mean_frac=mean / full if full > 0 else float("nan"),
                    worst_frac=worst / full if full > 0 else float("nan"),
                    p_front_intact=best_prob_full)

    lko = [dict(k=k, **stats(N - k)) for k in range(1, k_max + 1)]
    sub = [stats(n) for n in sizes if 1 <= n <= N]
    return full, lko, sub


def noise_robustness(F, raw, feas, sigma_k_pcm, sigma_f, B, seed):
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(feas)
    cyc = -F[idx, 0]
    s_cyc = np.zeros(len(idx))
    for a, i in enumerate(idx):
        r = raw[i]
        slope = (float(r["k_bol"]) - float(r["k_target"])) / max(cyc[a], 1.0)
        s_cyc[a] = math.sqrt(2.0) * sigma_k_pcm * 1e-5 / slope if slope > 0 else 0.0
    counts = np.zeros(len(idx))
    sizes = np.zeros(B, dtype=int)
    for b in range(B):
        Fp = F[idx].copy()
        Fp[:, 0] -= rng.standard_normal(len(idx)) * s_cyc      # minus cycle
        Fp[:, 1] += rng.standard_normal(len(idx)) * sigma_f
        nd = nondominated(Fp)
        counts += nd
        sizes[b] = int(nd.sum())
    return dict(idx=idx.tolist(), sigma_cycle=s_cyc.tolist(),
                p_front=(counts / B).tolist(),
                front_size_mean=float(sizes.mean()), front_size_std=float(sizes.std()),
                B=B, sigma_k_pcm=sigma_k_pcm, sigma_f=sigma_f)


# ----------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9.5,
        "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.framealpha": 0.9})
    return plt


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    print(f"  wrote {name}.pdf/.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--out", default="figs_c8_stab")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--sizes", default="24,30,36,42,48,54")
    ap.add_argument("--sigma-k", type=float, default=44.0,
                    help="1-sigma Monte Carlo noise on k at the depletion "
                         "fidelity [pcm], default 44 (validate_ktarget_burnup)")
    ap.add_argument("--sigma-f", type=float, default=0.010,
                    help="1-sigma noise on the core F_dH at the archive "
                         "fidelity, default 0.010 (seed pairs). Replace by "
                         "the c8_post_numbers.json value when available.")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    ck, raw, F, feas, ref = load(a.checkpoint)
    N = len(F)
    print(f"archive {a.checkpoint}: {N} designs, {int(feas.sum())} feasible, "
          f"hv_ref {ref.tolist()}")
    if a.check:
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    chrono = chronological(F, feas, ref, ck)
    front = chrono["final"]
    print(f"final front ({len(front)}): {front}")
    print(f"entered at evaluation: {chrono['entered']}")
    print(f"displaced: {chrono['displaced']}")
    for c in chrono["hv_check"]:
        print(f"  hv check n={c['n']}: checkpoint {c['checkpoint']:.4f} "
              f"recomputed {c['recomputed']:.4f} {'OK' if c['agree'] else 'MISMATCH'}")
    if chrono["hv_check"] and not all(c["agree"] for c in chrono["hv_check"]):
        raise SystemExit("hypervolume bookkeeping does not reproduce the "
                         "checkpoint's hv_history; stop and inspect.")

    full, contrib = contributions(F, front, ref)
    print(f"HV(front) = {full:.4f}")
    for m in sorted(contrib, key=lambda i: -contrib[i]["exclusive"]):
        print(f"  design {m:2d}: exclusive {contrib[m]['exclusive']:8.3f} "
              f"({100 * contrib[m]['fraction']:5.1f} %)  "
              f"{-F[m, 0]:.0f} EFPD, F_dH {F[m, 1]:.4f}")

    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    _, lko, sub = leave_k_out_exact(F, front, ref, N, a.k_max, sizes)
    print("leave-k-out (exact):")
    for r in lko:
        print(f"  k={r['k']}: mean {100 * r['mean_frac']:.2f} % of HV, "
              f"worst {100 * r['worst_frac']:.2f} %, front intact with "
              f"probability {r['p_front_intact']:.3f}")
    print("random subset of size n (exact expectation):")
    for r in sub:
        print(f"  n={r['n_keep']}: mean {100 * r['mean_frac']:.1f} % "
              f"(std {100 * r['std'] / full:.1f} %), worst {100 * r['worst_frac']:.1f} %")

    noise = noise_robustness(F, raw, feas, a.sigma_k, a.sigma_f, a.B, a.seed)
    p = dict(zip(noise["idx"], noise["p_front"]))
    print(f"noise (sigma_k {a.sigma_k:g} pcm, sigma_F {a.sigma_f:g}, B {a.B}): "
          f"front size {noise['front_size_mean']:.2f} +- {noise['front_size_std']:.2f}")
    for i in sorted(p, key=lambda i: -p[i]):
        tag = "front" if i in front else "     "
        print(f"  design {i:2d} {tag}: P(on front) {p[i]:.3f}  "
              f"sigma_cycle {noise['sigma_cycle'][noise['idx'].index(i)]:.0f} EFPD")

    res = dict(N=N, n_feasible=int(feas.sum()), hv_ref=ref.tolist(),
               chronological=dict(hv=chrono["hv"], front_size=chrono["size"],
                                  final_front=front,
                                  entered={str(k): v for k, v in chrono["entered"].items()},
                                  displaced={str(k): v for k, v in chrono["displaced"].items()},
                                  block_ends=chrono["block_ends"], hv_check=chrono["hv_check"]),
               hv_front=full,
               contributions={str(k): v for k, v in contrib.items()},
               leave_k_out=lko, random_subsets=sub, noise=noise)
    (out / "c8_front_stability.json").write_text(json.dumps(res, indent=2, default=float))
    print("  wrote c8_front_stability.json")

    # ---------------- figures ------------------------------------------------
    plt = setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3))
    n_ax = np.arange(1, N + 1)
    axs[0].plot(n_ax, chrono["hv"], "-", c="navy", lw=1.2)
    for e in chrono["block_ends"]:
        axs[0].axvline(e, c="0.7", lw=0.6, ls=":")
    axs[0].set_xlabel("evaluations")
    axs[0].set_ylabel("hypervolume of the feasible front")
    ax2 = axs[0].twinx()
    ax2.step(n_ax, chrono["size"], where="post", c="tab:red", lw=0.9)
    ax2.set_ylabel("front size", color="tab:red")
    ax2.grid(False)
    # membership timeline
    members = sorted(set(front) | set(int(d) for d in chrono["displaced"]))
    fronts_by_n = []
    for n in range(1, N + 1):
        idx = np.flatnonzero(feas[:n])
        fronts_by_n.append(set(int(i) for i in idx[nondominated(F[idx])]) if len(idx) else set())
    for row, m in enumerate(members):
        on = np.array([m in fronts_by_n[n - 1] for n in range(1, N + 1)])
        axs[1].scatter(n_ax[on], np.full(on.sum(), row), s=6,
                       c="navy" if m in front else "tab:red", marker="s")
    axs[1].set_yticks(range(len(members)))
    axs[1].set_yticklabels([f"design {m}" for m in members])
    axs[1].set_xlabel("evaluations")
    from matplotlib.lines import Line2D
    axs[1].legend(handles=[
        Line2D([], [], marker="s", ls="", c="navy", ms=4, label="final front member"),
        Line2D([], [], marker="s", ls="", c="tab:red", ms=4, label="later displaced")],
        loc="center right")
    for ax, t in zip(axs, "ab"):
        ax.set_title(f"({t})", loc="left")
    fig.tight_layout()
    save(fig, out, "c8_stab_chrono")

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.1))
    order = sorted(front, key=lambda i: -F[i, 0])
    axs[0].bar([str(i) for i in order], [100 * contrib[i]["fraction"] for i in order], color="navy")
    axs[0].set_xlabel("front member (archive index)")
    axs[0].set_ylabel("exclusive hypervolume [% of HV]")
    ks = [r["k"] for r in lko]
    axs[1].errorbar(ks, [100 * r["mean_frac"] for r in lko], yerr=[100 * r["std"] / full for r in lko],
                    fmt="o-", c="navy", capsize=3, label="mean $\\pm$ std")
    axs[1].plot(ks, [100 * r["worst_frac"] for r in lko], "v--", c="tab:red", label="worst case")
    axs[1].set_xlabel("evaluations removed at random, $k$")
    axs[1].set_ylabel("hypervolume retained [%]")
    axs[1].set_xticks(ks)
    axs[1].legend(loc="lower left")
    pf = [p[i] for i in noise["idx"]]
    cols = ["navy" if i in front else "0.6" for i in noise["idx"]]
    axs[2].bar([str(i) for i in noise["idx"]], pf, color=cols)
    axs[2].set_xlabel("feasible design (archive index)")
    axs[2].set_ylabel("P(on the front) under noise")
    axs[2].tick_params(axis="x", labelsize=6.5, rotation=90)
    axs[2].set_ylim(0, 1.02)
    for ax, t in zip(axs, "abc"):
        ax.set_title(f"({t})", loc="left")
    fig.tight_layout()
    save(fig, out, "c8_stab_robust")

    # ---------------- table --------------------------------------------------
    lines = [
        r"% c8_stab_table.tex -- written by c8_front_stability.py.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{threeparttable}",
        r"  \caption[Stability of the Campaign 8 front]{Entry, exclusive hypervolume contribution and noise robustness of the Campaign 8 front members.}",
        r"  \label{tab:c8-front-stability}",
        r"  \begin{tabular}{rrrrrr}",
        r"    \toprule",
        r"    Design & Cycle [EFPD] & $F_{\Delta H}$ & Entered at & Excl.\ HV [\%] & $P$(front) \\",
        r"    \midrule",
    ]
    for i in order:
        lines.append(f"    {i} & {-F[i, 0]:.0f} & {F[i, 1]:.4f} & {chrono['entered'][i]} & "
                     f"{100 * contrib[i]['fraction']:.1f} & {p[i]:.2f} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}\footnotesize",
        r"    \item Entered at is the evaluation count at which the design first became a member of the feasible non-dominated set and it was never displaced afterwards. Excl.\ HV is the hypervolume lost when the design alone is removed, as a percentage of the front hypervolume at the campaign reference point. $P$(front) is the probability that the design remains non-dominated when the objectives of every feasible design are perturbed by Gaussian Monte Carlo noise, "
        f"with $\\sigma_k = {a.sigma_k:g}$\\,pcm propagated to the cycle length and $\\sigma_F = {a.sigma_f:g}$ on $F_{{\\Delta H}}$, over {a.B} replicates.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    (out / "c8_stab_table.tex").write_text("\n".join(lines) + "\n")
    print("  wrote c8_stab_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
