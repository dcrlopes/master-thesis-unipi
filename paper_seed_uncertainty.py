#!/usr/bin/env python3
"""
paper_seed_uncertainty.py -- the uncertainty OpenMC reports against the one
measured between independent seeds, for the paper's error model.

Twice the in-run value was too small: the campaign k_bol (194 pcm reported,
264 pcm between seeds, 21 designs, Oct 2026) and the eight-layer cycle length
(8 d reported, 15.9 d between seeds). This script measures the ratio on every
stored quantity and, where statepoints are at hand, looks for the cause in the
batch-to-batch correlation of k.

PART A, no transport, from the repository (default):
  1. k_bol of the fresh assembly: record (4000 x 60) minus high statistics
     (20 000 x 160) over the designs in repro_c9/kbol_*_hifi.json, against the
     sigma OpenMC printed for the campaign solve (kbol_test.log).
  2. k at every depletion state of the eight-layer core: between-seed standard
     deviation against the stored in-run k_sd, every design with 3 or more
     seeds at 20 000 x 160 (c9*_dep_core3d*/runs.json), pooled over states.
  3. The cycle length: stored sigma_efpd (propagated from k_sd) against the
     between-seed standard deviation, same groups, plus the 2-seed pairs of
     c10_dep_core3d_seed1/2 and the 3-seed groups of the proxy study.
PART B, on wks720 (--statepoints GLOB): for each statepoint, the per-batch
  k_generation of the active batches, its lag-1 to lag-5 autocorrelation, the
  naive standard error of the mean, the autocorrelation-corrected one, and
  OpenMC's combined k uncertainty. A positive lag-1 autocorrelation of rho
  inflates the true error by about sqrt((1 + rho) / (1 - rho)).

usage:  python paper_seed_uncertainty.py
        python paper_seed_uncertainty.py --statepoints 'openmc_runs_c10/case_0027/core3d/dep_*/openmc_simulation_n*.h5'
writes figs_paper/seed_uncertainty.md (appends the statepoint table when given)
"""
import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

HIFI_TR = (20000, 160)


def part_a():
    lines = ["# In-run against between-seed uncertainty", ""]
    # 1. k_bol
    c9 = json.load(open("out_c9a/optimization_checkpoint.json"))["all_raw"]
    c10 = json.load(open("out_c10/optimization_checkpoint.json"))["all_raw"]
    diffs, hifi_sd = [], []
    for p in sorted(glob.glob("repro_c9/kbol_*_hifi.json")):
        m = re.match(r"kbol_(c9|c10)_(\d+)_hifi", Path(p).stem)
        rec = (c9 if m.group(1) == "c9" else c10)[int(m.group(2))]["k_bol"]
        diffs.append((rec - float(json.load(open(p))["k"][0])) * 1e5)
    sig_campaign, sig_hifi = [], []
    for log in ("kbol_test.log", "kbol_queue.log"):
        if not Path(log).exists():
            continue
        txt = Path(log).read_text(errors="replace"); pos = 0
        for mm in re.finditer(r"^\[(kbol_\S+)\] threads", txt, flags=re.M):
            pm = re.findall(r"Combined k-effective\s*=\s*[\d.]+\s*\+/-\s*([\d.]+)", txt[pos:mm.start()]); pos = mm.end()
            if len(pm) >= 4:
                (sig_campaign if mm.group(1).endswith("_campaign") else sig_hifi).append(float(pm[-4]) * 1e5)
    sd_obs = st.stdev(diffs); s_h = st.mean(sig_hifi) if sig_hifi else 64.0
    sd_campaign = (sd_obs ** 2 - s_h ** 2) ** 0.5
    rep = st.mean(sig_campaign) if sig_campaign else float("nan")
    lines += ["## 1. Fresh-assembly k_bol, campaign solve 4000 x 60", "",
              f"- {len(diffs)} designs, record minus high statistics: mean {st.mean(diffs):+.0f} pcm, SD {sd_obs:.0f} pcm.",
              f"- High-statistics solves: {s_h:.0f} pcm reported (mean of {len(sig_hifi)}), removed in quadrature.",
              f"- Campaign solve: **{sd_campaign:.0f} pcm between seeds against {rep:.0f} pcm reported** (ratio {sd_campaign / rep:.2f}).", ""]
    # 2 and 3. depletion states and cycle length, groups of seeds at full fidelity
    groups = defaultdict(list)
    for p in sorted(glob.glob("c9*_dep_core3d*/runs.json")) + sorted(glob.glob("c10_dep_core3d_seed*/runs.json")):
        for key, r in json.load(open(p)).items():
            tr = r.get("transport") or {}
            if (tr.get("particles"), tr.get("batches")) == HIFI_TR and r.get("layers") == 8 and r.get("k_sd"):
                groups[(p.split("_dep_")[0][:3], key)].append(r)
    var_b, var_in, n_states = [], [], 0
    rows = ["| Design | Seeds | States | k: between-seed SD [pcm] | k: in-run sd [pcm] | Ratio | EFPD: between-seed SD [d] | EFPD: in-run sigma [d] | Ratio |",
            "|---|---|---|---|---|---|---|---|---|"]
    for (camp, key), runs in sorted(groups.items()):
        if len(runs) < 2:
            continue
        nst = min(len(r["k_hist"]) for r in runs)
        kb, ki = [], []
        for j in range(nst):
            vals = [r["k_hist"][j] for r in runs]; sds = [r["k_sd"][j] for r in runs if r["k_sd"][j] is not None]
            if len(sds) < len(runs):
                continue
            kb.append(st.variance(vals)); ki.append(st.mean(s ** 2 for s in sds))
        if len(runs) >= 3:
            var_b += kb; var_in += ki; n_states += len(kb)
        eb = st.stdev(r["efpd"] for r in runs); ei = st.mean(r["sigma_efpd"] for r in runs if r.get("sigma_efpd"))
        skb, ski = (st.mean(kb)) ** 0.5 * 1e5, (st.mean(ki)) ** 0.5 * 1e5
        rows.append(f"| {camp.upper()}-{key[1:]} | {len(runs)} | {len(kb)} | {skb:.0f} | {ski:.0f} | {skb / ski:.2f} | {eb:.1f} | {ei:.1f} | {eb / ei:.2f} |")
    pooled = (st.mean(var_b) / st.mean(var_in)) ** 0.5 if var_in else float("nan")
    lines += ["## 2 and 3. Eight-layer core at 20 000 x 160: k at each depletion state, and the cycle length", "", *rows, "",
              f"Pooled over {n_states} states of the designs with 3 or more seeds: **between-seed k error = {pooled:.2f} x the in-run sd**. "
              f"The cycle-length sigma is propagated from the in-run k sd of the two bracketing states, so it inherits the same factor, "
              f"and the interpolation between states two to four MWd/kgHM apart adds to it.", ""]
    return "\n".join(lines)


def part_b(pattern):
    import openmc
    rows = ["## 4. Batch-to-batch correlation of k in the statepoints", "",
            "| Statepoint | Active batches | k combined +/- (OpenMC) [pcm] | Naive SE of the batch mean [pcm] | rho(1) | rho(2) | rho(5) | Corrected SE [pcm] | Corrected / OpenMC |",
            "|---|---|---|---|---|---|---|---|---|"]
    ratios = []
    for p in sorted(glob.glob(pattern)):
        with openmc.StatePoint(p, autolink=False) as sp:
            kg = np.asarray(sp.k_generation, float)[sp.n_inactive:]
            k_rep = sp.keff.std_dev * 1e5
        n = len(kg); x = kg - kg.mean()
        rho = [float((x[:-l] * x[l:]).sum() / (x * x).sum()) for l in (1, 2, 5)]
        naive = kg.std(ddof=1) / n ** 0.5 * 1e5
        # Bartlett-type correction with the first ten lags
        acf = [float((x[:-l] * x[l:]).sum() / (x * x).sum()) for l in range(1, min(10, n // 4))]
        neff = n / (1 + 2 * sum(acf))
        corr = kg.std(ddof=1) / max(neff, 1.0) ** 0.5 * 1e5
        ratios.append(corr / k_rep)
        rows.append(f"| {Path(p).parent.name}/{Path(p).name} | {n} | {k_rep:.0f} | {naive:.0f} | {rho[0]:+.2f} | {rho[1]:+.2f} | {rho[2]:+.2f} | {corr:.0f} | {corr / k_rep:.2f} |")
    rows += ["", f"Mean corrected / reported over {len(ratios)} statepoints: **{st.mean(ratios):.2f}**. OpenMC's combined estimator treats the active "
             f"batches as independent; a positive lag-1 correlation of the fission source between batches makes its uncertainty too small by about "
             f"sqrt((1 + rho) / (1 - rho)).", ""]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--statepoints", default=None, help="glob of statepoints for part B (wks720)")
    ap.add_argument("--out", default="figs_paper")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True); f = out / "seed_uncertainty.md"
    if a.statepoints:
        txt = (f.read_text() if f.exists() else "# In-run against between-seed uncertainty\n\n") + "\n" + part_b(a.statepoints)
    else:
        txt = part_a()
    f.write_text(txt); print(txt); print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
