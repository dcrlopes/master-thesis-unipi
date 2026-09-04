#!/usr/bin/env python3
"""
c7_khist_from_log.py -- Campaign 7 k histories recovered from out_c7/run.log.gz.

WHY FROM THE LOG. The Campaign 7 depletion files (openmc_runs_c7/) sit on the
stopped AWS volume. The run log, committed at out_c7/run.log.gz, carries every
depletion eigenvalue: each "[openmc.deplete] t=<s>" line is followed by the
transport output of that state, whose "Combined k-effective" line is the k.
Solves that are not depletion states (assembly BOL, core BOL, control screens)
print a "Combined k-effective" without a preceding deplete line and are
ignored. At a chunk restart (write_rates fix) the restart state prints a
deplete line with NO transport: the next deplete line follows at once. That
state is the last state of the previous chunk and is not repeated.

VALIDATION, per case, against out_c7/optimization_checkpoint.json
  n_states  == n_dep_solves of the record
  k[0]      == k_bol of the record within --match-tol (the archive k_bol comes
               from the same depletion t=0 transport at 4000 x 60)
  crossing  reproduces cycle_length within one depletion step
A case failing any check is written with status "MISMATCH" and excluded from
the summary table. Do not quote a mismatched case.

DEFINITIONS as in c8_reactivity_swing.py: rho = 1e5 (k-1)/k, hump =
rho_peak - rho_BOL (xenon-free BOL), swing_peak = rho_peak - rho(k_target).
Burnup from time with the specific power of the record,
bu_eoc_mwd_kg * 1000 / cycle_length, campaign value 9.98 W/gHM.

USAGE
  python c7_khist_from_log.py --selftest
  python c7_khist_from_log.py --log out_c7/run.log.gz \
      --checkpoint out_c7/optimization_checkpoint.json --out kh_c7
OUTPUTS in <out>/: k_histories.json, hump_summary.csv, hump_table.tex,
  k_histories.pdf/png, report.txt
"""
from __future__ import annotations
import argparse, csv, gzip, json, re, sys
from pathlib import Path
import numpy as np

RE_DEP = re.compile(r"\[openmc\.deplete\] t=([0-9.eE+-]+)")
RE_K = re.compile(r"Combined k-effective\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
RE_CASE = re.compile(r"\[case (\d{4})\]")
SPEC_POWER_CAMPAIGN = 9.98
XE_POINT = 0.5


def rho(k):
    return 1e5 * (k - 1.0) / k


def parse_log(lines):
    """Yield (case_index, [(t_s, k, sd), ...]) in log order."""
    states, pending = [], None
    for line in lines:
        m = RE_DEP.search(line)
        if m:
            pending = float(m.group(1)); continue
        m = RE_K.search(line)
        if m:
            if pending is not None:
                t = pending; pending = None
                if states and abs(states[-1][0] - t) < 1.0:
                    continue          # duplicate time, keep the first
                states.append((t, float(m.group(1)), float(m.group(2))))
            continue
        m = RE_CASE.search(line)
        if m:
            yield int(m.group(1)), states
            states, pending = [], None


def hump(bu, k, k_target):
    bu = np.asarray(bu, float); k = np.asarray(k, float)
    op = bu > 1e-9
    i_xe = int(np.argmin(np.abs(bu - XE_POINT)))
    j = int(np.argmax(k[op])) if op.any() else 0
    k_peak = float(k[op][j]) if op.any() else float(k[0]); b_peak = float(bu[op][j]) if op.any() else 0.0
    crossing = None
    for i in range(len(bu) - 1, 0, -1):
        if k[i - 1] > k_target >= k[i]:
            crossing = float(bu[i - 1] + (k[i - 1] - k_target) * (bu[i] - bu[i - 1]) / (k[i - 1] - k[i])); break
    return dict(k_bol=float(k[0]), k_xe=float(k[i_xe]), k_peak=k_peak, b_peak=b_peak,
                hump_pcm=rho(k_peak) - rho(float(k[0])), hump_vs_xe_pcm=rho(k_peak) - rho(float(k[i_xe])),
                swing_peak_pcm=rho(k_peak) - rho(k_target), swing_xe_pcm=rho(float(k[i_xe])) - rho(k_target),
                bol_bounds_cycle=bool(k_peak <= k[0]), crossing_mwd_kg=crossing)


def selftest():
    log = [
        "[openmc.deplete] t=0.0 s, dt=1 s", " Combined k-effective        = 1.20000 +/- 0.00200",
        "[openmc.deplete] t=4327203.05 s, dt=1", " Combined k-effective        = 1.18000 +/- 0.00200",
        "[openmc.deplete] t=116834482.39 (final operator evaluation)", " Combined k-effective        = 1.19000 +/- 0.00200",
        " Combined k-effective        = 0.95000 +/- 0.00200",      # a control solve, no deplete line: ignored
        "[openmc.deplete] t=116834482.39 s, dt=1",                 # restart state, no transport
        "[openmc.deplete] t=151452106.80 s, dt=1", " Combined k-effective        = 1.10000 +/- 0.00200",
        "[openmc.deplete] t=186069731.22 (final operator evaluation)", " Combined k-effective        = 1.05000 +/- 0.00200",
        "  [case 0007] e=(1/1) ... k_bol=1.2000 [5 solves, 1 min]",
    ]
    out = list(parse_log(log))
    assert len(out) == 1 and out[0][0] == 7
    st = out[0][1]
    assert [round(s[1], 2) for s in st] == [1.20, 1.18, 1.19, 1.10, 1.05], st
    print("selftest OK"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="out_c7/run.log.gz")
    ap.add_argument("--checkpoint", default="out_c7/optimization_checkpoint.json")
    ap.add_argument("--out", default="kh_c7")
    ap.add_argument("--match-tol", type=float, default=2e-4, help="k[0] vs archive k_bol")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ck = json.load(open(a.checkpoint)); recs = ck["all_raw"]
    opener = gzip.open if a.log.endswith(".gz") else open
    with opener(a.log, "rt", errors="replace") as f:
        cases = dict(parse_log(f))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = []; P = L.append
    P(f"=== Campaign 7 k histories from {a.log}: {len(cases)} cases found, {len(recs)} records ===")
    results, rows = {}, []
    for i, r in enumerate(recs):
        st = cases.get(i)
        if not st:
            results[str(i)] = dict(status="NO_LOG_STATES"); P(f"  case {i:4d}: no depletion states in the log"); continue
        t = np.array([s[0] for s in st]) / 86400.0; k = np.array([s[1] for s in st]); sd = np.array([s[2] for s in st])
        sp = r["bu_eoc_mwd_kg"] * 1000.0 / r["cycle_length"] if r["cycle_length"] > 0 else SPEC_POWER_CAMPAIGN
        bu = t * sp / 1000.0
        problems = []
        if len(k) != r["n_dep_solves"]: problems.append(f"n_states {len(k)} != n_dep_solves {r['n_dep_solves']}")
        if abs(k[0] - r["k_bol"]) > a.match_tol: problems.append(f"k[0] {k[0]:.5f} != k_bol {r['k_bol']:.5f}")
        h = hump(bu, k, r["k_target"])
        if h["crossing_mwd_kg"] is not None and r["cycle_length"] > 0:
            efpd_log = h["crossing_mwd_kg"] * 1000.0 / sp
            step = float(np.max(np.diff(t))) if len(t) > 1 else 0.0
            if abs(efpd_log - r["cycle_length"]) > max(step, 1.0) and not r.get("censored"):
                problems.append(f"crossing {efpd_log:.0f} EFPD != cycle_length {r['cycle_length']:.0f}")
        else:
            efpd_log = None
        status = "OK" if not problems else "MISMATCH"
        results[str(i)] = dict(status=status, problems=problems, t_days=t.tolist(), bu=bu.tolist(), k=k.tolist(), k_sd=sd.tolist(),
                               spec_power=sp, k_target=r["k_target"], cycle_length=r["cycle_length"],
                               bu_eoc=r["bu_eoc_mwd_kg"], keff_core_bol=r["keff_core_bol"], k_allre=r.get("k_allre"),
                               e_in=r["enrich_inner"], e_out=r["enrich_outer"], gd_wt=r["gd_wt"], pins=r.get("gd_pins_used"),
                               pitch=r["pitch"], refl=r["refl_thick"], efpd_from_log=efpd_log, noise_pcm=float(np.median(sd) * 1e5), **h)
        flag = "" if status == "OK" else "  <-- " + "; ".join(problems)
        P(f"  case {i:4d}: {len(k):2d} states, k_bol {k[0]:.4f}, peak {h['k_peak']:.4f} at {h['b_peak']:5.1f} MWd/kg, "
          f"hump {h['hump_pcm']:+6.0f} pcm, swing peak->EOC {h['swing_peak_pcm']:6.0f} pcm, EFPD {r['cycle_length']:5.0f}{flag}")
        if status == "OK":
            rows.append((i, r["enrich_inner"], r["enrich_outer"], r["gd_wt"], r.get("gd_pins_used"), r["pitch"], r["refl_thick"], h, r["cycle_length"]))
    ok = [i for i in results if results[i]["status"] == "OK"]
    bad = [i for i in results if results[i]["status"] != "OK"]
    P(f"validated {len(ok)} of {len(recs)} cases; excluded: {bad or 'none'}")
    hp = [results[i]["hump_pcm"] for i in ok]
    if hp:
        P(f"hump over validated cases: min {min(hp):+.0f}, median {np.median(hp):+.0f}, max {max(hp):+.0f} pcm; "
          f"above BOL by more than 400 pcm: {sum(1 for x in hp if x > 400)} cases")
    (out / "k_histories.json").write_text(json.dumps(results, indent=1))
    with open(out / "hump_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "e_in", "e_out", "gd_wt", "pins", "pitch", "refl", "k_bol", "k_xe", "k_peak", "b_peak", "hump_pcm", "swing_peak_pcm", "swing_xe_pcm", "efpd", "status"])
        for i in results:
            r = results[i]
            if r["status"] != "OK": w.writerow([i] + [""] * 14 + [r["status"]]); continue
            w.writerow([i, f"{r['e_in']:.2f}", f"{r['e_out']:.2f}", f"{r['gd_wt']:.2f}", r["pins"], f"{r['pitch']:.3f}", f"{r['refl']:.2f}",
                        f"{r['k_bol']:.4f}", f"{r['k_xe']:.4f}", f"{r['k_peak']:.4f}", f"{r['b_peak']:.1f}", f"{r['hump_pcm']:+.0f}",
                        f"{r['swing_peak_pcm']:.0f}", f"{r['swing_xe_pcm']:.0f}", f"{r['cycle_length']:.0f}", "OK"])
    T = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption[Reactivity hump of the Campaign 7 designs]{Gadolinium hump and burnup reactivity swing of the Campaign 7 designs, recovered from the run log.}",
         r"  \label{tab:c7-hump}", r"  \begin{tabular}{@{}r r r r r r r r@{}}", r"    \toprule",
         r"    ID & $e_\mathrm{in}/e_\mathrm{out}$ [wt\%] & Gd [wt\%] & $k_\mathrm{BOL}$ & $k_\mathrm{peak}$ & $B_\mathrm{peak}$ [MWd/kgHM] & hump [pcm] & $\Delta\rho_\mathrm{peak\to EOC}$ [pcm] \\",
         r"    \midrule"]
    for i, ei, eo, gd, pins, pitch, refl, h, efpd in sorted(rows, key=lambda x: -x[7]["hump_pcm"])[:15]:
        T.append(f"    {i} & {ei:.2f}/{eo:.2f} & {gd:.2f} & {h['k_bol']:.4f} & {h['k_peak']:.4f} & {h['b_peak']:.1f} & {h['hump_pcm']:+.0f} & {h['swing_peak_pcm']:.0f} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / "hump_table.tex").write_text("\n".join(T) + "\n")
    (out / "report.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
        fig, ax = plt.subplots(figsize=(6.4, 4.0)); cmap = plt.get_cmap("viridis")
        gds = [results[i]["gd_wt"] for i in ok]; norm = plt.Normalize(min(gds), max(gds))
        for i in ok:
            r = results[i]; ax.plot(r["bu"], r["k"], "-", lw=0.7, color=cmap(norm(r["gd_wt"])), alpha=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([]); fig.colorbar(sm, ax=ax, label="Gd$_2$O$_3$ [wt%]")
        ax.set_xlabel("Burnup, MWd/kgHM"); ax.set_ylabel("$k_\\infty$ of the depleting assembly")
        fig.savefig(out / "k_histories.pdf", bbox_inches="tight"); fig.savefig(out / "k_histories.png", dpi=300, bbox_inches="tight")
    except Exception as ex:
        print(f"no figure: {ex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
