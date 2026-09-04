#!/usr/bin/env python3
"""
c8_reactivity_swing.py -- k history, gadolinium hump and reactivity swing of
the Campaign 8 candidates. No transport. Written 4 September 2026 against
commit f60b78a of branch campaign8.

INPUTS (all already in the repository)
  kt_burnup/summary.json          single-shot depletion of ten designs at the
                                  campaign fidelity (validate_ktarget_burnup.py):
                                  fields k_hist, k_sd_hist, bu_hist, kt_table
  out_c8/optimization_checkpoint.json   archive: keff_core_bol, k_allre, k_re12,
                                  cycle_length, bu_eoc_mwd_kg, design variables
  boron_c8/runs.json              boron sweep, ARO at 0/500/1000/1500 ppm
  confirm3d_c8/summary.json       optional, 3D margins at 1000 ppm (BOL)
  khist_c8/khist.json             optional, the chunked-archive histories from
                                  c8_hump2.py, used only as a cross-check

DEFINITIONS, every k is the k_inf of the depleting assembly at 1000 ppm unless
the name says "core"
  rho(k)        = 1e5 (k - 1) / k                                  [pcm]
  rho_BOL       xenon-free beginning of life, bu = 0
  rho_Xe        first operating point, bu = 0.5 MWd/kgHM, equilibrium xenon
  rho_peak      maximum over the operating points (bu > 0)
  rho_EOC       rho(k_target): by the Route B construction the core is critical
                when the assembly reaches k_target
  hump          rho_peak - rho_BOL      (> 0: the BOL screen is not the bound)
  swing_peak    rho_peak - rho_EOC      burnup reactivity swing from the
                                        operating maximum to end of cycle
  swing_Xe      rho_Xe   - rho_EOC      the same from the first operating point
  xenon_step    rho_BOL  - rho_Xe       xenon build-up plus the first 0.5 MWd/kg
  rho_core_BOL  1e5 (1 - 1/keff_core_bol), the 2D core at 1000 ppm, xenon-free
  w_B           differential boron worth on the core, ARO, pcm/ppm, from the
                boron sweep, on the 0-1000 and 1000-1500 ppm intervals
  c_BOL         1000 + rho_core_BOL / w_B(1000-1500)   critical boron at BOL,
                xenon-free, 2D core, linear extrapolation above 1000 ppm
  c_peak        1000 + (rho_core_BOL + max(hump,0)) / w_B(1000-1500)
                the same at the operating maximum
  hump_core     LF_BOL * hump, the hump carried to the core: with
                k_core = k_inf / LF a reactivity difference on the assembly is
                LF times larger on the core, and kt_burnup shows LF constant
                over the cycle within noise (drift -152 to +149 pcm)
  margin_peak   (BOL rodded margin of the 3D confirmation) - max(hump_core, 0)

APPROXIMATIONS, stated in the report and to be repeated in the thesis
  1. The swing is an assembly quantity (k_inf), the boron worth a core quantity
     at BOL. Dividing one by the other assumes the boron worth does not change
     over the cycle. It does, through the spectrum. Treat c_peak as an
     estimate to one significant figure.
  2. Linear extrapolation above 1500 ppm ignores further self-shielding, so
     c_BOL and c_peak are lower bounds.
  3. The depletion fidelity of the campaign gives 150 to 250 pcm noise per
     state. A hump below about 400 pcm is not resolved.

USAGE
  python c8_reactivity_swing.py --selftest
  python c8_reactivity_swing.py                       # defaults below
  python c8_reactivity_swing.py --out swing_c8 --designs 47 42 23 29 21 44 59 1 53 31 13

OUTPUTS in <out>/
  swing.json          one record per design
  swing_table.tex     booktabs table, Pisa-style caption with list entry
  swing_khist.pdf/png k against burnup, peaks marked, k_target dotted
  swing_report.txt    the console report
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

XE_POINT = 0.5
NOISE_RESOLVED_PCM = 400.0


def rho(k):
    return 1e5 * (k - 1.0) / k


def analyse_history(bu, k, ksd, k_target):
    bu = np.asarray(bu, float); k = np.asarray(k, float); ksd = np.asarray(ksd, float)
    op = bu > 1e-9
    if op.sum() == 0:
        raise ValueError("history has only the BOL point")
    i_xe = int(np.argmin(np.abs(bu - XE_POINT)))
    j = int(np.argmax(k[op]))
    k_peak = float(k[op][j]); b_peak = float(bu[op][j]); sd_peak = float(ksd[op][j])
    crossing = None
    for i in range(len(bu) - 1, 0, -1):
        if k[i - 1] > k_target >= k[i]:
            crossing = float(bu[i - 1] + (k[i - 1] - k_target) * (bu[i] - bu[i - 1]) / (k[i - 1] - k[i]))
            break
    r = dict(k_bol=float(k[0]), k_xe=float(k[i_xe]), k_peak=k_peak, b_peak=b_peak, k_target=float(k_target),
             crossing_mwd_kg=crossing, sd_bol_pcm=float(ksd[0]) * 1e5, sd_peak_pcm=sd_peak * 1e5,
             noise_pcm=float(np.median(ksd[ksd > 0]) * 1e5) if np.any(ksd > 0) else float("nan"))
    r["rho_bol"] = rho(r["k_bol"]); r["rho_xe"] = rho(r["k_xe"]); r["rho_peak"] = rho(k_peak)
    r["rho_eoc"] = rho(float(k_target))
    r["hump_pcm"] = r["rho_peak"] - r["rho_bol"]
    r["hump_vs_xe_pcm"] = r["rho_peak"] - r["rho_xe"]
    r["hump_resolved"] = bool(abs(r["hump_pcm"]) >= NOISE_RESOLVED_PCM)
    r["swing_peak_pcm"] = r["rho_peak"] - r["rho_eoc"]
    r["swing_xe_pcm"] = r["rho_xe"] - r["rho_eoc"]
    r["swing_bol_pcm"] = r["rho_bol"] - r["rho_eoc"]
    r["xenon_step_pcm"] = r["rho_bol"] - r["rho_xe"]
    r["bol_bounds_cycle"] = bool(k_peak <= r["k_bol"])
    return r


def boron_worth(runs, idx):
    """ARO differential worth on the core from the boron sweep, pcm/ppm."""
    out = {}
    def k_at(ppm):
        ks = [v["keff"] for kk, v in runs.items()
              if kk.startswith(f"{idx}|ARO|{ppm:.1f}|")]
        return float(np.mean(ks)) if ks else None
    k0, k1000, k1500 = k_at(0.0), k_at(1000.0), k_at(1500.0)
    if k0 and k1000:
        out["w_0_1000"] = (rho(k0) - rho(k1000)) / 1000.0
    if k1000 and k1500:
        out["w_1000_1500"] = (rho(k1000) - rho(k1500)) / 500.0
    if k1000:
        out["rho_core_1000_boron_sweep"] = rho(k1000)
    return out


def selftest():
    bu = [0, 0.5, 1.5, 3.5, 7.5, 11.5, 15.5]
    k = [1.20, 1.18, 1.19, 1.21, 1.19, 1.12, 1.05]
    sd = [2e-3] * 7
    r = analyse_history(bu, k, sd, 1.08)
    assert abs(r["k_peak"] - 1.21) < 1e-12 and abs(r["b_peak"] - 3.5) < 1e-12
    assert r["hump_pcm"] > 0 and not r["bol_bounds_cycle"]
    assert abs(r["swing_peak_pcm"] - (rho(1.21) - rho(1.08))) < 1e-9
    assert r["crossing_mwd_kg"] is not None and 11.5 < r["crossing_mwd_kg"] < 15.5
    k2 = [1.25, 1.20, 1.19, 1.18, 1.15, 1.10, 1.05]
    r2 = analyse_history(bu, k2, sd, 1.08)
    assert r2["bol_bounds_cycle"] and r2["hump_pcm"] < 0
    runs = {"9|ARO|0.0|0": {"keff": 1.20}, "9|ARO|1000.0|0": {"keff": 1.13}, "9|ARO|1500.0|0": {"keff": 1.10}}
    w = boron_worth(runs, 9)
    assert abs(w["w_0_1000"] - (rho(1.20) - rho(1.13)) / 1000) < 1e-12
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kt-summary", default="kt_burnup/summary.json")
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--boron-runs", default="boron_c8/runs.json")
    ap.add_argument("--confirm3d", default="confirm3d_c8/summary.json")
    ap.add_argument("--khist", default="khist_c8/khist.json", help="chunked-archive cross-check, optional")
    ap.add_argument("--designs", type=int, nargs="*", default=None, help="default: every design in kt_burnup")
    ap.add_argument("--out", default="swing_c8")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    kt = json.load(open(a.kt_summary))
    ck = json.load(open(a.checkpoint))["all_raw"]
    runs = json.load(open(a.boron_runs)) if Path(a.boron_runs).exists() else {}
    c3 = json.load(open(a.confirm3d)) if Path(a.confirm3d).exists() else {}
    kh = json.load(open(a.khist)) if Path(a.khist).exists() else {}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    by_idx = {int(r["idx"]): r for r in kt}
    designs = a.designs or sorted(by_idx)
    L = []; P = L.append
    P("=== C8 k history and reactivity swing (assembly k_inf at 1000 ppm, single-shot depletion) ===")
    missing = [d for d in designs if d not in by_idx]
    if missing:
        P(f"NOTE: not in {a.kt_summary}: {missing}. Run validate_ktarget_burnup.py on them first.")
    results = {}
    for i in designs:
        if i not in by_idx:
            continue
        rec = by_idx[i]; arc = ck[i]
        h = analyse_history(rec["bu_hist"], rec["k_hist"], rec["k_sd_hist"], rec["kt_table"])
        h.update(enrich=arc["enrich"], gd_wt=arc["gd_wt"], gd_pins=arc.get("gd_pins_used"),
                 refl_thick=arc["refl_thick"], efpd_archive=arc["cycle_length"],
                 efpd_single_shot=rec["efpd_here"], bu_eoc_archive=arc["bu_eoc_mwd_kg"],
                 keff_core_bol=arc["keff_core_bol"], rho_core_bol=rho(arc["keff_core_bol"]),
                 k_allre_2d=arc["k_allre"], k_re12_2d=arc.get("k_re12"),
                 bu=list(map(float, rec["bu_hist"])), k=list(map(float, rec["k_hist"])),
                 k_sd=list(map(float, rec["k_sd_hist"])))
        w = boron_worth(runs, i); h.update(w)
        h["lf_bol"] = float(rec.get("lf_bol", h["k_bol"] / arc["keff_core_bol"]))
        h["lf_drift_pcm"] = rec.get("drift_pcm"); h["lf_drift_sd_pcm"] = rec.get("drift_sd_pcm")
        h["hump_core_pcm"] = h["lf_bol"] * h["hump_pcm"]
        hump_pos = max(h["hump_core_pcm"], 0.0)
        if "w_1000_1500" in w:
            h["c_bol_ppm"] = 1000.0 + h["rho_core_bol"] / w["w_1000_1500"]
            h["c_peak_ppm"] = 1000.0 + (h["rho_core_bol"] + hump_pos) / w["w_1000_1500"]
            h["swing_peak_ppm_equiv"] = h["swing_peak_pcm"] / w["w_1000_1500"]
        s3 = c3.get(str(i), {})
        for st in ("ARI", "RE12"):
            for tier in ("2D", "3D"):
                key = f"{st}_margin{tier}_pcm"
                if key in s3:
                    h[f"{st}_margin_bol_{tier.lower()}_pcm"] = s3[key]
                    h[f"{st}_margin_peak_{tier.lower()}_pcm"] = s3[key] - hump_pos
        if str(i) in kh:
            hk = kh[str(i)]
            h["chunked_k_peak"] = hk.get("k_peak"); h["chunked_hump_vs_bol_pcm"] = hk.get("hump_vs_bol_pcm")
            if h["chunked_k_peak"] is not None:
                h["chunked_minus_single_shot_peak_pcm"] = 1e5 * (h["chunked_k_peak"] - h["k_peak"]) / h["k_peak"]
        results[str(i)] = h

    order = sorted(results, key=lambda s: results[s]["enrich"])
    P(f"{'idx':>3} {'e':>5} {'Gd':>5} {'pins':>4} | {'k_BOL':>7} {'k_Xe':>7} {'k_peak':>7} {'B_pk':>5} | "
      f"{'hump':>6} {'swPk':>6} {'swXe':>6} {'xeSt':>5} | {'EFPD ss/arc':>11} | {'w1k':>5} {'c_BOL':>5} {'c_pk':>5} | "
      f"{'humpC':>6} {'M16 3D pk':>9} {'M8 3D pk':>8}")
    for s in order:
        h = results[s]
        P(f"{s:>3} {h['enrich']:5.2f} {h['gd_wt']:5.2f} {h['gd_pins']:4} | {h['k_bol']:7.4f} {h['k_xe']:7.4f} "
          f"{h['k_peak']:7.4f} {h['b_peak']:5.1f} | {h['hump_pcm']:+6.0f} {h['swing_peak_pcm']:6.0f} "
          f"{h['swing_xe_pcm']:6.0f} {h['xenon_step_pcm']:5.0f} | {h['efpd_single_shot']:5.0f}/{h['efpd_archive']:5.0f} | "
          f"{h.get('w_1000_1500', float('nan')):5.2f} {h.get('c_bol_ppm', float('nan')):5.0f} {h.get('c_peak_ppm', float('nan')):5.0f} | "
          f"{h['hump_core_pcm']:+6.0f} {h.get('ARI_margin_peak_3d_pcm', float('nan')):9.0f} {h.get('RE12_margin_peak_3d_pcm', float('nan')):8.0f}")
    P("")
    P("columns: hump = rho_peak - rho_BOL; swPk = swing from the operating maximum to EOC; swXe = from the first")
    P("operating point; xeSt = BOL to first point; c_BOL, c_pk = critical boron estimate at BOL and at the")
    P("maximum, ppm, linear above 1000 ppm (lower bound); humpC = LF_BOL x hump, the hump on the core;")
    P("M16/M8 3D pk = four-bank and two-bank 3D margins at 1000 ppm reduced by humpC (BOL margin when negative).")
    P("The 1010 pcm screen (k <= 0.99) applies to these margins. Designs below it at the maximum are not")
    P("controllable by that bank set over the whole cycle, whatever the BOL screen said.")
    for st, lab in (("ARI", "four banks"), ("RE12", "two banks")):
        fail = [s for s in order if results[s].get(f"{st}_margin_peak_3d_pcm", 1e9) < 1010.1]
        P(f"  below 1010 pcm at the maximum, {lab}, 3D: {fail or 'none'}")
    unresolved = [s for s in order if not results[s]["hump_resolved"]]
    above = [s for s in order if not results[s]["bol_bounds_cycle"] and results[s]["hump_resolved"]]
    P(f"designs whose operating maximum is above the xenon-free BOL by more than {NOISE_RESOLVED_PCM:.0f} pcm: {above or 'none'}")
    P(f"designs whose hump is within noise (not resolved): {unresolved or 'none'}")
    if any("chunked_minus_single_shot_peak_pcm" in results[s] for s in order):
        P("cross-check against the chunked archive (khist_c8): peak difference in pcm")
        for s in order:
            if "chunked_minus_single_shot_peak_pcm" in results[s]:
                P(f"  {s:>3}: {results[s]['chunked_minus_single_shot_peak_pcm']:+.0f}")
    else:
        P("khist_c8/khist.json not present: chunked-archive cross-check skipped (run c8_hump2.py)")

    (out / "swing.json").write_text(json.dumps(results, indent=1))
    (out / "swing_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- LaTeX table, Pisa style: noun-phrase caption, list entry, units in header
    T = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption[Reactivity swing of the Campaign 8 candidates]{Gadolinium hump and burnup reactivity swing of the Campaign 8 candidates.}",
         r"  \label{tab:c8-swing}",
         r"  \begin{threeparttable}",
         r"  \begin{tabular}{@{}r r r r r r r r r r@{}}", r"    \toprule",
         r"    ID & $e$ [wt\%] & $k_\mathrm{BOL}$ & $k_\mathrm{peak}$ & $B_\mathrm{peak}$ [MWd/kgHM] & hump [pcm] & $\Delta\rho_\mathrm{peak\to EOC}$ [pcm] & $\Delta\rho_\mathrm{Xe\to EOC}$ [pcm] & $c_\mathrm{BOL}$ [ppm] & $c_\mathrm{peak}$ [ppm] \\",
         r"    \midrule"]
    for s in order:
        h = results[s]
        hump_txt = f"{h['hump_pcm']:+.0f}" if h["hump_resolved"] else "n.r."
        T.append(f"    {s} & {h['enrich']:.2f} & {h['k_bol']:.4f} & {h['k_peak']:.4f} & {h['b_peak']:.1f} & {hump_txt} & "
                 f"{h['swing_peak_pcm']:.0f} & {h['swing_xe_pcm']:.0f} & {h.get('c_bol_ppm', float('nan')):.0f} & {h.get('c_peak_ppm', float('nan')):.0f} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}",
          r"  \begin{tablenotes}\footnotesize",
          r"    \item Assembly $k_\infty$ at 1000\,ppm from the single-shot depletion of \texttt{validate\_ktarget\_burnup.py} at the campaign fidelity. The hump is $\rho_\mathrm{peak} - \rho_\mathrm{BOL}$, xenon-free BOL, and is marked n.r. when below the 400\,pcm resolution of the depletion noise. $\Delta\rho$ is measured to $\rho(k_\mathrm{target})$. The boron concentrations are linear extrapolations above 1000\,ppm with the 1000 to 1500\,ppm differential worth of the core and are lower bounds.",
          r"  \end{tablenotes}", r"  \end{threeparttable}", r"\end{table}"]
    (out / "swing_table.tex").write_text("\n".join(T) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        cmap = plt.get_cmap("viridis"); es = [results[s]["enrich"] for s in order]
        norm = plt.Normalize(min(es), max(es))
        for s in order:
            h = results[s]
            ax.plot(h["bu"], h["k"], "-", marker=".", ms=3, lw=1.0, color=cmap(norm(h["enrich"])), label=s)
            ax.plot(h["b_peak"], h["k_peak"], "v", color=cmap(norm(h["enrich"])), ms=5)
            ax.axhline(h["k_target"], color="0.5", lw=0.5, ls=":")
        ax.set_xlabel("Burnup, MWd/kgHM"); ax.set_ylabel("$k_\\infty$ of the depleting assembly, 1000 ppm")
        ax.legend(fontsize=7, ncol=3, title="design")
        fig.savefig(out / "swing_khist.pdf", bbox_inches="tight"); fig.savefig(out / "swing_khist.png", dpi=300, bbox_inches="tight")
        print(f"wrote {out}/swing.json, swing_table.tex, swing_report.txt, swing_khist.pdf")
    except Exception as ex:
        print(f"wrote {out}/swing.json, swing_table.tex, swing_report.txt (no figure: {ex})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
