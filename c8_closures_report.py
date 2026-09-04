#!/usr/bin/env python3
"""
c8_closures_report.py -- the four closures of the three-dimensional
confirmation of the Campaign 8 candidates, in numbers. No transport.
Written 4 September 2026 against commit f60b78a of branch campaign8.

The Notion task "Run the 3D confirmation study, carrying four separate
closures" requires a scope paragraph listing four commitments and the
measured number that closes each. This script reads every confirmation
output present and writes the numbers, so the paragraph is written from
data rather than from memory. Every input is optional: an absent stage is
reported as such and its sentence becomes an \\openpoint{}.

CLOSURE 1  second tier of the fidelity strategy
   kt_burnup/summary.json     single-shot depletion at the campaign fidelity,
                              core solves at BOL and EOC, per design: leakage
                              factor drift and the implied cycle correction
   swing_c8/swing.json        single-shot cycle length against the archive
CLOSURE 2  axial correction
   confirm3d_c8/summary.json  L_ax_hw = k2D/k3D_hw per rod state at 1000 ppm,
                              against the campaign 1.0289 (ktarget_table_c8)
   confirm3d_c8_noparked      parked-rod share of the unrodded factor
CLOSURE 3  benchmark departures carried as sensitivities
   confirm3d_c8_refl956       reflector 0.956 steel against campaign 0.90, ARO
   confirm3d_c8_rabs4229      absorber radius 0.4229 against 0.4331, rodded
   coolant density 0.72 vs 0.752 g/cm3: not exposed as a flag, stated only
CLOSURE 4  rod-stack comparison
   confirm3d_c8/summary.json  benchmark stack (12.04 cm unrodded, 30.48 cm
                              AIC, B4C above) against the 2D full-height B4C
   confirm3d_c8_fullb4c       3D core with the full-height B4C stack, which
                              separates the stack effect from the axial
                              leakage inside the rodded factor
ALSO       confirm3d_c8_0ppm (boron-free necessary condition),
           confirm3d_c8_dc37 (downcomer 3.7 cm), swing_c8 (hump-adjusted
           margins)

USAGE
  python c8_closures_report.py [--out closures_c8]
OUTPUTS in <out>/: closures.json, closures_report.txt, closures_table.tex,
  closures_scope.tex (draft paragraph, \\openpoint{} where a stage is absent)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

CAMPAIGN_LAX = 1.0289
SCREEN_PCM = 1010.1     # k <= 0.99 on the reactivity axis


def rho(k):
    return 1e5 * (k - 1.0) / k


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def rng(xs, fmt="{:.4f}"):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (fmt.format(min(xs)) + " to " + fmt.format(max(xs))) if xs else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="closures_c8")
    ap.add_argument("--ktarget-table", default="ktarget_table_c8.json")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = []; P = L.append
    C = {}     # everything the paragraph needs
    absent = []

    kt = load(a.ktarget_table)
    lax_c = float(kt.get("axial_leakage_factor", CAMPAIGN_LAX)) if kt else CAMPAIGN_LAX
    lax_rng = kt.get("axial_leakage_range", [None, None]) if kt else [None, None]
    C["campaign_lax"] = lax_c; C["campaign_lax_range"] = lax_rng

    # ---------------- closure 1: second tier -------------------------------
    kb = load("kt_burnup/summary.json")
    if kb:
        d = {int(r["idx"]): r for r in kb}
        C["c1_designs"] = sorted(d)
        C["c1_drift_pcm"] = {i: [d[i]["drift_pcm"], d[i]["drift_sd_pcm"]] for i in d}
        C["c1_defpd"] = {i: d[i]["defpd_days"] for i in d}
        C["c1_resid_eoc_pcm"] = {i: d[i]["resid_eoc_pcm"] for i in d}
        within = [i for i in d if abs(d[i]["drift_pcm"]) <= 2 * d[i]["drift_sd_pcm"]]
        C["c1_drift_within_2sigma"] = within
        C["c1_efpd_single_vs_archive"] = {i: [d[i]["efpd_here"], d[i]["efpd_checkpoint"]] for i in d}
        P("CLOSURE 1, second tier (single-shot depletion, core solves at BOL and EOC)")
        P(f"  designs: {sorted(d)}")
        P(f"  leakage-factor drift BOL to EOC: {rng([d[i]['drift_pcm'] for i in d], '{:+.0f}')} pcm, "
          f"sd {rng([d[i]['drift_sd_pcm'] for i in d], '{:.0f}')} pcm; within 2 sigma: {len(within)} of {len(d)}")
        P(f"  EOC residual to the table: {rng([d[i]['resid_eoc_pcm'] for i in d], '{:+.0f}')} pcm")
        P(f"  implied cycle correction: {rng([d[i]['defpd_days'] for i in d], '{:+.0f}')} EFPD")
        P(f"  single-shot minus archived cycle length: "
          f"{rng([d[i]['efpd_here'] - d[i]['efpd_checkpoint'] for i in d], '{:+.0f}')} EFPD (restart consistency)")
    else:
        absent.append("kt_burnup/summary.json (closure 1)"); P("CLOSURE 1: kt_burnup/summary.json absent")

    # ---------------- closure 2: axial ------------------------------------
    c3 = load("confirm3d_c8/summary.json")
    if c3:
        st = {"ARO": "unrodded", "RE12": "two-bank", "ARI": "four-bank"}
        C["c2_designs"] = sorted(c3, key=int)
        for s in st:
            vals = [c3[i].get(f"{s}_Lax_hw") for i in c3]
            C[f"c2_lax_{s}"] = {i: c3[i].get(f"{s}_Lax_hw") for i in c3}
            P(f"CLOSURE 2, L_ax_hw {st[s]:9s}: {rng(vals)}  (campaign {lax_c}, water-slab study {lax_rng})")
        aro = {i: c3[i]["ARO_Lax_hw"] for i in c3 if "ARO_Lax_hw" in c3[i]}
        e = {i: c3[i]["design"]["enrich_inner"] for i in c3}
        lo = [aro[i] for i in aro if e[i] < 7.0]; hi = [aro[i] for i in aro if e[i] >= 7.0]
        C["c2_lax_ARO_below7"] = rng(lo); C["c2_lax_ARO_above7"] = rng(hi)
        C["c2_gap_to_campaign_pcm"] = {i: 1e5 * (aro[i] - lax_c) / lax_c for i in aro}
        P(f"  unrodded factor by enrichment: below 7 wt% {rng(lo)}, above {rng(hi)}; "
          f"gap to campaign value {rng(list(C['c2_gap_to_campaign_pcm'].values()), '{:+.0f}')} pcm")
        for s in ("RE12", "ARI"):
            m2 = [c3[i].get(f"{s}_margin2D_pcm") for i in c3]; m3 = [c3[i].get(f"{s}_margin3D_pcm") for i in c3]
            cons = [b - a_ for a_, b in zip(m2, m3) if a_ is not None and b is not None]
            C[f"c4_screen_conservatism_{s}_pcm"] = {i: c3[i][f"{s}_margin3D_pcm"] - c3[i][f"{s}_margin2D_pcm"]
                                                   for i in c3 if f"{s}_margin3D_pcm" in c3[i]}
            P(f"  2D screen conservatism, {st[s]}: {rng(cons, '{:.0f}')} pcm (3D margin minus 2D margin)")
    else:
        absent.append("confirm3d_c8/summary.json (closures 2 and 4)"); P("CLOSURE 2: confirm3d_c8/summary.json absent")
    npk = load("confirm3d_c8_noparked/summary.json")
    if npk and c3:
        for i in npk:
            if i in c3 and "ARO_3Dhw" in npk[i]:
                C.setdefault("c2_parked_rod_pcm", {})[i] = rho(c3[i]["ARO_3Dhw"]["k"]) - rho(npk[i]["ARO_3Dhw"]["k"])
        P(f"  parked-rod share of the unrodded factor: {C.get('c2_parked_rod_pcm')} pcm (3D with parked rods minus without)")
    else:
        absent.append("confirm3d_c8_noparked (stage E, optional)")

    # ---------------- closure 3: departures --------------------------------
    r956 = load("confirm3d_c8_refl956/summary.json")
    if r956 and c3:
        for i in r956:
            if i in c3 and "ARO_3Dhw" in r956[i]:
                C.setdefault("c3_refl956_pcm", {})[i] = rho(r956[i]["ARO_3Dhw"]["k"]) - rho(c3[i]["ARO_3Dhw"]["k"])
                C.setdefault("c3_refl956_lax", {})[i] = r956[i].get("ARO_Lax_hw")
        P(f"CLOSURE 3a, reflector 0.956 vs 0.90 steel, unrodded 3D: {C.get('c3_refl956_pcm')} pcm "
          f"(positive: the benchmark reflector is more reactive); L_ax with 0.956: {C.get('c3_refl956_lax')}")
    else:
        absent.append("confirm3d_c8_refl956 (closure 3a)"); P("CLOSURE 3a: confirm3d_c8_refl956 absent")
    r4229 = load("confirm3d_c8_rabs4229/summary.json")
    if r4229 and c3:
        for i in r4229:
            for s in ("ARI", "RE12"):
                if i in c3 and f"{s}_margin3D_pcm" in r4229[i] and f"{s}_margin3D_pcm" in c3[i]:
                    C.setdefault(f"c3_rabs4229_{s}_pcm", {})[i] = r4229[i][f"{s}_margin3D_pcm"] - c3[i][f"{s}_margin3D_pcm"]
        P(f"CLOSURE 3b, absorber radius 0.4229 vs 0.4331, change of the 3D margin: four-bank {C.get('c3_rabs4229_ARI_pcm')}, "
          f"two-bank {C.get('c3_rabs4229_RE12_pcm')} pcm (negative: the campaign radius overstates the worth)")
    else:
        absent.append("confirm3d_c8_rabs4229 (closure 3b)"); P("CLOSURE 3b: confirm3d_c8_rabs4229 absent")
    P("CLOSURE 3c, coolant density 0.72 vs 0.752 g/cm3: no flag in confirm3d.py, stated as a departure only")

    # ---------------- closure 4: rod stack ---------------------------------
    fb = load("confirm3d_c8_fullb4c/summary.json")
    if fb and c3:
        for i in fb:
            for s in ("ARI", "RE12"):
                if i in c3 and f"{s}_3Dhw" in fb[i] and f"{s}_3Dhw" in c3[i]:
                    C.setdefault(f"c4_stack_effect_{s}_pcm", {})[i] = rho(c3[i][f"{s}_3Dhw"]["k"]) - rho(fb[i][f"{s}_3Dhw"]["k"])
                    C.setdefault(f"c4_axial_only_{s}_lax", {})[i] = fb[i].get(f"{s}_Lax_hw")
        P(f"CLOSURE 4, benchmark stack minus full-height B4C on the 3D core: four-bank {C.get('c4_stack_effect_ARI_pcm')}, "
          f"two-bank {C.get('c4_stack_effect_RE12_pcm')} pcm (positive: the benchmark rod is weaker)")
        P(f"  rodded axial factor with the full-B4C stack (leakage only): four-bank {C.get('c4_axial_only_ARI_lax')}, "
          f"two-bank {C.get('c4_axial_only_RE12_lax')}")
    else:
        absent.append("confirm3d_c8_fullb4c (closure 4 separation)"); P("CLOSURE 4: confirm3d_c8_fullb4c absent, the screen conservatism above is the closure as run")

    # ---------------- also in scope ----------------------------------------
    z = load("confirm3d_c8_0ppm/summary.json")
    if z:
        C["also_0ppm_ARI_margin3D_pcm"] = {i: z[i].get("ARI_margin3D_pcm") for i in z}
        C["also_0ppm_pass_screen"] = [i for i in z if (z[i].get("ARI_margin3D_pcm") or -1e9) >= SCREEN_PCM]
        P(f"ALSO, 0 ppm four-bank 3D margin: {rng(list(C['also_0ppm_ARI_margin3D_pcm'].values()), '{:.0f}')} pcm; "
          f"designs passing the 1010 pcm screen at 0 ppm: {C['also_0ppm_pass_screen']}")
    dc = load("confirm3d_c8_dc37/summary.json")
    if dc and c3:
        for i in dc:
            if i in c3 and "ARO_3Dhw" in dc[i]:
                C.setdefault("also_dc37_pcm", {})[i] = rho(dc[i]["ARO_3Dhw"]["k"]) - rho(c3[i]["ARO_3Dhw"]["k"])
        P(f"ALSO, downcomer 3.7 cm (reflector cut to 3.969 cm), unrodded 3D: {C.get('also_dc37_pcm')} pcm")
    sw = load("swing_c8/swing.json")
    if sw:
        C["also_hump_core_pcm"] = {i: sw[i]["hump_core_pcm"] for i in sw}
        C["also_two_bank_fail_at_peak_3d"] = [i for i in sw if sw[i].get("RE12_margin_peak_3d_pcm", 1e9) < SCREEN_PCM]
        C["also_four_bank_fail_at_peak_3d"] = [i for i in sw if sw[i].get("ARI_margin_peak_3d_pcm", 1e9) < SCREEN_PCM]
        P(f"ALSO, hump carried to the core: {rng(list(C['also_hump_core_pcm'].values()), '{:+.0f}')} pcm; "
          f"below the screen at the operating maximum in 3D: two-bank {C['also_two_bank_fail_at_peak_3d']}, "
          f"four-bank {C['also_four_bank_fail_at_peak_3d']}")
    if absent:
        P(f"ABSENT STAGES: {absent}")
    C["absent"] = absent
    (out / "closures.json").write_text(json.dumps(C, indent=1, default=float))
    (out / "closures_report.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))

    # ---------------- LaTeX: table and draft paragraph ----------------------
    def op(s): return "\\openpoint{" + s + "}"
    rows = []
    rows.append(("1 Second tier", "leakage-factor drift BOL to EOC [pcm]",
                 rng([v[0] for v in C.get("c1_drift_pcm", {}).values()], "{:+.0f}") if kb else "pending",
                 "implied cycle correction [EFPD]", rng(list(C.get("c1_defpd", {}).values()), "{:+.0f}") if kb else "pending"))
    rows.append(("2 Axial correction", "$L_\\mathrm{ax,hw}$ unrodded",
                 rng(list(C.get("c2_lax_ARO", {}).values())) if c3 else "pending",
                 "gap to the campaign 1.0289 [pcm]", rng(list(C.get("c2_gap_to_campaign_pcm", {}).values()), "{:+.0f}") if c3 else "pending"))
    rows.append(("3a Reflector 0.956", "$\\Delta\\rho$ unrodded 3D [pcm]",
                 rng(list(C.get("c3_refl956_pcm", {}).values()), "{:+.0f}") if r956 else "pending", "", ""))
    rows.append(("3b Absorber 0.4229", "$\\Delta$ four-bank margin [pcm]",
                 rng(list(C.get("c3_rabs4229_ARI_pcm", {}).values()), "{:+.0f}") if r4229 else "pending",
                 "$\\Delta$ two-bank margin [pcm]", rng(list(C.get("c3_rabs4229_RE12_pcm", {}).values()), "{:+.0f}") if r4229 else "pending"))
    rows.append(("4 Rod stack", "screen conservatism four-bank [pcm]",
                 rng(list(C.get("c4_screen_conservatism_ARI_pcm", {}).values()), "{:.0f}") if c3 else "pending",
                 "stack effect four-bank [pcm]", rng(list(C.get("c4_stack_effect_ARI_pcm", {}).values()), "{:+.0f}") if fb else "pending"))
    T = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption[Closures of the three-dimensional confirmation]{Measured closure of the four commitments carried by the three-dimensional confirmation.}",
         r"  \label{tab:c8-closures}", r"  \begin{tabular}{@{}l l r l r@{}}", r"    \toprule",
         r"    Closure & Quantity & Value & Quantity & Value \\", r"    \midrule"]
    for r in rows:
        T.append("    " + " & ".join(r) + r" \\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out / "closures_table.tex").write_text("\n".join(T) + "\n")

    S = ["%% Draft scope paragraph for the head of the confirmation section.",
         "%% Generated by c8_closures_report.py. Numbers come from the confirmation",
         "%% outputs; sentences of absent stages are \\openpoint{} boxes.", "",
         "The three-dimensional confirmation closes four commitments made earlier",
         "in this work. Each is listed with the quantity that closes it.", "",
         r"\begin{enumerate}"]
    if kb:
        S.append(f"  \\item The second tier of the fidelity strategy of Section~\\ref{{sec:meth-fidelity}}. "
                 f"Ten front designs were depleted in a single shot at the campaign fidelity with core solves at beginning and end of cycle. "
                 f"The leakage factor drifts by {rng([v[0] for v in C['c1_drift_pcm'].values()], '{:+.0f}')}\\,pcm between the two, "
                 f"within two standard deviations for {len(C['c1_drift_within_2sigma'])} of {len(C['c1_designs'])} designs, "
                 f"and the implied cycle correction is {rng(list(C['c1_defpd'].values()), '{:+.0f}')}\\,EFPD.")
    else:
        S.append("  \\item " + op("Closure 1: run validate\\_ktarget\\_burnup.py on the candidates and quote the drift and the cycle correction."))
    if c3:
        S.append(f"  \\item The axial correction of Section~\\ref{{sec:meth-lax}}. On the core with hardware the unrodded factor is "
                 f"{rng(list(C['c2_lax_ARO'].values()))}, against the campaign value {lax_c} of the water-slab study, "
                 f"a gap of {rng(list(C['c2_gap_to_campaign_pcm'].values()), '{:+.0f}')}\\,pcm. "
                 f"The two-bank factor is {rng(list(C['c2_lax_RE12'].values()))} and the four-bank factor {rng(list(C['c2_lax_ARI'].values()))}.")
    else:
        S.append("  \\item " + op("Closure 2: the confirmation summary is absent."))
    s3 = "  \\item The three departures from the benchmark inputs stated in Section~\\ref{sec:meth-model}. "
    s3 += (f"The benchmark reflector at 0.956 steel changes the unrodded three-dimensional eigenvalue by "
           f"{rng(list(C['c3_refl956_pcm'].values()), '{:+.0f}')}\\,pcm. " if r956 else op("Closure 3a: reflector sensitivity pending. ") + " ")
    s3 += (f"The benchmark absorber radius 0.4229\\,cm changes the four-bank margin by "
           f"{rng(list(C['c3_rabs4229_ARI_pcm'].values()), '{:+.0f}')}\\,pcm and the two-bank margin by "
           f"{rng(list(C['c3_rabs4229_RE12_pcm'].values()), '{:+.0f}')}\\,pcm. " if r4229 else op("Closure 3b: absorber-radius sensitivity pending. ") + " ")
    s3 += "The coolant density is not exposed as a sensitivity and remains a stated departure."
    S.append(s3)
    if c3:
        s4 = (f"  \\item The rod-stack comparison. The two-dimensional screen with full-height boron carbide is conservative by "
              f"{rng(list(C['c4_screen_conservatism_ARI_pcm'].values()), '{:.0f}')}\\,pcm for four banks and "
              f"{rng(list(C['c4_screen_conservatism_RE12_pcm'].values()), '{:.0f}')}\\,pcm for two banks against the benchmark rod on the three-dimensional core. ")
        s4 += (f"Of this, the stack itself accounts for {rng(list(C['c4_stack_effect_ARI_pcm'].values()), '{:+.0f}')}\\,pcm with four banks and "
               f"{rng(list(C['c4_stack_effect_RE12_pcm'].values()), '{:+.0f}')}\\,pcm with two banks, the remainder being axial leakage."
               if fb else op("Closure 4: the full-B4C stack run that separates the stack from the leakage is pending."))
        S.append(s4)
    S += [r"\end{enumerate}", ""]
    if sw:
        S.append(f"The beginning-of-life margins above are reduced by the gadolinium hump of the operating maximum, "
                 f"{rng(list(C['also_hump_core_pcm'].values()), '{:+.0f}')}\\,pcm on the core. "
                 f"At the maximum the two-bank set at 1000\\,ppm on the three-dimensional core excludes designs "
                 f"{', '.join(C['also_two_bank_fail_at_peak_3d']) or 'none'}. %% AUTHOR: physics of the hump and its consequence for the candidate.")
    (out / "closures_scope.tex").write_text("\n".join(S) + "\n")
    print(f"wrote {out}/closures.json, closures_report.txt, closures_table.tex, closures_scope.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
