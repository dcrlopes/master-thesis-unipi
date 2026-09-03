#!/usr/bin/env python3
"""
c8_boron_3d_analysis.py -- first analysis of boron_c8 and confirm3d_c8.

Reads the two summary/runs pairs, recomputes design 53 ARO with the smoke
run excluded, estimates sigma(F_dH) from the confirm3d seed pairs, and
writes five figures plus a text report. No OpenMC needed.

Terminology (from confirm3d.py and zoning.py, campaign8 branch):
  ARI  = RE1..RE4, the sixteen regulating-bank assemblies (ALL-RE).
         The sixteen SH assemblies are NOT inserted. This is the thesis's
         operational-controllability state, not a scram state.
  RE12 = RE1 + RE2, eight assemblies.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

UP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/claude/figs")
OUT.mkdir(parents=True, exist_ok=True)

bor = json.load(open(UP / "summary.json"))          # boron_c8/summary.json
c3r = json.load(open(UP / "runs_1.json"))            # confirm3d_c8/runs.json
c3s = json.load(open(UP / "summary_1.json"))         # confirm3d_c8/summary.json
blog = (UP / "boron_c8.log").read_text()

def rho(k): return 1.0e5 * (k - 1.0) / k

# ---- design variables from the boron log header ------------------------
dv = {}
for line in blog.splitlines():
    if line.strip().startswith("design ") and "archive k_core" in line:
        t = line.split(); idx = int(t[1].rstrip(":"))
        kv = {p.split("=")[0]: float(p.split("=")[1]) for p in t[2:6]}
        dv[idx] = dict(enr=kv["e"], gd=kv["gd"], refl=kv["refl"], pins=kv["pins"])

ORDER = sorted(bor, key=lambda i: dv[int(i)]["enr"])   # by enrichment
PPM = [0.0, 500.0, 1000.0, 1500.0]

# ---- confirm3d: rebuild per-design stats from runs.json, smoke excluded --
def is_smoke(rec): return rec["wall_s"] < 60 or rec["sd"] > 1.5e-3
c3 = {}
pairs = {"ARO": [], "rodded": []}
for key, rec in c3r.items():
    idx, st, mode = key.split("|")[:3]
    if is_smoke(rec): continue
    c3.setdefault(idx, {}).setdefault(f"{st}_{mode}", []).append(rec)
c3stat = {}
for idx, d in c3.items():
    r = {}
    for st in ("ARO", "ARI", "RE12"):
        for mode in ("2D", "3Dhw"):
            recs = d.get(f"{st}_{mode}", [])
            ks = [x["keff"] for x in recs]; fs = [x["fdh"] for x in recs]
            r[f"{st}_{mode}"] = dict(k=np.mean(ks), F=np.mean(fs), n=len(ks),
                                     k_sd=np.std(ks, ddof=1) if len(ks) > 1 else np.nan,
                                     F_sd=np.std(fs, ddof=1) if len(fs) > 1 else np.nan)
            if len(fs) == 2:
                pairs["ARO" if st == "ARO" else "rodded"].append(fs[0] - fs[1])
        k2, k3 = r[f"{st}_2D"]["k"], r[f"{st}_3Dhw"]["k"]
        r[f"{st}_Lax"] = k2 / k3
        r[f"{st}_dk"] = rho(k2) - rho(k3)
        r[f"{st}_m2D"] = -rho(k2); r[f"{st}_m3D"] = -rho(k3)
    c3stat[idx] = r

# sigma(F) from seed pairs: sigma = sqrt(mean(delta^2)/2)
sigF = {k: float(np.sqrt(np.mean(np.square(v)) / 2.0)) for k, v in pairs.items()}
sigF["all"] = float(np.sqrt(np.mean(np.square(pairs["ARO"] + pairs["rodded"])) / 2.0))

# ---- report -------------------------------------------------------------
rep = []
P = rep.append
P("=== Design 53 ARO, smoke run excluded (seed 1 only) vs summary.json (smoke averaged in) ===")
s53 = c3stat["53"]; j53 = c3s["53"]
P(f"  k2D   {s53['ARO_2D']['k']:.5f}  (summary {j53['ARO_2D']['k']:.5f})")
P(f"  k3D   {s53['ARO_3Dhw']['k']:.5f}  (summary {j53['ARO_3Dhw']['k']:.5f})")
P(f"  L_ax  {s53['ARO_Lax']:.5f}  (summary {j53['ARO_Lax_hw']:.5f})")
P(f"  dk    {s53['ARO_dk']:.0f} pcm  (summary {j53['ARO_dk_pcm']:.0f})")
P(f"  F2D   {s53['ARO_2D']['F']:.3f}  (summary {j53['ARO_2D']['F']:.3f})   F3D {s53['ARO_3Dhw']['F']:.3f}  (summary {j53['ARO_3Dhw']['F']:.3f})")
P("")
P(f"=== sigma(F_dH) per solve from confirm3d seed pairs (150000 x 200, 80 inactive) ===")
P(f"  ARO    {sigF['ARO']:.4f}  (n = {len(pairs['ARO'])} pairs)")
P(f"  rodded {sigF['rodded']:.4f}  (n = {len(pairs['rodded'])} pairs)")
P(f"  pooled {sigF['all']:.4f}")
P("")
P("=== confirm3d per design (corrected), 1000 ppm ===")
P(f"{'idx':>4} {'enr':>5} | {'Lax ARO':>8} {'Lax RE12':>8} {'Lax ARI':>8} | {'dk ARO':>7} {'dk RE12':>7} {'dk ARI':>7} | {'M16 2D':>7} {'M16 3D':>7} | {'M8 2D':>7} {'M8 3D':>7}")
for idx in sorted(c3stat, key=lambda i: dv[int(i)]["enr"]):
    r = c3stat[idx]
    P(f"{idx:>4} {dv[int(idx)]['enr']:5.2f} | {r['ARO_Lax']:8.4f} {r['RE12_Lax']:8.4f} {r['ARI_Lax']:8.4f} | "
      f"{r['ARO_dk']:7.0f} {r['RE12_dk']:7.0f} {r['ARI_dk']:7.0f} | {r['ARI_m2D']:7.0f} {r['ARI_m3D']:7.0f} | "
      f"{r['RE12_m2D']:7.0f} {r['RE12_m3D']:7.0f}")
lax_aro = [c3stat[i]["ARO_Lax"] for i in c3stat]
P(f"  mean L_ax(ARO) hardware = {np.mean(lax_aro):.4f}, range {min(lax_aro):.4f} to {max(lax_aro):.4f}; campaign used 1.0289")
P("")

# ---- ARI (ALL-RE) axial gain vs enrichment, linear fit for extrapolation --
e5 = np.array([dv[int(i)]["enr"] for i in c3stat]); g5 = np.array([c3stat[i]["ARI_dk"] for i in c3stat])
A = np.vstack([e5, np.ones_like(e5)]).T
slope, icpt = np.linalg.lstsq(A, g5, rcond=None)[0]
resid = g5 - (slope * e5 + icpt)
P(f"=== ALL-RE axial gain vs enrichment: dk = {slope:.1f} * enr + {icpt:.0f} pcm, rms resid {np.sqrt(np.mean(resid**2)):.0f} pcm ===")
gain3d = {}
for idx in bor:
    e = dv[int(idx)]["enr"]
    if idx in c3stat: gain3d[idx] = (c3stat[idx]["ARI_dk"], "measured")
    else:            gain3d[idx] = (slope * e + icpt, "extrapolated")
P("")
P("=== Zero-boron ALL-RE (M16) margin: 2D measured, 3D by adding the 1000 ppm axial gain ===")
P(f"{'idx':>4} {'enr':>5} {'kcore':>7} {'EFPD':>6} | {'M16(0) 2D':>10} {'gain':>6} {'src':>12} {'M16(0) 3D':>10} | {'M16(1000) 2D':>12}")
for idx in ORDER:
    b = bor[idx]; m0 = b["margin_ARI_pcm"]["0.0"]; g, src = gain3d[idx]
    P(f"{idx:>4} {dv[int(idx)]['enr']:5.2f} {b['archive']['keff_core_bol']:7.4f} {b['archive']['cycle_length']:6.0f} | "
      f"{m0:10.0f} {g:6.0f} {src:>12} {m0+g:10.0f} | {b['margin_ARI_pcm']['1000.0']:12.0f}")
P("  caveat: the gain is measured at 1000 ppm and applied at 0 ppm. The rodded L_ax also carries")
P("  the rod-stack departure (2D full-height B4C r=0.4331 vs 3D hybrid AIC/B4C with a 12 cm unrodded")
P("  bottom), so it is a 2D-to-3D hardware correction, not a pure axial leakage factor.")
P("")

# ---- boron-study vs confirm3d 2D F at 1000 ppm (fidelity bias check) ----
P("=== F_dH at 1000 ppm: boron study (100k x 170) minus confirm3d 2D (150k x 200, 2 seeds) ===")
diffs = []
for idx in c3stat:
    for st in ("ARO", "ARI", "RE12"):
        fb = c3r_key = None
        fb = json.load(open(UP / "runs.json"))[f"{idx}|{st}|1000.0|0"]["fdh"]
        fc = c3stat[idx][f"{st}_2D"]["F"]
        diffs.append(fb - fc); P(f"  d{idx:>2} {st:4s}  {fb:.3f} - {fc:.3f} = {fb-fc:+.3f}")
d = np.array(diffs)
P(f"  mean {d.mean():+.3f}, {np.sum(d>0)} of {len(d)} positive, sd {d.std(ddof=1):.3f}")
P("")

# ---- boron worth table --------------------------------------------------
P("=== Boron study, ordered by enrichment ===")
P(f"{'idx':>4} {'enr':>5} {'gd':>5} {'pins':>4} | {'dρ/dc':>6} {'0-500':>6} {'500-1k':>6} {'1k-1.5k':>7} | {'W_RE 0':>7} {'W_RE 1k':>7} {'W_RE12 1k':>9} | {'share':>5}")
for idx in ORDER:
    b = bor[idx]; di = b["differential_by_interval_pcm_per_ppm"]; v = dv[int(idx)]
    P(f"{idx:>4} {v['enr']:5.2f} {v['gd']:5.2f} {v['pins']:4.0f} | {b['differential_worth_pcm_per_ppm']:6.3f} "
      f"{di['0.0-500.0']:6.3f} {di['500.0-1000.0']:6.3f} {di['1000.0-1500.0']:7.3f} | "
      f"{b['worth_ARI_pcm']['0.0']:7.0f} {b['worth_ARI_pcm']['1000.0']:7.0f} {b['worth_RE12_pcm']['1000.0']:9.0f} | {b['boron_share_of_holddown']:5.3f}")
(OUT / "analysis_report.txt").write_text("\n".join(rep))
print("\n".join(rep))

# ======================= FIGURES =========================================
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
cmap = plt.get_cmap("viridis")
def col(idx): return cmap((dv[int(idx)]["enr"] - 3.0) / (11.5 - 3.0))
dep = {"23", "21", "44", "1"}   # boron-dependent under ALL-RE at 0 ppm

# Fig 1: differential boron worth vs enrichment ---------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for idx in ORDER:
    b = bor[idx]; v = dv[int(idx)]
    ax[0].scatter(v["enr"], b["differential_worth_pcm_per_ppm"], s=30 + 4 * v["pins"], color=col(idx),
                  edgecolor="k", zorder=3)
    ax[0].annotate(f"d{idx}\n{v['pins']:.0f} pins", (v["enr"], b["differential_worth_pcm_per_ppm"]),
                   textcoords="offset points", xytext=(8, -4), fontsize=8)
ax[0].set_xlabel("Enrichment, wt% $^{235}$U"); ax[0].set_ylabel("Boron differential worth, ARO, 0 to 1000 ppm  [pcm/ppm]")
ax[0].set_title("Boron worth falls monotonically with enrichment\n(marker size = gadolinia pin count)")
ax[0].set_ylim(2, 9.5)
w = 0.25; x = np.arange(len(ORDER))
for j, (lab, key) in enumerate([("0-500", "0.0-500.0"), ("500-1000", "500.0-1000.0"), ("1000-1500", "1000.0-1500.0")]):
    ax[1].bar(x + (j - 1) * w, [bor[i]["differential_by_interval_pcm_per_ppm"][key] for i in ORDER], w, label=f"{lab} ppm")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"d{i}\n{dv[int(i)]['enr']:.1f}%" for i in ORDER], fontsize=8)
ax[1].set_ylabel("Interval differential worth  [pcm/ppm]"); ax[1].legend(fontsize=8)
ax[1].set_title("Self-shielding: worth decreases with concentration\n(1σ per bar ≈ 0.085 pcm/ppm)")
fig.tight_layout(); fig.savefig(OUT / "fig1_boron_worth_vs_enrichment.png"); plt.close(fig)

# Fig 2: ALL-RE margin vs boron, and zero-boron 2D vs 3D ------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for idx in ORDER:
    m = [bor[idx]["margin_ARI_pcm"][f"{c}"] for c in PPM]
    ax[0].plot(PPM, m, "-o", color=col(idx), label=f"d{idx} ({dv[int(idx)]['enr']:.1f}%)", lw=1.6, ms=4)
ax[0].axhline(0, color="k", lw=1); ax[0].axhspan(-3000, 0, color="red", alpha=0.07)
ax[0].axvline(1000, color="gray", ls="--", lw=0.8)
ax[0].set_xlabel("Soluble boron  [ppm]"); ax[0].set_ylabel("$M_{16}$: subcriticality under ALL-RE (RE1 to RE4)  [pcm]")
ax[0].set_title("Four designs are supercritical under ALL-RE at 0 ppm (2D, BOL)")
ax[0].legend(fontsize=7, ncol=2)
m2 = [bor[i]["margin_ARI_pcm"]["0.0"] for i in ORDER]
m3 = [bor[i]["margin_ARI_pcm"]["0.0"] + gain3d[i][0] for i in ORDER]
hatch = ["" if gain3d[i][1] == "measured" else "//" for i in ORDER]
ax[1].bar(x - 0.2, m2, 0.4, label="2D measured", color="#4c72b0")
bars = ax[1].bar(x + 0.2, m3, 0.4, label="3D: 2D + ALL-RE axial gain", color="#dd8452")
for bb, h in zip(bars, hatch): bb.set_hatch(h)
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"d{i}\n{dv[int(i)]['enr']:.1f}%" for i in ORDER], fontsize=8)
ax[1].set_ylabel("$M_{16}(0\\ \\mathrm{ppm})$  [pcm]"); ax[1].legend(fontsize=8)
ax[1].set_title("Zero-boron ALL-RE margin. Hatched = gain extrapolated\n(d44, d1, d13 have no 3D run)")
fig.tight_layout(); fig.savefig(OUT / "fig2_allre_margin_vs_boron.png"); plt.close(fig)

# Fig 3: ARO F_dH vs boron ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
runs_b = json.load(open(UP / "runs.json"))
for idx in ORDER:
    f = [runs_b[f"{idx}|ARO|{c}|0"]["fdh"] for c in PPM]
    ax.plot(PPM, f, "-o", color=col(idx), label=f"d{idx} ({dv[int(idx)]['enr']:.1f}%)", lw=1.6, ms=4)
ax.set_xlabel("Soluble boron  [ppm]"); ax.set_ylabel("$F_{\\Delta H}$, all rods out (single seed)")
ax.set_title(f"Unrodded peaking falls with boron in 8 of 8 designs\n(1σ per point ≈ {sigF['ARO']:.3f} at confirm3d fidelity, larger here)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig3_fdh_aro_vs_boron.png"); plt.close(fig)

# Fig 4: L_ax by state ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 4.2))
c3o = sorted(c3stat, key=lambda i: dv[int(i)]["enr"]); xx = np.arange(len(c3o))
for j, (st, lab) in enumerate([("ARO", "ARO"), ("RE12", "RE1+RE2 (8)"), ("ARI", "ALL-RE (16)")]):
    ax.bar(xx + (j - 1) * 0.27, [c3stat[i][f"{st}_Lax"] for i in c3o], 0.27, label=lab)
ax.axhline(1.0289, color="k", ls="--", lw=1, label="campaign $L_{ax}$ = 1.0289 (water-slab study)")
ax.set_ylim(1.010, 1.034)
ax.set_xticks(xx); ax.set_xticklabels([f"d{i}\n{dv[int(i)]['enr']:.1f}%" for i in c3o], fontsize=8)
ax.set_ylabel("$L_{ax,hw} = k_{2D} / k_{3D,hw}$")
ax.set_title("2D-to-3D factor by rod state (d53 ARO corrected, seed 1 only)\nRodded values include the rod-stack departure, not only axial leakage")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig4_lax_by_state.png"); plt.close(fig)

# Fig 5: F 2D vs 3D per state --------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for j, st in enumerate(["ARO", "RE12", "ARI"]):
    for k, idx in enumerate(c3o):
        r = c3stat[idx]; xpos = j * (len(c3o) + 1) + k
        ax.plot([xpos, xpos], [r[f"{st}_2D"]["F"], r[f"{st}_3Dhw"]["F"]], color="gray", lw=1.2, zorder=1)
        ax.scatter(xpos, r[f"{st}_2D"]["F"], color="#4c72b0", zorder=3, s=28, label="2D" if (j, k) == (0, 0) else None)
        ax.scatter(xpos, r[f"{st}_3Dhw"]["F"], color="#dd8452", zorder=3, s=28, marker="s", label="3D hardware" if (j, k) == (0, 0) else None)
        ax.text(xpos, r[f"{st}_2D"]["F"] + 0.035, f"d{idx}", ha="center", fontsize=7)
ax.axhline(2.0, color="red", ls="--", lw=1, label="screening bound 2.0")
ax.set_xticks([j * (len(c3o) + 1) + (len(c3o) - 1) / 2 for j in range(3)]); ax.set_xticklabels(["ARO", "RE1+RE2", "ALL-RE"])
ax.set_ylabel("$F_{\\Delta H}$ (mean of 2 seeds)")
ax.set_title(f"3D lowers peaking in every state; all five pass 2.0 under RE1+RE2 in 3D\n(1σ ≈ {sigF['ARO']:.3f} ARO, {sigF['rodded']:.3f} rodded)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig5_fdh_2d_vs_3d.png"); plt.close(fig)
print("\nfigures ->", sorted(p.name for p in OUT.glob("*.png")))
