"""
make_bg_k_hierarchy_c9.py -- multiplication ladder of the Campaign 9 champion,
C9-47, for Figure fig:bg-k of the Theoretical Background chapter.

All values are read from the Campaign 9 post-analysis archive, at the same
state: beginning of life, 1000 ppm soluble boron, all rods out.

  assembly k_inf   kt_burnup_c9/runs.json       d47_s0_seed1, d47_s0_seed2
                   reflective 17x17 assembly, 10 000 x 120 (30 inactive)
  2D core k_eff    confirm3d_c9_front/runs.json 47|ARO|2D|seed 0, 1
  3D core k_eff    confirm3d_c9_front/runs.json 47|ARO|3Dhw|seed 0, 1
                   both 150 000 x 200 (80 inactive)

The two seeds of each level are averaged and their standard deviations
combined as sqrt(s1^2 + s2^2) / 2. Reactivity steps use rho = (k - 1) / k.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(sys.argv[1])          # master-thesis-unipi
OUT_PDF = Path(sys.argv[2])       # thesis/images/bg_k_hierarchy.pdf
OUT_PNG = Path(sys.argv[3])       # preview png

kt = json.loads((ROOT / "kt_burnup_c9" / "runs.json").read_text())
cf = json.loads((ROOT / "confirm3d_c9_front" / "runs.json").read_text())


def combine(pairs):
    ks = [p[0] for p in pairs]
    sds = [p[1] for p in pairs]
    k = sum(ks) / len(ks)
    sd = math.sqrt(sum(s * s for s in sds)) / len(sds)
    return k, sd


asm = combine([(kt[f"d47_s0_seed{s}"]["kinf"], kt[f"d47_s0_seed{s}"]["kinf_sd"]) for s in (1, 2)])
key = "47|ARO|{geo}|{seed}|None|None|None|benchmark|False|150000|200|80|1000.0"
c2d = combine([(cf[key.format(geo="2D", seed=s)]["keff"], cf[key.format(geo="2D", seed=s)]["sd"]) for s in (0, 1)])
c3d = combine([(cf[key.format(geo="3Dhw", seed=s)]["keff"], cf[key.format(geo="3Dhw", seed=s)]["sd"]) for s in (0, 1)])


def rho_pcm(k):
    return (k - 1.0) / k * 1e5


step1 = rho_pcm(c2d[0]) - rho_pcm(asm[0])
step2 = rho_pcm(c3d[0]) - rho_pcm(c2d[0])

for name, (k, sd) in (("assembly k_inf", asm), ("2D core k_eff", c2d), ("3D core k_eff", c3d)):
    print(f"{name:15s} {k:.5f} +/- {sd:.5f}   rho = {rho_pcm(k):8.0f} pcm")
print(f"assembly -> 2D core: {step1:.0f} pcm   2D -> 3D: {step2:.0f} pcm")

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10.6, 3.2))

levels = [
    ("17 × 17 assembly", "Reflective boundaries", r"$k_\infty$", asm,
     "10 000 particles × 120 batches", "30 inactive, 2 seeds"),
    ("2D core, 32 assemblies", "Reflector, vacuum boundary", r"$k_\mathrm{eff}$", c2d,
     "150 000 particles × 200 batches", "80 inactive, 2 seeds"),
    ("3D core, finite height", "With structural components", r"$k_\mathrm{eff}$", c3d,
     "150 000 particles × 200 batches", "80 inactive, 2 seeds"),
]
W, H, GAP, X0, Y0 = 3.3, 2.4, 1.45, 0.15, 0.35
xs = [X0 + i * (W + GAP) for i in range(3)]

for (title, sub, sym, (k, sd), set1, set2), x in zip(levels, xs):
    ax.add_patch(FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0.04",
                                fc="#eef3f8", ec="#33506b", lw=1.2))
    cx = x + W / 2
    ax.text(cx, Y0 + H - 0.32, title, ha="center", va="center", fontsize=9.5, weight="bold",
            color="#173a5e")
    ax.text(cx, Y0 + H - 0.66, sub, ha="center", va="center", fontsize=8.5, color="#33506b")
    ax.text(cx, Y0 + 1.12, f"{sym} = {k:.5f} $\\pm$ {sd:.5f}", ha="center", va="center",
            fontsize=9.8, color="#173a5e")
    ax.text(cx, Y0 + 0.58, set1, ha="center", va="center", fontsize=7.6, color="#444444")
    ax.text(cx, Y0 + 0.32, set2, ha="center", va="center", fontsize=7.6, color="#444444")

arrow_labels = [("Radial", "leakage", step1), ("Axial leakage", "+ structures", step2)]
for i, (l1, l2, step) in enumerate(arrow_labels):
    x_start = xs[i] + W + 0.18
    x_end = xs[i + 1] - 0.18
    y = Y0 + H / 2
    ax.add_patch(FancyArrowPatch((x_start, y), (x_end, y), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, color="#7a2f2f"))
    xm = (x_start + x_end) / 2
    ax.text(xm, y + 0.55, l1, ha="center", va="center", fontsize=7.5, color="#7a2f2f")
    ax.text(xm, y + 0.32, l2, ha="center", va="center", fontsize=7.5, color="#7a2f2f")
    ax.text(xm, y - 0.30, f"{step:,.0f}".replace(",", " ").replace("-", "−"), ha="center", va="center",
            fontsize=8.2, weight="bold", color="#7a2f2f")
    ax.text(xm, y - 0.52, "pcm", ha="center", va="center", fontsize=7.5, color="#7a2f2f")

ax.set_xlim(0, xs[-1] + W + 0.15)
ax.set_ylim(0.15, Y0 + H + 0.2)
ax.axis("off")
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print("written", OUT_PDF)
