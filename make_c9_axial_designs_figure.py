#!/usr/bin/env python3
"""make_c9_axial_designs_figure.py -- Figure 5.19, the seven designs depleted in 3D.

For each design the archived assembly cycle length, the uniform 3D value where the
one-layer run exists, and the eight-layer value, with the arrow from the archive to
the eight-layer value and the margin over the mission floor.

usage (repository root):
    python make_c9_axial_designs_figure.py OUT.pdf [OUT.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
FLOOR, SIGMA = 1826.0, 12.0          # d; statistical error of one layered run

RUNS = {                             # design: (one-layer run, eight-layer run)
    1:  ("c9_dep_core3d_d1_L1", "c9_dep_core3d_d1_L8"),
    27: ("c9_dep_core3d_d27_L1", "c9_dep_core3d_d27_L8"),
    24: (None, "c9_dep_core3d_d24_L8"),
    16: (None, "c9_dep_core3d_d16_L8"),
    11: (None, "c9_dep_core3d_d11_L8"),
    35: ("c9_dep_core3d_d35_L1", "c9_dep_core3d_d35_L8"),
    47: ("c9_dep_core3d_L1", "c9_dep_core3d"),
    32: (None, "c9_dep_core3d_d32_L8"),
}

raw = json.load(open("out_c9/optimization_checkpoint.json"))["all_raw"]


def efpd(folder, i):
    """Cycle length of design i from its run record, or from the summary when
    the record is not archived."""
    p = Path(folder) / "runs.json"
    if p.exists():
        return json.load(open(p))[f"d{i}"]["efpd"]
    return json.load(open(Path(folder) / "summary.json"))[str(i)]["efpd_3d"]


D = []
for i, (one, eight) in RUNS.items():
    D.append(dict(i=i, arch=raw[i]["cycle_length"], one=efpd(one, i) if one else None,
                  eight=efpd(eight, i), e=raw[i]["enrich"], gd=raw[i]["gd_wt"]))
D.sort(key=lambda d: -d["eight"])

C_ARCH, C_ONE, C_OK, C_NOISE, C_NO = "#D55E00", "#009E73", "#0072B2", "#8C8C8C", "#222222"
fig, ax = plt.subplots(figsize=(9.2, 4.2))

ax.axhline(FLOOR, color="#CC79A7", ls="--", lw=1.1, zorder=1)
ax.text(-0.42, FLOOR - 42, "Mission floor 1826 d", color="#CC79A7", fontsize=8, va="top")

seen = set()
for x, d in enumerate(D):
    m = d["eight"] - FLOOR
    colour, lab = ((C_OK, "3D, 8 layers, meets the floor") if m > 2 * SIGMA else
                   (C_NOISE, "3D, 8 layers, within noise") if m > -2 * SIGMA else
                   (C_NO, "3D, 8 layers, misses the floor"))
    ax.annotate("", xy=(x, d["eight"]), xytext=(x, d["arch"]),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.0, shrinkA=4, shrinkB=4))
    ax.plot(x, d["arch"], "D", mfc="none", mec=C_ARCH, ms=8, mew=1.4, zorder=3,
            label=None if "a" in seen else "Assembly depletion (campaign)")
    if d["one"] is not None:
        ax.plot(x + 0.22, d["one"], "o", mfc="none", mec=C_ONE, ms=7, mew=1.4, zorder=3,
                label=None if "o" in seen else "3D, uniform axial burnup (1 layer)")
        seen.add("o")
    ax.plot(x, d["eight"], "o", color=colour, ms=7, zorder=3,
            label=None if lab in seen else lab)
    seen.update({"a", lab})
    ax.annotate(f"{m:+.0f} d", xy=(x + 0.10, 0.5 * (d["arch"] + d["eight"])),
                fontsize=8, color="0.25", va="center")

ax.set_xticks(range(len(D)))
ax.set_xticklabels([f"C9-{d['i']}\n{d['e']:.2f} / {d['gd']:.2f}" for d in D], fontsize=8.5)
ax.set_xlabel(r"Design, with enrichment / gadolinia [wt\%]".replace("\\", ""))
ax.set_ylabel("Cycle length [d]")
ax.set_xlim(-0.5, len(D) - 0.3)
ax.set_ylim(1500, 3000)
ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.95)
ax.grid(alpha=0.25, lw=0.6, axis="y")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight")

for d in D:
    print(f"C9-{d['i']:<3} archive {d['arch']:6.0f}  1 layer "
          f"{('%6.0f' % d['one']) if d['one'] else '    --'}  8 layers {d['eight']:6.0f}  "
          f"margin {d['eight'] - FLOOR:+5.0f}")
print("wrote", OUT)
