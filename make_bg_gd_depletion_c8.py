"""
make_bg_gd_depletion_c8.py -- life curves for Figure fig:bg-dep of the
Theoretical Background chapter, from measured campaign depletion histories.

Data (reflective 17x17 assembly, campaign depletion, 1000 ppm):
  khist_c8/khist.json      Campaign 8 histories (c8_khist_hump.py)
  kh_c9/k_histories.json   Campaign 9 histories recovered from out_c9.log
                           (c7_khist_from_log.py, 60 of 60 cases validated)
Designs:
  C8-47  12 poisoned pins at 2.88 wt% Gd2O3, falls from beginning of life
  C8-31  40 poisoned pins at 3.38 wt% Gd2O3, strong burnout rise
  C9-4   32 poisoned pins at 4.05 wt% Gd2O3, 12.92 wt% enrichment, the
         longest cycle in Campaigns 8 and 9 below the discharge limit
The end-of-cycle burnup and target of each design are the archive values.
The top axis converts burnup to effective full-power days with the specific
power of the archive (EFPD = 1000 B / P_spec).

Usage: python make_bg_gd_depletion_c8.py <repo root> <out.pdf> <out.png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1])
OUT_PDF = Path(sys.argv[2])
OUT_PNG = Path(sys.argv[3])
B_UO2 = 55.0        # UO2 in Zr cladding, ASSEMBLY average (Song and Sanchez 2026);
                    # the curves are assembly-average burnup, so this is the comparable basis

c8 = json.loads((ROOT / "khist_c8" / "khist.json").read_text())
c9 = json.loads((ROOT / "kh_c9" / "k_histories.json").read_text())
c9_raw = json.loads((ROOT / "out_c9" / "optimization_checkpoint.json").read_text())["all_raw"]

p_spec = c8["47"]["analysis"]["specific_power_w_per_g"]


def from_c8(did):
    d = c8[did]
    return dict(bu=d["curve"]["bu"], k=d["curve"]["k"], bc=d["analysis"]["crossing_bu_mwd_kg"],
                kt=d["analysis"]["k_target"], rec=d["record"])


def from_c9(idx):
    r = c9_raw[idx]
    assert c9[str(idx)]["status"] == "OK"
    return dict(bu=c9[str(idx)]["bu"], k=c9[str(idx)]["k"], bc=r["bu_eoc_mwd_kg"],
                kt=r["k_target"], rec=r)


designs = [("C8-47", from_c8("47"), "#173a5e", "o"),
           ("C8-31", from_c8("31"), "#B5651D", "s"),
           ("C9-4", from_c9(4), "#6A3D9A", "^")]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.8, 4.8))

for name, d, color, marker in designs:
    r = d["rec"]
    label = (f"{name}: {r['enrich']:.2f} wt% U-235, {r['gd_pins_used']} pins "
             f"at {r['gd_wt']:.2f} wt% Gd$_2$O$_3$")
    ax.plot(d["bu"], d["k"], "-", color=color, lw=1.8, marker=marker, ms=4.2, label=label)
    ax.plot([d["bc"], d["bc"]], [0.98, d["kt"]], ":", color=color, lw=1.2)
    print(f"{name}: k_target {d['kt']:.4f}  EOC {d['bc']:.1f} MWd/kgHM = "
          f"{1000 * d['bc'] / p_spec:.0f} EFPD")

k_t = sum(d["kt"] for _, d, _, _ in designs) / len(designs)
ax.axhline(k_t, color="#2E6F4E", lw=1.3)
ax.text(39.0, k_t - 0.006, "Leakage-corrected target $\\approx$ "
        f"{k_t:.3f}", color="#2E6F4E", fontsize=8.3, va="top")

# end-of-cycle labels below the target line; C9-4 sits left of its line
for name, d, color, _ in designs:
    efpd = 1000.0 * d["bc"] / p_spec
    left = name == "C9-4"
    ax.text(d["bc"] + (-0.8 if left else 0.8), 1.000,
            f"EOC {name}\n{d['bc']:.1f} MWd/kgHM\n{efpd:.0f} EFPD",
            color=color, fontsize=7.6, va="bottom", ha="right" if left else "left")

d31 = designs[1][1]
i_pk = max(range(len(d31["k"])), key=lambda i: d31["k"][i])
ax.annotate(f"Gadolinia burnout peak\nat {d31['bu'][i_pk]:.1f} MWd/kgHM",
            xy=(d31["bu"][i_pk], d31["k"][i_pk]), xytext=(4.0, 1.225),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#B5651D"),
            fontsize=8.0, color="#B5651D")



ax.set_xlabel("Burnup $B$ [MWd/kgHM]")
ax.set_ylabel(r"Assembly $k_\infty$")
ax.set_xlim(0, 80)
ax.set_ylim(0.98, 1.36)
ax.grid(alpha=0.25, lw=0.5)
ax.axvline(B_UO2, color="#B23A48", lw=1.5,
           label="UO$_2$ in Zr cladding, 55 MWd/kgHM")

ax.legend(fontsize=7.8, loc="upper right", frameon=True, bbox_to_anchor=(0.9, 1.0))

top = ax.secondary_xaxis("top", functions=(lambda b: 1000.0 * b / p_spec,
                                           lambda e: e * p_spec / 1000.0))
top.set_xlabel("Effective full-power days")

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print("written", OUT_PDF)
