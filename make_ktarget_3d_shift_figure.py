"""Effect of the axial correction on the end-of-cycle target (Campaign 8).

(a) 2D and 3D leakage-corrected targets against reflector thickness.
(b) k_inf history of one Campaign 8 design with both targets and crossings.
(c) Cycle-length shift of every Campaign 8 design with a resolved crossing.

Data: ktarget_table_c8.json, khist_c8/khist.json.  Plot only, runs locally.
usage: python make_ktarget_3d_shift_figure.py OUT.pdf OUT.png
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2]
L_AX = 1.0289
EXAMPLE = "29"

tab = json.load(open("ktarget_table_c8.json"))
kh = json.load(open("khist_c8/khist.json"))


def shift(v):
    """Target shift (pcm of k), local slope (pcm per MWd/kgHM), dB, dEFPD."""
    a, c = v["analysis"], v["curve"]
    bu, k = np.array(c["bu"]), np.array(c["k"])
    kt3 = a["k_target"]
    kt2 = kt3 / L_AX
    bx = a["crossing_bu_mwd_kg"]
    i = min(max(np.searchsorted(bu, bx) - 1, 0), len(bu) - 2)
    slope = (k[i] - k[i + 1]) / (bu[i + 1] - bu[i])
    db = (kt3 - kt2) / slope
    return kt3, kt2, slope, bx, db, db / a["specific_power_w_per_g"] * 1000.0


rows = {n: shift(v) for n, v in kh.items() if v["analysis"]["resolved"]}
d_efpd = np.array([r[5] for r in rows.values()])
assert len(rows) == 10 and 400 < d_efpd.min() and d_efpd.max() < 560

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 4.0))

# (a) targets
t = np.array(tab["refl_thick_cm"])
k2, k3 = np.array(tab["k_target_2d_fit"]), np.array(tab["k_target"])
ax1.plot(t, k3, "o-", color="#B23A48", label="Finite-height target, 3D")
ax1.plot(t, k2, "s-", color="#3f6d8c", label="Two-dimensional target, 2D")
ax1.annotate("", xy=(2.6, k3[1]), xytext=(2.6, k2[1]),
             arrowprops=dict(arrowstyle="<->", color="black"))
ax1.text(2.7, (k2[1] + k3[1]) / 2, f"+{(k3[1] - k2[1]) * 1e5:.0f} pcm\n(factor {L_AX})",
         va="center", fontsize=8.5)
ax1.set_xlabel("Reflector thickness (cm)")
ax1.set_ylabel("End-of-cycle target $k_\\mathrm{target}$")
ax1.set_title("(a) Targets at pitch 1.26 cm")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_ylim(1.045, 1.100)

# (b) one history
v = kh[EXAMPLE]
bu, k = np.array(v["curve"]["bu"]), np.array(v["curve"]["k"])
kt3, kt2, slope, bx, db, de = rows[EXAMPLE]
ax2.plot(bu, k, "o-", color="#4d4d4d", label=f"Assembly $k_\\infty$, design C8-{EXAMPLE}")
ax2.plot([bu[-1], bx + db], [k[-1], kt2], ":", color="#4d4d4d",
         label="Linear extension at the end-of-cycle slope")
ax2.axhline(kt3, color="#B23A48", lw=1.2, label="Finite-height target")
ax2.axhline(kt2, color="#3f6d8c", lw=1.2, label="Two-dimensional target")
ax2.axvline(bx, color="#B23A48", ls="--", lw=0.9)
ax2.axvline(bx + db, color="#3f6d8c", ls="--", lw=0.9)
ax2.annotate("", xy=(bx, 1.04), xytext=(bx + db, 1.04),
             arrowprops=dict(arrowstyle="<->"))
ax2.text(bx + db / 2, 1.043, f"{db:.1f} MWd/kgHM\n{de:.0f} EFPD", ha="center", fontsize=8.5)
ax2.set_xlim(bx - 12, bx + db + 4)
ax2.set_ylim(1.03, 1.14)
ax2.set_xlabel("Burnup (MWd/kgHM)")
ax2.set_ylabel("$k_\\infty$")
ax2.set_title("(b) End of cycle on one design")
ax2.legend(fontsize=7.5, loc="upper right")

# (c) all resolved designs
order = sorted(rows, key=lambda n: rows[n][3])
ax3.bar(range(len(order)), [rows[n][5] for n in order], color="#5f9a7f")
ax3.axhline(np.median(d_efpd), color="black", ls="--", lw=1,
            label=f"Median {np.median(d_efpd):.0f} EFPD")
ax3.set_xticks(range(len(order)))
ax3.set_xticklabels([f"C8-{n}" for n in order], rotation=60, fontsize=8)
ax3.set_ylabel("Cycle length removed by the correction (EFPD)")
ax3.set_title("(c) Campaign 8 designs, by end-of-cycle burnup")
ax3.legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
for n in order:
    kt3, kt2, s, bx, db, de = rows[n]
    print(f"C8-{n}: slope {s * 1e5:.0f} pcm/(MWd/kgHM)  dB {db:.2f}  dEFPD {de:.0f}")
print("median dEFPD", round(float(np.median(d_efpd))),
      "median slope", round(float(np.median([r[2] for r in rows.values()])) * 1e5))
