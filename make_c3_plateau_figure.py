#!/usr/bin/env python3
"""make_c3_plateau_figure.py -- Campaign 3 plateau figure for Section 5.2.2.

Stored against resolved peaking for the nine Campaign 3 candidates that were
re-solved at higher statistics. No transport: the figure reads the ranking runs
and the campaign archive.

  stored    the single-seed value of the archive, at 16000 particles
  measured  the mean of eight seeds at 256000 particles, with its 95 % interval
  adjusted  the mean of five seeds at 64000 particles, less the mean bias of
            that setting against 256000 measured on the five candidates that
            were run at both

Candidate indices of the ranking study are mapped to the design names of the
thesis by the stored peaking factor and the cycle length together.

usage (plot only, runs anywhere):
    python make_c3_plateau_figure.py RANK_RUNS.json CHECKPOINT.json OUT.pdf [OUT.png]
"""
import json
import re
import statistics as st
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
runs = json.load(open(RUNS))
arch = json.load(open(CKPT))["all_raw"]

# candidate index -> archive index of the design
CAND = {0: 51, 1: 18, 2: 49, 3: 46, 4: 52, 5: 53, 6: 50, 7: 45, 8: 43}

by = {}
for k, v in runs.items():
    m = re.match(r"c(\d+)_p(\d+)_s(\d+)", k)
    by.setdefault((int(m.group(1)), int(m.group(2))), []).append(v["fdh"])

# the bias of the 64000 setting, measured where both settings exist
bias = [st.mean(by[(c, 64000)]) - st.mean(by[(c, 256000)])
        for c in CAND if (c, 256000) in by and (c, 64000) in by]
b_mean, b_sd = st.mean(bias), st.stdev(bias)

rows = []
for c, idx in CAND.items():
    r = arch[idx]
    hi = by.get((c, 256000))
    lo = by.get((c, 64000))
    if hi:
        val = st.mean(hi)
        err = 1.96 * st.stdev(hi) / np.sqrt(len(hi))
        kind = "measured"
    else:
        val = st.mean(lo) - b_mean
        err = 1.96 * np.hypot(st.stdev(lo) / np.sqrt(len(lo)), b_sd)
        kind = "adjusted"
    rows.append({"name": f"C3-{idx}", "stored": r["peaking"], "val": val,
                 "err": err, "kind": kind, "efpd": r.get("cycle_length", 0)})

rows.sort(key=lambda d: d["val"])
x = np.arange(len(rows))

C_ST, C_ME, C_AD, C_BAND = "0.55", "#1f3b57", "#a03236", "#9ecfb0"
fig, ax = plt.subplots(figsize=(8.4, 4.4))

# the plateau: the designs whose measured values are not separated
plate = [d for d in rows if d["kind"] == "measured" and d["val"] < 1.1250]
if plate:
    ax.axhspan(min(d["val"] - d["err"] for d in plate),
               max(d["val"] + d["err"] for d in plate),
               color=C_BAND, alpha=0.35, zorder=0)
    ax.text(len(rows) - 0.4, max(d["val"] + d["err"] for d in plate) + 0.0004,
            f"Plateau, {len(plate)} designs", ha="right", va="bottom",
            fontsize=8.6, color="#2f6b4f")

for xi, d in zip(x, rows):
    ax.plot([xi, xi], [d["stored"], d["val"]], color=C_ST, lw=0.9, zorder=1)
    ax.plot(xi, d["stored"], "o", ms=6, mfc="white", mec=C_ST, mew=1.2, zorder=3)
    col = C_ME if d["kind"] == "measured" else C_AD
    ax.errorbar(xi, d["val"], yerr=d["err"], fmt="s", ms=6, color=col,
                capsize=3, lw=1.3, zorder=4)
    ax.annotate(f"{d['efpd']:.0f}", xy=(xi, d["val"] - d["err"]),
                xytext=(0, -9), textcoords="offset points",
                ha="center", fontsize=7.6, color="0.35")

ax.plot([], [], "o", ms=6, mfc="white", mec=C_ST, mew=1.2,
        label="Stored, 16000 particles, one seed")
ax.plot([], [], "s", ms=6, color=C_ME, label="Measured, 256000 particles, eight seeds")
ax.plot([], [], "s", ms=6, color=C_AD, label="Measured at 64000 and bias-adjusted")
ax.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

ax.set_xticks(x)
ax.set_xticklabels([d["name"] for d in rows], fontsize=9)
ax.set_xlabel("Campaign 3 candidate, ordered by resolved peaking factor")
ax.set_ylabel(r"Assembly $F_{\Delta H}$")
ax.set_xlim(-0.6, len(rows) - 0.4)
ax.grid(alpha=0.25, lw=0.6, axis="y")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
if len(sys.argv) > 4:
    fig.savefig(sys.argv[4], dpi=200, bbox_inches="tight")
print("wrote", OUT, "| bias 64k", f"{b_mean:+.5f}", "sd", f"{b_sd:.5f}")
for d in rows:
    print(f"  {d['name']:>6} stored {d['stored']:.4f}  {d['kind']:>8} "
          f"{d['val']:.4f} +- {d['err']:.4f}  {d['efpd']:.0f} EFPD")
