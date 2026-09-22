"""Source-convergence figure: Shannon entropy per batch for a reflective assembly,
a typical core solve and the slowest core solve. Plot only, runs locally.

usage: python make_entropy_figure.py entropy_traces.json OUT.pdf OUT.png
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2], sys.argv[3]
chosen = json.load(open(SRC))["chosen"]
PANELS = [("assembly", "Reflective assembly"),
          ("core_typical", "Core, typical case"),
          ("core_slowest", "Core, slowest case")]

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
for ax, (key, title) in zip(axes, PANELS):
    r = chosen[key]
    H = np.array(r["entropy"])
    b = np.arange(1, H.size + 1)
    ax.plot(b, H, color="#3f6d8c", lw=1.0, label="Shannon entropy")
    ax.axvline(r["n_inactive"], color="black", ls="--", lw=1.0,
               label=f"End of inactive batches ({r['n_inactive']})")
    ax.axvline(r["conv"], color="#B23A48", ls=":", lw=1.6,
               label=f"Stationary source from batch {r['conv']}")
    ax.set_title(title)
    ax.set_xlabel("Batch")
    ax.legend(fontsize=7.5, loc="lower right")
axes[0].set_ylabel("Shannon entropy (bits)")
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
print({k: chosen[k]["conv"] for k, _ in PANELS})
