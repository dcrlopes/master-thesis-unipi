"""Source-convergence figure: Shannon entropy per batch for a reflective assembly,
a typical core solve and the slowest core solve. Plot only, runs locally.

usage: python make_entropy_figure.py entropy_traces.json OUT.pdf OUT.png [HIST.pdf HIST.png]

With the two optional paths it also writes a second figure: the convergence
batch of every core solve in the archive, divided by its inactive-batch count
(a value at or below 1 means the source was stationary before tallying began).
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT_PDF, OUT_PNG = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(SRC))
chosen = data["chosen"]
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

if len(sys.argv) == 6:
    core = [r for r in data["summary"] if r["kind"] == "core"]
    ratio = np.array([r["conv"] / r["n_inactive"] for r in core])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(ratio, bins=np.linspace(0, max(1.2, ratio.max() * 1.05), 25),
            color="#3f6d8c", edgecolor="white")
    ax.axvline(1.0, color="black", ls="--", lw=1.0, label="End of inactive batches")
    ax.set_xlabel("Convergence batch / number of inactive batches")
    ax.set_ylabel("Core solves")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(sys.argv[4], bbox_inches="tight")
    fig.savefig(sys.argv[5], dpi=170, bbox_inches="tight")
    print(f"core solves {ratio.size}, ratio median {np.median(ratio):.2f}, "
          f"max {ratio.max():.2f}, above 1: {(ratio > 1).sum()}")
