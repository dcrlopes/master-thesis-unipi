"""
plot_bg_pin_power_c9.py -- Figures fig:bg-pin-asm and fig:bg-pin-core from the
maps written by make_bg_pin_power_c9.py (C9-47, fresh fuel, rods out, 1000 ppm).

The mesh tally stores bins with y increasing upwards; the lattice masks use
(row from top, column from left). The maps are flipped to lattice order and
the flip is checked against the guide-tube positions (zero fission).

Usage: python plot_bg_pin_power_c9.py <dir with pinmaps_c9.npz> <images dir> <png dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

SRC, IMG, PNG = (Path(p) for p in sys.argv[1:4])
d = np.load(SRC / "pinmaps_c9.npz")
s = json.loads((SRC / "pinmaps_c9.json").read_text())
N = 17

asm = np.flipud(d["asm"])
core = np.flipud(d["core"])
gd, gt = d["gd"], d["gt"]
assert np.all(asm[gt] == 0.0) and np.all(asm[~gt] > 0.0), "assembly orientation check failed"

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
CMAP = plt.get_cmap("jet").copy()      # blue (low) to red (high)
CMAP.set_bad("#bdbdbd")

# ---- assembly --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(np.ma.masked_equal(asm, 0.0), cmap=CMAP, origin="upper")
vmin, vmax = float(asm[asm > 0].min()), float(asm.max())
for r in range(N):
    for c in range(N):
        if gt[r, c]:
            continue
        v = asm[r, c]
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=4.6,
                # white on the dark ends of the jet scale, black in between
                color="white" if not (0.2 <= (v - vmin) / (vmax - vmin) <= 0.8) else "black")
        if gd[r, c]:
            ax.add_patch(Circle((c, r), 0.47, fill=False, ec="white", lw=1.4))
rmax, cmax = np.unravel_index(np.argmax(asm), asm.shape)
ax.add_patch(Rectangle((cmax - 0.5, rmax - 0.5), 1, 1, fill=False, ec="black", lw=2.0))
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Relative pin power")
ax.set_title(f"Assembly, reflective boundaries: $F_{{\\Delta H}}$ = {s['fdh_asm']:.3f}", fontsize=9)
fig.savefig(IMG / "bg_pin_power_asm.pdf", bbox_inches="tight")
fig.savefig(PNG / "bg_pin_power_asm.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---- core ------------------------------------------------------------------
ny, nx = core.shape
fig, ax = plt.subplots(figsize=(6.0, 5.2))
im = ax.imshow(np.ma.masked_equal(core, 0.0), cmap=CMAP, origin="upper")
cm = np.flipud(d["rmap"]) if False else d["rmap"]        # ring map is already row-from-top
na = nx // N
for i in range(na + 1):
    ax.axhline(i * N - 0.5, color="white", lw=0.5)
    ax.axvline(i * N - 0.5, color="white", lw=0.5)
rmax, cmax = np.unravel_index(np.argmax(core), core.shape)
ax.add_patch(Circle((cmax, rmax), 2.2, fill=False, ec="black", lw=1.8))
ax.annotate(f"Hottest pin, {core[rmax, cmax]:.3f}", xy=(cmax + 1.6, rmax - 1.6),
            xytext=(nx - 1, 8), ha="right", va="center", color="black", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Relative pin power")
ax.set_title(f"Core, 32 assemblies: $F_{{\\Delta H}}$ = {s['fdh_core']:.3f}", fontsize=9)
fig.savefig(IMG / "bg_pin_power_core.pdf", bbox_inches="tight")
fig.savefig(PNG / "bg_pin_power_core.png", dpi=170, bbox_inches="tight")
plt.close(fig)

print(json.dumps({k: s[k] for k in ("k_asm", "k_asm_sd", "fdh_asm", "k_core", "k_core_sd",
                                   "fdh_core", "archive_peaking", "archive_peaking_asm",
                                   "gd_pins_used")}, indent=1))
print("hottest core pin (row, col from top-left):", (int(rmax), int(cmax)),
      "assembly", (int(rmax) // N, int(cmax) // N))
