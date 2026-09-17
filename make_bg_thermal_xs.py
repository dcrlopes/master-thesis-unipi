#!/usr/bin/env python3
"""
make_bg_thermal_xs.py -- absorption cross sections for Figure fig:bg-xs of the
Theoretical Background chapter, read from the ENDF/B-VII.1 HDF5 library used by
every transport solve of the thesis (Chadwick et al., Nucl. Data Sheets 112, 2011).

Absorption is the sum of fission (MT 18) and the neutron-disappearance
reactions MT 102-107, evaluated pointwise at 294 K. The 2200 m/s value is the
cross section at 0.0253 eV.

Step 1 (OpenMC environment, WSL):  python make_bg_thermal_xs.py extract <out.npz>
Step 2 (numpy + matplotlib only):   python make_bg_thermal_xs.py plot <in.npz> <out.pdf> <out.png>
"""
import sys

import numpy as np

NUCLIDES = ["U238", "U235", "B10", "Gd155", "Gd157", "Xe135"]
LABELS = {"U238": "U-238", "U235": "U-235", "B10": "B-10",
          "Gd155": "Gd-155", "Gd157": "Gd-157", "Xe135": "Xe-135"}
MTS = [18, 102, 103, 104, 105, 106, 107]
E_TH = 0.0253


def extract(out):
    import os
    from pathlib import Path
    import openmc.data
    lib = Path(os.environ["OPENMC_CROSS_SECTIONS"]).parent / "neutron"
    E = np.logspace(-3, 2, 6000)
    data = {"E": E}
    for n in NUCLIDES:
        nuc = openmc.data.IncidentNeutron.from_hdf5(lib / f"{n}.h5")
        T = "294K"
        sig = np.zeros_like(E)
        sig_th = 0.0
        for mt in MTS:
            if mt in nuc.reactions and T in nuc.reactions[mt].xs:
                xs = nuc.reactions[mt].xs[T]
                sig += xs(E)
                sig_th += float(xs(E_TH))
        data[n] = sig
        data[n + "_th"] = sig_th
        print(f"{n:6s} sigma_a(0.0253 eV, 294 K) = {sig_th:12.2f} b")
    np.savez(out, **data)


def plot(src, pdf, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(src)
    colors = {"U238": "#7f8c99", "U235": "#3a78b8", "B10": "#6d4c9f",
              "Gd155": "#c9920e", "Gd157": "#8a5a00", "Xe135": "#444444"}
    styles = {"Xe135": "--"}
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for n in NUCLIDES:
        th = float(d[n + "_th"])
        # three significant figures, thin-space thousands separator
        val = (f"{float(f'{th:.3g}'):,.0f}".replace(",", " ")
               if th >= 100 else f"{th:.2f}")
        ax.loglog(d["E"], d[n], styles.get(n, "-"), color=colors[n], lw=1.4,
                  label=f"{LABELS[n]}: {val} b")
    ax.axvline(E_TH, color="#B23A48", lw=1.0, ls=":")
    ax.text(E_TH * 1.1, 2.0e-1, "0.0253 eV\n(2200 m/s)", color="#B23A48", fontsize=8, va="bottom")
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-1, 1e7)
    ax.set_xlabel("Neutron energy [eV]")
    ax.set_ylabel(r"Absorption cross section $\sigma_a$ [b]")
    ax.grid(alpha=0.25, lw=0.5, which="major")
    leg = ax.legend(title="Value at 0.0253 eV", fontsize=7.8, title_fontsize=8,
                    loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=3)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    print("written", pdf)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        plot(*sys.argv[2:5])
