#!/usr/bin/env python3
r"""
c9_axial_correction.py -- the burnup-dependent axial correction, derived from
the paired 3D depletions, validated, written for use, and plotted.

WHAT IT DERIVES
    For each design depleted both with one axial layer (uniform burnup) and
    with eight (resolved burnup), at the same burnup states,

        A(B) = [k_1(B) / k_1(0)] / [k_8(B) / k_8(0)]

    is the reactivity the resolved core loses beyond the uniform one. The
    correction table is its mean over the paired designs, in pcm,
    rho_A(B) = 1e5 ln A(B). It is held at zero up to 3.5 MWd/kgHM, where the
    measured values scatter around zero within the transport noise, and held
    at its last value beyond the data.

    Applied like the axial leakage factor of the k-target table:

        k_target(t_refl, B) = k_target(t_refl) x exp(rho_A(B) / 1e5)
        L_ax,eff(B)         = L_ax x exp(rho_A(B) / 1e5)

VALIDATION
    Leave one design out: the table of the other designs converts the left-out
    design's uniform history into a predicted layered history, whose crossing
    of the relative end-of-cycle target is compared with the measured layered
    cycle length.

OUTPUTS (in --out)
    axial_correction.json   the table, L_ax,eff(B), per-design values, validation
    c9_axial_correction.pdf/.png   (a) k histories of C9-47, uniform and layered
                                   (b) rho_A(B) per design and the table,
                                       right axis L_ax,eff(B)
    c9_axial_peaking.pdf/.png      C9-47: (a) F_z against burnup for 1, 4, 8
                                   and 12 layers (b) F_dH, 1 and 12 layers
                                   (c) axial power profile, 12 layers

USAGE (repository root, reads c9_dep_core3d*/runs.json)
    python c9_axial_correction.py --out figs_c9_axial
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPEC = 9.9834                  # W/gHM, core specific power used by every depletion
L_AX = 1.0289                  # axial leakage factor of the k-target table
HOLD_ZERO_BELOW = 3.5          # MWd/kgHM
FDH_SD = 0.025                 # seed s.d. of one core F_dH at ~20000-25000 particles (ladder_core_c9)
PAIRS = {47: ("c9_dep_core3d_L1", "c9_dep_core3d"),
         35: ("c9_dep_core3d_d35_L1", "c9_dep_core3d_d35_L8"),
         27: ("c9_dep_core3d_d27_L1", "c9_dep_core3d_d27_L8"),
         1:  ("c9_dep_core3d_d1_L1", "c9_dep_core3d_d1_L8")}
LAYER_RUNS_47 = {1: "c9_dep_core3d_L1", 4: "c9_dep_core3d_L4", 8: "c9_dep_core3d", 12: "c9_dep_core3d_L12"}

C = {47: "#0072B2", 35: "#D55E00", 27: "#009E73", 1: "#CC79A7"}
C_GREY = "#444444"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "pdf.fonttype": 42,
})


def load(d):
    return next(iter(json.loads((Path(d) / "runs.json").read_text()).values()))


def pair_data():
    D = {}
    for d, (p1, p8) in PAIRS.items():
        r1, r8 = load(p1), load(p8)
        n = min(len(r1["k_hist"]), len(r8["k_hist"]))
        b1 = np.array(r1["bu_hist"]); b8 = np.array(r8["bu_hist"])
        if not np.allclose(b1[:n], b8[:n]):
            raise RuntimeError(f"C9-{d}: the two runs do not share burnup states")
        k1 = np.array(r1["k_hist"]); k8 = np.array(r8["k_hist"])
        rho = 1e5 * np.log((k1[:n] / k1[0]) / (k8[:n] / k8[0]))
        D[d] = dict(b=b1, k1=k1, k8=k8, bn=b1[:n], rho=rho, ratio=float(r8["k_target_ratio"]),
                    e1=float(r1["efpd"]), e8=float(r8["efpd"]))
    return D


def table(D, designs):
    grid = sorted({float(b) for d in designs for b in D[d]["bn"]})
    vals = []
    for g in grid:
        v = [D[d]["rho"][list(D[d]["bn"]).index(g)] for d in designs if g in list(D[d]["bn"])]
        vals.append(0.0 if g <= HOLD_ZERO_BELOW else float(np.mean(v)))
    return np.array(grid), np.array(vals)


def rho_at(B, grid, vals):
    return np.interp(B, grid, vals, left=0.0, right=vals[-1])


def crossing(b, k, target):
    """Last downward crossing after the first state, linear in burnup (as core_geometry)."""
    out = None
    for i in range(1, len(b)):
        if k[i - 1] > target >= k[i]:
            out = b[i - 1] + (b[i] - b[i - 1]) * (k[i - 1] - target) / (k[i - 1] - k[i])
    return out


def validate(D):
    rows = []
    for d in D:
        grid, vals = table(D, [x for x in D if x != d])
        x = D[d]
        kn = (x["k1"] / x["k1"][0]) / np.exp(rho_at(x["b"], grid, vals) / 1e5)
        B = crossing(x["b"], kn, x["ratio"])
        pred = B * 1000 / SPEC if B is not None else None
        rows.append(dict(design=d, uniform=x["e1"], predicted=pred, measured=x["e8"],
                         error=(pred - x["e8"]) if pred is not None else None))
    e = [r["error"] for r in rows if r["error"] is not None]
    return rows, float(np.sqrt(np.mean(np.square(e)))), float(np.max(np.abs(e)))


def fig_correction(D, grid, vals, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    x = D[47]
    a.plot(x["b"], x["k1"] / x["k1"][0], "o-", color=C_GREY, ms=3.5, label="1 layer (uniform burnup)")
    a.plot(x["b"][:len(x["k8"])], x["k8"] / x["k8"][0], "s-", color=C[47], ms=3.5, label="8 layers (resolved burnup)")
    a.axhline(x["ratio"], color="#CC79A7", ls="--", lw=1.0, label="End-of-cycle target")
    a.set_xlabel("Core-average burnup [MWd/kgHM]")
    a.set_ylabel(r"$k(B)\,/\,k(0)$, C9-47")
    a.legend(loc="lower left", frameon=False)
    a.text(0.97, 0.95, "(a)", transform=a.transAxes, fontweight="bold", ha="right", va="top")

    for d in D:
        b.plot(D[d]["bn"], D[d]["rho"], "o", ms=4, color=C[d], label=f"C9-{d}")
    b.plot(grid, vals, "-", color="black", lw=1.6, label="Correction (mean)")
    b.set_xlabel("Core-average burnup [MWd/kgHM]")
    b.set_ylabel(r"$\rho_A(B)$ [pcm]")
    b.legend(loc="upper left", frameon=False, ncol=2)
    b.set_ylim(-400, 3300)
    r = b.twinx()
    lo, hi = b.get_ylim()
    r.set_ylim(L_AX * np.exp(lo / 1e5), L_AX * np.exp(hi / 1e5))
    r.set_ylabel(r"$L_\mathrm{ax,eff}(B)$")
    r.grid(False)
    b.text(0.97, 0.05, "(b)", transform=b.transAxes, fontweight="bold", ha="right", va="bottom")
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"c9_axial_correction.{ext}", **kw)
    plt.close(fig)


def fig_peaking(out):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    cols = {1: C_GREY, 4: "#E69F00", 8: C[47], 12: "#56B4E9"}
    for L, p in LAYER_RUNS_47.items():
        r = load(p); bu = r["bu_hist"]; st = r["states"]
        a.plot(bu, [s["fz"] for s in st], "o-", ms=3, color=cols[L], label=f"{L} layer" + ("s" if L > 1 else ""))
        if L in (1, 12):
            b.errorbar(np.array(bu) + (0.25 if L == 12 else -0.25), [s["fdh"] for s in st], yerr=FDH_SD,
                       fmt="o-", ms=3, capsize=2, lw=1.0, color=cols[L], label=f"{L} layer" + ("s" if L > 1 else ""))
    a.set_xlabel("Burnup [MWd/kgHM]"); a.set_ylabel(r"Axial peaking $F_z$")
    a.legend(loc="lower left", frameon=False)
    a.set_title("(a)", loc="left", fontweight="bold")
    b.set_xlabel("Burnup [MWd/kgHM]"); b.set_ylabel(r"Radial peaking $F_{\Delta H}$")
    b.legend(loc="lower left", frameon=False)
    b.set_title("(b)", loc="left", fontweight="bold")

    r = load(LAYER_RUNS_47[12]); n = len(r["states"][0]["layer_share"])
    z = (np.arange(n) + 0.5) * 120.0 / n
    shades = ["#999999", "#56B4E9", "#0072B2", "#000000"]
    for j, Bt in enumerate((0.0, 7.5, 13.5, 21.5)):
        i = r["bu_hist"].index(Bt)
        sh = np.array(r["states"][i]["layer_share"]) * n
        c.plot(sh, z, "-o", ms=2.5, color=shades[j], label=f"B = {Bt:g}")
    c.set_xlabel("Layer power / average"); c.set_ylabel("Height [cm]")
    c.set_ylim(0, 120); c.set_xlim(0.2, 1.6)
    c.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=6.5)
    c.set_title("(c)", loc="left", fontweight="bold")
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"c9_axial_peaking.{ext}", **kw)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figs_c9_axial")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = pair_data()
    grid, vals = table(D, list(D))
    rows, rms, mx = validate(D)
    res = dict(
        definition="rho_A(B) = 1e5 ln{[k_1(B)/k_1(0)] / [k_8(B)/k_8(0)]}, mean over the paired designs",
        apply="k_target(t_refl, B) = k_target(t_refl) * exp(rho_A(B)/1e5);  L_ax_eff(B) = L_ax * exp(rho_A(B)/1e5)",
        hold_zero_below_mwd_kg=HOLD_ZERO_BELOW, L_ax=L_AX, designs=list(D),
        burnup_mwd_kg=grid.tolist(), rho_A_pcm=[round(v, 1) for v in vals],
        L_ax_eff=[round(L_AX * np.exp(v / 1e5), 5) for v in vals],
        per_design={int(d): dict(burnup=D[d]["bn"].tolist(), rho_A_pcm=[round(v, 1) for v in D[d]["rho"]])
                    for d in D},
        validation=dict(method="leave one design out, uniform history -> predicted layered cycle",
                        rows=rows, rms_d=rms, max_abs_d=mx))
    (out / "axial_correction.json").write_text(json.dumps(res, indent=1))
    print("burnup [MWd/kgHM]:", grid.tolist())
    print("rho_A [pcm]      :", [round(v) for v in vals])
    print("L_ax,eff         :", [round(float(L_AX * np.exp(v / 1e5)), 4) for v in vals])
    for r in rows:
        print(f"  C9-{r['design']:<3} uniform {r['uniform']:6.0f} d -> predicted {r['predicted']:6.0f} d, "
              f"measured {r['measured']:6.0f} d, error {r['error']:+5.0f} d")
    print(f"leave-one-out: rms {rms:.0f} d, max {mx:.0f} d")
    fig_correction(D, grid, vals, out)
    fig_peaking(out)
    print(f"wrote {out}/axial_correction.json, c9_axial_correction.pdf/.png, c9_axial_peaking.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
