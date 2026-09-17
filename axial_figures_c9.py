#!/usr/bin/env python3
r"""
axial_figures_c9.py -- draw the axial and radial power figures from the
tally arrays that axial_shape_c9.py saved. No OpenMC, no transport, numpy
and matplotlib only, so it runs on the laptop as well as on wks720.

Reads <out>/d<idx>/<state>_s<seed>.npz (asm, pin, ax, edges) and
<out>/runs.json (k, sd per solve), averages the seeds of each state, and
writes to <out>/figs/:

  per state, one panel per design, the comparison the front needs
    <state>_assembly.pdf         normalised assembly power, annotated
    <state>_pins.pdf             normalised pin power
    <state>_axial.pdf            power per axial bin in MW, all designs
    <state>_shape.pdf            axial shape per unit length, all designs
    <state>_rpd_vs_champion.pdf  relative difference of each assembly map
                                 against the champion, max and rms printed
  per design, one panel per state, the rod effect on one lattice
    d<idx>_assembly.pdf, d<idx>_pins.pdf, d<idx>_axial.pdf

Two axial figures exist because they answer different questions. Power per
bin in MW is what the NuScale benchmark tabulates and is comparable with it,
but the grid bins are shorter than the others, so part of every dip is bin
length. The shape per unit length removes that and is the one to use when
comparing designs with each other.

Terms:
    F_dH   radial hot channel factor, max pin over mean pin, dimensionless
    F_z    axial peaking, max over bins of local per-length power over the
           length-weighted mean, dimensionless
    AO     axial offset, (P_top - P_bottom) / (P_top + P_bottom),
           dimensionless, positive when power is top-skewed
    P_i    power in axial bin i, normalised so the bins sum to the core
           power, MW

USAGE
    python -c "import numpy, matplotlib; print('env ok')" && \
    python axial_figures_c9.py --out axial_c9 --champion 47 --power-mw 48

Flags:
    --out DIR         the directory axial_shape_c9.py wrote (default axial_c9)
    --champion N      design the RPD maps are taken against (default: the
                      lowest index present, so pass it explicitly)
    --power-mw P      core power the axial bins are normalised to (default 48)
    --states ...      restrict to some states (default: every state found)
    --png             also write a PNG next to each PDF
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axial_shape_c9 import derive, STATE_ORDER, NL   # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9,
                     "pdf.fonttype": 42})
COLS = {"ARO": "#0072B2", "RE12": "#D55E00", "ARI": "#009E73"}
MARK = {"ARO": "o-", "RE12": "s--", "ARI": "^-."}
LETTERS = "ABCDEFGH"


# ------------------------------------------------------------------ loading --
def load(out, power_mw, states=None):
    """Return results[idx][state] with seed-averaged arrays and figures of merit."""
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else {}
    edges = None; results = {}
    for ddir in sorted(out.glob("d*")):
        m = re.fullmatch(r"d(\d+)", ddir.name)
        if not m:
            continue
        idx = int(m.group(1)); per_state = {}
        for st in STATE_ORDER:
            if states and st not in states:
                continue
            files = sorted(ddir.glob(f"{st}_s*.npz"))
            if not files:
                continue
            asm, pin, ax, ks, sds = [], [], [], [], []
            for f in files:
                z = np.load(f)
                asm.append(z["asm"]); pin.append(z["pin"]); ax.append(z["ax"])
                if edges is None:
                    edges = z["edges"]
                seed = int(re.search(r"_s(\d+)", f.stem).group(1))
                for key, rec in runs.items():
                    a, s, sd_, *_ = key.split("|")
                    if int(a) == idx and s == st and int(sd_) == seed:
                        ks.append(rec["keff"]); sds.append(rec["sd"])
            asm = np.mean(asm, axis=0); pin = np.mean(pin, axis=0); ax = np.mean(ax, axis=0)
            dv = derive(asm, pin, ax, edges, power_mw)
            f = np.ma.masked_equal(pin, 0.0)
            per_state[st] = dict(
                keff=float(np.mean(ks)) if ks else float("nan"),
                sd=float(np.sqrt(np.sum(np.square(sds))) / len(sds)) if sds else float("nan"),
                n_seeds=len(files), fdh=dv["fdh"], fz=dv["fz"], ao=dv["ao"],
                p_mw=dv["p_mw"], shape=dv["shape"], asm_norm=dv["asm_norm"],
                pin_norm=np.where(pin > 0, pin / f.mean(), np.nan))
        if "ARO" in per_state:
            k0 = per_state["ARO"]["keff"]
            for st in per_state:
                per_state[st]["rho_vs_ARO_pcm"] = float((1.0 / k0 - 1.0 / per_state[st]["keff"]) * 1e5)
        if per_state:
            results[idx] = per_state
    return results, edges


def grid_bands(edges):
    """Spacer-grid bands inside the active fuel, from the same function the
    transport used (axial_shape_c9.axial_edges on the default HardwareSpec).
    Drawn only if that function reproduces the stored edges, since a band
    guessed from bin lengths can mark a bin that is not a grid."""
    try:
        import hardware3d as hw
        from axial_shape_c9 import axial_edges
        e_ref, bands, _ = axial_edges(hw.HardwareSpec(), hw)
    except Exception as exc:                      # noqa: BLE001
        print(f"  WARNING: grid bands not recovered ({exc}), none drawn")
        return []
    e_ref = np.asarray(e_ref, dtype=float); edges = np.asarray(edges, dtype=float)
    if e_ref.shape != edges.shape or not np.allclose(e_ref, edges, atol=1e-6):
        print("  WARNING: stored edges differ from the default HardwareSpec, "
              "no grid bands drawn")
        return []
    return [(float(g0), float(g1)) for g0, g1 in bands]


def save(fig, path, png):
    fig.savefig(path)
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ------------------------------------------------------------ per state figs --
def by_state(results, edges, bands, fuel, h, champion, figdir, png):
    zc = 0.5 * (edges[:-1] + edges[1:]) - fuel[0]
    ids = sorted(results)
    for st in STATE_ORDER:
        have = [i for i in ids if st in results[i]]
        if not have:
            continue
        n = len(have); ncol = min(4, n); nrow = int(np.ceil(n / ncol))

        maps = {i: results[i][st]["asm_norm"] for i in have}
        vmin = min(np.nanmin(m) for m in maps.values()); vmax = max(np.nanmax(m) for m in maps.values())
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol + 0.9, 3.4 * nrow), squeeze=False)
        for ax, i in zip(axes.ravel(), have):
            m = maps[i]; ny, nx = m.shape
            im = ax.imshow(m, cmap="jet", vmin=vmin, vmax=vmax, origin="lower")
            for r in range(ny):
                for c in range(nx):
                    if np.isfinite(m[r, c]):
                        ax.text(c, r, f"{m[r, c]:.3f}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(nx), list(LETTERS[:nx])); ax.set_yticks(range(ny), range(1, ny + 1))
            ax.set_title(f"C9-{i}, $F_{{\\Delta H}}$ = {results[i][st]['fdh']:.3f}", fontsize=8)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="assembly power, normalised")
        fig.suptitle(f"{st}, assembly power on the hardware model", fontsize=9)
        save(fig, figdir / f"{st}_assembly.pdf", png)

        pins = {i: results[i][st]["pin_norm"] for i in have}
        pmax = max(np.nanmax(m) for m in pins.values())
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 0.9, 3.6 * nrow), squeeze=False)
        for ax, i in zip(axes.ravel(), have):
            im = ax.imshow(pins[i], cmap="jet", vmin=0, vmax=pmax, origin="lower", interpolation="nearest")
            ax.set_title(f"C9-{i}", fontsize=8); ax.axis("off")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="pin power, normalised")
        fig.suptitle(f"{st}, radial pin power on the hardware model", fontsize=9)
        save(fig, figdir / f"{st}_pins.pdf", png)

        for key, xlabel, name in (("p_mw", "power per axial bin (MW)", "axial"),
                                  ("shape", "axial shape, local over mean (dimensionless)", "shape")):
            fig, ax = plt.subplots(figsize=(4.4, 5.8))
            for g0, g1 in bands:
                ax.axhspan(g0 - fuel[0], g1 - fuel[0], color="#999999", alpha=0.25, lw=0)
            for i in have:
                r = results[i][st]
                ax.plot(r[key], zc, "o-", ms=3, lw=1.0,
                        label=f"C9-{i}, $F_z$ {r['fz']:.3f}, AO {r['ao']:+.3f}")
            if key == "shape":
                ax.axvline(1.0, color="#444444", lw=0.8, ls=":")
            ax.set_xlabel(xlabel); ax.set_ylabel("height above the fuel bottom (cm)")
            ax.set_ylim(0, h); ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=6.5, loc="lower right")
            ax.set_title(f"{st}, shaded bands are spacer grids", fontsize=8)
            save(fig, figdir / f"{st}_{name}.pdf", png)

        if champion in maps and n > 1:
            ref = maps[champion]; others = [i for i in have if i != champion]
            nc = min(3, len(others)); nr = int(np.ceil(len(others) / nc))
            rpds = {i: np.abs(maps[i] - ref) / ref * 100.0 for i in others}
            vmax = max(np.nanmax(v) for v in rpds.values())
            fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc + 0.9, 3.4 * nr), squeeze=False)
            for ax, i in zip(axes.ravel(), others):
                v = rpds[i]; ny, nx = v.shape
                im = ax.imshow(v, cmap="YlOrRd", vmin=0, vmax=vmax, origin="lower")
                for r in range(ny):
                    for c in range(nx):
                        if np.isfinite(v[r, c]):
                            ax.text(c, r, f"{v[r, c]:.2f}", ha="center", va="center", fontsize=6,
                                    color="white" if v[r, c] > 0.6 * vmax else "black")
                ax.set_xticks(range(nx), list(LETTERS[:nx])); ax.set_yticks(range(ny), range(1, ny + 1))
                ax.set_title(f"C9-{i} vs C9-{champion}: max {np.nanmax(v):.2f} %, "
                             f"rms {np.sqrt(np.nanmean(v ** 2)):.2f} %", fontsize=7.5)
            for ax in axes.ravel()[len(others):]:
                ax.axis("off")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="relative difference (%)")
            fig.suptitle(f"{st}, assembly power relative to C9-{champion}", fontsize=9)
            save(fig, figdir / f"{st}_rpd_vs_champion.pdf", png)


# ----------------------------------------------------------- per design figs --
def by_design(results, edges, bands, fuel, h, figdir, png):
    zc = 0.5 * (edges[:-1] + edges[1:]) - fuel[0]
    for idx, ps in sorted(results.items()):
        sts = [s for s in STATE_ORDER if s in ps]
        if len(sts) < 2:
            continue
        vmin = min(np.nanmin(ps[s]["asm_norm"]) for s in sts); vmax = max(np.nanmax(ps[s]["asm_norm"]) for s in sts)
        fig, axes = plt.subplots(1, len(sts), figsize=(3.6 * len(sts) + 0.8, 3.6), squeeze=False)
        for ax, s in zip(axes[0], sts):
            m = ps[s]["asm_norm"]; ny, nx = m.shape
            im = ax.imshow(m, cmap="jet", vmin=vmin, vmax=vmax, origin="lower")
            for r in range(ny):
                for c in range(nx):
                    if np.isfinite(m[r, c]):
                        ax.text(c, r, f"{m[r, c]:.3f}", ha="center", va="center", fontsize=6.5)
            ax.set_xticks(range(nx), list(LETTERS[:nx])); ax.set_yticks(range(ny), range(1, ny + 1))
            ax.set_title(f"{s}, k = {ps[s]['keff']:.5f}", fontsize=8)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="assembly power, normalised")
        fig.suptitle(f"C9-{idx}, rod effect on the assembly power", fontsize=9)
        save(fig, figdir / f"d{idx}_assembly.pdf", png)

        pmax = max(np.nanmax(ps[s]["pin_norm"]) for s in sts)
        fig, axes = plt.subplots(1, len(sts), figsize=(4.2 * len(sts) + 0.8, 4.2), squeeze=False)
        for ax, s in zip(axes[0], sts):
            im = ax.imshow(ps[s]["pin_norm"], cmap="jet", vmin=0, vmax=pmax, origin="lower", interpolation="nearest")
            ax.set_title(f"{s}, $F_{{\\Delta H}}$ = {ps[s]['fdh']:.3f}", fontsize=8); ax.axis("off")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="pin power, normalised")
        fig.suptitle(f"C9-{idx}, rod effect on the pin power", fontsize=9)
        save(fig, figdir / f"d{idx}_pins.pdf", png)

        fig, ax = plt.subplots(figsize=(4.2, 5.6))
        for g0, g1 in bands:
            ax.axhspan(g0 - fuel[0], g1 - fuel[0], color="#999999", alpha=0.25, lw=0)
        for s in sts:
            r = ps[s]
            ax.plot(r["p_mw"], zc, MARK[s], color=COLS[s], ms=4, lw=1.1,
                    label=f"{s}, $F_z$ {r['fz']:.3f}, AO {r['ao']:+.3f}, "
                          f"$\\rho$ {r.get('rho_vs_ARO_pcm', 0):+.0f} pcm")
        ax.set_xlabel("power per axial bin (MW)"); ax.set_ylabel("height above the fuel bottom (cm)")
        ax.set_ylim(0, h); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.set_title(f"C9-{idx}, rod effect on the axial profile", fontsize=8)
        save(fig, figdir / f"d{idx}_axial.pdf", png)


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="axial_c9")
    ap.add_argument("--champion", type=int, default=None)
    ap.add_argument("--power-mw", type=float, default=48.0)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if not out.exists():
        print(f"ABORT: {out} not found"); return 1
    results, edges = load(out, a.power_mw, a.states)
    if not results:
        print(f"ABORT: no <state>_s<seed>.npz files under {out}/d*/"); return 1
    bands = grid_bands(edges); fuel = (float(edges[0]), float(edges[-1])); h = fuel[1] - fuel[0]
    champion = a.champion if a.champion is not None else min(results)
    print(f"designs {sorted(results)}, states per design "
          f"{ {i: sorted(ps) for i, ps in results.items()} }")
    print(f"{len(edges) - 1} axial bins, {len(bands)} grid bands, champion C9-{champion}")

    figdir = out / "figs"; figdir.mkdir(parents=True, exist_ok=True)
    by_state(results, edges, bands, fuel, h, champion, figdir, a.png)
    by_design(results, edges, bands, fuel, h, figdir, a.png)

    print("\n  design state    k        F_dH   F_z    AO      rho vs ARO")
    for i, ps in sorted(results.items()):
        for s in STATE_ORDER:
            if s in ps:
                r = ps[s]
                print(f"  C9-{i:<3d} {s:5s} {r['keff']:.5f}  {r['fdh']:.3f}  {r['fz']:.3f}  "
                      f"{r['ao']:+.3f}  {r.get('rho_vs_ARO_pcm', 0):+7.0f} pcm  ({r['n_seeds']} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
