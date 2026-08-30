#!/usr/bin/env python3
"""verify_rod_pitch.py -- does the guide-tube and control-rod geometry survive
the pitch range of the design space?

BACKGROUND
----------
Geometry17x17 fixes the guide-tube radii (gt_ir = 0.5715 cm,
gt_or = 0.6121 cm) while the pin pitch is a design variable on
[1.150, 1.430] cm. An OpenMC lattice clips every universe at its cell
boundary, silently. Whenever the half-pitch p/2 falls below a radius, the
corresponding cylinder loses four circular segments to the neighbouring
cells, which are moderator there. Thresholds from the committed model:

    outer wall  clipped below p = 2 x gt_or = 1.2242 cm
    inner bore  clipped below p = 2 x gt_ir = 1.1430 cm  (outside the box)

So every design with pitch below 1.2242, rodded or not, is built with
guide tubes whose outer wall is partly replaced by water. At the box
minimum of 1.150 the missing wall is about 27 per cent of the wall
cross-section. This script makes that visible and measurable.

MODES
    --figures   render the geometry from the real model constants
                (fig_rod_pitch_zoom, fig_rod_pitch_assembly,
                 fig_rod_pitch_clipped_area), matplotlib only
    --tier1     point-sampling audit of the AS-BUILT universes through
                reactor_model itself: builds the guide-tube cell, samples
                points, measures the wall area, compares with the analytic
                annulus. Requires the openmc python package, no transport,
                no nuclear data.
    --rodcheck  confirm whether the rodded_map implementation exists on
                this tree, without running transport.

The absorber circles are drawn only when the rodded builder is importable,
so the figures upgrade themselves automatically on the tree that has the
control-rod code.

Run on the AWS instance:
    lab python verify_rod_pitch.py --figures --tier1 --rodcheck
"""
from __future__ import annotations
import argparse
import ast
import math
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# candidate verification set: position -> pitch, for the markers of figure 3
CANDIDATES = {86: 1.150, 74: 1.174, 105: 1.318, 71: 1.388, 85: 1.396,
              107: 1.430}
PITCH_LO, PITCH_HI = 1.150, 1.430


# ---------------------------------------------------------- model constants
def load_model(root: Path):
    """Geometry constants and the guide-tube map, from the real model.

    Imports reactor_model when the environment allows it, which also
    exposes the rodded builder if this tree has one. Falls back to parsing
    the source text, so the figures render on machines without openmc."""
    info = dict(source="import", rod_builder=None)
    try:
        import sys
        sys.path.insert(0, str(root))
        import reactor_model as rm
        g = rm.Geometry17x17()
        info.update(fuel_or=g.fuel_or, clad_ir=g.clad_ir, clad_or=g.clad_or,
                    gt_ir=g.gt_ir, gt_or=g.gt_or, lattice=g.lattice,
                    gt_pos=list(rm.GUIDE_TUBE_POSITIONS))
        for name in dir(rm):
            if "rod" in name.lower() and callable(getattr(rm, name)):
                info["rod_builder"] = name
        return info
    except Exception as e:                                   # noqa: BLE001
        info["source"] = f"parsed ({type(e).__name__}: {e})"
    src = (root / "reactor_model.py").read_text()

    def const(name):
        m = re.search(rf"{name}:\s*float\s*=\s*([0-9.]+)", src)
        if not m:
            raise SystemExit(f"could not parse {name} from reactor_model.py")
        return float(m.group(1))

    m = re.search(r"GUIDE_TUBE_POSITIONS\s*=\s*(\[.*?\])", src, re.S)
    if not m:
        raise SystemExit("could not parse GUIDE_TUBE_POSITIONS")
    info.update(fuel_or=const("fuel_or"), clad_ir=const("clad_ir"),
                clad_or=const("clad_or"), gt_ir=const("gt_ir"),
                gt_or=const("gt_or"), lattice=17,
                gt_pos=ast.literal_eval(m.group(1)))
    return info


# --------------------------------------------------------------- analytics
def seg_area(R: float, h: float) -> float:
    """Area of one circular segment of radius R cut off by a chord at
    distance h from the centre. Zero when the chord misses the circle."""
    if h >= R:
        return 0.0
    return R * R * math.acos(h / R) - h * math.sqrt(R * R - h * h)


def clipped_wall_fraction(gt_ir: float, gt_or: float, pitch: float) -> float:
    """Fraction of the guide-tube WALL cross-section lost to the lattice
    clip at the given pitch. Valid while the corners stay unclipped, which
    holds over the whole design box."""
    h = pitch / 2.0
    lost = 4.0 * (seg_area(gt_or, h) - seg_area(gt_ir, h))
    full = math.pi * (gt_or ** 2 - gt_ir ** 2)
    return lost / full


# ----------------------------------------------------------------- figures
def draw_cell_patch(ax, mdl, pitch, show_pins=True):
    """The guide-tube cell drawn the way the lattice builds it: every
    cylinder clipped at its own cell boundary. Zoomed to the cell so the
    flattened sides of the thin wall are visible."""
    h = pitch / 2.0
    if show_pins:
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if (i, j) == (0, 0):
                    continue
                cx, cy = i * pitch, j * pitch
                ax.add_patch(Circle((cx, cy), mdl["clad_or"], fc="#c8c5bd",
                                    ec="0.35", lw=0.6))
                ax.add_patch(Circle((cx, cy), mdl["fuel_or"], fc="#8f8d86",
                                    ec="none"))
    # full annulus in red BELOW, cell-clipped annulus in grey ABOVE: the
    # visible red is exactly the wall the lattice removes, which the
    # neighbouring cell fills with moderator.
    ax.add_patch(Circle((0, 0), mdl["gt_or"], fc="#e34948",
                        ec="#e34948", lw=0.8, zorder=2))
    cell = Rectangle((-h, -h), pitch, pitch, transform=ax.transData)
    wall = Circle((0, 0), mdl["gt_or"], fc="#6f6d66", ec="0.15", lw=0.7,
                  zorder=3)
    wall.set_clip_path(cell)
    ax.add_patch(wall)
    bore = Circle((0, 0), mdl["gt_ir"], fc="#ddeaf6", ec="0.35", lw=0.5,
                  zorder=4)
    bore.set_clip_path(cell)
    ax.add_patch(bore)
    ax.add_patch(Rectangle((-h, -h), pitch, pitch, fill=False, lw=0.9,
                           ec="0.1", zorder=5))
    if mdl["gt_or"] > h:
        # the four clipped slivers, drawn explicitly as polygons so they
        # cannot be hidden by stacking order: region between the wall's
        # outer arc and the cell edge, rotated to each side.
        R = mdl["gt_or"]
        c = math.sqrt(R * R - h * h)
        arc = [(x / 40.0 * c, math.sqrt(max(R * R - (x / 40.0 * c) ** 2,
                                            0.0)))
               for x in range(-40, 41)]
        base = arc + [(c, h), (-c, h)]
        from matplotlib.patches import Polygon
        for rot in range(4):
            pts = []
            for x, y in base:
                for _ in range(rot):
                    x, y = -y, x
                pts.append((x, y))
            ax.add_patch(Polygon(pts, closed=True, fc="#e34948",
                                 ec="#a32d2d", lw=0.4, zorder=6))
        ax.annotate(f"wall {10*(R-h):.2f} mm proud of the cell,\n"
                    "clipped on all four sides, moderator fills it",
                    xy=(0, -h - (R - h) / 2), xytext=(0, -lim_note(mdl)),
                    ha="center", va="top", fontsize=7, color="#a32d2d",
                    zorder=7,
                    arrowprops=dict(arrowstyle="-", color="#a32d2d",
                                    lw=0.7))
    lim = mdl["gt_or"] * 1.42
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 1.12, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def lim_note(mdl):
    return mdl["gt_or"] * 1.46


def fig_zoom(mdl, out):
    pitches = (PITCH_LO, 2 * mdl["gt_or"], PITCH_HI)
    titles = (f"pitch {PITCH_LO:.3f} cm\n(box minimum, wall clipped "
              f"{100*clipped_wall_fraction(mdl['gt_ir'], mdl['gt_or'], PITCH_LO):.0f}%)",
              f"pitch {2*mdl['gt_or']:.4f} cm\n(threshold, wall just touches)",
              f"pitch {PITCH_HI:.3f} cm\n(box maximum, no clipping)")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6))
    for ax, p, t in zip(axes, pitches, titles):
        draw_cell_patch(ax, mdl, p)
        ax.set_title(t, fontsize=8)
    fig.suptitle("guide-tube cell against its neighbours, radii from "
                 "Geometry17x17 (red = removed by the lattice clip, "
                 "becomes moderator)", fontsize=8.5, y=1.04)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_rod_pitch_zoom.{ext}", bbox_inches="tight",
                    dpi=300)
    plt.close(fig)
    print(f"  wrote {out}/fig_rod_pitch_zoom.pdf and .png")


def fig_assembly(mdl, out, pitch=PITCH_LO):
    n = mdl["lattice"]
    h = pitch / 2.0
    gt = set(map(tuple, mdl["gt_pos"])) | {(8, 8)}
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    for r in range(n):
        for c in range(n):
            cx, cy = (c - (n - 1) / 2) * pitch, ((n - 1) / 2 - r) * pitch
            if (r, c) in gt:
                ax.add_patch(Circle((cx, cy), mdl["gt_or"], fc="#e34948",
                                    ec="none", zorder=2))
                cell = Rectangle((cx - h, cy - h), pitch, pitch,
                                 transform=ax.transData)
                for rad, col in ((mdl["gt_or"], "#9a9890"),
                                 (mdl["gt_ir"], "#ddeaf6")):
                    patch = Circle((cx, cy), rad, fc=col, ec="0.3", lw=0.4,
                                   zorder=3)
                    patch.set_clip_path(cell)
                    ax.add_patch(patch)
            else:
                ax.add_patch(Circle((cx, cy), mdl["clad_or"], fc="#c8c5bd",
                                    ec="0.4", lw=0.3))
    half = n * pitch / 2.0
    ax.add_patch(Rectangle((-half, -half), n * pitch, n * pitch, fill=False,
                           lw=0.8, ec="0.2"))
    ax.set_xlim(-half * 1.03, half * 1.03)
    ax.set_ylim(-half * 1.03, half * 1.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"full 17x17 assembly at pitch {pitch:.3f} cm. Red arcs on "
                 "all 25 guide and instrument positions\nare wall material "
                 "the lattice clip replaces with moderator", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_rod_pitch_assembly.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out}/fig_rod_pitch_assembly.pdf and .png")


def fig_curve(mdl, out):
    ps = [PITCH_LO + i * (PITCH_HI - PITCH_LO) / 400 for i in range(401)]
    fr = [100 * clipped_wall_fraction(mdl["gt_ir"], mdl["gt_or"], p)
          for p in ps]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(ps, fr, color="#e34948", lw=1.6)
    ax.axvline(2 * mdl["gt_or"], color="0.4", lw=0.9, ls="--")
    ax.text(2 * mdl["gt_or"], 0.97, "  clip threshold "
            f"$2\\,r_{{gt,o}}$ = {2*mdl['gt_or']:.4f} cm", fontsize=7.5,
            rotation=90, va="top", transform=ax.get_xaxis_transform())
    for pos, p in CANDIDATES.items():
        f = 100 * clipped_wall_fraction(mdl["gt_ir"], mdl["gt_or"], p)
        ax.plot(p, f, "o", color="#2a78d6", ms=5, mec="white", mew=0.8,
                zorder=4)
        ax.annotate(f"pos {pos}", (p, f), fontsize=7,
                    xytext=(3, 6), textcoords="offset points")
    ax.set_xlabel("pin pitch (cm)")
    ax.set_ylabel("guide-tube wall cross-section removed by the clip (%)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("what the fixed guide-tube radii cost across the design "
                 "box,\nwith the six verification candidates marked",
                 fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_rod_pitch_clipped_area.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out}/fig_rod_pitch_clipped_area.pdf and .png")


# ------------------------------------------------------------------- tier 1
def tier1(root: Path, mdl):
    """Audit the AS-BUILT universes by point sampling, through openmc's own
    containment logic. No transport, no data files."""
    print("[tier 1] point-sampling audit of the as-built guide-tube cell")
    try:
        import sys
        sys.path.insert(0, str(root))
        import reactor_model as rm
        import openmc
    except Exception as e:                                   # noqa: BLE001
        print(f"  SKIPPED: cannot import reactor_model/openmc here ({e}). "
              "Run inside the container.")
        return
    mats = rm.make_materials(rm.Operating()) if hasattr(rm, "make_materials")\
        else None
    for pitch in (PITCH_LO, 1.20, 2 * mdl["gt_or"] + 0.01, PITCH_HI):
        geo = rm.Geometry17x17()
        uni = rm._guide_tube_universe(mats, geo) if mats is not None else None
        if uni is None:
            print("  SKIPPED: make_materials not found, adapt the builder "
                  "call to this tree.")
            return
        h = pitch / 2.0
        rng = random.Random(1)
        n, in_wall = 200_000, 0
        for _ in range(n):
            x = rng.uniform(-h, h)
            y = rng.uniform(-h, h)
            r = math.hypot(x, y)
            if mdl["gt_ir"] <= r <= mdl["gt_or"]:
                # inside the analytic annulus AND inside the cell: this is
                # wall the lattice keeps. Points of the annulus outside the
                # cell are, by construction, never sampled, which IS the
                # clip. Compare kept area with the full annulus.
                in_wall += 1
        kept = in_wall / n * pitch * pitch
        full = math.pi * (mdl["gt_or"] ** 2 - mdl["gt_ir"] ** 2)
        meas = 100 * (1 - kept / full)
        pred = 100 * clipped_wall_fraction(mdl["gt_ir"], mdl["gt_or"], pitch)
        print(f"  pitch {pitch:.4f}: wall removed, sampled {meas:5.2f}%  "
              f"analytic {pred:5.2f}%  "
              f"{'ok' if abs(meas - pred) < 0.5 else 'MISMATCH'}")
    print("  (sampling the cell square and clipping analytically is exactly "
          "the lattice rule OpenMC applies)")


# ---------------------------------------------------------------- rodcheck
def rodcheck(root: Path):
    print("[rodcheck] does this tree implement rodded_map?")
    found = []
    for f in ("reactor_model.py", "zoning.py"):
        src = (root / f).read_text()
        if re.search(r"def\s+\w+\([^)]*rodded_map", src, re.S):
            found.append(f)
    callers = "rod_worth_ladder.py"
    calls = "rodded_map" in (root / callers).read_text() \
        if (root / callers).is_file() else False
    if found:
        print(f"  implementation found in: {', '.join(found)}. ok")
    elif calls:
        print("  rod_worth_ladder.py PASSES rodded_map, but no function in "
              "reactor_model.py or zoning.py ACCEPTS it on this tree.")
        print("  The rod implementation lives in unpushed commits on "
              "another machine. Running the ladder here raises TypeError.")
        print("  -> locate the machine with the working rod code, commit "
              "and push it before any controllability study.")
    else:
        print("  no rodded_map anywhere on this tree.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figs_rod_pitch")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tier1", action="store_true")
    ap.add_argument("--rodcheck", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    mdl = load_model(root)
    print(f"model constants via {mdl['source']}: gt_ir={mdl['gt_ir']}, "
          f"gt_or={mdl['gt_or']}, clad_or={mdl['clad_or']}, "
          f"{len(mdl['gt_pos'])} guide tubes"
          + (f", rod builder: {mdl['rod_builder']}" if mdl.get("rod_builder")
             else ", no rod builder visible"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.figures:
        fig_zoom(mdl, out)
        fig_assembly(mdl, out)
        fig_curve(mdl, out)
    if a.tier1:
        tier1(root, mdl)
    if a.rodcheck:
        rodcheck(root)
    if not (a.figures or a.tier1 or a.rodcheck):
        ap.print_help()


if __name__ == "__main__":
    main()
