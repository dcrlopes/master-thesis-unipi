#!/usr/bin/env python3
r"""
c9_reverse_retro.py
===================
The mirror of c8_reformulation_retro.py. That script asked what Campaign 8
would have selected under the Campaign 9 objectives. This one asks what
Campaign 8's objectives would have selected FROM the Campaign 9 archive,
and which Campaign 9 front members the original formulation would have
discarded. No OpenMC. Seconds.

Three questions.

 1. Under (maximise EFPD, minimise F_dH) with the Campaign 8 constraints,
    which Campaign 9 designs are non-dominated, and do they include any
    of the Campaign 9 front?
 2. Where does each Campaign 9 front member rank under the Campaign 8
    objectives? A low rank is a design the original formulation would
    never have kept.
 3. Cross-campaign overlap: the Campaign 8 champion (47) and the
    retrospective's second member (13) against the Campaign 9 front, in
    design space, by normalised distance.

Usage
  python c9_reverse_retro.py --c9 out_c9/optimization_checkpoint.json \
      --c8 out_c8/optimization_checkpoint.json --manifest c9_post/c9_front.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np


def nondom(F):
    n = len(F); keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False; break
    return np.where(keep)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c9", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--c8", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--manifest", default="c9_post/c9_front.json")
    ap.add_argument("--out", default="c9_post")
    ap.add_argument("--c8-front", nargs="*", type=int, default=[47, 42, 23, 29, 21, 44, 59, 1])
    ap.add_argument("--c8-candidates", nargs="*", type=int, default=[47, 13])
    a = ap.parse_args(argv)

    d9 = json.load(open(a.c9)); r9 = d9["all_raw"]; cn9 = d9["constraint_names"]
    d8 = json.load(open(a.c8)); r8 = d8["all_raw"]; cn8 = d8["constraint_names"]
    m = json.load(open(a.manifest)); front9 = m["front"]
    dv = d9["design_variables"]
    L = []; P = L.append

    # 1. C9 archive under the C8 formulation: same constraint set as C8 (drop g_efpd, g_ctrl_peak)
    c8_like = [c for c in cn9 if c in cn8]
    feas = [i for i, x in enumerate(r9) if all(x[c] <= 0 for c in c8_like)]
    F8 = np.array([[-r9[i]["cycle_length"], r9[i]["peaking"]] for i in feas])
    nd = [feas[j] for j in nondom(F8)] if len(feas) else []
    P("1. THE CAMPAIGN 9 ARCHIVE SCORED UNDER THE CAMPAIGN 8 OBJECTIVES")
    P(f"   constraints applied: {c8_like}  (g_efpd and g_ctrl_peak dropped, as in C8)")
    P(f"   feasible {len(feas)} of {len(r9)}; non-dominated in (max EFPD, min F): {sorted(nd)}")
    P(f"   of which on the Campaign 9 front: {sorted(set(nd) & set(front9))}")
    P(f"   Campaign 9 front members the C8 formulation would discard: "
      f"{sorted(set(front9) - set(nd))} of {len(front9)}")
    P("")
    P(f"   {'id':>3} {'e':>5} {'Gd':>5} {'EFPD':>6} {'F':>6} {'c_max':>6}  C8-nondominated  C9-front")
    for i in sorted(set(nd) | set(front9), key=lambda i: -r9[i]["cycle_length"]):
        x = r9[i]
        P(f"   {i:>3} {x['enrich']:5.2f} {x['gd_wt']:5.2f} {x['cycle_length']:6.0f} {x['peaking']:6.3f} "
          f"{x['c_max']:6.0f}  {'yes' if i in nd else 'no':>15}  {'yes' if i in front9 else 'no':>8}")

    # 2. rank of each C9 front member under the C8 objectives (by non-dominated sorting depth)
    P("")
    P("2. NON-DOMINATION DEPTH OF THE CAMPAIGN 9 FRONT UNDER THE CAMPAIGN 8 OBJECTIVES")
    remaining = list(feas); depth = {}; level = 1
    while remaining:
        Fr = np.array([[-r9[i]["cycle_length"], r9[i]["peaking"]] for i in remaining])
        for j in nondom(Fr): depth[remaining[j]] = level
        remaining = [i for i in remaining if i not in depth]; level += 1
    for i in front9:
        P(f"   id {i:>3}: depth {depth.get(i, 'infeasible under C8')}  "
          f"(1 = would have been on the C8-style front, higher = discarded earlier)")

    # 3. overlap with C8 candidates in design space
    P("")
    P("3. CROSS-CAMPAIGN OVERLAP, normalised design-space distance to the Campaign 9 front")
    X9 = np.array([[r9[i][v] for v in dv] for i in range(len(r9))], float)
    lo, hi = X9.min(0), X9.max(0); nrm = lambda v: (np.array(v) - lo) / np.where(hi > lo, hi - lo, 1)
    for c in a.c8_candidates:
        x8 = r8[c]; v8 = nrm([x8[v] for v in dv])
        dist = [(float(np.linalg.norm(v8 - nrm([r9[i][v] for v in dv]))), i) for i in front9]
        dist.sort(); dmin, imin = dist[0]
        x9 = r9[imin]
        P(f"   C8 design {c:>2} ({x8['enrich']:.2f} wt%, Gd {x8['gd_wt']:.2f}, pins {x8['gd_pins_used']:.0f}, "
          f"refl {x8['refl_thick']:.2f}, F {x8['peaking']:.3f}, EFPD {x8['cycle_length']:.0f})")
        P(f"      nearest C9 front member id {imin} ({x9['enrich']:.2f} wt%, Gd {x9['gd_wt']:.2f}, "
          f"pins {x9['gd_pins_used']:.0f}, refl {x9['refl_thick']:.2f}, F {x9['peaking']:.3f}, "
          f"EFPD {x9['cycle_length']:.0f}, c_max {x9['c_max']:.0f}), distance {dmin:.3f} "
          f"({'inside' if dmin < 0.14 else 'outside'} the 0.14 infill separation)")
    txt = "\n".join(L); print(txt)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "c9_reverse_retro.txt").write_text(txt + "\n")
    (out / "c9_reverse_retro.json").write_text(json.dumps(dict(
        c8_like_constraints=c8_like, feasible_under_c8=feas, nondominated_under_c8=sorted(int(i) for i in nd),
        c9_front=front9, discarded_by_c8=sorted(int(i) for i in set(front9) - set(nd)),
        depth={int(k): int(v) for k, v in depth.items() if k in front9}), indent=2))
    print(f"\nwrote {out}/c9_reverse_retro.txt and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
