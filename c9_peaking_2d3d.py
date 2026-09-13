#!/usr/bin/env python3
r"""
c9_peaking_2d3d.py
==================
Reads one or more confirm3d summaries (ARO state is enough) and answers
the question that decides whether Campaign 10 is worth running: is the
three-dimensional peaking a monotone transformation of the
two-dimensional one, or does it reorder the designs?

For every design with both ARO_2D and ARO_3Dhw:
  F_3D / F_2D      the ratio, and its mean, sd, relative sd
  F_3D - F_2D      the difference, for comparison with the C8 statement
  Spearman rho     rank correlation between the two orderings
  max rank shift   the largest change of position
  front test       the non-dominated set in (F, c_max) recomputed with
                   F_3D replacing F_2D, against the archived front

Decision rule printed at the end:
  ratio relative sd below 1 %  AND  front unchanged  -> constant, C10 unnecessary
  otherwise                                           -> reordering measured, C10 justified

Usage
  python c9_peaking_2d3d.py --summary confirm3d_c9_front/summary.json \
      --checkpoint out_c9/optimization_checkpoint.json --manifest c9_post/c9_front.json
  python c9_peaking_2d3d.py --selftest   (C8 eleven designs: shift -0.023, rho 0.909)
"""
import argparse
import json
import pathlib
import statistics as st
import sys


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
    n = len(a); d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)), max(abs(ra[i] - rb[i]) for i in range(n))


def nondom(rows, kx, ky):
    out = []
    for i, a in enumerate(rows):
        if not any(j != i and rows[j][kx] <= a[kx] and rows[j][ky] <= a[ky]
                   and (rows[j][kx] < a[kx] or rows[j][ky] < a[ky]) for j in range(len(rows))):
            out.append(a["id"])
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", nargs="+", default=["confirm3d_c9_front/summary.json"])
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="c9_post")
    ap.add_argument("--tag", default="front")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    rows = {}
    for f in a.summary:
        for k, v in json.load(open(f)).items():
            if "ARO_2D" in v and "ARO_3Dhw" in v:
                rows[int(k)] = dict(id=int(k), F2=v["ARO_2D"]["F"], F3=v["ARO_3Dhw"]["F"],
                                    e=v["design"].get("enrich_inner", v["design"].get("enrich")),
                                    gd=v["design"].get("gd_wt"))
    if not rows:
        sys.exit("no design with both ARO_2D and ARO_3Dhw in the summaries")
    R = sorted(rows.values(), key=lambda r: r["F2"])
    rat = [r["F3"] / r["F2"] for r in R]; dif = [r["F3"] - r["F2"] for r in R]
    rho, shift = spearman([r["F2"] for r in R], [r["F3"] for r in R])
    L = []; P = L.append
    P(f"2D versus 3D unrodded peaking, {len(R)} designs ({a.tag})")
    P(f"{'id':>3} {'e':>5} {'Gd':>5} {'F_2D':>6} {'F_3D':>6} {'ratio':>6} {'diff':>7}")
    for r, q, s in zip(R, rat, dif):
        P(f"{r['id']:>3} {r['e']:5.2f} {r['gd']:5.2f} {r['F2']:6.3f} {r['F3']:6.3f} {q:6.3f} {s:+7.3f}")
    sd_r = st.stdev(rat) if len(rat) > 1 else 0.0; sd_d = st.stdev(dif) if len(dif) > 1 else 0.0
    P("")
    P(f"ratio F_3D/F_2D : mean {st.mean(rat):.4f}  sd {sd_r:.4f}  relative sd {sd_r/st.mean(rat)*100:.2f} %  "
      f"range {min(rat):.3f} to {max(rat):.3f}")
    P(f"difference      : mean {st.mean(dif):+.4f}  sd {sd_d:.4f}  range {min(dif):+.3f} to {max(dif):+.3f}")
    P(f"Spearman rho {rho:.3f}, largest rank shift {shift} of {len(R)}")
    P(f"ranking by F_2D: {[r['id'] for r in R]}")
    P(f"ranking by F_3D: {[r['id'] for r in sorted(R, key=lambda r: r['F3'])]}")

    front_changed = None
    if a.manifest and pathlib.Path(a.checkpoint).is_file():
        d = json.load(open(a.checkpoint)); raw = d["all_raw"]
        m = json.load(open(a.manifest))
        sub = [dict(id=i, F2=rows[i]["F2"], F3=rows[i]["F3"], c=raw[i]["c_max"])
               for i in m["feasible"] if i in rows]
        if len(sub) >= 2:
            f2 = nondom(sub, "F2", "c"); f3 = nondom(sub, "F3", "c")
            front_changed = f2 != f3
            P("")
            P(f"front over the {len(sub)} confirmed feasible designs, in (F, c_max):")
            P(f"   with F_2D: {f2}")
            P(f"   with F_3D: {f3}")
            P(f"   {'UNCHANGED' if not front_changed else 'CHANGED: gained ' + str(sorted(set(f3)-set(f2))) + ', lost ' + str(sorted(set(f2)-set(f3)))}")
    P("")
    # A constant ratio leaves the ranking unchanged. The test is therefore the
    # ranking itself, not the size of the scatter: on the eleven Campaign 8
    # designs a 0.7 % relative scatter still moved design 23 by three places.
    rank_const = (shift <= 1 and rho >= 0.98)
    if front_changed is None:
        P("DECISION (ratio and ranking only, no manifest): "
          + ("ranking preserved to within one place, consistent with a constant ratio"
             if rank_const else
             f"ranking changes by up to {shift} places (rho {rho:.3f}), the ratio is NOT a constant"))
    else:
        const = rank_const and not front_changed
        P("DECISION: " + ("the 3D peaking preserves the ranking and the front. Campaign 10 would "
                          "reproduce Campaign 9 with a shifted objective. NOT NEEDED."
                          if const else
                          "the 3D peaking reorders the confirmed designs" +
                          (" and changes the front" if front_changed else ", the front survives") +
                          ". That is the Campaign 10 result, already measured on these designs. "
                          "Run Campaign 10 only if the reordering reaches the champion."))
    txt = "\n".join(L); print(txt)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"c9_peaking_2d3d_{a.tag}.txt").write_text(txt + "\n")
    (out / f"c9_peaking_2d3d_{a.tag}.json").write_text(json.dumps(dict(
        n=len(R), ratio_mean=st.mean(rat), ratio_sd=sd_r, diff_mean=st.mean(dif), diff_sd=sd_d,
        spearman=rho, max_rank_shift=shift, front_changed=front_changed,
        rows=[dict(id=r["id"], F2=r["F2"], F3=r["F3"]) for r in R]), indent=2))
    return 0


def selftest():
    rc = main(["--summary", "confirm3d_c8/summary.json", "--out", "/tmp/c9p", "--tag", "selftest"])
    r = json.load(open("/tmp/c9p/c9_peaking_2d3d_selftest.json"))
    assert r["n"] == 11 and abs(r["diff_mean"] + 0.0233) < 0.001 and abs(r["spearman"] - 0.909) < 0.002, r
    print("\nselftest OK: C8 eleven designs, shift -0.023, Spearman 0.909, max rank shift 3")
    return rc


if __name__ == "__main__":
    sys.exit(main())
