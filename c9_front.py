#!/usr/bin/env python3
r"""
c9_front.py
===========
Reads the Campaign 9 checkpoint and writes the manifest every later stage
of the post-analysis consumes: the feasible set, the Pareto front in
(F_dH, c_max), the two-bank subset, the designs under the boron ceiling,
and the champion. No OpenMC. Seconds.

The front is recomputed here rather than read from pareto_X, because the
archive's own front is the optimiser's view (feasible under the campaign
constraints) and the post-analysis also wants the sub-ceiling view.

Outputs (in --out, default c9_post)
  c9_front.json     ids and tables consumed by run_c9_post.sh
  c9_status.txt     the human-readable summary

Usage
  python c9_front.py --checkpoint out_c9/optimization_checkpoint.json
  python c9_front.py --selftest        (runs against out_c8 with c8 objectives)
"""
import argparse
import json
import pathlib
import sys


def nondominated(rows, keys):
    """Indices of rows non-dominated under minimisation of every key."""
    out = []
    for i, a in enumerate(rows):
        dom = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if all(b[k] <= a[k] for k in keys) and any(b[k] < a[k] for k in keys):
                dom = True
                break
        if not dom:
            out.append(i)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--out", default="c9_post")
    ap.add_argument("--ceiling", type=float, default=2763.0,
                    help="MTC boron ceiling, ppm (design 47, 12.8 MPa)")
    ap.add_argument("--objectives", nargs=2, default=["peaking", "c_max"],
                    help="two archived keys, both minimised")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    d = json.load(open(a.checkpoint))
    raw, cn = d["all_raw"], d["constraint_names"]
    k1, k2 = a.objectives
    for i, x in enumerate(raw):
        x["_id"] = i
    feas = [x for x in raw if all(x[c] <= 0 for c in cn)]
    front = [feas[i] for i in nondominated(feas, [k1, k2])]
    front.sort(key=lambda x: x[k2])
    under = [x for x in feas if x.get("c_max", 0) <= a.ceiling]
    # two-bank set: RE12 margin at the operating maximum >= the screen
    two_bank = []
    for x in feas:
        if "k_re12" in x and x["k_re12"] is not None:
            m_bol = 1e5 * (1 - x["k_re12"]) / x["k_re12"]
            m_peak = m_bol - x.get("hump_core_pcm", 0.0)
            if m_peak >= x.get("ctrl_screen_pcm", 1010.1):
                two_bank.append(x)
    champion = front[0] if front else None

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    P = lines.append
    P(f"Campaign 9 status, {len(raw)} evaluations, checkpoint {a.checkpoint}")
    P(f"objectives {d['objectives']}  constraints {cn}")
    P(f"feasible {len(feas)}  under {a.ceiling:.0f} ppm {len(under)}  front {len(front)}  two-bank {len(two_bank)}")
    P("")
    P("PARETO FRONT in (F_dH, c_max), feasible only, sorted by c_max")
    hdr = (f"{'id':>3} {'e':>5} {'Gd':>5} {'pin':>4} {'refl':>5} {'EFPD':>6} "
           f"{'F':>6} {'c_BOL':>6} {'c_max':>6} {'hump':>6} {'nB':>3} {'status':>10} {'2bk':>4}")
    P(hdr); P("-" * len(hdr))
    tb = {x["_id"] for x in two_bank}
    for x in front:
        P(f"{x['_id']:>3} {x['enrich']:5.2f} {x['gd_wt']:5.2f} {x['gd_pins_used']:4.0f} "
          f"{x['refl_thick']:5.2f} {x['cycle_length']:6.0f} {x[k1]:6.3f} "
          f"{x.get('c_bol', float('nan')):6.0f} {x[k2]:6.0f} {x.get('hump_core_pcm', 0):+6.0f} "
          f"{x.get('n_boron_solves', 0):3d} {x.get('c_max_status', '-'):>10} "
          f"{'yes' if x['_id'] in tb else 'no':>4}")
    P("")
    P("FEASIBLE, NOT ON THE FRONT, sorted by c_max")
    for x in sorted((y for y in feas if y not in front), key=lambda y: y[k2]):
        P(f"{x['_id']:>3} {x['enrich']:5.2f} {x['gd_wt']:5.2f} {x['gd_pins_used']:4.0f} "
          f"{x['refl_thick']:5.2f} {x['cycle_length']:6.0f} {x[k1]:6.3f} "
          f"{x.get('c_bol', float('nan')):6.0f} {x[k2]:6.0f} {x.get('hump_core_pcm', 0):+6.0f}")
    P("")
    for c in cn:
        v = [x[c] for x in raw]
        P(f"  {c:<12} ok {sum(1 for y in v if y<=0):2d}/{len(v)}  min {min(v):+9.3f}  max {max(v):+9.3f}")
    P("")
    P(f"hypervolume history {[round(float(h), 3) for h in d['hv_history']]}")
    if champion:
        P(f"champion (lowest c_max on the front): id {champion['_id']}, "
          f"{champion['enrich']:.2f} wt%, Gd {champion['gd_wt']:.2f} wt% on "
          f"{champion['gd_pins_used']:.0f} pins, refl {champion['refl_thick']:.2f} cm, "
          f"c_max {champion[k2]:.0f} ppm, F {champion[k1]:.3f}, EFPD {champion['cycle_length']:.0f}")
    txt = "\n".join(lines)
    (out / "c9_status.txt").write_text(txt + "\n")
    print(txt)

    man = dict(checkpoint=a.checkpoint, objectives=[k1, k2], ceiling_ppm=a.ceiling,
               n_eval=len(raw),
               feasible=[x["_id"] for x in feas],
               under_ceiling=[x["_id"] for x in under],
               front=[x["_id"] for x in front],
               two_bank=[x["_id"] for x in two_bank],
               champion=champion["_id"] if champion else None,
               front_c_bol={str(x["_id"]): x.get("c_bol") for x in front},
               front_c_max={str(x["_id"]): x.get(k2) for x in front})
    (out / "c9_front.json").write_text(json.dumps(man, indent=2))
    print(f"\nwrote {out}/c9_front.json and {out}/c9_status.txt")
    return 0


def selftest():
    # the Campaign 8 archive has the same structure with the C8 objectives:
    # cycle length is MAXIMISED there, so pass its negative as a proxy key
    d = json.load(open("out_c8/optimization_checkpoint.json"))
    for x in d["all_raw"]:
        x["neg_efpd"] = -x["cycle_length"]
    p = pathlib.Path("out_c8_selftest.json")
    p.write_text(json.dumps(d))
    try:
        rc = main(["--checkpoint", str(p), "--out", "c9_post_selftest",
                   "--objectives", "neg_efpd", "peaking"])
        m = json.loads(pathlib.Path("c9_post_selftest/c9_front.json").read_text())
        # the C8 front is {47, 42, 23, 29, 21, 44, 59, 1}
        assert set(m["front"]) == {47, 42, 23, 29, 21, 44, 59, 1}, m["front"]
        print("\nselftest OK: reproduces the Campaign 8 front", sorted(m["front"]))
    finally:
        p.unlink(missing_ok=True)
        import shutil
        shutil.rmtree("c9_post_selftest", ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
