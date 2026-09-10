#!/usr/bin/env python
"""
apply_c8_reformulation_partb.py
===============================
Three corrections to Part B of c8_reformulation_retro.py.

1. OBJECTIVE AT THE OPERATING MAXIMUM
   Part A of the same script proves that a beginning-of-life control
   screen admits designs that are not controllable at their own
   mid-cycle maximum. Part B then uses the beginning-of-life critical
   boron as an objective, which rewards exactly the designs whose
   reactivity is temporarily suppressed by unburned gadolinium. The
   objective becomes

       c_max = c_BOL + max(lf_bol * hump_pcm, 0) / w_B(e)

   using the same lift factor and hump field Part A already reads, and
   the same 400 pcm noise floor c8_hump2.py uses when it reports a hump
   as unresolved. No extra transport.

2. CONTROL SCREEN IN PART B
   The 1010.1 pcm screen is applied at the operating maximum as a third
   condition, on ARI_margin_bol_3d_pcm, the sixteen regulating-bank
   assemblies. That is k_RE, the constraint g_ctrl of Campaign 7 onward,
   and it is the screen Part A's own retrospective is about.

   The two-bank reading RE12 is NOT used as a constraint. The
   methodology chapter states that g_ctrl,12 is recorded and not
   constrained on purpose, so that the front can be split afterwards
   into designs controllable with two banks and designs needing all
   four, without shrinking the feasible region in advance. It is
   therefore carried here as a label column only.

   IMPORTANT: lf_bol, hump_pcm and the margin fields exist only in
   swing_c8, which covers eleven designs. The screen is therefore
   applied WHERE MEASURED and the remaining designs are labelled n/a
   rather than dropped. Dropping them for missing data would be a worse
   error than not screening them.

3. VALIDITY GUARD ON THE PROXY
   The power law is refitted at runtime on the designs carrying a
   measured worth. Its support is the reactivity range of those designs.
   Design 27 was reported at 381 ppm, which implies about -4412 pcm at
   the reference concentration, that is subcritical unrodded while
   delivering 1638 EFPD. Any design whose reactivity or resulting
   concentration falls outside the fitted range is now flagged.

USAGE
    python apply_c8_reformulation_partb.py --selftest
    python apply_c8_reformulation_partb.py --check
    python apply_c8_reformulation_partb.py
    python apply_c8_reformulation_partb.py --revert

--check  verifies every anchor matches exactly once and prints them,
         writing nothing.
--revert restores the newest backup this script wrote.

No command-line interface of the target script is changed, so main()
and its argument parser are not touched.
"""

import argparse
import datetime
import pathlib
import sys

TARGET = "c8_reformulation_retro.py"
MARKER = "HUMP_NOISE_PCM"

# --------------------------------------------------------------- anchor 1 --
A1_OLD = ('F_LIMIT = 1.65             # AP1000-class design limit used on '
          'the front figure\n')
A1_NEW = A1_OLD + (
    "HUMP_NOISE_PCM = 400.0     # c8_hump2.py reports smaller humps as "
    "unresolved\n")

# --------------------------------------------------------------- anchor 2 --
A2_OLD = ('    P("  objective   minimise c_BOL, the critical boron at '
          'beginning of life")\n')
A2_NEW = (
    '    P("  objective   minimise c_max, the peak critical boron over the '
    'cycle")\n'
    '    P(f"  screen      four-bank ARI margin >= {SCREEN_PCM:.1f} pcm at '
    'the operating maximum,")\n'
    '    P("              applied only to designs carrying a measured '
    'margin")\n'
    '    P("  recorded    two-bank RE12 margin, a label and not a '
    'constraint")\n')

# --------------------------------------------------------------- anchor 3 --
A3_OLD = """        c_bol = PPM_REF + rho_pcm(kc) / (a * e ** b)
        cand.append((i, f, c_bol, cyc, e, r.get("gd_wt"), r.get("gd_pins_used")))
"""
A3_NEW = """        rho_i = rho_pcm(kc)
        w_i = a * e ** b
        c_bol = PPM_REF + rho_i / w_i

        # objective at the operating maximum, not at beginning of life
        sr = (swing or {}).get(str(i)) or {}
        lf, h = sr.get("lf_bol"), sr.get("hump_pcm")
        resolved = (lf is not None and h is not None and h >= HUMP_NOISE_PCM)
        if resolved:
            c_obj = c_bol + (lf * h) / w_i
            corr = "hump"
        elif lf is not None and h is not None:
            c_obj, corr = c_bol, "flat"
        else:
            c_obj, corr = c_bol, "n/a"

        # control screen at the operating maximum, four banks, where measured
        drop = max(lf * h, 0.0) if resolved else 0.0
        m16 = sr.get("ARI_margin_bol_3d_pcm")
        if m16 is None:
            scr = "n/a"
        else:
            if m16 - drop < SCREEN_PCM:
                n_screened += 1
                continue
            scr = "pass"

        # two-bank reading, RECORDED and not constrained, per the
        # methodology chapter: the front is split afterwards, not shrunk
        m8 = sr.get("RE12_margin_bol_3d_pcm")
        re12 = "n/a" if m8 is None else ("2bk" if m8 - drop >= SCREEN_PCM
                                         else "4bk")

        # validity guard: is the proxy being used inside its support?
        flag = ""
        if rho_i < rho_lo or rho_i > rho_hi:
            flag = "!rho"
            n_outside += 1
        elif c_obj < c_lo or c_obj > c_hi:
            flag = "!ppm"
            n_outside += 1

        cand.append((i, f, c_obj, cyc, e, r.get("gd_wt"),
                     r.get("gd_pins_used"), c_bol, corr, scr, re12, flag))
"""

# --------------------------------------------------------------- anchor 4 --
A4_OLD = """    # build the candidate set
    cand = []
"""
A4_NEW = """    # support of the refitted proxy, for the validity guard
    rho_fit = [rho_pcm(raw[i]["keff_core_bol"]) for i in ids]
    c_fit = [PPM_REF + rho_pcm(raw[i]["keff_core_bol"])
             / (a * raw[i]["enrich"] ** b) for i in ids]
    rho_lo, rho_hi = min(rho_fit), max(rho_fit)
    c_lo, c_hi = min(c_fit), max(c_fit)
    P(f"  proxy support: rho_BOL {rho_lo:.0f} to {rho_hi:.0f} pcm, "
      f"c_BOL {c_lo:.0f} to {c_hi:.0f} ppm")
    P("  values outside that range are flagged and must not be quoted")
    P("")

    # build the candidate set
    cand = []
    n_screened = 0
    n_outside = 0
"""

# --------------------------------------------------------------- anchor 5 --
A5_OLD = """    P(f"  designs satisfying both constraints: {len(cand)} of {len(raw)}")
"""
A5_NEW = """    P(f"  designs satisfying both constraints: "
      f"{len(cand) + n_screened} of {len(raw)}")
    P(f"  rejected by the control screen at the operating maximum: "
      f"{n_screened}")
    P(f"  flagged outside the proxy support: {n_outside}")
"""

# --------------------------------------------------------------- anchor 6 --
A6_OLD = """    P(f"  {'idx':>4} {'e wt%':>6} {'Gd wt%':>7} {'pins':>5} {'EFPD':>6} "
      f"{'F_dH':>6} {'c_BOL':>7}  status")
    for i, f, c, cyc, e, gd, pins in sorted(cand, key=lambda t: t[2]):
        tag = "FRONT" if i in front else "dominated"
        gd_s = f"{gd:7.2f}" if gd is not None else "    n/a"
        pn_s = f"{pins:5.0f}" if pins is not None else "  n/a"
        P(f"  {i:>4} {e:>6.2f} {gd_s} {pn_s} {cyc:>6.0f} {f:>6.3f} "
          f"{c:>7.0f}  {tag}")
"""
A6_NEW = """    P(f"  {'idx':>4} {'e wt%':>6} {'Gd wt%':>7} {'pins':>5} {'EFPD':>6} "
      f"{'F_dH':>6} {'c_BOL':>7} {'c_max':>7} {'corr':>5} {'ctrl':>5} "
      f"{'RE12':>5} {'flag':>5}  status")
    for (i, f, c, cyc, e, gd, pins, c_bol, corr, scr, re12, flag) in sorted(
            cand, key=lambda t: t[2]):
        tag = "FRONT" if i in front else "dominated"
        gd_s = f"{gd:7.2f}" if gd is not None else "    n/a"
        pn_s = f"{pins:5.0f}" if pins is not None else "  n/a"
        P(f"  {i:>4} {e:>6.2f} {gd_s} {pn_s} {cyc:>6.0f} {f:>6.3f} "
          f"{c_bol:>7.0f} {c:>7.0f} {corr:>5} {scr:>5} {re12:>5} "
          f"{flag:>5}  {tag}")
    P("")
    P("  ctrl is the four-bank screen at the operating maximum, the")
    P("  constraint. RE12 is the two-bank reading, recorded and not")
    P("  constrained: 2bk means the design is controllable with RE1 and")
    P("  RE2 alone, 4bk means it needs all four banks.")
"""

ANCHORS = [(A1_OLD, A1_NEW), (A2_OLD, A2_NEW), (A3_OLD, A3_NEW),
           (A4_OLD, A4_NEW), (A5_OLD, A5_NEW), (A6_OLD, A6_NEW)]


def transform(text):
    for n, (old, new) in enumerate(ANCHORS, 1):
        c = text.count(old)
        if c != 1:
            raise SystemExit(f"anchor {n} matches {c} times, expected 1")
        text = text.replace(old, new)
    return text


def selftest():
    sample = ("head\n" + A1_OLD + "mid1\n" + A2_OLD + "mid2\n" + A4_OLD
              + "mid3\n" + A3_OLD + "mid4\n" + A5_OLD + "mid5\n" + A6_OLD
              + "tail\n")
    out = transform(sample)
    assert MARKER in out, "constant not inserted"
    assert "c_max, the peak critical boron" in out, "objective text missing"
    assert "n_screened += 1" in out, "screen not inserted"
    assert "proxy support" in out, "guard not inserted"
    assert out.count("c_bol = PPM_REF + rho_i / w_i") == 1, "c_bol lost"
    for tok in ("head", "mid1", "mid2", "mid3", "mid4", "mid5", "tail"):
        assert tok in out, f"surrounding text {tok} damaged"
    # the screen must be skipped, not failed, when no margin was measured
    assert 'if m16 is None:' in out and 'scr = "n/a"' in out
    # the four-bank margin is the constraint
    assert 'ARI_margin_bol_3d_pcm' in out, "four-bank margin not used"
    assert 'if m16 - drop < SCREEN_PCM' in out, "screen not on four banks"
    # the two-bank reading must be recorded, never a reason to drop a design
    assert 'RE12_margin_bol_3d_pcm' in out, "two-bank label missing"
    i8 = out.index('m8 = sr.get("RE12_margin_bol_3d_pcm")')
    tail = out[i8:i8 + 400]
    assert "continue" not in tail, "two-bank reading must not drop a design"
    # ordering: the guard block must precede the candidate loop it feeds
    assert out.index("rho_lo, rho_hi") < out.index("if rho_i < rho_lo")
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--path", default=TARGET)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    p = pathlib.Path(a.path)

    if a.revert:
        b = sorted(p.parent.glob(p.name + ".bak_*"))
        if not b:
            raise SystemExit("no backup found")
        p.write_text(b[-1].read_text(encoding="utf-8"), encoding="utf-8")
        print("restored from " + b[-1].name)
        return 0

    if not p.exists():
        raise SystemExit("not found: " + str(p))
    text = p.read_text(encoding="utf-8")

    if MARKER in text:
        print("already applied, nothing to do")
        return 0

    if a.check:
        ok = True
        for n, (old, _) in enumerate(ANCHORS, 1):
            c = text.count(old)
            print(f"anchor {n}: {c} match(es)")
            if c != 1:
                ok = False
            else:
                print("  " + old.strip().splitlines()[0][:70])
        print("ALL ANCHORS UNIQUE" if ok else "ANCHOR PROBLEM, do not apply")
        print("nothing written")
        return 0 if ok else 1

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = p.with_name(p.name + ".bak_" + stamp)
    bak.write_text(text, encoding="utf-8")
    p.write_text(transform(text), encoding="utf-8")
    print(f"wrote {p}, backup {bak.name}")
    print(f"review with: git diff -- {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
