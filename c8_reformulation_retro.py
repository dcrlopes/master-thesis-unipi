#!/usr/bin/env python3
"""
c8_reformulation_retro.py
=========================
What would the optimiser have done under a different formulation?

No transport. Reads archives only. Seconds to run.

MOTIVATION
----------
Two defects of the Campaign 7 and Campaign 8 formulation are visible only
after the fact.

  1. The control screen g_ctrl is evaluated at beginning of life. The
     gadolinium hump appears later, so a design can pass the screen and
     still be uncontrollable over the cycle. The evaluator already builds
     the whole k history inside _cycle_length and discards everything
     except k_hist[0] and the end-of-cycle crossing, so the hump costs
     nothing extra to use.

  2. Cycle length was an objective, yet every confirmed design exceeds
     five years at capacity factor one. Peaking was an objective, yet
     every confirmed design falls in a narrow band. Neither discriminates.
     Meanwhile the quantity that actually separates the designs, the
     reactivity the control system must hold, was never in the
     formulation at all, so gadolinia was seen only as a cost.

PART A, hump-aware control screen
---------------------------------
The margins are recomputed at the operating maximum instead of at
beginning of life,

    margin_peak = margin_bol - max(LF_bol * hump, 0)

and the designs whose verdict changes are named. Campaign 7 carries 58
recovered histories and is the meaningful sample. Campaign 8 carries the
eleven designs of the working set.

PART B, reformulated objectives
-------------------------------
Cycle length and peaking become constraints, and the objectives become

    minimise  F_dH            the unrodded hot channel factor
    minimise  c_BOL           the critical boron concentration at BOL

subject to a mission cycle length and a peaking limit. The Pareto set of
that problem is computed over the whole 60-design Campaign 8 archive and
compared with the original front.

c_BOL is not measured for all 60 designs. It is estimated as

    c_BOL = 1000 + rho_core_BOL / w_B(e),     w_B(e) = a * e**b

where the power law is FITTED IN THIS SCRIPT from the designs that do
carry a measured differential worth, and the fit residual is reported so
the proxy can be judged rather than trusted. On the eleven measured
designs the proxy reproduces the swing report to within a few ppm.

WHY c_BOL IS AN OBJECTIVE AND NOT A CONSTRAINT
----------------------------------------------
A constraint needs a bound. The bound here is the concentration at which
the moderator temperature coefficient turns positive, and that has not
been computed for this lattice. Guessing it would silently delete designs.
An objective needs no bound, and once the coefficient is known the limit
is drawn as a line across the resulting front. The line can move later
without rerunning anything.

USAGE (from the repository root, conda env openmc-env not required)
    python c8_reformulation_retro.py --selftest
    python c8_reformulation_retro.py
    python c8_reformulation_retro.py --mission-efpd 1461 --f-limit 1.65
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from pathlib import Path

SCREEN_PCM = 1010.1        # k <= 0.99
PPM_REF = 1000.0           # campaign reference boron concentration
MISSION_EFPD = 1826.0      # five years at capacity factor 1.0
F_LIMIT = 1.65             # AP1000-class design limit used on the front figure


# ------------------------------------------------------------------ helpers --
def rho_pcm(k):
    """Reactivity in pcm from a multiplication factor."""
    return 1e5 * (k - 1.0) / k


def margin_pcm(k):
    """Subcriticality margin in pcm. Positive when k < 1."""
    return 1e5 * (1.0 - k) / k


def fit_power_law(e, w):
    """Least squares fit of w = a * e**b in log space.

    Returns (a, b, rms_rel) where rms_rel is the relative root mean square
    residual on w, so the caller can judge the proxy.
    """
    x = [math.log(v) for v in e]
    y = [math.log(v) for v in w]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    a = math.exp(my - b * mx)
    res = [(a * ei ** b) / wi - 1.0 for ei, wi in zip(e, w)]
    return a, b, math.sqrt(sum(r * r for r in res) / n)


def pareto_min(points):
    """Indices of the non-dominated set for simultaneous minimisation.

    points is a list of (key, (f1, f2)). A point is dominated when another
    is no worse in both objectives and strictly better in at least one.
    """
    keep = []
    for k, (a1, a2) in points:
        dominated = any(
            (b1 <= a1 and b2 <= a2) and (b1 < a1 or b2 < a2)
            for j, (b1, b2) in points if j != k)
        if not dominated:
            keep.append(k)
    return keep


def selftest():
    assert abs(margin_pcm(0.99) - SCREEN_PCM) < 0.5
    assert abs(rho_pcm(1.01) - 990.1) < 0.5

    a, b, rms = fit_power_law([2.0, 4.0, 8.0], [8.0, 4.0, 2.0])
    assert abs(b + 1.0) < 1e-9 and abs(a - 16.0) < 1e-9 and rms < 1e-12

    # a strictly better point dominates, a trade-off point does not
    pts = [("a", (1.0, 5.0)), ("b", (2.0, 4.0)), ("c", (3.0, 9.0))]
    assert sorted(pareto_min(pts)) == ["a", "b"]

    # duplicate points must both survive rather than annihilate each other
    pts = [("a", (1.0, 1.0)), ("b", (1.0, 1.0))]
    assert sorted(pareto_min(pts)) == ["a", "b"]

    # hump-aware screen: a positive hump can cross the screen, a negative
    # hump must leave the margin untouched
    assert 2990.0 - max(1.08 * 2000.0, 0.0) < SCREEN_PCM
    assert 1330.0 - max(1.08 * -1600.0, 0.0) == 1330.0
    print("selftest OK")
    return 0


def preflight(paths):
    print("=" * 74)
    print(" PREFLIGHT")
    print("=" * 74)
    print(f"  host        : {socket.gethostname()}")
    print(f"  cwd         : {os.getcwd()}")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  conda env   : {os.environ.get('CONDA_DEFAULT_ENV', 'none')}")
    print("  no transport required, openmc not imported")
    missing = [p for p in paths if not Path(p).is_file()]
    for p in paths:
        print(f"  {'MISSING' if p in missing else 'found  '} : {p}")
    if paths[0] in missing:
        sys.exit(f"FAIL: {paths[0]} is required. Run from ~/master-thesis-unipi.")
    print("preflight OK\n")
    return missing


# ------------------------------------------------------- part A, the screen --
def part_a(P, khist_c8, kh_c7, swing, screen):
    P("=" * 74)
    P("PART A. Control screen evaluated at the operating maximum")
    P("=" * 74)
    P("  margin_peak = margin_bol - max(LF_bol * hump, 0)")
    P(f"  screen {screen:.1f} pcm, equivalent to k <= 0.99")
    P("")

    # --- Campaign 7, 58 recovered histories -------------------------------
    if kh_c7:
        rows = []
        for key, r in kh_c7.items():
            if r.get("status") != "OK":
                continue
            k_allre, k_bol, kc = r.get("k_allre"), r.get("k_bol"), r.get("keff_core_bol")
            hump = r.get("hump_pcm")
            if None in (k_allre, k_bol, kc, hump):
                continue
            lf = k_bol / kc
            m_bol = margin_pcm(k_allre)
            m_pk = m_bol - max(lf * hump, 0.0)
            rows.append((int(key), r.get("gd_wt"), r.get("pins"), hump, m_bol, m_pk))
        rows.sort()

        pass_bol = [r[0] for r in rows if r[4] >= screen]
        pass_pk = [r[0] for r in rows if r[5] >= screen]
        lost = sorted(set(pass_bol) - set(pass_pk))

        P(f"CAMPAIGN 7, {len(rows)} designs with a validated history")
        P(f"  pass at beginning of life      : {len(pass_bol)} of {len(rows)}")
        P(f"  pass at the operating maximum  : {len(pass_pk)} of {len(rows)}")
        P(f"  admitted by the BOL screen and NOT controllable over the cycle:")
        P(f"    {lost if lost else 'none'}   ({len(lost)} designs)")
        if lost:
            P("")
            P(f"  {'idx':>4} {'Gd wt%':>7} {'pins':>5} {'hump':>7} "
              f"{'m_BOL':>8} {'m_peak':>8}")
            for idx, gd, pins, h, mb, mp in rows:
                if idx in lost:
                    gd_s = f"{gd:7.2f}" if gd is not None else "    n/a"
                    pn_s = f"{pins:5.0f}" if pins is not None else "  n/a"
                    P(f"  {idx:>4} {gd_s} {pn_s} {h:>+7.0f} {mb:>8.0f} {mp:>8.0f}")
        P("")
        P("  Every design in that list was accepted as feasible by the campaign")
        P("  and is not controllable at its own mid-cycle maximum. This is the")
        P("  cost of a beginning-of-life screen, measured rather than asserted.")
    else:
        P("CAMPAIGN 7: kh_c7/k_histories.json not found, section skipped")
    P("")

    # --- Campaign 8 working set -------------------------------------------
    if swing and khist_c8:
        P("CAMPAIGN 8, eleven-design working set")
        P(f"  {'idx':>4} {'Gd wt%':>7} {'pins':>5} {'hump ss':>8} {'hump arc':>9} "
          f"{'RE12 bol':>9} {'RE12 pk':>8}")
        flip = []
        for key in sorted(swing, key=int):
            r = swing[key]
            lf, h = r.get("lf_bol"), r.get("hump_pcm")
            mb = r.get("RE12_margin_bol_3d_pcm")
            if None in (lf, h, mb):
                continue
            mp = mb - max(lf * h, 0.0)
            an = (khist_c8.get(key) or {}).get("analysis", {})
            ha = an.get("hump_vs_bol_pcm")
            P(f"  {int(key):>4} {r.get('gd_wt', float('nan')):>7.2f} "
              f"{r.get('gd_pins', float('nan')):>5.0f} {h:>+8.0f} "
              f"{(f'{ha:+9.0f}' if ha is not None else '      n/a')} "
              f"{mb:>9.0f} {mp:>8.0f}")
            if (mb >= screen) != (mp >= screen):
                flip.append(int(key))
        P(f"  verdict changed by the hump: {flip if flip else 'none'}")
    P("")


# ------------------------------------------- part B, the reformulated front --
def part_b(P, raw, swing, mission, f_limit):
    P("=" * 74)
    P("PART B. Reformulated problem on the 60-design Campaign 8 archive")
    P("=" * 74)
    P(f"  constraint  cycle length >= {mission:.0f} EFPD")
    P(f"  constraint  F_dH <= {f_limit:.2f}")
    P("  objective   minimise F_dH")
    P("  objective   minimise c_BOL, the critical boron at beginning of life")
    P("")

    # fit the boron worth law on the designs that carry a measured worth
    e_m, w_m, ids = [], [], []
    for key, r in (swing or {}).items():
        w = r.get("w_1000_1500")
        if w and w > 0:
            i = int(key)
            e_m.append(raw[i]["enrich"])
            w_m.append(w)
            ids.append(i)
    if len(e_m) < 3:
        P("  FAIL: fewer than three measured boron worths, cannot fit the proxy")
        return None
    a, b, rms = fit_power_law(e_m, w_m)
    P(f"  boron worth law fitted on {len(e_m)} measured designs:")
    P(f"    w_B(e) = {a:.3f} * e ** ({b:.4f})   pcm per ppm")
    P(f"    relative rms residual {100 * rms:.1f} per cent")

    # validate the proxy against the swing report where c_BOL was measured
    err = []
    for i in ids:
        r = swing[str(i)]
        if r.get("c_bol_ppm"):
            c_hat = PPM_REF + rho_pcm(raw[i]["keff_core_bol"]) / (a * raw[i]["enrich"] ** b)
            err.append(abs(c_hat - r["c_bol_ppm"]))
    if err:
        P(f"    proxy against the measured c_BOL: max error "
          f"{max(err):.0f} ppm over {len(err)} designs")
    P("")

    # build the candidate set
    cand = []
    for i, r in enumerate(raw):
        cyc, f, kc, e = (r.get("cycle_length"), r.get("peaking"),
                         r.get("keff_core_bol"), r.get("enrich"))
        if None in (cyc, f, kc, e) or kc <= 0:
            continue
        if r.get("censored"):
            continue
        if cyc < mission or f > f_limit:
            continue
        c_bol = PPM_REF + rho_pcm(kc) / (a * e ** b)
        cand.append((i, f, c_bol, cyc, e, r.get("gd_wt"), r.get("gd_pins_used")))

    P(f"  designs satisfying both constraints: {len(cand)} of {len(raw)}")
    if not cand:
        P("  the constraint pair is empty, relax the mission or the limit")
        return None

    front = pareto_min([(i, (f, c)) for i, f, c, *_ in cand])
    P(f"  Pareto set of the reformulated problem: {sorted(front)}")
    P("")
    P(f"  {'idx':>4} {'e wt%':>6} {'Gd wt%':>7} {'pins':>5} {'EFPD':>6} "
      f"{'F_dH':>6} {'c_BOL':>7}  status")
    for i, f, c, cyc, e, gd, pins in sorted(cand, key=lambda t: t[2]):
        tag = "FRONT" if i in front else "dominated"
        gd_s = f"{gd:7.2f}" if gd is not None else "    n/a"
        pn_s = f"{pins:5.0f}" if pins is not None else "  n/a"
        P(f"  {i:>4} {e:>6.2f} {gd_s} {pn_s} {cyc:>6.0f} {f:>6.3f} "
          f"{c:>7.0f}  {tag}")
    P("")
    P("  Read the gadolinia columns down the c_BOL ordering. The designs at")
    P("  the low-boron end are the ones carrying more gadolinia, which is the")
    P("  reward the original two-objective formulation never offered.")
    return front


# ------------------------------------------------------------------- driver --
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c8/optimization_checkpoint.json")
    ap.add_argument("--swing", default="swing_c8/swing.json")
    ap.add_argument("--khist-c8", default="khist_c8/khist.json")
    ap.add_argument("--kh-c7", default="kh_c7/k_histories.json")
    ap.add_argument("--mission-efpd", type=float, default=MISSION_EFPD,
                    help="cycle-length constraint, default 1826, five years at "
                         "capacity factor one")
    ap.add_argument("--f-limit", type=float, default=F_LIMIT,
                    help="peaking constraint, default 1.65")
    ap.add_argument("--screen", type=float, default=SCREEN_PCM)
    ap.add_argument("--out", default="reformulation_c8")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    preflight([a.checkpoint, a.swing, a.khist_c8, a.kh_c7])

    def load(p):
        try:
            return json.loads(Path(p).read_text())
        except Exception:
            return None

    raw = json.loads(Path(a.checkpoint).read_text())["all_raw"]
    swing, khist_c8, kh_c7 = load(a.swing), load(a.khist_c8), load(a.kh_c7)

    L = []
    P = L.append
    P("=== Campaign 8 retrospective: what a different formulation would have "
      "produced ===")
    P("No transport. Archives only.")
    P("")
    part_a(P, khist_c8, kh_c7, swing, a.screen)
    front = part_b(P, raw, swing, a.mission_efpd, a.f_limit)

    txt = "\n".join(L)
    print(txt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(txt + "\n", encoding="utf-8")
    (out / "front.json").write_text(json.dumps(
        dict(mission_efpd=a.mission_efpd, f_limit=a.f_limit,
             screen_pcm=a.screen, reformulated_front=sorted(front or [])),
        indent=1), encoding="utf-8")
    print(f"\nwrote {out}/report.txt and {out}/front.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
