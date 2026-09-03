#!/usr/bin/env python3
"""derive_kmax.py -- turn measured control-bank worth into a DERIVED upper
reactivity bound, replacing the historical k_max = 1.35.

LOGIC
-----
A design is controllable by a given bank group when that group alone holds
it subcritical by the operating margin. In reactivity terms

    rho_excess  <=  rho_bank - margin

and since rho_excess = (1 - 1/k0) * 1e5, the largest admissible k0 is

    k_max_ctrl = 1 / (1 - (rho_bank_min - margin) * 1e-5)

The conservative choice is the MINIMUM worth over the screened designs.
Every term is measured, so no adopted number survives in the bound.

WHICH GROUP TO DERIVE FROM
--------------------------
    ALLRE   the four regulating banks, 16 CRAs (default). Operational
            control without chemical shim, the criterion of this study.
    RE12    RE1 + RE2 only, 8 CRAs. Much stricter.
    SCRAM   all seven banks, 32 CRAs. Closest to the licensing shutdown
            criterion, though a true shutdown margin also requires the
            highest-worth cluster stuck out and the cold, xenon-free state.

INPUT
-----
The JSON written by rod_bank_worth.py --screen. Handles both layouts:
  v3 nested   state["states"]["ALLRE"]["worth_pcm"|"margin_pcm"]
  v1 flat     state["re_worth_pcm"], state["margin_pcm"]

USAGE
    python derive_kmax.py banks_screen_v3/banks_B4C.json --margin 1000
    python derive_kmax.py banks_screen_v3/banks_B4C.json --group SCRAM
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def pull(state: dict, group: str):
    """(worth_pcm, margin_pcm) for one screened design, either layout."""
    st = state.get("states")
    if isinstance(st, dict):
        if group not in st:
            raise SystemExit(
                f"design {state.get('idx')} has no '{group}' state. "
                f"Available: {sorted(st)}. Re-run the screen with "
                f"--screen-states including {group}, or pass --group.")
        return float(st[group]["worth_pcm"]), float(st[group]["margin_pcm"])
    if "re_worth_pcm" in state:            # v1 flat layout, ALLRE only
        if group != "ALLRE":
            raise SystemExit("this file is the older flat format and holds "
                             "only the ALL-RE state, so --group must be "
                             "ALLRE.")
        return float(state["re_worth_pcm"]), float(state["margin_pcm"])
    raise SystemExit(f"unrecognised record for design {state.get('idx')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("screen_json")
    ap.add_argument("--group", default="ALLRE",
                    choices=["RE12", "ALLRE", "SCRAM"],
                    help="bank group the bound is derived from "
                         "(default ALLRE)")
    ap.add_argument("--margin", type=float, default=1000.0,
                    help="required subcriticality, in pcm of reactivity "
                         "(default 1000)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default ctrl_kmax_<group>.json)")
    a = ap.parse_args()
    out = Path(a.out or f"ctrl_kmax_{a.group}.json")

    data = json.loads(Path(a.screen_json).read_text())
    rows = [s for s in data.get("states", []) if s.get("mode") == "screen"]
    if not rows:
        raise SystemExit("no screen states in this file. Run "
                         "rod_bank_worth.py --screen first.")

    recs = []
    for s in rows:
        worth, margin = pull(s, a.group)
        recs.append(dict(idx=int(s["idx"]), k0=float(s["k0"]),
                         excess=float(s["excess_pcm"]),
                         worth=worth, margin=margin))
    recs.sort(key=lambda r: r["worth"])

    print(f"bank group: {a.group}   margin: {a.margin:.0f} pcm   "
          f"designs: {len(recs)}")
    print(f"{'idx':>4} {'k0':>9} {'excess':>9} {'worth':>9} {'margin':>9}"
          f"  verdict")
    for r in recs:
        print(f"{r['idx']:>4} {r['k0']:>9.5f} {r['excess']:>9.0f} "
              f"{r['worth']:>9.0f} {r['margin']:>9.0f}  "
              f"{'ok' if r['margin'] >= a.margin else 'NO'}")

    worths = [r["worth"] for r in recs]
    w_min = min(worths)
    w_med = worths[len(worths) // 2]
    for tag, w in (("conservative (minimum worth)", w_min),
                   ("median worth, for context", w_med)):
        allow = w - a.margin
        kmax = 1.0 / (1.0 - allow * 1e-5)
        print(f"\n{tag}: rho = {w:.0f} pcm, margin = {a.margin:.0f} pcm"
              f"\n  controllable excess = {allow:.0f} pcm  ->  "
              f"k_max_ctrl = {kmax:.4f}")

    allow = w_min - a.margin
    kmax = 1.0 / (1.0 - allow * 1e-5)
    out.write_text(json.dumps(dict(
        source=str(a.screen_json), group=a.group, n_designs=len(recs),
        margin_pcm=a.margin, worth_min_pcm=w_min, worth_median_pcm=w_med,
        controllable_excess_pcm=allow, k_max_ctrl=kmax,
        designs=recs), indent=2))
    print(f"\nwrote {out}")
    print(f"Campaign 7: --k-basis core --k-max {kmax:.3f} as a prescreen, "
          f"with --ctrl-margin {a.margin:.0f} as the measured truth "
          f"criterion. The bound applies the worst case to every design; "
          f"g_ctrl measures each one.")


if __name__ == "__main__":
    main()
