#!/usr/bin/env python3
"""derive_kmax.py -- suggestion 2: turn the measured regulating-bank worth
into a DERIVED upper reactivity bound, replacing the historical 1.35.

LOGIC
-----
A design is operationally controllable when the four regulating banks alone
hold it subcritical by the operating margin, because the SH banks are
reserved for scram. In reactivity terms

    rho_excess  <=  rho_RE - margin

and since rho_excess = (1 - 1/k0) * 1e5, the largest admissible k0 is

    k_max_ctrl = 1 / (1 - (rho_RE_min - margin) * 1e-5)

The conservative choice for rho_RE is the MINIMUM worth over the screened
designs, because bank worth varies with the design (design 71 proved by how
much). Every term is measured: no adopted numbers remain in the bound.

INPUT
-----
The JSON written by rod_bank_worth.py --screen (v2 banks, ALL-RE = the
complete inner sixteen). Reads every state with mode == "screen".

USAGE
    python derive_kmax.py banks_screen/banks_B4C.json --margin 1000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("screen_json")
    ap.add_argument("--margin", type=float, default=1000.0,
                    help="operating subcriticality margin under ALL-RE, "
                         "in pcm of reactivity (default 1000)")
    ap.add_argument("--out", default="ctrl_kmax.json")
    a = ap.parse_args()

    data = json.loads(Path(a.screen_json).read_text())
    rows = [s for s in data.get("states", []) if s.get("mode") == "screen"]
    if not rows:
        raise SystemExit("no screen states in this file. Run "
                         "rod_bank_worth.py --screen first.")

    print(f"{'idx':>4} {'k0':>8} {'rho_RE (pcm)':>13} {'excess':>8} "
          f"{'margin':>8}  verdict")
    for s in sorted(rows, key=lambda r: r["re_worth_pcm"]):
        print(f"{s['idx']:>4} {s['k0']:>8.5f} {s['re_worth_pcm']:>13.0f} "
              f"{s['excess_pcm']:>8.0f} {s['margin_pcm']:>8.0f}  "
              f"{s['verdict']}")

    rho_min = min(s["re_worth_pcm"] for s in rows)
    rho_med = sorted(s["re_worth_pcm"] for s in rows)[len(rows) // 2]
    for tag, rho in (("conservative (min worth)", rho_min),
                     ("median worth, for context", rho_med)):
        allow = rho - a.margin
        kmax = 1.0 / (1.0 - allow * 1e-5)
        print(f"\n{tag}: rho_RE = {rho:.0f} pcm, margin = {a.margin:.0f} "
              f"pcm\n  controllable excess = {allow:.0f} pcm  ->  "
              f"k_max_ctrl = {kmax:.4f}")

    allow = rho_min - a.margin
    result = dict(source=str(a.screen_json), n_designs=len(rows),
                  margin_pcm=a.margin, rho_re_min_pcm=rho_min,
                  rho_re_median_pcm=rho_med,
                  k_max_ctrl=1.0 / (1.0 - allow * 1e-5),
                  designs=[dict(idx=s["idx"], k0=s["k0"],
                                rho_re_pcm=s["re_worth_pcm"])
                           for s in rows])
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {a.out}")
    print("Use as: --k-basis core --k-max <k_max_ctrl> in Campaign 7, or "
          "prefer the direct g_ctrl constraint (--ctrl-margin), which "
          "measures each design instead of applying the worst case to all.")


if __name__ == "__main__":
    main()
