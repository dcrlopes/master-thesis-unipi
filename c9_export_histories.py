#!/usr/bin/env python3
r"""
c9_export_histories.py -- export the assembly depletion history of every
Campaign 9 design from openmc_runs_c9/case_NNNN into one small JSON, so the
axial correction can be developed away from wks720.

Each history is checked: the campaign's own crossing function applied to it
must reproduce the archived cycle length, which confirms the reader and the
case-to-design mapping.

USAGE (repository root, on wks720)
    python c9_export_histories.py --out figs_c9_axial/c9_histories.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from c9_axial_rescore import fast_k_history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="out_c9/optimization_checkpoint.json")
    ap.add_argument("--workdir", default="openmc_runs_c9")
    ap.add_argument("--out", default="figs_c9_axial/c9_histories.json")
    a = ap.parse_args(argv)
    import reactor_model as rm
    import core_geometry as cg
    spec = rm.core_specific_power_w_per_g(rm.Operating(), rm.Geometry17x17())
    raw = json.loads(Path(a.checkpoint).read_text())["all_raw"]
    out, ok, bad = {}, 0, []
    for i, r in enumerate(raw):
        try:
            bu, k = fast_k_history(Path(a.workdir) / f"case_{i:04d}", spec)
        except Exception as exc:
            bad.append((i, f"read failed: {exc!r}")); continue
        b0 = cg.eoc_crossing_burnup(bu, k, float(r["k_target"]))
        e0 = b0 * 1000.0 / spec if b0 is not None else None
        match = e0 is not None and abs(e0 - float(r["cycle_length"])) < 0.5
        ok += match
        if not match:
            bad.append((i, f"reread {e0} against archive {r['cycle_length']}"))
        out[str(i)] = dict(bu=[float(x) for x in bu], k=[float(x) for x in k],
                           k_target=float(r["k_target"]), cycle_length=float(r["cycle_length"]),
                           enrich=float(r["enrich"]), gd_wt=float(r["gd_wt"]), reread_matches=bool(match))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(spec_power=spec, designs=out), indent=0))
    print(f"{ok} of {len(raw)} histories reproduce their archived cycle length")
    for i, why in bad:
        print(f"  C9-{i}: {why}")
    print(f"wrote {a.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
