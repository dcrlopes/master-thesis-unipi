#!/usr/bin/env python3
"""mtc_extra_lattices.py -- rows of the lattices scanned after the logarithmic
fit of eq:mtc-ceiling-fit (C9-69, C9-70, C9-27), for the open markers of
c9_ceiling_two_pressures.py. All three are read from mtc_front_table.py
output, the method of the 8 lattices of the fit.

usage (repository root):
    python mtc_front_table.py --checkpoint out_c9a/optimization_checkpoint.json \
        --glob 'mtc_c9a_d69_p128_core3d' --pressure 12.8 --out c9_post/d69_p128
    (same for d69_p155, d70_p128, d70_p155 with out_c9a, and d27_p128,
     d27_p155 with out_c9 and --glob 'mtc_c9_d27_p..._core3d')
    python c9_post/mtc_extra_lattices.py
"""
import json
from pathlib import Path

for tag in ("p128", "p155"):
    rows = []
    for d in ("d69", "d70", "d27"):
        rows += json.loads(Path(f"c9_post/{d}_{tag}/mtc_ceiling_table.json").read_text())
    Path(f"c9_post/mtc_extra_{tag}.json").write_text(json.dumps(rows, indent=1))
    print(tag, [(r["idx"], round(r["ceiling"]), round(r["sigma"])) for r in rows])
