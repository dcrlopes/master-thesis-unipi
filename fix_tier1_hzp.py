#!/usr/bin/env python3
"""Add a hot-zero-power state to tier1_coefficients.py and make it resume,
so the shutdown-margin decomposition (Doppler defect and moderator defect)
comes from separated states rather than one 600 K extrapolation. Rerunning
the same command then computes ONLY the new state."""
import py_compile, shutil
from pathlib import Path

P = Path("tier1_coefficients.py"); s = P.read_text()
if '"hzp"' in s:
    raise SystemExit("already patched.")
A = '"cold":     (300.0, 300.0, 1000.0, RHO[300.0]),'
assert s.count(A) == 1
s = s.replace(A, A + chr(10) + '        "hzp":      (580.0, 580.0, 1000.0, '
              'RHO[580.0]),')
B = "    res = {}"
assert s.count(B) == 1
s = s.replace(B, """    res = {}
    prev = out / f"tier1_idx{args.idx}.json"
    if prev.exists():
        res.update(json.loads(prev.read_text()).get("states", {}))
        print(f"resuming, {len(res)} states already done: {list(res)}")""")
C = "        for name, (fT, mT, ppm, rho) in states.items():"
assert s.count(C) == 1
s = s.replace(C, C + chr(10) + "            if name in res:" + chr(10) +
              "                continue")
D = "        boron_worth_pcm=pcm(\"boron0\", \"nominal\"),"
assert s.count(D) == 1
s = s.replace(D, D + chr(10) +
    '        doppler_defect_pcm=pcm("hzp", "nominal"),' + chr(10) +
    '        moderator_defect_pcm=pcm("cold", "hzp"),')
shutil.copy(P, "tier1_coefficients.py.hzp.bak")
P.write_text(s); py_compile.compile(str(P), doraise=True)
print("patched tier1_coefficients.py (backup .hzp.bak)")
