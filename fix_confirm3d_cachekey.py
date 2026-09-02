#!/usr/bin/env python3
"""
fix_confirm3d_cachekey.py -- three corrections, anchor-verified.

  P1  confirm3d.py: the runs.json cache key omits the transport fidelity, so a
      --smoke solve (5000 x 40) is reused by a later production run under the
      same key. This happened to design 53 ARO seed 0 in confirm3d_c8. The key
      gains |particles|batches|inactive|boron_ppm.
  P2  confirm3d.py: new --boron-ppm flag (default 1000.0, the campaign value)
      so the 3D margins can be measured at 0 ppm instead of extrapolated.
  P3  boron_worth.py: LaTeX header M_{2} -> M_{8}. The subscript is the number
      of rodded assemblies (16 under ALL-RE, 8 under RE1+RE2), not the number
      of banks.
  M   confirm3d_c8/runs.json: existing keys are migrated to the new format so
      the production solves are still found. Smoke entries (sd > 1.5e-3 or
      wall < 60 s) are tagged with the smoke fidelity and therefore no longer
      collide with production.

USAGE (on wks720, branch campaign8, working tree clean):
  python fix_confirm3d_cachekey.py --check     # report state, change nothing
  python fix_confirm3d_cachekey.py --apply     # patch + migrate, .bak written
  python fix_confirm3d_cachekey.py --revert    # restore all .bak files

Flags:
  --particles/--batches/--inactive  the production fidelity the existing
        confirm3d_c8 solves were run with. Defaults are the script defaults
        (150000, 200, 80). Confirm against the launch command before --apply.
  --out  the confirm3d output directory holding runs.json (default confirm3d_c8)
"""
import argparse, json, shutil, sys
from pathlib import Path

# (file, old, new) exact anchors. Each old must occur exactly once.
PATCHES = [
    ("confirm3d.py",
     'key = f"{idx}|{st}|{mode}|{s}|{a.refl_override}|{a.refl_steel_vol}|{a.cr_abs_radius}|{a.rod_stack}|{a.no_parked_rods}"',
     'key = (f"{idx}|{st}|{mode}|{s}|{a.refl_override}|{a.refl_steel_vol}|{a.cr_abs_radius}|{a.rod_stack}|{a.no_parked_rods}"\n'
     '                           f"|{fid[\'particles\']}|{fid[\'batches\']}|{fid[\'inactive\']}|{a.boron_ppm}")'),
    ("confirm3d.py",
     '    ap.add_argument("--out", default="confirm3d_c8")',
     '    ap.add_argument("--boron-ppm", type=float, default=1000.0,\n'
     '                    help="soluble boron of every solve, campaign 1000.0. Use a separate --out per value")\n'
     '    ap.add_argument("--out", default="confirm3d_c8")'),
    ("confirm3d.py",
     '    geo, op = rm.Geometry17x17(), rm.Operating()',
     '    geo, op = rm.Geometry17x17(), rm.Operating()\n'
     '    op.boron_ppm = float(a.boron_ppm)   # Operating is a mutable dataclass\n'
     '    print(f"boron  : {op.boron_ppm:.1f} ppm in every solve")'),
    ("boron_worth.py",
     '"$M_{16}(1000)$ & $M_{16}(0)$ & $M_{2}(1000)$ & $M_{2}(0)$ & boron share',
     '"$M_{16}(1000)$ & $M_{16}(0)$ & $M_{8}(1000)$ & $M_{8}(0)$ & boron share'),
]

def count(text, s):
    return text.count(s)

def check(a):
    ok = True
    for f, old, new in PATCHES:
        p = Path(f)
        if not p.exists():
            print(f"  {f}: MISSING"); ok = False; continue
        t = p.read_text()
        n_old, n_new = count(t, old), count(t, new)
        # an anchor that survives inside its own replacement is discounted
        if n_new == 1 and n_old == count(new, old):
            state = "applied"
        elif n_old == 1 and n_new == 0:
            state = "pending"
        else:
            state = f"UNEXPECTED (old x{n_old}, new x{n_new})"; ok = False
        print(f"  {f}: {state}")
    rj = Path(a.out) / "runs.json"
    if rj.exists():
        d = json.loads(rj.read_text())
        nfld = {k.count("|") for k in d}
        smoke = [k for k, r in d.items() if r["sd"] > 1.5e-3 or r["wall_s"] < 60]
        print(f"  {rj}: {len(d)} entries, key fields {sorted(nfld)} "
              f"({'migrated' if nfld == {12} else 'old format' if nfld == {8} else 'MIXED'}), "
              f"smoke-like entries: {smoke if smoke else 'none'}")
    else:
        print(f"  {rj}: not found (nothing to migrate)")
    return ok

def apply(a):
    for f, old, new in PATCHES:
        p = Path(f); t = p.read_text()
        if count(t, new) == 1 and count(t, old) == count(new, old):
            print(f"  {f}: already applied, skipped"); continue
        if count(t, old) != 1:
            print(f"  {f}: expected exactly one occurrence of the anchor, found {count(t, old)}. ABORT"); return 1
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists(): shutil.copy2(p, bak)
        p.write_text(t.replace(old, new, 1)); print(f"  {f}: patched ({bak.name} written)")
    rj = Path(a.out) / "runs.json"
    if rj.exists():
        d = json.loads(rj.read_text())
        if {k.count("|") for k in d} == {8}:
            shutil.copy2(rj, rj.with_suffix(".json.bak"))
            prod = f"|{a.particles}|{a.batches}|{a.inactive}|{a.boron_ppm}"
            smk = f"|5000|40|15|{a.boron_ppm}"
            nd, ns = {}, 0
            for k, r in d.items():
                is_smoke = r["sd"] > 1.5e-3 or r["wall_s"] < 60
                nd[k + (smk if is_smoke else prod)] = r; ns += is_smoke
            rj.write_text(json.dumps(nd, indent=1))
            print(f"  {rj}: migrated {len(nd)} keys, {ns} tagged as smoke (runs.json.bak written)")
        else:
            print(f"  {rj}: not in the old 8-field format, left untouched")
    return 0

def revert(a):
    for f in {p[0] for p in PATCHES}:
        bak = Path(f + ".bak")
        if bak.exists(): shutil.copy2(bak, f); print(f"  {f}: restored from {bak.name}")
    rj = Path(a.out) / "runs.json"; bak = rj.with_suffix(".json.bak")
    if bak.exists(): shutil.copy2(bak, rj); print(f"  {rj}: restored")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true"); g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    ap.add_argument("--out", default="confirm3d_c8")
    ap.add_argument("--particles", type=int, default=150000); ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--inactive", type=int, default=80); ap.add_argument("--boron-ppm", type=float, default=1000.0)
    a = ap.parse_args()
    if a.check: sys.exit(0 if check(a) else 1)
    if a.apply:
        print("state before:"); check(a); print("applying:"); rc = apply(a)
        print("state after:"); check(a); sys.exit(rc)
    if a.revert: sys.exit(revert(a))
