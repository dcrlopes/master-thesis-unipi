#!/usr/bin/env python3
"""
repair_c4_cli.py -- restores the Campaign-4 command-line wiring that was lost
when run_optimization.py was overwritten with a copy built from the `main`
branch (which never had it), and then committed.

WHAT WAS LOST vs WHAT SURVIVED
  survived : the per-iteration checkpoint wiring (opt.checkpoint_path /
             opt.checkpoint_meta) and the meta dict that references
             args.core_particles ...
  lost     : the three argparse definitions those references depend on, the
             evaluator constructor kwargs, and the --resume guard on core
             transport settings.

  The result is a file that fails on "--core-particles" as an unrecognized
  argument AND would fail on args.core_particles even without the flags.

  reactor_optimization.py, openmc_evaluator.py and core_geometry.py are NOT
  affected: reactor_optimization.py is identical on both branches, and the
  other two were never overwritten.

WHAT THIS SCRIPT ADDS BACK
  1. --core-particles / --core-batches / --core-inactive (defaults
     100000 / 170 / 60, the settings measured in the core_rescore screen)
  2. core_particles= / core_batches= / core_inactive= on the OpenMCEvaluator
     construction
  3. the --resume guard: core transport settings must match the checkpoint,
     mirroring the existing guard on assembly transport settings, because
     every evaluation sharing a checkpoint must use identical fidelity

USAGE
    cd ~/master-thesis-unipi
    python3 repair_c4_cli.py --check     # verify, change nothing
    python3 repair_c4_cli.py             # apply (.bak written)
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--runner", default="run_optimization.py")
args = ap.parse_args()

p = Path(args.runner)
if not p.exists():
    sys.exit(f"ERROR: {p} not found. Run from the repository root.")
s = p.read_text()

# --------------------------------------------------------------------------- #
# what is present / missing                                                    #
# --------------------------------------------------------------------------- #
has_flags = '"--core-particles"' in s
has_ctor = "core_particles=args.core_particles" in s
has_guard = 'prev_meta.get("core_transport")' in s
has_meta = "args.core_particles" in s

print(f"  argparse --core-* flags : {'present' if has_flags else 'MISSING'}")
print(f"  evaluator kwargs        : {'present' if has_ctor else 'MISSING'}")
print(f"  --resume core guard     : {'present' if has_guard else 'MISSING'}")
print(f"  checkpoint meta wiring  : {'present' if has_meta else 'MISSING'}")

if has_flags and has_ctor and has_guard:
    sys.exit("\nNothing to repair.")

A_PART = re.search(r'^(\s*)(ap|p|parser)\.add_argument\(\s*"--particles"', s, re.M)
A_CTOR = """    ev = OpenMCEvaluator(spec, k_target=k_target_arg, transport=transport,
                         workdir="openmc_runs", **schedule)"""
A_GUARD = """        prev_tr = prev_meta.get("transport")"""

bad = False
for name, cond in (("--particles anchor", A_PART is not None),
                   ("evaluator construction", s.count(A_CTOR) == 1),
                   ("resume guard anchor", s.count(A_GUARD) == 1)):
    print(f"  {'OK ' if cond else 'BAD'} {name}")
    bad |= not cond
if bad:
    sys.exit("\nERROR: anchors do not match. Stop and inspect by hand.")
print("all anchors matched.")
if args.check:
    sys.exit(0)

# --------------------------------------------------------------------------- #
if not has_flags:
    indent, obj = A_PART.group(1), A_PART.group(2)
    ins = (f'{indent}# CAMPAIGN 4: transport settings for the core-BOL '
           f'peaking solve\n'
           f'{indent}{obj}.add_argument("--core-particles", type=int, '
           f'default=100000)\n'
           f'{indent}{obj}.add_argument("--core-batches", type=int, '
           f'default=170)\n'
           f'{indent}{obj}.add_argument("--core-inactive", type=int, '
           f'default=60)\n')
    s = s[:A_PART.start()] + ins + s[A_PART.start():]

if not has_ctor:
    s = s.replace(A_CTOR, """    ev = OpenMCEvaluator(spec, k_target=k_target_arg, transport=transport,
                         core_particles=args.core_particles,
                         core_batches=args.core_batches,
                         core_inactive=args.core_inactive,
                         workdir="openmc_runs", **schedule)""")

if not has_guard:
    s = s.replace(A_GUARD, """        prev_core = prev_meta.get("core_transport")
        cur_core = {"particles": args.core_particles,
                    "batches": args.core_batches,
                    "inactive": args.core_inactive}
        if prev_core is not None and dict(prev_core) != cur_core:
            raise SystemExit(
                "core transport settings differ from the checkpoint: "
                f"{prev_core} vs {cur_core}. Every evaluation sharing a "
                "checkpoint must use identical core settings.")
        prev_tr = prev_meta.get("transport")""")

shutil.copy(p, p.with_suffix(".py.bak"))
p.write_text(s)
print(f"\nwritten: {p}   (backup: {p.with_suffix('.py.bak')})")
print("""
verify (all four must appear):
  python3 -c "import ast; ast.parse(open('run_optimization.py').read()); print('parse OK')"
  grep -n 'core-particles\\|core_particles=args\\|prev_core\\|checkpoint_path' run_optimization.py
""")
