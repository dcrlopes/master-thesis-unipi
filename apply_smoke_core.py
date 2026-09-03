#!/usr/bin/env python3
"""
apply_smoke_core.py -- make --smoke cheap on the CORE solves too.

THE DEFECT
----------
--smoke already lowers the DEPLETION transport (800 x 30, 10 inactive) and
shortens the burnup schedule, but the core-class solves keep the production
defaults of --core-particles 100000, --core-batches 170, --core-inactive 60.

In Campaign 7 that meant two full-fidelity core solves per smoke design
(unrodded + ALL-RE). In Campaign 8 it is three (unrodded + ALL-RE + RE1+RE2),
so an unmodified smoke run spends roughly 18 core-class solves at about
100 s each, some 30 minutes, on solves whose only job is to prove the
wiring. The smoke test stops being a smoke test.

THE FIX
-------
Inside the --smoke branch, lower the core fidelity to 8000 x 60 with 20
inactive batches, but ONLY for the flags the user did not type. Detection is
exact: the flag string is looked up in sys.argv rather than compared against
the argparse default, so passing --core-particles 100000 --smoke still gives
you a full-fidelity smoke run when you want one.

Because every downstream consumer reads args.core_particles (the evaluator
construction, the two-fidelity rescore block and the meta record), mutating
args here propagates everywhere, and the checkpoint honestly records the
fidelity that was actually used.

Nothing outside the --smoke branch is touched, so production runs are
bit-for-bit unchanged.

USAGE
    python3 apply_smoke_core.py --check     report, change nothing
    python3 apply_smoke_core.py            apply, writing the .bak
    python3 apply_smoke_core.py --revert   restore from the .bak
    python3 apply_smoke_core.py --selftest compile + behaviour checks
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path

SMOKE_CORE = dict(core_particles=8000, core_batches=60, core_inactive=20)

EDITS = []

EDITS.append(dict(
    path="run_optimization.py",
    name="S1 smoke lowers the core-solve fidelity",
    old='''        print(">>> SMOKE TEST <<<")
''',
    new='''        # CAMPAIGN 8: --smoke lowered only the DEPLETION transport, so the
        # core-class solves still ran at 100000 x 170. With three of them per
        # design (unrodded, ALL-RE, RE1+RE2) a smoke run cost about half an
        # hour on solves that exist only to prove the wiring. Lower them here,
        # but ONLY for flags the user did not type, so an explicit
        # --core-particles still wins. Detection is exact (sys.argv), not a
        # comparison against the argparse default.
        _smoke_core = {"--core-particles": ("core_particles", 8000),
                       "--core-batches":   ("core_batches", 60),
                       "--core-inactive":  ("core_inactive", 20)}
        _typed = " ".join(_sys.argv[1:])
        for _flag, (_attr, _val) in _smoke_core.items():
            if _flag not in _typed:
                setattr(args, _attr, _val)
        print(">>> SMOKE TEST <<<")
        print(f"    core solves at {args.core_particles} x "
              f"{args.core_batches} ({args.core_inactive} inactive); "
              f"pass --core-particles to override")
''',
))

EDITS.append(dict(
    path="run_optimization.py",
    name="S2 import sys",
    old='''import argparse
import platform
''',
    new='''import argparse
import platform
import sys as _sys
''',
))


def files():
    return sorted({e["path"] for e in EDITS})


def check(apply: bool) -> int:
    ok = True
    for e in EDITS:
        text = Path(e["path"]).read_text()
        n_old, n_new = text.count(e["old"]), text.count(e["new"])
        inside = n_new * e["new"].count(e["old"])
        n_old_outside = n_old - inside
        state = ("APPLIED" if (n_new == 1 and n_old_outside == 0) else
                 "ready" if (n_new == 0 and n_old == 1) else
                 f"FAIL (old x{n_old_outside} outside, new x{n_new})")
        print(f"  {e['name']:<45s} {state}")
        if state.startswith("FAIL"):
            ok = False
    if not ok:
        print("Anchors are not unique. The file changed since this patch was "
              "written. Apply the intent by hand.")
        return 2
    if not apply:
        return 0
    for p in files():
        shutil.copy2(p, p + ".bak.smoke")
        print(f"  backup -> {p}.bak.smoke")
    for e in EDITS:
        p = Path(e["path"])
        t = p.read_text()
        if e["old"] in t:
            p.write_text(t.replace(e["old"], e["new"], 1))
            print(f"  applied {e['name']}")
    return 0


def revert() -> int:
    rc = 0
    for p in files():
        b = Path(p + ".bak.smoke")
        if b.exists():
            shutil.copy2(b, p)
            print(f"  reverted {p}")
        else:
            print(f"  MISSING {b}, cannot revert {p}")
            rc = 2
    return rc


def selftest() -> int:
    print("selftest:")
    src = Path("run_optimization.py").read_text()
    py_compile.compile("run_optimization.py", doraise=True)
    print("  compiles: run_optimization.py")
    if "import sys as _sys" not in src:
        print("  FAIL: sys not imported")
        return 2
    # the override must live INSIDE the smoke branch, above the else
    i_smoke = src.index('_smoke_core = {"--core-particles"')
    i_else = src.index("        # 24 + n_iter*6 real evaluations per session")
    if not i_smoke < i_else:
        print("  FAIL: the override is not inside the --smoke branch")
        return 2
    print("  override sits inside the --smoke branch")
    # the production defaults must be untouched
    for flag, val in (("--core-particles", 100000),
                      ("--core-batches", 170),
                      ("--core-inactive", 60)):
        if f'ap.add_argument("{flag}", type=int, default={val})' not in src:
            print(f"  FAIL: production default for {flag} changed")
            return 2
    print("  production defaults unchanged: 100000 x 170, 60 inactive")
    # behaviour of the flag-detection rule, exercised standalone
    rule = {"--core-particles": ("core_particles", 8000)}
    for argv, expect in ((["--smoke"], 8000),
                         (["--smoke", "--core-particles", "100000"], 100000)):
        class A:
            core_particles = 100000
        a = A()
        typed = " ".join(argv)
        for flag, (attr, val) in rule.items():
            if flag not in typed:
                setattr(a, attr, val)
        if a.core_particles != expect:
            print(f"  FAIL: argv {argv} gave {a.core_particles}, "
                  f"expected {expect}")
            return 2
    print("  flag detection: bare --smoke lowers, explicit flag wins")
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    print(f"python  : {sys.version.split()[0]}")
    print(f"cwd     : {Path.cwd()}")
    print("OpenMC required: no (string edits, compile and behaviour checks)")
    missing = [p for p in files() if not Path(p).exists()]
    if missing:
        print(f"FAIL: not in the repository root, missing {missing}")
        return 2
    if a.revert:
        return revert()
    if a.selftest:
        return selftest()
    rc = check(apply=not a.check)
    if rc == 0 and not a.check:
        rc = selftest()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
