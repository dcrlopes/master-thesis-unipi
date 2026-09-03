#!/usr/bin/env python3
"""
apply_campaign4.py -- applies the Campaign-4 change (core-BOL peaking as the
in-loop objective) to openmc_evaluator.py and run_optimization.py.

WHY THIS EXISTS
  The Campaign-4 change is small but touches four places in two files. Doing
  it by hand risks a silent mismatch; this script asserts that every anchor
  string appears EXACTLY ONCE before it edits anything, writes .bak copies,
  and refuses to run twice (it detects its own marker).

WHAT IT CHANGES
  openmc_evaluator.py
    1. _design_seed(design)            -> _design_seed(design, salt="")
       so the core solve gets its own reproducible stream.
    2. new method _bol_core_peaking()  -> core model + pin-resolved mesh over
       the whole 6x6-minus-corners footprint, masked max/mean, plus a
       Shannon-entropy source-convergence sentinel (the finite core, unlike a
       reflective single assembly, has a REAL spatial transient).
    3. evaluate(): "peaking" and "g_peak" now carry the CORE value; the
       assembly value is retained as "peaking_asm" (diagnostic + training
       data for the bridge model), and keff_core_bol / core_entropy_conv are
       recorded for free.
    4. __init__: three new kwargs core_particles / core_batches /
       core_inactive (defaults 100000 / 170 / 60, i.e. the settings used in
       the core_rescore screen).
  run_optimization.py
    5. --core-particles / --core-batches / --core-inactive CLI flags, passed
       to the evaluator and recorded in the checkpoint metadata.

USAGE
    cd ~/master-thesis-unipi
    python3 apply_campaign4.py            # edit in place, .bak written
    python3 apply_campaign4.py --check    # verify only, change nothing
"""
import argparse
import shutil
import sys
from pathlib import Path

MARKER = "_bol_core_peaking"

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true",
                help="verify anchors and report; make no changes")
ap.add_argument("--evaluator", default="openmc_evaluator.py")
ap.add_argument("--runner", default="run_optimization.py")
args = ap.parse_args()

ev_p, ro_p = Path(args.evaluator), Path(args.runner)
for p in (ev_p, ro_p):
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Run this from the repository root.")

ev, ro = ev_p.read_text(), ro_p.read_text()

if MARKER in ev:
    sys.exit("Already applied (found _bol_core_peaking). Nothing to do.")

# --------------------------------------------------------------------------- #
# anchors -- every one must appear exactly once                                #
# --------------------------------------------------------------------------- #
A_SEED_DEF = 'def _design_seed(design: dict) -> int:'
A_SEED_BODY = """    key = _json.dumps({k: round(float(v), 10)
                       for k, v in sorted(design.items())})
    return 1 + _zlib.crc32(key.encode()) % 2_000_000_000"""
A_INIT_SIG = """                 max_burnup: float = DEFAULT_MAX_BURNUP,
                 verbose: bool = True):"""
A_INIT_BODY = """        self.verbose = verbose"""
A_EVAL = """        peaking = self._bol_peaking(design, case)
        (cycle_efpd, k_bol, k_target_used,
         censored, bu_eoc, n_solves) = self._cycle_length(design, case)"""
A_RES = """            "peaking":      peaking,                    # objective (minimise)"""
A_GPEAK = """            "g_peak":  peaking - 2.0,                   # peaking <= 2.0"""
A_CARRY = """            "k_bol":   k_bol,                           # carried for plots"""
A_BOLPEAK_DEF = "    def _bol_peaking(self, design: dict, case: Path) -> float:"

checks = [("_design_seed def", A_SEED_DEF, ev), ("_design_seed body", A_SEED_BODY, ev),
          ("__init__ signature", A_INIT_SIG, ev), ("__init__ tail", A_INIT_BODY, ev),
          ("evaluate() head", A_EVAL, ev), ("result peaking", A_RES, ev),
          ("result g_peak", A_GPEAK, ev), ("result k_bol", A_CARRY, ev),
          ("_bol_peaking def", A_BOLPEAK_DEF, ev)]
ok = True
for name, anchor, text in checks:
    n = text.count(anchor)
    print(f"  {'OK ' if n == 1 else 'BAD'} {name}: found {n}x")
    ok &= (n == 1)
if not ok:
    sys.exit("\nERROR: anchors do not match this file version. Stop and ask "
             "before editing by hand.")
print("all anchors matched.")
if args.check:
    sys.exit(0)

# --------------------------------------------------------------------------- #
# 1. salted seed                                                               #
# --------------------------------------------------------------------------- #
ev = ev.replace(A_SEED_DEF, 'def _design_seed(design: dict, salt: str = "") -> int:')
ev = ev.replace(A_SEED_BODY, """    key = _json.dumps({k: round(float(v), 10)
                       for k, v in sorted(design.items())}) + salt
    return 1 + _zlib.crc32(key.encode()) % 2_000_000_000""")

# --------------------------------------------------------------------------- #
# 2. __init__ kwargs                                                           #
# --------------------------------------------------------------------------- #
ev = ev.replace(A_INIT_SIG, """                 max_burnup: float = DEFAULT_MAX_BURNUP,
                 core_particles: int = 100000,
                 core_batches: int = 170,
                 core_inactive: int = 60,
                 verbose: bool = True):""")
ev = ev.replace(A_INIT_BODY, """        # CAMPAIGN 4: transport settings for the core-BOL peaking solve.
        # Defaults are the core_rescore screen settings; the inactive count is
        # deliberately generous because the finite core has a real source
        # transient (one screened design converged only at batch 94).
        self.core_particles = int(core_particles)
        self.core_batches = int(core_batches)
        self.core_inactive = int(core_inactive)

        self.verbose = verbose""")

# --------------------------------------------------------------------------- #
# 3. new method, inserted directly before _bol_peaking                         #
# --------------------------------------------------------------------------- #
NEW_METHOD = '''    # ------------------------------------------------------------------ #
    # CAMPAIGN 4: BOL radial peaking on the FULL 32-assembly CORE.        #
    #                                                                    #
    # Campaign 3 optimised the single-assembly proxy. The core_rescore    #
    # study (36 feasible designs, 3-8 seeds each) measured                #
    # Spearman rho(assembly, core) = +0.89 GLOBALLY but a scrambled       #
    # ordering inside the near-optimal set: assembly ranks 1/2/3 fell to  #
    # core ranks 11/9/12, while assembly #14 rose to core #4. A linear    #
    # bridge correction F_core/F_asm = 1.400 - 0.0045 refl + 0.343 pitch  #
    # explains only R^2 = 0.746 -- too weak to optimise through. One      #
    # extra BOL transport solve (~2 min of a ~110 min evaluation, ~3%)    #
    # measures the true quantity instead of correcting a proxy.           #
    # ------------------------------------------------------------------ #
    def _bol_core_peaking(self, design: dict, case: Path) -> dict:
        m = rm.make_core_model(design, self.op, self.geo,
                               particles=self.core_particles,
                               batches=self.core_batches,
                               inactive=self.core_inactive)
        model = m[0] if isinstance(m, tuple) else m
        model.settings.seed = _design_seed(design, salt="core")

        N = self.geo.lattice
        pitch = design.get("pitch", 1.26)
        try:
            nx = ny = cg.CORE_MAP_32.shape[0]
        except Exception:
            nx = ny = 6
        half = nx * N * pitch / 2.0

        mesh = openmc.RegularMesh()
        mesh.dimension = (nx * N, ny * N)
        mesh.lower_left = (-half, -half)
        mesh.upper_right = (half, half)
        t = openmc.Tally(name="core_pin_fission")
        t.filters = [openmc.MeshFilter(mesh)]
        t.scores = ["fission"]
        model.tallies = openmc.Tallies([t])

        cdir = case / "core_bol"
        cdir.mkdir(parents=True, exist_ok=True)
        sp_path = model.run(cwd=str(cdir), output=False)
        with openmc.StatePoint(sp_path) as sp:
            keff = float(sp.keff.nominal_value)
            v = sp.get_tally(name="core_pin_fission").get_values(
                scores=["fission"]).reshape(ny * N, nx * N)
            H = np.asarray(getattr(sp, "entropy", []), dtype=float)

        # mask guide tubes, removed corners and reflector (all zero-fission)
        f = np.ma.masked_equal(v, 0.0)
        fdh = float((f / f.mean()).max())

        # source-convergence sentinel
        conv = None
        if H.size:
            tail = H[self.core_inactive + (len(H) - self.core_inactive) // 2:]
            mu, sd = float(tail.mean()), float(tail.std(ddof=1))
            Hs = np.convolve(H, np.ones(3) / 3.0, mode="same")
            Hs[0], Hs[-1] = H[0], H[-1]
            bad = np.where(~((Hs >= mu - 3 * sd) & (Hs <= mu + 3 * sd)))[0]
            conv = int(bad[-1]) + 2 if len(bad) else 1
        return dict(fdh_core=fdh, keff_core=keff, entropy_conv=conv)

'''
ev = ev.replace(A_BOLPEAK_DEF, NEW_METHOD + A_BOLPEAK_DEF)

# --------------------------------------------------------------------------- #
# 4. evaluate(): swap the objective                                            #
# --------------------------------------------------------------------------- #
ev = ev.replace(A_EVAL, """        peaking = self._bol_peaking(design, case)      # assembly (diagnostic)
        core = self._bol_core_peaking(design, case)   # CAMPAIGN 4 objective
        (cycle_efpd, k_bol, k_target_used,
         censored, bu_eoc, n_solves) = self._cycle_length(design, case)""")
ev = ev.replace(A_RES,
                '            "peaking":      core["fdh_core"],           '
                '# objective (minimise): CORE F_dh')
ev = ev.replace(A_GPEAK,
                '            "g_peak":  core["fdh_core"] - 2.0,          '
                '# CORE peaking <= 2.0')
ev = ev.replace(A_CARRY, """            "peaking_asm": peaking,       # assembly F_dh: diagnostic and
                                          # training data for the bridge model
            "keff_core_bol": core["keff_core"],   # free Route-B closure check
            "core_entropy_conv": core["entropy_conv"],  # flag if > inactive
            "k_bol":   k_bol,                           # carried for plots""")

# --------------------------------------------------------------------------- #
# 5. run_optimization.py: CLI flags + metadata                                 #
# --------------------------------------------------------------------------- #
import re

# 5a. CLI flags, inserted next to the existing --particles flag
m = re.search(r'^(\s*)(ap|p|parser)\.add_argument\(\s*["\']--particles["\']',
              ro, re.M)
if not m:
    sys.exit(f"ERROR: could not find the --particles argument in {ro_p}.")
indent, obj = m.group(1), m.group(2)
ins = (f'{indent}# CAMPAIGN 4: transport settings for the core-BOL '
       f'peaking solve\n'
       f'{indent}{obj}.add_argument("--core-particles", type=int, '
       f'default=100000)\n'
       f'{indent}{obj}.add_argument("--core-batches", type=int, '
       f'default=170)\n'
       f'{indent}{obj}.add_argument("--core-inactive", type=int, '
       f'default=60)\n')
ro = ro[:m.start()] + ins + ro[m.start():]

# 5b. pass the three settings into the evaluator constructor
A_CTOR = """    ev = OpenMCEvaluator(spec, k_target=k_target_arg, transport=transport,
                         workdir="openmc_runs", **schedule)"""
if ro.count(A_CTOR) != 1:
    sys.exit(f"ERROR: OpenMCEvaluator(...) construction not found verbatim in "
             f"{ro_p} (found {ro.count(A_CTOR)}x). Wire the three "
             "core_* kwargs by hand.")
ro = ro.replace(A_CTOR, """    ev = OpenMCEvaluator(spec, k_target=k_target_arg, transport=transport,
                         core_particles=args.core_particles,
                         core_batches=args.core_batches,
                         core_inactive=args.core_inactive,
                         workdir="openmc_runs", **schedule)""")

# 5c. record the core settings and the objective definition in the checkpoint
A_META = """                               meta={"k_target": k_target_arg,
                                     "smoke": bool(args.smoke),
                                     "transport": dict(transport),"""
if ro.count(A_META) != 1:
    sys.exit(f"ERROR: checkpoint meta dict not found verbatim in {ro_p} "
             f"(found {ro.count(A_META)}x). Add core_transport by hand.")
ro = ro.replace(A_META, """                               meta={"k_target": k_target_arg,
                                     "smoke": bool(args.smoke),
                                     "transport": dict(transport),
                                     "core_transport": {
                                         "particles": args.core_particles,
                                         "batches": args.core_batches,
                                         "inactive": args.core_inactive},
                                     "objective_def":
                                         "peaking = core BOL F_dh "
                                         "(Campaign 4)",""")

# 5d. extend the --resume guard so a core-settings mismatch is caught too,
#     exactly as the existing guard does for the assembly transport settings
A_GUARD = """        prev_tr = prev_meta.get("transport")"""
if ro.count(A_GUARD) == 1:
    ro = ro.replace(A_GUARD, """        prev_core = prev_meta.get("core_transport")
        cur_core = {"particles": args.core_particles,
                    "batches": args.core_batches,
                    "inactive": args.core_inactive}
        if prev_core is not None and dict(prev_core) != cur_core:
            raise SystemExit(
                "core transport settings differ from the checkpoint: "
                f"{prev_core} vs {cur_core}. Every evaluation sharing a "
                "checkpoint must use identical core settings.")
        prev_tr = prev_meta.get("transport")""")
else:
    print("\n!! note: --resume guard anchor not found; core settings will not "
          "be checked on resume. Keep the --core-* flags identical by hand.")

if not args.check:
    shutil.copy(ev_p, ev_p.with_suffix(".py.bak"))
    shutil.copy(ro_p, ro_p.with_suffix(".py.bak"))
    ev_p.write_text(ev)
    ro_p.write_text(ro)
    print(f"\nwritten: {ev_p} and {ro_p}   (backups: *.py.bak)")
    print("""
Applied in full -- no manual editing required:
  evaluator : salted seed, _bol_core_peaking(), objective + g_peak swapped
              to CORE F_dh, assembly value kept as peaking_asm
  runner    : --core-particles / --core-batches / --core-inactive flags,
              wired into the evaluator, recorded in the checkpoint meta,
              and guarded on --resume

Next:
  conda activate openmc-env
  python -c "import openmc_evaluator, run_optimization; print('import OK')"
""")
