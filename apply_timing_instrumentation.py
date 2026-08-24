#!/usr/bin/env python
"""
apply_timing_instrumentation.py -- record wall-clock cost INSIDE the campaign,
so that Campaign 5 never needs a reconstruction.

Same pattern as apply_campaign4.py and fix_infeasible_infill.py: every edit is
anchored on an exact, unique text fragment of the CURRENT campaign4-branch
files, the anchors are all verified before anything is written, and each file
gets a .bak copy first. Run from the repository root:

    python apply_timing_instrumentation.py --check     # verify anchors only
    python apply_timing_instrumentation.py             # apply
    python apply_timing_instrumentation.py --no-workdir   # skip patch 3c

What it changes
  1. openmc_evaluator.py
     a. imports time and datetime
     b. evaluate_one() times the three phases and stores, per design,
        t_eval_s, t_asm_bol_s, t_core_bol_s, t_deplete_s, t_start_utc
        in the raw record (hence in the checkpoint and the results JSON)
     c. the per-case console line prints the evaluation minutes
  2. reactor_optimization.py
     a. ActiveLearningMOO gets a phase_log list
     b. run() times the DOE evaluation and, per iteration, the GP fit, the
        NSGA-II search, the acquisition, the truth evaluation, the
        hypervolume and the checkpoint write, and prints a one-line budget
     c. save_checkpoint() persists phase_log, load_checkpoint() restores it
  3. run_optimization.py
     a. checkpoint meta records host, cpu_count, openmc_version, started_utc
     b. the FINAL checkpoint write reuses the same meta as the per-iteration
        write (the current final write drops core_transport and
        objective_def, so a resume silently skips the core-settings check)
     c. --workdir flag (default openmc_runs), recorded in meta. This was the
        pending item before Campaign 5.

Nothing changes in the physics, the seeds, the acquisition or the transport
settings. The extra keys in the raw records are ignored by _seed_from_raw(),
so old and new checkpoints remain mutually resumable.

After applying: run the smoke test once
    python run_optimization.py --smoke --ktarget-table ktarget_table.json --out smoke_t
and check that smoke_t/optimization_checkpoint.json contains "phase_log" and
that each all_raw entry has "t_eval_s".
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# (anchor, replacement) pairs per file. Each anchor must occur EXACTLY once.  #
# --------------------------------------------------------------------------- #
EVALUATOR = "openmc_evaluator.py"
OPTIMIZER = "reactor_optimization.py"
DRIVER = "run_optimization.py"

# Each entry is (name, anchor, replacement) or (name, anchor, replacement,
# already_signature). When the anchor is absent but the signature is already in
# the file, that change exists in the repository (apply_campaign5.py made some
# of them) and the patch is reported as satisfied instead of failing.
PATCHES: dict[str, list[tuple]] = {
    EVALUATOR: [
        ("1a imports",
         "import math\nimport os\nfrom collections import Counter\n",
         "import math\nimport os\nimport time\nfrom collections import Counter\n"
         "from datetime import datetime, timezone\n"),
        ("1b phase timers",
         "        peaking = self._bol_peaking(design, case)      # assembly (diagnostic)\n"
         "        core = self._bol_core_peaking(design, case)   # CAMPAIGN 4 objective\n"
         "        (cycle_efpd, k_bol, k_target_used,\n"
         "         censored, bu_eoc, n_solves) = self._cycle_length(design, case)\n",
         "        # wall-clock instrumentation: one timer per transport phase, so the\n"
         "        # cost tables of the thesis come from the archive, not from a\n"
         "        # statepoint reconstruction.\n"
         "        t_wall0 = time.time()\n"
         "        t0 = time.perf_counter()\n"
         "        peaking = self._bol_peaking(design, case)      # assembly (diagnostic)\n"
         "        t_asm = time.perf_counter() - t0\n"
         "        t0 = time.perf_counter()\n"
         "        core = self._bol_core_peaking(design, case)   # CAMPAIGN 4 objective\n"
         "        t_core = time.perf_counter() - t0\n"
         "        t0 = time.perf_counter()\n"
         "        (cycle_efpd, k_bol, k_target_used,\n"
         "         censored, bu_eoc, n_solves) = self._cycle_length(design, case)\n"
         "        t_dep = time.perf_counter() - t0\n"),
        ("1b raw keys",
         "            \"n_dep_solves\": n_solves,     # transport solves spent on depletion\n"
         "        }\n",
         "            \"n_dep_solves\": n_solves,     # transport solves spent on depletion\n"
         "            # wall-clock cost of this evaluation [s] and its phases\n"
         "            \"t_eval_s\":     t_asm + t_core + t_dep,\n"
         "            \"t_asm_bol_s\":  t_asm,\n"
         "            \"t_core_bol_s\": t_core,\n"
         "            \"t_deplete_s\":  t_dep,\n"
         "            \"t_start_utc\":  datetime.fromtimestamp(\n"
         "                t_wall0, timezone.utc).isoformat(timespec=\"seconds\"),\n"
         "        }\n"),
        ("1c console line",
         "                  f\"[{n_solves} solves]\")\n",
         "                  f\"[{n_solves} solves, \"\n"
         "                  f\"{(t_asm + t_core + t_dep) / 60.0:.1f} min]\")\n"),
    ],
    OPTIMIZER: [
        ("2a phase_log attribute",
         "        self.history = []          # hypervolume per iteration\n",
         "        self.history = []          # hypervolume per iteration\n"
         "        self.phase_log: list[dict] = []   # wall time per phase, per iteration\n"),
        ("2b DOE timing",
         "            F0, G0, raw0 = self.evaluator.evaluate(X0)\n"
         "            self._add(X0, F0, G0, raw0)\n"
         "            self.history.append(self._hv())\n",
         "            t_ev0 = time.perf_counter()\n"
         "            F0, G0, raw0 = self.evaluator.evaluate(X0)\n"
         "            t_ev = time.perf_counter() - t_ev0\n"
         "            self._add(X0, F0, G0, raw0)\n"
         "            t_hv0 = time.perf_counter()\n"
         "            self.history.append(self._hv())\n"
         "            self.phase_log.append(dict(\n"
         "                stage=\"DOE\", iteration=0, n_eval=int(len(X0)),\n"
         "                t_eval_s=t_ev, t_hv_s=time.perf_counter() - t_hv0))\n"),
        ("2b fit + NSGA timing",
         "        for it in range(self.cfg.n_iter):\n"
         "            obj_sur = self._new_surrogate().fit(self.X, self.F)\n"
         "            con_sur = (self._new_surrogate().fit(self.X, self.G)\n"
         "                       if self.spec.n_constr else None)\n"
         "\n"
         "            # NSGA-II on the surrogate (cheap):\n"
         "            prob = _SurrogateProblem(self.spec, obj_sur, con_sur)\n"
         "            algo = NSGA2(pop_size=self.cfg.nsga_pop, sampling=LHS())\n"
         "            res = minimize(prob, algo,\n"
         "                           (\"n_gen\", self.cfg.nsga_gen),\n"
         "                           seed=self.cfg.seed + it, verbose=False)\n"
         "            cand = self._least_infeasible_candidates(res)\n",
         "        for it in range(self.cfg.n_iter):\n"
         "            t_fit0 = time.perf_counter()\n"
         "            obj_sur = self._new_surrogate().fit(self.X, self.F)\n"
         "            con_sur = (self._new_surrogate().fit(self.X, self.G)\n"
         "                       if self.spec.n_constr else None)\n"
         "            t_fit = time.perf_counter() - t_fit0\n"
         "\n"
         "            # NSGA-II on the surrogate (cheap):\n"
         "            t_nsga0 = time.perf_counter()\n"
         "            prob = _SurrogateProblem(self.spec, obj_sur, con_sur)\n"
         "            algo = NSGA2(pop_size=self.cfg.nsga_pop, sampling=LHS())\n"
         "            res = minimize(prob, algo,\n"
         "                           (\"n_gen\", self.cfg.nsga_gen),\n"
         "                           seed=self.cfg.seed + it, verbose=False)\n"
         "            t_nsga = time.perf_counter() - t_nsga0\n"
         "            t_acq0 = time.perf_counter()\n"
         "            cand = self._least_infeasible_candidates(res)\n"),
        ("2b evaluation + HV timing",
         "            Xinf = np.array(chosen)\n"
         "\n"
         "            # evaluate the infill points with the TRUTH:\n"
         "            Finf, Ginf, rawinf = self.evaluator.evaluate(Xinf)\n"
         "            self._add(Xinf, Finf, Ginf, rawinf)\n"
         "            self.history.append(self._hv())\n",
         "            Xinf = np.array(chosen)\n"
         "            t_acq = time.perf_counter() - t_acq0\n"
         "\n"
         "            # evaluate the infill points with the TRUTH:\n"
         "            t_ev0 = time.perf_counter()\n"
         "            Finf, Ginf, rawinf = self.evaluator.evaluate(Xinf)\n"
         "            t_ev = time.perf_counter() - t_ev0\n"
         "            self._add(Xinf, Finf, Ginf, rawinf)\n"
         "            t_hv0 = time.perf_counter()\n"
         "            self.history.append(self._hv())\n"
         "            t_hv = time.perf_counter() - t_hv0\n"
         "            self.phase_log.append(dict(\n"
         "                stage=\"infill\", iteration=len(self.phase_log),\n"
         "                n_eval=int(len(Xinf)), n_archive_before=int(len(self.X) - len(Xinf)),\n"
         "                t_fit_s=t_fit, t_nsga_s=t_nsga, t_acq_s=t_acq,\n"
         "                t_eval_s=t_ev, t_hv_s=t_hv, t_ckpt_s=None))\n"
         "            if verbose:\n"
         "                print(f\"           budget: evaluation {t_ev / 60.0:.1f} min | \"\n"
         "                      f\"optimiser {t_fit + t_nsga + t_acq + t_hv:.1f} s \"\n"
         "                      f\"(fit {t_fit:.1f}, NSGA-II {t_nsga:.1f}, \"\n"
         "                      f\"acquisition {t_acq:.2f}, HV {t_hv:.3f})\")\n"),
        ("2b checkpoint timing",
         "                try:\n"
         "                    self.save_checkpoint(\n"
         "                        ckpt_path, meta=getattr(self, \"checkpoint_meta\", None))\n",
         "                try:\n"
         "                    t_ck0 = time.perf_counter()\n"
         "                    self.save_checkpoint(\n"
         "                        ckpt_path, meta=getattr(self, \"checkpoint_meta\", None))\n"
         "                    self.phase_log[-1][\"t_ckpt_s\"] = time.perf_counter() - t_ck0\n"),
        ("2c save phase_log",
         "            \"n_real_evaluations\": r[\"n_real_evaluations\"],\n"
         "        }\n"
         "        if meta:\n",
         "            \"n_real_evaluations\": r[\"n_real_evaluations\"],\n"
         "            \"phase_log\": list(self.phase_log),   # wall time per phase\n"
         "        }\n"
         "        if meta:\n"),
        ("2c load phase_log",
         "        self.history = list(ckpt.get(\"hv_history\", []))\n",
         "        self.history = list(ckpt.get(\"hv_history\", []))\n"
         "        self.phase_log = list(ckpt.get(\"phase_log\", []))\n"),
    ],
    DRIVER: [
        ("3a platform import",
         "import argparse\n",
         "import argparse\nimport platform\nfrom datetime import datetime, timezone\n"),
        ("3a meta host fields",
         "                           \"geometry\": \"v2-envelope\",\n"
         "                           \"omp_threads\": n_threads}\n",
         "                           \"geometry\": \"v2-envelope\",\n"
         "                           \"omp_threads\": n_threads,\n"
         "                           # provenance for the cost tables\n"
         "                           \"host\": platform.node(),\n"
         "                           \"cpu_count\": os.cpu_count(),\n"
         "                           \"openmc_version\": _openmc_version(),\n"
         "                           \"workdir\": getattr(args, \"workdir\", \"openmc_runs\"),\n"
         "                           \"started_utc\": datetime.now(timezone.utc)\n"
         "                               .isoformat(timespec=\"seconds\")}\n"),
        ("3b final write reuses meta",
         "    ckpt = opt.save_checkpoint(ckpt_out,\n"
         "                               meta={\"k_target\": k_target_arg,\n"
         "                                     \"smoke\": bool(args.smoke),\n"
         "                                     \"transport\": dict(transport),\n"
         "                                     \"schedule\": dict(schedule),\n"
         "                                     \"geometry\": \"v2-envelope\",\n"
         "                                     \"omp_threads\": n_threads})\n",
         "    # identical meta to the per-iteration checkpoint, so a resume always\n"
         "    # sees core_transport and objective_def whichever write came last\n"
         "    ckpt = opt.save_checkpoint(ckpt_out, meta=opt.checkpoint_meta)\n",
         "opt.save_checkpoint(ckpt_out, meta=opt.checkpoint_meta)"),
        ("3a version helper",
         "def main():\n    ap = argparse.ArgumentParser()\n",
         "def _openmc_version() -> str:\n"
         "    try:\n"
         "        import openmc\n"
         "        return str(openmc.__version__)\n"
         "    except Exception:\n"
         "        return \"unknown\"\n"
         "\n\n"
         "def main():\n    ap = argparse.ArgumentParser()\n"),
    ],
}

WORKDIR_PATCHES: list[tuple] = [
    ("3c --workdir flag",
     "    ap.add_argument(\"--out\", default=\".\", help=\"output directory\")\n",
     "    ap.add_argument(\"--out\", default=\".\", help=\"output directory\")\n"
     "    ap.add_argument(\"--workdir\", default=\"openmc_runs\",\n"
     "                    help=\"directory for the per-design OpenMC case folders \"\n"
     "                         \"(case_NNNN). Give every campaign its own, e.g. \"\n"
     "                         \"openmc_runs_c5, so that campaigns never overwrite \"\n"
     "                         \"each other's statepoints (default: openmc_runs)\")\n",
     "ap.add_argument(\"--workdir\""),
    ("3c evaluator workdir",
     "                         workdir=\"openmc_runs\", **schedule)\n",
     "                         workdir=args.workdir, **schedule)\n",
     "workdir=args.workdir"),
]


def _unpack(patch):
    """(name, anchor, replacement) or (name, anchor, replacement, signature)."""
    already = patch[3] if len(patch) > 3 else None
    return patch[0], patch[1], patch[2], already


def verify(path: Path, patches):
    """Return (problems, satisfied). A patch counts as satisfied when its anchor
    is gone but its signature is already in the file."""
    text = path.read_text()
    problems, satisfied = [], []
    for patch in patches:
        name, anchor, _, already = _unpack(patch)
        if already and already in text:
            satisfied.append(f"{path.name}: '{name}' already present, skipped")
            continue
        n = text.count(anchor)
        if n == 1:
            continue
        problems.append(f"{path.name}: anchor '{name}' found {n} times (need 1)")
    return problems, satisfied


def apply(path: Path, patches) -> None:
    text = path.read_text()
    for patch in patches:
        name, anchor, repl, already = _unpack(patch)
        if already and already in text:
            print(f"  skipped {name} (already in the repository)")
        elif text.count(anchor) == 1:
            text = text.replace(anchor, repl, 1)
            print(f"  applied {name}")
        else:
            raise RuntimeError(f"unexpected state for patch {name}")
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify anchors, write nothing")
    ap.add_argument("--no-workdir", action="store_true",
                    help="do not add the --workdir flag (patch 3c)")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args()
    root = Path(args.root)

    plan = {k: list(v) for k, v in PATCHES.items()}
    if not args.no_workdir:
        plan[DRIVER] = plan[DRIVER] + WORKDIR_PATCHES

    problems, satisfied = [], []
    for fname, patches in plan.items():
        p = root / fname
        if not p.exists():
            problems.append(f"missing file {p}")
            continue
        pb, sa = verify(p, patches)
        problems += pb
        satisfied += sa
    for sa in satisfied:
        print("  already there: " + sa)
    if problems:
        print("ANCHOR CHECK FAILED, nothing written:")
        for pr in problems:
            print("  - " + pr)
        sys.exit(1)
    print("all anchors verified")
    if args.check:
        return

    for fname, patches in plan.items():
        p = root / fname
        bak = p.with_suffix(p.suffix + ".bak")
        shutil.copy2(p, bak)
        print(f"{fname} -> backup {bak.name}")
        apply(p, patches)

    import py_compile
    for fname in plan:
        py_compile.compile(str(root / fname), doraise=True)
    print("all patched files compile. Next: run the --smoke test once.")


if __name__ == "__main__":
    main()
