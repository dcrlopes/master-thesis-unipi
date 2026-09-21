#!/usr/bin/env python3
r"""
apply_hv_stop.py
================
Wires the automatic stopping rule of hv_stop.py into the active-learning
loop, on branch main. Everything is OFF by default: without --hv-stop and
--max-evals the loop runs exactly --iters iterations as it always has, so
Campaigns 1 to 9 reproduce. Independent of the objective set.

WHAT CHANGES
------------
reactor_optimization.py
  + OptimizerConfig gains hv_stop, hv_tol, hv_patience, max_evals.
  + run() refuses to START an iteration whose infill batch would exceed
    max_evals, and after each iteration evaluates the two-condition rule of
    methodology.tex (gain below tol for `patience` consecutive iterations,
    and no censored design on the feasible Pareto front).
  + three health indicators are stored per iteration in phase_log, where
    until now they were only printed: n_candidates, n_margin_feasible,
    relax_depth, plus n_infill_feasible and hv_gain.
  + the reason the loop ended is kept on the optimiser (stop_info) and
    written to the checkpoint under the new top-level key "stop".

run_optimization.py
  + flags --hv-stop, --hv-tol, --hv-patience, --max-evals.
  + under --hv-stop the rule decides, so --iters becomes an upper bound.
    With no --iters it is sized from --max-evals. An unattended run with
    neither is refused.
  + a stop_rule block in the checkpoint meta.

REVERT removes exactly these edits by the inverse anchored replacement. It
does not restore a backup, so a later patch to the same files survives.

REQUIRES hv_stop.py next to the framework files.

USAGE, from the repository root on main
    python apply_hv_stop.py --selftest
    python apply_hv_stop.py --check
    python apply_hv_stop.py
    python apply_hv_stop.py --revert
"""

import argparse
import datetime
import pathlib
import sys

MARKER = "HV-STOP"
RO, RUN = "reactor_optimization.py", "run_optimization.py"

EDITS = []


def edit(fname, name, old, new):
    EDITS.append((fname, name, old, new))


# ====================================================== reactor_optimization ==
edit(RO, "config fields",
     "                                   # constraints to rank first; 0 restores\n"
     "                                   # the pure-uncertainty ordering\n",
     "                                   # constraints to rank first; 0 restores\n"
     "                                   # the pure-uncertainty ordering\n"
     "    # HV-STOP: automatic stopping (hv_stop.py). All off by default, so every\n"
     "    # earlier campaign reproduces. hv_stop enables the two-condition rule of\n"
     "    # methodology.tex; max_evals is a budget on real evaluations that the\n"
     "    # loop never exceeds.\n"
     "    hv_stop: bool = False\n"
     "    hv_tol: float = 0.01\n"
     "    hv_patience: int = 3\n"
     "    max_evals: int | None = None\n")

edit(RO, "stop_info attribute",
     "        self._hv_ref_frozen = None # fixed reference point (set after Stage 1)\n",
     "        self._hv_ref_frozen = None # fixed reference point (set after Stage 1)\n"
     "        self.stop_info = None      # HV-STOP: why the last run() ended\n")

edit(RO, "front censored counter",
     "    @staticmethod\n"
     "    def _least_infeasible_candidates(res):\n",
     "    def _n_front_censored(self):\n"
     "        \"\"\"HV-STOP: censored designs on the current feasible Pareto front.\"\"\"\n"
     "        feas = self._feasible_mask()\n"
     "        if feas.sum() == 0:\n"
     "            return 0\n"
     "        idx = np.flatnonzero(feas)[self._nondominated(self.F[feas])]\n"
     "        return int(sum(bool(self.raw[i].get(\"censored\")) for i in idx))\n"
     "\n"
     "    @staticmethod\n"
     "    def _least_infeasible_candidates(res):\n")

edit(RO, "lazy import at the top of run",
     "    def run(self, verbose=True):\n"
     "        t0 = time.time()\n",
     "    def run(self, verbose=True):\n"
     "        t0 = time.time()\n"
     "        # HV-STOP: imported only when asked for, so the framework does not\n"
     "        # depend on hv_stop.py for any earlier campaign.\n"
     "        _hvs = None\n"
     "        self.stop_info = None\n"
     "        if self.cfg.hv_stop or self.cfg.max_evals is not None:\n"
     "            import hv_stop as _hvs\n")

edit(RO, "budget check before an iteration",
     "        for it in range(self.cfg.n_iter):\n"
     "            t_fit0 = time.perf_counter()\n",
     "        for it in range(self.cfg.n_iter):\n"
     "            # HV-STOP: never start an iteration that would exceed the budget\n"
     "            if _hvs is not None and _hvs.cap_blocks(\n"
     "                    len(self.X), self.cfg.n_infill, self.cfg.max_evals):\n"
     "                self.stop_info = dict(\n"
     "                    stop=True, by=\"cap\", n_evals=int(len(self.X)),\n"
     "                    reason=f\"budget: {len(self.X)} evaluations done, one more \"\n"
     "                           f\"batch of {self.cfg.n_infill} would exceed \"\n"
     "                           f\"{self.cfg.max_evals}\")\n"
     "                if verbose:\n"
     "                    print(f\"[HV-STOP] {self.stop_info['reason']}\")\n"
     "                break\n"
     "            t_fit0 = time.perf_counter()\n")

edit(RO, "relaxation depth, initialise",
     "            chosen = []\n"
     "            min_sep = float(getattr(self.cfg, \"infill_min_sep\", 0.05))\n",
     "            chosen = []\n"
     "            relax_depth = 0             # HV-STOP health indicator\n"
     "            min_sep = float(getattr(self.cfg, \"infill_min_sep\", 0.05))\n")

edit(RO, "relaxation depth, count",
     "                    min_sep *= 0.5          # relax and rescan the ranking\n",
     "                    min_sep *= 0.5          # relax and rescan the ranking\n"
     "                    relax_depth += 1        # HV-STOP health indicator\n")

edit(RO, "health indicators into phase_log",
     "                t_fit_s=t_fit, t_nsga_s=t_nsga, t_acq_s=t_acq,\n"
     "                t_eval_s=t_ev, t_hv_s=t_hv, t_ckpt_s=None))\n",
     "                t_fit_s=t_fit, t_nsga_s=t_nsga, t_acq_s=t_acq,\n"
     "                t_eval_s=t_ev, t_hv_s=t_hv, t_ckpt_s=None,\n"
     "                # HV-STOP: the indicators methodology.tex asks to inspect\n"
     "                # before the rule is trusted, stored and not only printed\n"
     "                n_candidates=int(len(cand)),\n"
     "                n_margin_feasible=int(eligible.sum()),\n"
     "                relax_depth=int(relax_depth),\n"
     "                n_infill_feasible=(int(np.all(Ginf <= 1e-9, axis=1).sum())\n"
     "                                   if self.spec.n_constr else int(len(Xinf))),\n"
     "                hv_gain=((self.history[-1] - self.history[-2]) / self.history[-2]\n"
     "                         if len(self.history) > 1 and self.history[-2] > 0\n"
     "                         else None)))\n")

edit(RO, "rule evaluated before the checkpoint",
     "            # crash-safe: persist the FULL archive after every iteration, so an\n",
     "            # HV-STOP: evaluate the rule BEFORE the checkpoint, so the decision\n"
     "            # is inside the file that a resume or a post-analysis reads.\n"
     "            if _hvs is not None and self.cfg.hv_stop:\n"
     "                _dec = _hvs.decide(self.history, self._n_front_censored(),\n"
     "                                   self.cfg.hv_tol, self.cfg.hv_patience)\n"
     "                if verbose:\n"
     "                    _g = _dec[\"last_gain\"]\n"
     "                    print(\"           [HV-STOP] gain \"\n"
     "                          + (f\"{100 * _g:.3f} %\" if _g is not None else \"undefined\")\n"
     "                          + f\" | {_dec['reason']}\")\n"
     "                if _dec[\"stop\"]:\n"
     "                    self.stop_info = dict(\n"
     "                        by=\"rule\", n_evals=int(len(self.X)),\n"
     "                        warnings=_hvs.health_warnings(self.phase_log,\n"
     "                                                      self.cfg.hv_patience),\n"
     "                        notes=_hvs.health_notes(self.phase_log,\n"
     "                                                self.cfg.hv_patience),\n"
     "                        **_dec)\n"
     "                    for _n in self.stop_info[\"notes\"]:\n"
     "                        print(f\"           [HV-STOP] note    {_n}\")\n"
     "                    for _w in self.stop_info[\"warnings\"]:\n"
     "                        print(f\"           [HV-STOP] WARNING {_w}\")\n"
     "\n"
     "            # crash-safe: persist the FULL archive after every iteration, so an\n")

edit(RO, "break after the checkpoint",
     "                except Exception as exc:            # never kill a live campaign\n"
     "                    print(f\"           WARNING: checkpoint failed: {exc}\")\n"
     "\n"
     "        if verbose:\n"
     "            print(f\"Done in {time.time()-t0:.1f}s, \"\n",
     "                except Exception as exc:            # never kill a live campaign\n"
     "                    print(f\"           WARNING: checkpoint failed: {exc}\")\n"
     "\n"
     "            if self.stop_info is not None and self.stop_info.get(\"stop\"):\n"
     "                break                   # HV-STOP: rule met, state is on disk\n"
     "\n"
     "        if self.stop_info is None:      # HV-STOP: the --iters bound ended the run\n"
     "            self.stop_info = dict(stop=False, by=\"iters\", n_evals=int(len(self.X)),\n"
     "                                  reason=f\"{self.cfg.n_iter} iterations requested \"\n"
     "                                         f\"and done, rule not met\")\n"
     "        if verbose:\n"
     "            print(f\"Done in {time.time()-t0:.1f}s, \"\n")

edit(RO, "stop reason into the checkpoint",
     "            \"phase_log\": list(self.phase_log),   # wall time per phase\n"
     "        }\n",
     "            \"phase_log\": list(self.phase_log),   # wall time per phase\n"
     "            \"stop\": getattr(self, \"stop_info\", None),   # HV-STOP\n"
     "        }\n")

# ========================================================== run_optimization ==
edit(RUN, "command-line flags",
     "                         \"THIS run (handy on --resume, e.g. --iters 3 to add 3 \"\n"
     "                         \"more rounds of infill)\")\n",
     "                         \"THIS run (handy on --resume, e.g. --iters 3 to add 3 \"\n"
     "                         \"more rounds of infill)\")\n"
     "    # HV-STOP\n"
     "    ap.add_argument(\"--hv-stop\", action=\"store_true\",\n"
     "                    help=\"stop automatically on the rule of methodology.tex: \"\n"
     "                         \"hypervolume gain below --hv-tol for --hv-patience \"\n"
     "                         \"consecutive iterations AND no censored design on \"\n"
     "                         \"the front. --iters then acts as an upper bound\")\n"
     "    ap.add_argument(\"--hv-tol\", type=float, default=0.01,\n"
     "                    help=\"relative hypervolume gain that counts as quiet \"\n"
     "                         \"(default 0.01, that is 1 percent)\")\n"
     "    ap.add_argument(\"--hv-patience\", type=int, default=3,\n"
     "                    help=\"consecutive quiet iterations required (default 3)\")\n"
     "    ap.add_argument(\"--max-evals\", type=int, default=None,\n"
     "                    help=\"budget on real evaluations, DOE included. The loop \"\n"
     "                         \"never starts an iteration that would exceed it\")\n")

edit(RUN, "config and iteration bound",
     "    cfg.feas_kappa = float(args.feas_kappa)          # FEAS-MARGIN\n",
     "    cfg.feas_kappa = float(args.feas_kappa)          # FEAS-MARGIN\n"
     "    # HV-STOP: under the rule --iters is only an upper bound. With no\n"
     "    # --iters it is sized from the budget, generously, and cap_blocks()\n"
     "    # inside the loop does the exact accounting, on a resume as well.\n"
     "    cfg.hv_stop = bool(args.hv_stop)\n"
     "    cfg.hv_tol = float(args.hv_tol)\n"
     "    cfg.hv_patience = int(args.hv_patience)\n"
     "    cfg.max_evals = args.max_evals\n"
     "    if args.hv_stop and args.iters is None:\n"
     "        if args.max_evals is None:\n"
     "            raise SystemExit(\"--hv-stop needs --max-evals or --iters: an \"\n"
     "                             \"unattended run must have an upper bound.\")\n"
     "        cfg.n_iter = -(-int(args.max_evals) // max(1, cfg.n_infill))\n"
     "    if args.hv_stop or args.max_evals is not None:\n"
     "        print(f\"STOP RULE: \"\n"
     "              + (f\"gain < {100 * cfg.hv_tol:g} % for {cfg.hv_patience} \"\n"
     "                 f\"consecutive iterations and no censored front member\"\n"
     "                 if cfg.hv_stop else \"rule off\")\n"
     "              + f\" | budget {cfg.max_evals} evaluations\"\n"
     "              + f\" | iteration bound {cfg.n_iter}, the budget is enforced \"\n"
     "                f\"inside the loop\")\n")

edit(RUN, "stop_rule block in the meta",
     "                           \"objective_set\": args.objective_set,\n",
     "                           \"objective_set\": args.objective_set,\n"
     "                           # HV-STOP\n"
     "                           \"stop_rule\": {\"enabled\": bool(args.hv_stop),\n"
     "                                         \"tol\": args.hv_tol,\n"
     "                                         \"patience\": args.hv_patience,\n"
     "                                         \"max_evals\": args.max_evals},\n")


# ==================================================================== driver ==
def selftest():
    import ast
    seen = set()
    for f, name, old, new in EDITS:
        assert old and new and old != new, name
        assert (f, old) not in seen, f"duplicate anchor {name}"
        seen.add((f, old))
        assert MARKER in new and MARKER not in old, f"{name}: marker must come from the edit"
    # the inserted method must be valid python on its own
    m = next(new for f, n, old, new in EDITS if n == "front censored counter")
    body = m.split("    @staticmethod")[0]
    ast.parse("class _T:\n" + body)
    # inverse edits must be unambiguous: no `new` may contain another `new`
    news = [(f, new) for f, _, _, new in EDITS]
    for i, (fi, a) in enumerate(news):
        for j, (fj, b) in enumerate(news):
            assert i == j or fi != fj or a not in b, "overlapping edits"
    # round trip on the real files, in memory: apply every edit, then the
    # inverse edits in reverse order, and demand the original text back
    root = pathlib.Path(".")
    if all((root / f).is_file() for f in (RO, RUN)):
        for fname in (RO, RUN):
            orig = (root / fname).read_text(encoding="utf-8")
            if MARKER in orig:
                print(f"  round trip skipped for {fname}: already patched"); continue
            t = orig
            for f, name, old, new in EDITS:
                if f == fname:
                    assert t.count(old) == 1, f"{name}: anchor matches {t.count(old)} times"
                    t = t.replace(old, new, 1)
            for f, name, old, new in reversed(EDITS):
                if f == fname:
                    assert t.count(new) == 1, f"{name}: inserted block not unique"
                    t = t.replace(new, old, 1)
            assert t == orig, f"{fname}: round trip does not restore the file"
            print(f"  round trip exact for {fname}")
    print(f"selftest OK ({len(EDITS)} edits, "
          f"{sum(f == RO for f, *_ in EDITS)} in {RO}, {sum(f == RUN for f, *_ in EDITS)} in {RUN})")
    return 0


def _load():
    root, texts = pathlib.Path("."), {}
    for fname in sorted({f for f, *_ in EDITS}):
        p = root / fname
        if not p.is_file():
            sys.exit(f"FAIL: {fname} not found. Run from the repository root.")
        texts[fname] = p.read_text(encoding="utf-8")
    return root, texts


def _write(root, texts, tag):
    import py_compile
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for fname, text in texts.items():
        p = root / fname
        bak = root / f"{fname}.bak_hvstop_{tag}_{stamp}"
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        p.write_text(text, encoding="utf-8")
        print(f"  wrote {fname}  (backup {bak.name})")
    for fname in texts:
        py_compile.compile(str(root / fname), doraise=True)
    print("  both files compile")


def run(mode):
    root, texts = _load()
    if mode == "apply" and not (root / "hv_stop.py").is_file():
        sys.exit("FAIL: hv_stop.py must sit next to the framework files.")
    if any(MARKER in t for t in texts.values()):
        if mode == "check":
            print("already applied (marker found)"); return 0
        sys.exit("Already applied (marker found). Nothing to do.")
    bad = 0
    for fname, name, old, new in EDITS:
        n = texts[fname].count(old)
        bad += (n != 1)
        print(f"  [{'OK  ' if n == 1 else 'FAIL'}] {fname:24s} {n} match  {name}")
        if n == 1:
            texts[fname] = texts[fname].replace(old, new, 1)
    if bad:
        sys.exit(f"\n{bad} anchor(s) did not match exactly once. Nothing written.")
    if mode == "check":
        print("\nall anchors unique, nothing written"); return 0
    _write(root, texts, "pre")
    return 0


def revert():
    root, texts = _load()
    if not any(MARKER in t for t in texts.values()):
        print("not applied (no marker). Nothing to do."); return 0
    bad = 0
    for fname, name, old, new in reversed(EDITS):
        n = texts[fname].count(new)
        bad += (n != 1)
        print(f"  [{'OK  ' if n == 1 else 'FAIL'}] {fname:24s} {n} match  {name}")
        if n == 1:
            texts[fname] = texts[fname].replace(new, old, 1)
    if bad:
        sys.exit(f"\n{bad} inserted block(s) not found exactly once: the files were edited "
                 f"after the patch. Nothing written. Use the .bak_hvstop_pre_* backup by hand.")
    if any(MARKER in t for t in texts.values()):
        sys.exit("FAIL: marker still present after the inverse edits. Nothing written.")
    _write(root, texts, "applied")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.revert:
        sys.exit(revert())
    sys.exit(run("check" if a.check else "apply"))
