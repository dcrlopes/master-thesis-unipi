#!/usr/bin/env python3
"""
fix_infeasible_infill.py -- two robustness fixes exposed by Campaign 4, the
first campaign in which NO design is feasible.

BUG 1 (the crash)
  reactor_optimization.run() does

      res  = minimize(prob, NSGA2(...), ...)
      cand = np.atleast_2d(res.X)

  pymoo sets res.X = None when the run ends with an EMPTY feasible set.
  np.atleast_2d(None) is a (1,1) object array, so the very next line

      _, std = obj_sur.predict(cand)

  reaches StandardScaler with one column and raises

      ValueError: X has 1 features, but StandardScaler is expecting 5

  Campaign 4 survived iteration 1 (the constraint surrogate, fitted on 36
  points, still predicted a feasible pocket) and died at the start of
  iteration 2 -- once six more real evaluations had confirmed infeasibility,
  the surrogate stopped predicting any feasible region at all.

  FIX: when res.X is None or empty, fall back to the final population ranked
  by constraint violation, i.e. infill on the LEAST-INFEASIBLE designs. That
  is the behaviour a constrained NSGA-II is supposed to have when nothing is
  feasible, and it is exactly what a zero-feasible campaign needs: the search
  is driven toward the constraint boundary instead of stopping.

BUG 2 (why the crash cost 11 hours)
  save_checkpoint() is called ONCE, after run() returns. Any exception in the
  loop discards every completed infill evaluation -- six of them here, about
  11 hours of transport.

  FIX: optional per-iteration checkpointing. Set opt.checkpoint_path (and
  opt.checkpoint_meta) and the optimiser writes the full archive after every
  iteration. Crash-safe by construction, and harmless if unset.

USAGE
    cd ~/master-thesis-unipi
    python3 fix_infeasible_infill.py --check     # verify anchors only
    python3 fix_infeasible_infill.py             # apply (.bak written)
"""
import argparse
import shutil
import sys
from pathlib import Path

MARKER = "_least_infeasible_candidates"

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--optimization", default="reactor_optimization.py")
ap.add_argument("--runner", default="run_optimization.py")
args = ap.parse_args()

op_p, ru_p = Path(args.optimization), Path(args.runner)
for p in (op_p, ru_p):
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Run from the repository root.")
op, ru = op_p.read_text(), ru_p.read_text()

if MARKER in op:
    sys.exit("Already applied. Nothing to do.")

# --------------------------------------------------------------------------- #
A_CAND = """            cand = np.atleast_2d(res.X)"""
A_RUNDEF = """    # ---- main loop ----------------------------------------------------------
    def run(self, verbose=True):
        t0 = time.time()"""
A_ITEREND = """            if verbose:
                print(f"[Stage 2] iter {it+1}/{self.cfg.n_iter}: "
                      f"+{len(Xinf)} real evals "
                      f"(total {self.evaluator.n_calls}), "
                      f"HV={self.history[-1]:.4g}")"""
A_RUNCALL = """    res = opt.run(verbose=True)"""

checks = [("candidate extraction", A_CAND, op),
          ("run() definition", A_RUNDEF, op),
          ("iteration epilogue", A_ITEREND, op),
          ("opt.run() call", A_RUNCALL, ru)]
bad = False
for name, anchor, text in checks:
    n = text.count(anchor)
    print(f"  {'OK ' if n == 1 else 'BAD'} {name}: found {n}x")
    bad |= (n != 1)
if bad:
    sys.exit("\nERROR: anchors do not match this file version.")
print("all anchors matched.")
if args.check:
    sys.exit(0)

# --------------------------------------------------------------------------- #
# FIX 1: least-infeasible fallback                                            #
# --------------------------------------------------------------------------- #
HELPER = '''    @staticmethod
    def _least_infeasible_candidates(res):
        """Candidate matrix from a pymoo Result, valid even when the run ended
        with NO feasible solution.

        pymoo reports res.X = None in that case. The information is still in
        the final population, so rank it by total constraint violation and hand
        back the least-infeasible designs: with an empty feasible set the
        useful search direction is toward the constraint boundary, which is
        precisely what a zero-feasible campaign must explore. Returns None only
        if the population is unavailable too, leaving the caller's random
        fallback in charge."""
        X = getattr(res, "X", None)
        if X is not None:
            X = np.atleast_2d(X)
            if X.ndim == 2 and X.shape[0] > 0 and X.dtype != object:
                return X
        pop = getattr(res, "pop", None)
        if pop is None or len(pop) == 0:
            return None
        Xp = np.atleast_2d(np.asarray(pop.get("X"), dtype=float))
        try:
            cv = np.asarray(pop.get("CV"), dtype=float).ravel()
        except Exception:
            return Xp
        if cv.size != Xp.shape[0]:
            return Xp
        return Xp[np.argsort(cv)]        # least-infeasible first

'''
op = op.replace(A_RUNDEF, HELPER + A_RUNDEF)

op = op.replace(A_CAND, """            cand = self._least_infeasible_candidates(res)
            if cand is None or cand.shape[0] == 0:
                # nothing usable came back: explore randomly this iteration
                cand = np.atleast_2d(
                    self.spec.design_space.lhs(max(self.cfg.n_infill * 4, 32),
                                               seed=self.cfg.seed + 777 + it))
            elif res.X is None and verbose:
                print(f"[Stage 2] iter {it+1}: surrogate NSGA-II found no "
                      f"feasible design; infilling on the {cand.shape[0]} "
                      f"least-infeasible population members.")""")

# --------------------------------------------------------------------------- #
# FIX 2: per-iteration checkpointing                                          #
# --------------------------------------------------------------------------- #
op = op.replace(A_ITEREND, A_ITEREND + """

            # crash-safe: persist the FULL archive after every iteration, so an
            # exception later in the loop can never discard completed real
            # evaluations (Campaign 4 lost six to a mid-loop crash).
            ckpt_path = getattr(self, "checkpoint_path", None)
            if ckpt_path:
                try:
                    self.save_checkpoint(
                        ckpt_path, meta=getattr(self, "checkpoint_meta", None))
                    if verbose:
                        print(f"           checkpoint written -> {ckpt_path}")
                except Exception as exc:            # never kill a live campaign
                    print(f"           WARNING: checkpoint failed: {exc}")""")

# --------------------------------------------------------------------------- #
# wire the runner: set the attributes before run()                            #
# --------------------------------------------------------------------------- #
ru = ru.replace(A_RUNCALL, """    # crash-safe per-iteration checkpointing (same path and meta as the final
    # write below, so a resume reads an identical file either way)
    opt.checkpoint_path = ckpt_out
    opt.checkpoint_meta = {"k_target": k_target_arg,
                           "smoke": bool(args.smoke),
                           "transport": dict(transport),
                           "core_transport": {
                               "particles": args.core_particles,
                               "batches": args.core_batches,
                               "inactive": args.core_inactive},
                           "objective_def":
                               "peaking = core BOL F_dh (Campaign 4)",
                           "schedule": dict(schedule),
                           "geometry": "v2-envelope",
                           "omp_threads": n_threads}

    res = opt.run(verbose=True)""")

shutil.copy(op_p, op_p.with_suffix(".py.bak"))
shutil.copy(ru_p, ru_p.with_suffix(".py.bak"))
op_p.write_text(op)
ru_p.write_text(ru)
print(f"\nwritten: {op_p} and {ru_p}   (backups: *.py.bak)")
print("""
verify:
  python3 -c "import ast; [ast.parse(open(f).read()) for f in
      ['reactor_optimization.py','run_optimization.py']]; print('parse OK')"

then resume; from the next iteration onward the log shows
  'checkpoint written -> ...' after every iteration, and if the surrogate
  finds no feasible design it says so and infills on the boundary instead of
  crashing.
""")
