"""
reactor_optimization.py
=======================

Dual-stage, active-learning, surrogate-assisted MULTI-OBJECTIVE optimization
framework for nuclear core design.  This is the "engine" for the second part of
your thesis.  It mirrors the methodology of the KIT platform you want to emulate
(surrogate models + NSGA-II + active learning) but is written so that the slow
"truth" evaluator is your OpenMC depletion model.

WHY THIS DESIGN
---------------
Each OpenMC depletion run is expensive (minutes to hours).  A genetic algorithm
needs thousands of evaluations.  Doing that directly is impossible.  The trick
(this is the whole point of the KIT framework) is:

    Stage 1  -- EXPLORE:  run a SMALL number of real OpenMC cases on a
                space-filling sample of designs and train a cheap surrogate
                (a machine-learning model) that predicts the objectives.

    Stage 2  -- EXPLOIT:  run the genetic algorithm (NSGA-II) on the *surrogate*
                (which is instantaneous).  Take the most promising / most
                UNCERTAIN designs it finds, evaluate ONLY those with the real
                OpenMC model, add them to the data set, retrain the surrogate,
                and repeat.  This "active learning" loop spends your expensive
                evaluations only where they matter.

The framework is deliberately split so you can:
  * run the WHOLE pipeline today with a fast synthetic evaluator (no OpenMC),
  * then swap in OpenMCEvaluator without touching the optimization logic,
  * later replace the Gaussian-Process surrogate with an MLP / CNN / GNN
    (Work-Plan stretch goals) by editing ONE class.

DEPENDENCIES
------------
    pip install numpy scipy scikit-learn pymoo matplotlib

Author: prepared as a thesis starter. Read it, change it, make it yours.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# ----- scikit-learn surrogates ------------------------------------------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ----- pymoo (NSGA-II + indicators) ------------------------------------------
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize


# =============================================================================
# 1.  DESIGN SPACE
# =============================================================================
@dataclass
class DesignVariable:
    """One continuous design knob the optimizer is allowed to turn."""
    name: str
    low: float
    high: float
    unit: str = ""

    def clip(self, x):
        return float(np.clip(x, self.low, self.high))


@dataclass
class DesignSpace:
    """The full set of design variables. Order matters: vectors follow it."""
    variables: list[DesignVariable]

    @property
    def n(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def xl(self) -> np.ndarray:
        return np.array([v.low for v in self.variables])

    @property
    def xu(self) -> np.ndarray:
        return np.array([v.high for v in self.variables])

    def lhs(self, n_samples: int, seed: int = 0, accept=None) -> np.ndarray:
        """Latin-Hypercube space-filling sample (Stage-1 DOE).

        accept : callable(dict) -> bool, optional
            A cheap ANALYTIC feasibility test (e.g. the vessel-fit constraint
            g_geom <= 0). When given, LHS rounds are drawn until n_samples
            ACCEPTED points are collected, so no expensive truth evaluation is
            ever spent on a design that is unbuildable by construction. The
            points still come from LHS rounds, so space-filling quality inside
            the feasible region is preserved.
        """
        sampler = LHS()
        if accept is None:
            unit = sampler(_BoxProblem(self.n), n_samples).get("X")  # in [0,1]
            return self.xl + unit * (self.xu - self.xl)
        kept = []
        for round_ in range(50):                      # hard stop: 50 rounds
            unit = sampler(_BoxProblem(self.n), n_samples).get("X")
            X = self.xl + unit * (self.xu - self.xl)
            for x in X:
                if accept(self.as_dict(x)):
                    kept.append(x)
                    if len(kept) >= n_samples:
                        return np.array(kept)
        raise RuntimeError(
            f"DesignSpace.lhs: could not collect {n_samples} accepted points "
            f"in 50 LHS rounds -- the accept() region may be (nearly) empty; "
            f"check the design-variable bounds against the constraint.")

    def as_dict(self, x: Sequence[float]) -> dict:
        return {v.name: float(xi) for v, xi in zip(self.variables, x)}


class _BoxProblem(Problem):
    """Tiny helper so pymoo's LHS sampler has something to sample in [0,1]^n."""
    def __init__(self, n):
        super().__init__(n_var=n, n_obj=1, xl=0.0, xu=1.0)
    def _evaluate(self, X, out, *a, **k):
        out["F"] = np.zeros((len(X), 1))


# =============================================================================
# 2.  WHAT WE OPTIMIZE  (objectives + constraints)
# =============================================================================
# Convention used everywhere below:
#   * objectives are returned as a dict {name: value} and ALL are MINIMIZED
#     internally (pymoo minimizes).  To MAXIMISE cycle length we minimise its
#     negative -- handled automatically via the `maximize` flag.
#   * constraints are returned as a dict {name: g} with the rule  g <= 0 is
#     FEASIBLE  (pymoo convention).  e.g. "enrichment <= 19.75%"  ->
#     g = enrichment - 19.75.
@dataclass
class Objective:
    name: str
    maximize: bool = False        # True -> we internally minimise (-value)
    label: str = ""               # for plots

    def to_min(self, value: float) -> float:
        return -value if self.maximize else value


@dataclass
class ProblemSpec:
    design_space: DesignSpace
    objectives: list[Objective]
    constraint_names: list[str]
    # ANALYTIC constraints: {name -> callable(design_dict) -> g}. These are
    # known in CLOSED FORM (e.g. the vessel-fit g_geom, pure geometry), so the
    # optimizer (a) screens the Stage-1 DOE with them for free and (b) feeds
    # NSGA-II their EXACT value instead of a surrogate prediction. Every name
    # here must also appear in constraint_names, and the truth evaluator must
    # still return it (so it lands in the archive/checkpoint like any other g).
    exact_constraints: dict = field(default_factory=dict)

    @property
    def n_obj(self):
        return len(self.objectives)

    @property
    def n_constr(self):
        return len(self.constraint_names)

    def exact_ok(self, design: dict) -> bool:
        """True iff the design satisfies every ANALYTIC constraint (g <= 0)."""
        return all(f(design) <= 0.0 for f in self.exact_constraints.values())


# =============================================================================
# 3.  EVALUATORS  (the "truth")
# =============================================================================
class Evaluator:
    """Base class.  Given a 2D array of designs X (n x n_var) return:
        F  : objectives, shape (n, n_obj)   -- already converted to MINIMISE
        G  : constraints, shape (n, n_constr) -- g<=0 feasible
        raw: list of dicts with the physical (un-converted) quantities, so you
             keep cycle-length in days, peaking as a factor, etc. for plots.
    Subclass and implement evaluate_one()."""

    def __init__(self, spec: ProblemSpec):
        self.spec = spec
        self.n_calls = 0

    def evaluate_one(self, design: dict) -> dict:
        """Return a dict with one entry per objective name and per constraint
        name, using PHYSICAL values (objectives NOT negated)."""
        raise NotImplementedError

    def evaluate(self, X: np.ndarray):
        X = np.atleast_2d(X)
        F = np.zeros((len(X), self.spec.n_obj))
        G = np.zeros((len(X), self.spec.n_constr))
        raw = []
        for i, x in enumerate(X):
            d = self.spec.design_space.as_dict(x)
            res = self.evaluate_one(d)
            raw.append({**d, **res})
            for j, obj in enumerate(self.spec.objectives):
                F[i, j] = obj.to_min(res[obj.name])
            for j, cname in enumerate(self.spec.constraint_names):
                G[i, j] = res[cname]
            self.n_calls += 1
        return F, G, raw


# ---------------------------------------------------------------------------
# 3a.  ANALYTIC evaluator -- a *fast* stand-in for OpenMC so you can run and
#      understand the whole pipeline TODAY.  The formulas are toy physics:
#      they reproduce the right qualitative trends (more enrichment & burnable
#      poison -> longer cycle; moderation optimum in pitch; reflector saves
#      neutrons; peaking worsens with aggressive zoning).  DO NOT use these
#      numbers in your thesis -- they exist only to exercise the optimizer.
# ---------------------------------------------------------------------------
class AnalyticEvaluator(Evaluator):
    def __init__(self, spec: ProblemSpec, noise: float = 0.0, seed: int = 0):
        super().__init__(spec)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def evaluate_one(self, d: dict) -> dict:
        e_in  = d["enrich_inner"]      # %
        e_out = d["enrich_outer"]      # %
        gd    = d["gd_wt"]             # wt% Gd2O3 in poisoned pins
        pitch = d["pitch"]             # cm
        refl  = d["refl_thick"]        # cm

        e_avg = 0.5 * (e_in + e_out)
        # moderation: optimum moderator-to-fuel ratio near pitch ~ 1.30 cm
        mod = np.exp(-((pitch - 1.30) / 0.18) ** 2)
        refl_gain = 1.0 - np.exp(-refl / 12.0)        # saturates with thickness

        # k at beginning of life (toy): rises with enrichment & moderation,
        # Gd depresses it at BOL.
        k_bol = (0.78 + 0.085 * e_avg) * (0.75 + 0.25 * mod) \
                * (1.0 + 0.06 * refl_gain) - 0.010 * gd
        # Cycle length proxy [EFPD]: more fissile -> longer; Gd lets you hold
        # more excess reactivity so a *moderate* amount extends the cycle, but
        # too much wastes neutrons (penalty term).
        cycle = (220.0 * (e_avg - 1.0) * mod * (0.9 + 0.1 * refl_gain)
                 + 90.0 * gd - 22.0 * gd ** 2)
        cycle = max(cycle, 1.0)
        # Radial power peaking: aggressive inner/outer mismatch and poor
        # moderation make it worse; Gd flattens it a little.
        peaking = (1.25 + 0.55 * abs(e_in - e_out) / 5.0
                   + 0.30 * (1.0 - mod) - 0.04 * gd)
        peaking = max(peaking, 1.0)

        if self.noise:
            cycle   *= 1.0 + self.noise * self.rng.standard_normal()
            peaking *= 1.0 + 0.3 * self.noise * self.rng.standard_normal()

        from core_geometry import geometry_margin
        return {
            "cycle_length": cycle,                 # objective (maximise)
            "peaking":      peaking,               # objective (minimise)
            # constraints (g<=0 feasible):
            "g_kmin":  1.02 - k_bol,               # need k_bol >= 1.02
            "g_kmax":  k_bol - 1.35,               # and  k_bol <= 1.35
            "g_enr":   max(e_in, e_out) - 19.75,   # LEU cap
            "g_peak":  peaking - 2.0,              # peaking <= 2.0
            "g_geom":  geometry_margin(pitch, refl),  # fits in the vessel
            "k_bol":   k_bol,                      # carried along for plots
        }


# ---------------------------------------------------------------------------
# 3b.  OpenMC evaluator -- the REAL thing.  Skeleton only: fill the two TODOs
#      with calls into your model-builder (see the Jupyter notebook).  The
#      optimization code never changes.
# ---------------------------------------------------------------------------
class OpenMCEvaluator(Evaluator):
    """Evaluate one core design by running an OpenMC depletion calculation.

    Expected workflow inside evaluate_one():
       1. build_model(design)  -> openmc.Model            (geometry+materials)
       2. run a depletion sweep -> k_inf(burnup) curve and pin-power tally
       3. cycle_length  = reactivity-limited burnup (k_inf crosses a target
                          threshold that accounts for leakage+control) divided
                          by specific power, in EFPD.
       4. peaking       = max pin (or assembly) power / average.

    See thesis Work-Plan Weeks 6-9.  Keep `power` consistent with 48 MWth over
    the heavy-metal mass actually present in your model.
    """

    def __init__(self, spec: ProblemSpec, build_model: Callable,
                 power_watts: float = 48e6,
                 burnup_steps_mwd_kg: Sequence[float] = (0.5, 1, 2, 4, 8, 12,
                                                         18, 24, 30, 36, 42),
                 k_target: float = 1.03,
                 chain_file: str | None = None,
                 workdir: str = "openmc_runs",
                 nthreads: int = 8):
        super().__init__(spec)
        self.build_model = build_model
        self.power_watts = power_watts
        self.burnup_steps = list(burnup_steps_mwd_kg)
        self.k_target = k_target
        self.chain_file = chain_file
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True, parents=True)
        self.nthreads = nthreads

    def evaluate_one(self, design: dict) -> dict:
        import openmc
        import openmc.deplete

        case_dir = self.workdir / f"case_{self.n_calls:04d}"
        case_dir.mkdir(exist_ok=True, parents=True)

        # 1) Build the model for this design (geometry + materials) -----------
        model = self.build_model(design)        # <-- from your notebook

        # 2) Depletion sweep ---------------------------------------------------
        # heavy-metal mass and specific power must be self-consistent. The
        # build_model() helper should set the volume of fissile materials so
        # OpenMC can compute the HM mass when `normalization_mode="fission-q"`.
        op = openmc.deplete.CoupledOperator(
            model, self.chain_file, diff_burnable_mats=True
        )
        # convert MWd/kgHM cumulative points into per-step day lengths:
        timesteps, units = self._burnup_to_timesteps()
        integrator = openmc.deplete.PredictorIntegrator(
            op, timesteps, power=self.power_watts, timestep_units=units
        )
        cwd = Path.cwd()
        try:
            import os
            os.chdir(case_dir)
            integrator.integrate()
            results = openmc.deplete.Results("depletion_results.h5")
        finally:
            os.chdir(cwd)

        # 3) Cycle length from the k_inf(burnup) curve ------------------------
        bu, keff = [], []
        for i in range(len(results)):
            k = results[i].get_keff()           # (mean, std)
            bu.append(results.get_step_where_time)  # placeholder; see notebook
        # ---- TODO: replace the two lines above with proper extraction:
        #   times = results.get_times(time_units="d")  # EFPD axis
        #   k_mean = [r.get_keff()[0] for r in results]
        #   then interpolate where k crosses self.k_target -> cycle_length (EFPD)
        cycle_length = self._cycle_from_curve(results)

        # 4) Power peaking from the pin/assembly fission tally ----------------
        peaking = self._peaking_from_tally(case_dir)

        # beginning-of-life k for the constraint:
        k_bol = float(results[0].get_keff()[0])

        e_in  = design["enrich_inner"]
        e_out = design["enrich_outer"]
        return {
            "cycle_length": cycle_length,
            "peaking":      peaking,
            "g_kmin":  1.02 - k_bol,
            "g_kmax":  k_bol - 1.35,
            "g_enr":   max(e_in, e_out) - 19.75,
            "g_peak":  peaking - 2.0,
            "k_bol":   k_bol,
        }

    # --- helpers you will finish in Weeks 6-9 --------------------------------
    def _burnup_to_timesteps(self):
        # Convert cumulative MWd/kgHM points to per-step increments.
        bu = np.array(self.burnup_steps, dtype=float)
        steps = np.diff(np.concatenate([[0.0], bu]))
        return list(steps), "MWd/kg"

    def _cycle_from_curve(self, results):
        # TODO Week 7: interpolate EFPD where k_inf crosses k_target.
        raise NotImplementedError("Wire up to results.get_times()/get_keff().")

    def _peaking_from_tally(self, case_dir):
        # TODO Week 7: read the fission-rate mesh tally, normalise, take max/avg.
        raise NotImplementedError("Wire up to the StatePoint fission tally.")


# =============================================================================
# 4.  SURROGATE MODELS  (predict objectives + give an uncertainty)
# =============================================================================
class Surrogate:
    """Base interface: fit(X, Y) then predict(X) -> (mean, std)."""
    def fit(self, X, Y):  raise NotImplementedError
    def predict(self, X): raise NotImplementedError


class GPSurrogate(Surrogate):
    """One Gaussian Process per output column. Native, well-calibrated
    uncertainty -- ideal to *drive* active learning.  Great for <~1000 points,
    which is exactly the regime of expensive OpenMC data."""
    def __init__(self):
        self.xscaler = StandardScaler()
        self.models: list[GaussianProcessRegressor] = []

    def fit(self, X, Y):
        Xs = self.xscaler.fit_transform(X)
        self.models = []
        n_out = Y.shape[1]
        for j in range(n_out):
            kernel = (ConstantKernel(1.0, (1e-3, 1e4))
                      * Matern(length_scale=np.ones(X.shape[1]),
                               length_scale_bounds=(1e-2, 1e3), nu=2.5)
                      + WhiteKernel(1e-3, (1e-8, 1e1)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=4,
                                          random_state=0)
            gp.fit(Xs, Y[:, j])
            self.models.append(gp)
        return self

    def predict(self, X):
        Xs = self.xscaler.transform(np.atleast_2d(X))
        means, stds = [], []
        for gp in self.models:
            m, s = gp.predict(Xs, return_std=True)
            means.append(m); stds.append(s)
        return np.column_stack(means), np.column_stack(stds)


class MLPEnsembleSurrogate(Surrogate):
    """A small ensemble of MLPs.  Mean = ensemble average; uncertainty =
    ensemble spread.  This is the bridge toward the KIT 'MLP' baseline and the
    later CNN/GNN stretch goals -- swap MLPRegressor for a torch model and keep
    the same fit/predict signature."""
    def __init__(self, n_models=5, hidden=(64, 64), seed=0):
        self.n_models = n_models
        self.hidden = hidden
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()
        self.seed = seed
        self.models = []

    def fit(self, X, Y):
        Xs = self.xscaler.fit_transform(X)
        Ys = self.yscaler.fit_transform(Y)
        self.models = []
        for k in range(self.n_models):
            m = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu",
                             solver="adam", max_iter=2000, alpha=1e-4,
                             random_state=self.seed + k)
            m.fit(Xs, Ys)
            self.models.append(m)
        return self

    def predict(self, X):
        Xs = self.xscaler.transform(np.atleast_2d(X))
        preds = np.stack([m.predict(Xs) for m in self.models])  # (E, n, out)
        mean_s = preds.mean(axis=0)
        std_s = preds.std(axis=0)
        mean = self.yscaler.inverse_transform(mean_s)
        std = std_s * self.yscaler.scale_       # un-scale the spread
        return mean, std


# =============================================================================
# 5.  SURROGATE-DRIVEN pymoo PROBLEM
# =============================================================================
class _SurrogateProblem(Problem):
    """A pymoo Problem whose objectives/constraints come from the surrogate.
    Instantaneous to evaluate -> NSGA-II can run thousands of generations.

    Constraints listed in spec.exact_constraints are NOT taken from the
    surrogate: their column of G is overwritten with the closed-form value
    (e.g. the vessel-fit g_geom). The acquisition therefore never proposes a
    geometrically unbuildable design, even at iteration 1 when the GP
    (Gaussian Process) has seen almost no data."""
    def __init__(self, spec, obj_surrogate, con_surrogate):
        super().__init__(n_var=spec.design_space.n,
                         n_obj=spec.n_obj,
                         n_ieq_constr=spec.n_constr,
                         xl=spec.design_space.xl,
                         xu=spec.design_space.xu)
        self.spec = spec
        self.obj_surrogate = obj_surrogate
        self.con_surrogate = con_surrogate
        self._exact_cols = [(spec.constraint_names.index(name), fn)
                            for name, fn in spec.exact_constraints.items()]

    def _evaluate(self, X, out, *a, **k):
        f_mean, _ = self.obj_surrogate.predict(X)
        out["F"] = f_mean
        if self.con_surrogate is not None:
            g_mean, _ = self.con_surrogate.predict(X)
            g_mean = np.atleast_2d(g_mean)
            for col, fn in self._exact_cols:
                g_mean[:, col] = [fn(self.spec.design_space.as_dict(x))
                                  for x in np.atleast_2d(X)]
            out["G"] = g_mean


# =============================================================================
# 6.  THE DUAL-STAGE ACTIVE-LEARNING OPTIMIZER
# =============================================================================
@dataclass
class OptimizerConfig:
    n_init: int = 24            # Stage-1 DOE size (real evaluations)
    n_iter: int = 8             # active-learning iterations
    n_infill: int = 6           # real evaluations added per iteration
    nsga_pop: int = 80          # NSGA-II population on the surrogate
    nsga_gen: int = 120         # NSGA-II generations on the surrogate
    surrogate: str = "gp"       # "gp" or "mlp"
    seed: int = 0
    hv_ref: tuple | None = None # reference point for hypervolume (in MIN space)


class ActiveLearningMOO:
    def __init__(self, spec: ProblemSpec, evaluator: Evaluator,
                 config: OptimizerConfig = OptimizerConfig()):
        self.spec = spec
        self.evaluator = evaluator
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.X = np.empty((0, spec.design_space.n))
        self.F = np.empty((0, spec.n_obj))
        self.G = np.empty((0, spec.n_constr))
        self.raw: list[dict] = []
        self.history = []          # hypervolume per iteration
        self._hv_ref_frozen = None # fixed reference point (set after Stage 1)

    # ---- surrogate factory --------------------------------------------------
    def _new_surrogate(self):
        return GPSurrogate() if self.cfg.surrogate == "gp" \
            else MLPEnsembleSurrogate(seed=self.cfg.seed)

    # ---- bookkeeping --------------------------------------------------------
    def _add(self, X, F, G, raw):
        self.X = np.vstack([self.X, X])
        self.F = np.vstack([self.F, F])
        self.G = np.vstack([self.G, G])
        self.raw.extend(raw)

    def _feasible_mask(self):
        return np.all(self.G <= 1e-9, axis=1) if self.spec.n_constr else \
            np.ones(len(self.F), dtype=bool)

    def _nondominated(self, F):
        """Pareto-front mask (minimisation)."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            dominated_by = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            dominated_by[i] = False
            if dominated_by.any():
                nd[i] = False
        return nd

    def _hv(self):
        feas = self._feasible_mask()
        if feas.sum() == 0:
            return 0.0
        Ff = self.F[feas]
        nd = self._nondominated(Ff)
        front = Ff[nd]
        # The reference point MUST be fixed across iterations, otherwise the
        # hypervolume is not a valid convergence diagnostic. We freeze it once
        # (either user-supplied, or derived from the first feasible points).
        if self._hv_ref_frozen is None:
            if self.cfg.hv_ref is not None:
                self._hv_ref_frozen = np.array(self.cfg.hv_ref, dtype=float)
            else:
                worst = front.max(axis=0)
                self._hv_ref_frozen = worst + 0.25 * np.abs(worst) + 1e-6
        ref = self._hv_ref_frozen
        # only count points dominating the reference point
        keep = np.all(front < ref, axis=1)
        if not keep.any():
            return 0.0
        return float(HV(ref_point=ref)(front[keep]))

    # ---- main loop ----------------------------------------------------------
    def run(self, verbose=True):
        t0 = time.time()
        # ---- STAGE 1 : space-filling DOE, evaluated with the TRUTH ----------
        # Skipped entirely on a RESUMED run: if load_checkpoint() has already
        # pre-seeded the data set, we keep the existing (X, F, G, raw, history,
        # frozen HV reference) intact and go straight to more active learning.
        if len(self.X) == 0:
            accept = (self.spec.exact_ok if self.spec.exact_constraints
                      else None)
            X0 = self.spec.design_space.lhs(self.cfg.n_init,
                                            seed=self.cfg.seed, accept=accept)
            F0, G0, raw0 = self.evaluator.evaluate(X0)
            self._add(X0, F0, G0, raw0)
            self.history.append(self._hv())
            if verbose:
                print(f"[Stage 1] {self.cfg.n_init} real evaluations done. "
                      f"HV={self.history[-1]:.4g}")
        else:
            if not self.history:
                self.history.append(self._hv())
            if verbose:
                print(f"[Resume] continuing from {len(self.X)} prior real "
                      f"evaluations. HV={self.history[-1]:.4g}")

        # ---- STAGE 2 : active-learning loop ---------------------------------
        for it in range(self.cfg.n_iter):
            obj_sur = self._new_surrogate().fit(self.X, self.F)
            con_sur = (self._new_surrogate().fit(self.X, self.G)
                       if self.spec.n_constr else None)

            # NSGA-II on the surrogate (cheap):
            prob = _SurrogateProblem(self.spec, obj_sur, con_sur)
            algo = NSGA2(pop_size=self.cfg.nsga_pop, sampling=LHS())
            res = minimize(prob, algo,
                           ("n_gen", self.cfg.nsga_gen),
                           seed=self.cfg.seed + it, verbose=False)
            cand = np.atleast_2d(res.X)

            # ---- infill / acquisition: pick the most UNCERTAIN candidates ----
            # (exploration). You can blend in predicted hypervolume gain later.
            _, std = obj_sur.predict(cand)
            score = (std / (std.max(axis=0) + 1e-12)).sum(axis=1)
            # de-duplicate against already-evaluated points
            order = np.argsort(-score)
            chosen, picked = [], 0
            for idx in order:
                x = cand[idx]
                if self.X.size and np.min(np.linalg.norm(self.X - x, axis=1)) < 1e-6:
                    continue
                chosen.append(x); picked += 1
                if picked >= self.cfg.n_infill:
                    break
            if not chosen:                      # fallback: random explore
                chosen = list(self.spec.design_space.lhs(self.cfg.n_infill,
                                                         seed=self.cfg.seed + 99 + it))
            Xinf = np.array(chosen)

            # evaluate the infill points with the TRUTH:
            Finf, Ginf, rawinf = self.evaluator.evaluate(Xinf)
            self._add(Xinf, Finf, Ginf, rawinf)
            self.history.append(self._hv())
            if verbose:
                print(f"[Stage 2] iter {it+1}/{self.cfg.n_iter}: "
                      f"+{len(Xinf)} real evals "
                      f"(total {self.evaluator.n_calls}), "
                      f"HV={self.history[-1]:.4g}")

        if verbose:
            print(f"Done in {time.time()-t0:.1f}s, "
                  f"{self.evaluator.n_calls} total real evaluations.")
        return self.results()

    # ---- final products -----------------------------------------------------
    def results(self) -> dict:
        feas = self._feasible_mask()
        Ff = self.F[feas]
        Xf = self.X[feas]
        raw_f = [r for r, m in zip(self.raw, feas) if m]
        nd = self._nondominated(Ff) if len(Ff) else np.array([], dtype=bool)
        return {
            "pareto_X": Xf[nd] if len(Ff) else np.empty((0, self.spec.design_space.n)),
            "pareto_F": Ff[nd] if len(Ff) else np.empty((0, self.spec.n_obj)),
            "pareto_raw": [r for r, m in zip(raw_f, nd) if m] if len(Ff) else [],
            "all_X": self.X, "all_F": self.F, "all_G": self.G, "all_raw": self.raw,
            "hv_history": self.history,
            "n_real_evaluations": self.evaluator.n_calls,
        }

    def save(self, path="optimization_results.json"):
        r = self.results()
        out = {
            "design_variables": self.spec.design_space.names,
            "objectives": [(o.name, "max" if o.maximize else "min")
                           for o in self.spec.objectives],
            "pareto_designs": [self.spec.design_space.as_dict(x)
                               for x in r["pareto_X"]],
            "pareto_raw": r["pareto_raw"],
            "hv_history": r["hv_history"],
            "n_real_evaluations": r["n_real_evaluations"],
        }
        Path(path).write_text(json.dumps(out, indent=2, default=float))
        return path

    # ---- checkpointing: FULL evaluation set, so a later run can RESUME -------
    def save_checkpoint(self, path="optimization_checkpoint.json", meta=None):
        """Write EVERY real evaluation (not just the Pareto subset) plus the
        hypervolume history and its frozen reference point. This is the file
        --resume reads: the surrogate must be retrained on the WHOLE data set
        (dominated points included), so the Pareto-only results file is NOT
        enough to continue from. `meta` (optional) stores run settings such as
        k_target so a resume can warn on a mismatch."""
        r = self.results()
        out = {
            "design_variables": self.spec.design_space.names,
            "objectives": [(o.name, "max" if o.maximize else "min")
                           for o in self.spec.objectives],
            "constraint_names": self.spec.constraint_names,
            "all_raw": r["all_raw"],                 # <-- the complete record
            "hv_history": r["hv_history"],
            "hv_ref": (self._hv_ref_frozen.tolist()
                       if self._hv_ref_frozen is not None else None),
            "n_real_evaluations": r["n_real_evaluations"],
        }
        if meta:
            out["meta"] = dict(meta)
        Path(path).write_text(json.dumps(out, indent=2, default=float))
        return path

    def _seed_from_raw(self, raw_list):
        """Rebuild (X, F, G, raw) from a list of physical raw dicts using this
        problem's spec -- the EXACT inverse of Evaluator.evaluate(), so the
        internal minimise-space sign convention (e.g. cycle_length stored
        negated) is reproduced faithfully."""
        names = self.spec.design_space.names
        X = np.array([[float(r[n]) for n in names] for r in raw_list])
        F = np.array([[obj.to_min(r[obj.name]) for obj in self.spec.objectives]
                      for r in raw_list])
        G = (np.array([[float(r[c]) for c in self.spec.constraint_names]
                       for r in raw_list])
             if self.spec.n_constr else np.empty((len(raw_list), 0)))
        self._add(X, F, G, [dict(r) for r in raw_list])

    def load_checkpoint(self, path):
        """Pre-seed this optimizer from a checkpoint written by
        save_checkpoint(). Call BEFORE run(): run() then skips the Stage-1 DOE
        and adds cfg.n_iter MORE active-learning iterations on top of the
        loaded data. Returns the number of evaluations loaded."""
        ckpt = json.loads(Path(path).read_text())
        if ckpt["design_variables"] != self.spec.design_space.names:
            raise ValueError(
                f"checkpoint design variables {ckpt['design_variables']} do not "
                f"match this problem {self.spec.design_space.names}; are you "
                f"resuming the right run?")
        ck_con = ckpt.get("constraint_names")
        if ck_con is not None and list(ck_con) != list(self.spec.constraint_names):
            raise ValueError(
                f"checkpoint constraints {ck_con} do not match this problem "
                f"{self.spec.constraint_names}. A checkpoint written before the "
                f"geometry fix (no g_geom, clipped-fuel core, old k_target "
                f"table) is PHYSICALLY inconsistent with the corrected model "
                f"-- start a fresh run instead of resuming it.")
        self._seed_from_raw(ckpt["all_raw"])
        self.history = list(ckpt.get("hv_history", []))
        hv_ref = ckpt.get("hv_ref")
        self._hv_ref_frozen = (np.array(hv_ref, dtype=float)
                               if hv_ref is not None else None)
        # continue case numbering so OpenMC scratch dirs never collide
        self.evaluator.n_calls = len(ckpt["all_raw"])
        return len(ckpt["all_raw"])


# =============================================================================
# 7.  A REACTOR-FLAVOURED EXAMPLE PROBLEM
# =============================================================================
def example_reactor_problem() -> ProblemSpec:
    """5 design variables, 2 objectives, 5 constraints -- the same SHAPE as
    your real OpenMC problem.  Edit freely.

    BOUNDS (updated for the corrected core geometry): the fuel envelope
    R_env = sqrt(13)*17*pitch plus the reflector must fit inside the vessel
    (inner radius 90 cm), so pitch and refl_thick are COUPLED through
    g_geom (see core_geometry.geometry_margin). The box below is the tight
    bounding box of the feasible slab:
      * refl_thick <= 19.5 cm  (19.51 cm is the most that EVER fits, at the
        minimum pitch 1.15; the old 25 cm bound was unbuildable at any pitch),
      * pitch <= 1.43 cm       (above ~1.436 cm not even the 2 cm minimum
        reflector fits; the old 1.45 bound had zero feasible reflector).
    The diagonal cut INSIDE the box is enforced by g_geom, which is analytic
    (exact_constraints), so the DOE and the acquisition never propose a
    design that cannot physically be built."""
    from core_geometry import geometry_margin

    ds = DesignSpace([
        DesignVariable("enrich_inner", 2.0, 19.75, "%"),
        DesignVariable("enrich_outer", 2.0, 19.75, "%"),
        DesignVariable("gd_wt",        0.0,  8.0,  "wt% Gd2O3"),
        DesignVariable("pitch",        1.15, 1.43, "cm"),
        DesignVariable("refl_thick",   2.0,  19.5, "cm"),
    ])
    objs = [
        Objective("cycle_length", maximize=True,  label="Cycle length [EFPD]"),
        Objective("peaking",      maximize=False, label="Power peaking factor"),
    ]
    constraints = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom"]
    exact = {"g_geom": lambda d: geometry_margin(d["pitch"], d["refl_thick"])}
    return ProblemSpec(ds, objs, constraints, exact_constraints=exact)


# =============================================================================
# 8.  DEMO  (runs with the analytic evaluator -- no OpenMC needed)
# =============================================================================
def _demo(make_plot=True, outdir="."):
    spec = example_reactor_problem()
    evaluator = AnalyticEvaluator(spec, noise=0.02, seed=1)
    cfg = OptimizerConfig(n_init=24, n_iter=8, n_infill=6,
                          nsga_pop=60, nsga_gen=80, surrogate="gp", seed=1)
    opt = ActiveLearningMOO(spec, evaluator, cfg)
    res = opt.run(verbose=True)
    path = opt.save(str(Path(outdir) / "optimization_results.json"))

    pf = res["pareto_F"]
    # convert back to physical units for reporting (cycle was negated)
    cyc = -pf[:, 0]
    peak = pf[:, 1]
    order = np.argsort(cyc)
    print("\nPareto-optimal trade-off (feasible):")
    print(f"{'cycle[EFPD]':>12} {'peaking':>9}  design")
    for i in order:
        d = res["pareto_raw"][i]
        print(f"{cyc[i]:12.1f} {peak[i]:9.3f}  "
              f"e_in={d['enrich_inner']:.2f} e_out={d['enrich_outer']:.2f} "
              f"Gd={d['gd_wt']:.2f} pitch={d['pitch']:.3f} refl={d['refl_thick']:.1f}")

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        allF = res["all_F"]
        ax[0].scatter(-allF[:, 0], allF[:, 1], s=18, c="lightgray",
                      label="all evaluations")
        ax[0].scatter(cyc, peak, s=42, c="crimson", zorder=3,
                      label="Pareto front")
        ax[0].set_xlabel("Cycle length [EFPD]  (maximise →)")
        ax[0].set_ylabel("Power peaking factor  (← minimise)")
        ax[0].set_title("Objective space")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(range(len(res["hv_history"])), res["hv_history"],
                   "o-", c="navy")
        ax[1].set_xlabel("Active-learning iteration")
        ax[1].set_ylabel("Hypervolume (feasible)")
        ax[1].set_title("Convergence")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        figpath = Path(outdir) / "optimization_demo.png"
        fig.savefig(figpath, dpi=130)
        print(f"\nSaved plot -> {figpath}")
    print(f"Saved results -> {path}")
    return res


if __name__ == "__main__":
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    _demo()
