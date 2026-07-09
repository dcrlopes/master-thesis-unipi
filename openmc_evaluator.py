"""
openmc_evaluator.py
===================
The REAL "truth" evaluator for reactor_optimization.ActiveLearningMOO.

It folds the proven logic from your notebook (01_first_simulation_*.ipynb,
Sections 3-6) into the Evaluator interface, so that each `design` dict becomes

        (cycle_length_EFPD, peaking)  +  constraints

by, per design:
  1. building a 17x17 assembly model WITH a pin-fission mesh tally,
  2. giving every fuel MATERIAL a volume (+ depletable=True) so OpenMC can
     normalise power per gram of heavy metal,
  3. running a PredictorIntegrator depletion at the CORE specific power,
  4. interpolating the burnup where k_inf crosses a *frozen, leakage-corrected*
     EOC target (your Section-4 correction; replaces the old hard-coded 1.03),
     and converting that burnup to EFPD,
  5. reading the BOL pin-power tally for the radial peaking factor F_dh.

NOTHING in reactor_optimization.py changes. You simply build an instance of this
class and hand it to ActiveLearningMOO instead of AnalyticEvaluator.

Requires: openmc (with cross sections + a depletion chain), and reactor_model.py
in the same folder. See run_optimization.py for the driver.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

import openmc
import openmc.deplete

import reactor_model as rm
from reactor_optimization import Evaluator, ProblemSpec


# Default burnup schedule (MWd/kgHM, INCREMENTS not cumulative) -- the same
# shape as notebook cell 24: coarse through the BOL Xe/Sm transient, then FINE
# across the EOC crossing region so the k=target point is resolved, not jumped.
DEFAULT_BURNUP_STEPS = (
    0.5, 1.0, 2.0, 4.0, 6.0,      # 0   -> 13.5  (coarse, BOL transient)
    4.0, 4.0,                     #     -> 21.5  (approaching EOC)
    2.0, 2.0, 2.0, 2.0,           #     -> 29.5  (FINE across crossing)
    3.0, 3.0,                     #     -> 35.5  (just past EOC, margin)
)


class OpenMCEvaluator(Evaluator):
    """Run an OpenMC assembly depletion per design and return the objectives.

    Parameters
    ----------
    spec : ProblemSpec
        Must match example_reactor_problem(): objectives {cycle_length, peaking},
        constraints {g_kmin, g_kmax, g_enr, g_peak}.
    k_target : float
        The FROZEN leakage-corrected EOC target  k_inf = 1.0 * (k_inf/k_eff).
        Measure it ONCE for your finalized reference geometry with
        measure_leakage_target.py and paste the number in. (Your 4.55/4.05,
        pitch 1.26, refl 15 reference gives ~1.085.)
    chain_file : str | None
        Path to the depletion chain. Falls back to $OPENMC_CHAIN_FILE.
    burnup_steps : sequence of float
        Per-step burnup increments in MWd/kgHM.
    transport : dict
        Particle/batch settings forwarded to make_assembly_model(). Keep these
        modest during the optimization (e.g. 4000 particles); the active-learning
        loop tolerates a little statistical noise.
    workdir : str
        Each design runs in workdir/case_NNNN/ so runs never collide.
    op, geo : optional
        Operating() and Geometry17x17() overrides (defaults match the notebook).
    """

    def __init__(self, spec: ProblemSpec, *,
                 k_target: float,
                 chain_file: str | None = None,
                 burnup_steps=DEFAULT_BURNUP_STEPS,
                 transport: dict | None = None,
                 workdir: str = "openmc_runs",
                 op=None, geo=None,
                 verbose: bool = True):
        super().__init__(spec)
        self.k_target = float(k_target)

        self.chain_file = chain_file or os.environ.get("OPENMC_CHAIN_FILE")
        if not self.chain_file:
            raise RuntimeError(
                "No depletion chain. Set OPENMC_CHAIN_FILE or pass chain_file=...")
        openmc.config["chain_file"] = self.chain_file

        self.burnup_steps = list(burnup_steps)
        self.transport = dict(transport or
                              dict(particles=4000, batches=60, inactive=20))
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.op = op or rm.Operating()
        self.geo = geo or rm.Geometry17x17()
        self.verbose = verbose

        # Specific power depends on geometry (active height, pellet radius,
        # assembly count) -- none of which are design variables here -- so it is
        # constant across the search and computed once. [W/gHM]
        self.spec_power = rm.core_specific_power_w_per_g(self.op, self.geo)

    # ------------------------------------------------------------------ #
    # one design -> objectives + constraints                             #
    # ------------------------------------------------------------------ #
    def evaluate_one(self, design: dict) -> dict:
        case = self.workdir / f"case_{self.n_calls:04d}"
        case.mkdir(parents=True, exist_ok=True)

        peaking = self._bol_peaking(design, case)
        cycle_efpd, k_bol = self._cycle_length(design, case)

        e_in = design["enrich_inner"]
        e_out = design["enrich_outer"]
        res = {
            "cycle_length": cycle_efpd,                 # objective (maximise)
            "peaking":      peaking,                    # objective (minimise)
            "g_kmin":  1.02 - k_bol,                    # need k_bol >= 1.02
            "g_kmax":  k_bol - 1.35,                    # and  k_bol <= 1.35
            "g_enr":   max(e_in, e_out) - 19.75,        # LEU cap
            "g_peak":  peaking - 2.0,                   # peaking <= 2.0
            "k_bol":   k_bol,                           # carried for plots
        }
        if self.verbose:
            print(f"  [case {self.n_calls:04d}] "
                  f"e=({e_in:5.2f}/{e_out:5.2f}) Gd={design['gd_wt']:4.2f} "
                  f"p={design['pitch']:.3f} refl={design['refl_thick']:5.1f} "
                  f"-> EFPD={cycle_efpd:7.0f} F_dh={peaking:.3f} "
                  f"k_bol={k_bol:.4f}")
        return res

    # ------------------------------------------------------------------ #
    # BOL radial peaking from a fresh-assembly mesh tally  (cell 10-12)   #
    # ------------------------------------------------------------------ #
    def _bol_peaking(self, design: dict, case: Path) -> float:
        model, _fuel_cells, _lat = rm.make_assembly_model(
            design, self.op, self.geo, bc="reflective", **self.transport)

        N = self.geo.lattice
        pitch = design.get("pitch", 1.26)
        half = N * pitch / 2.0

        mesh = openmc.RegularMesh()
        mesh.dimension = (N, N)
        mesh.lower_left = (-half, -half)
        mesh.upper_right = (half, half)

        t = openmc.Tally(name="pin_fission")
        t.filters = [openmc.MeshFilter(mesh)]
        t.scores = ["fission"]
        model.tallies = openmc.Tallies([t])

        sp_path = model.run(cwd=str(case / "bol"), output=False)
        with openmc.StatePoint(sp_path) as sp:
            fiss = sp.get_tally(name="pin_fission").get_values(
                scores=["fission"]).reshape((N, N))

        # normalise to the mean of FUELLED pins (guide tubes read 0 -> masked)
        fm = np.ma.masked_equal(fiss, 0.0)
        return float((fm / fm.mean()).max())

    # ------------------------------------------------------------------ #
    # cycle length from assembly depletion  (cells 23-25)                #
    # ------------------------------------------------------------------ #
    def _cycle_length(self, design: dict, case: Path):
        model, fuel_cells, _lat = rm.make_assembly_model(
            design, self.op, self.geo, bc="reflective", **self.transport)

        # give each fuel material a volume + mark depletable
        pin_vol = math.pi * self.geo.fuel_or ** 2 * self.geo.active_height
        counts = Counter(c.fill.id for c in fuel_cells)
        id2mat = {m.id: m for m in model.materials}
        for mat_id, npins in counts.items():
            m = id2mat[mat_id]
            m.volume = npins * pin_vol
            m.depletable = True

        op_dep = openmc.deplete.CoupledOperator(model, diff_burnable_mats=False)
        integrator = openmc.deplete.PredictorIntegrator(
            op_dep, self.burnup_steps,
            power_density=self.spec_power, timestep_units="MWd/kg")

        cwd = Path.cwd()
        try:
            os.chdir(case)
            integrator.integrate()
            results = openmc.deplete.Results("depletion_results.h5")
        finally:
            os.chdir(cwd)

        _t, k = results.get_keff()          # cumulative time, k with uncertainty
        kvals = k[:, 0]
        bu = np.cumsum([0.0] + self.burnup_steps)[:len(kvals)]
        k_bol = float(kvals[0])

        if kvals.min() <= self.k_target:
            # np.interp needs an increasing x; k decreases, so feed -k vs -target
            cycle_bu = float(np.interp(-self.k_target, -kvals, bu))
            efpd = cycle_bu * 1000.0 / self.spec_power
        else:
            # never reached target in the previewed window: floor at last burnup
            # (a finite, conservative value so the optimizer keeps a usable
            # gradient). Widen DEFAULT_BURNUP_STEPS if this fires often.
            efpd = float(bu[-1] * 1000.0 / self.spec_power)

        return efpd, k_bol


# Convenience: quick standalone check of a single design (not the optimizer).
if __name__ == "__main__":
    from reactor_optimization import example_reactor_problem

    spec = example_reactor_problem()
    ev = OpenMCEvaluator(spec, k_target=1.085,
                         transport=dict(particles=2000, batches=40, inactive=10))
    ref = {"enrich_inner": 4.55, "enrich_outer": 4.05, "gd_wt": 0.0,
           "pitch": 1.26, "refl_thick": 15.0}
    print("specific power:", round(ev.spec_power, 2), "W/gHM")
    print(ev.evaluate_one(ref))
