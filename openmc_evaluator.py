"""
openmc_evaluator.py
===================
The REAL "truth" evaluator for reactor_optimization.ActiveLearningMOO.

Per design it returns (cycle_length_EFPD, peaking) + constraints by:
  1. building a 17x17 assembly model WITH a pin-fission mesh tally,
  2. giving every fuel MATERIAL a volume (+ depletable=True) so OpenMC can
     normalise power per gram of heavy metal,
  3. running an ADAPTIVE depletion (see below) at the CORE specific power,
  4. interpolating the burnup where k_inf crosses a leakage-corrected EOC
     (End Of Cycle) target -- Route A frozen value or Route B per-design
     table -- and converting that burnup to EFPD,
  5. reading the BOL (Beginning Of Life) pin-power tally for the radial
     peaking factor F_dh,
  6. reporting the ANALYTIC vessel-fit constraint g_geom (core_geometry).

ADAPTIVE DEPLETION (replaces the fixed 13-step, 35.5 MWd/kgHM schedule)
-----------------------------------------------------------------------
The old fixed schedule imposed a hard ceiling of 35.5 MWd/kgHM = ~3556 EFPD:
54 % of the first campaign's evaluations hit it, turning their cycle lengths
into lower bounds instead of measurements -- and every SHORT design still paid
for all 13 steps even after crossing the target early. Both problems are gone:

    * a short BOL block (default 0.5,1,2,4,6 MWd/kg) resolves the Xe/Sm
      transient and the start of the Gd (gadolinium) burnout hump, then
    * the depletion marches in small restart chunks (default 2 steps x
      4 MWd/kg) and STOPS as soon as k has PEAKED and fallen through the
      design's k_target -- or aborts early if the peak clearly cannot reach
      the target (never-critical designs cost ~6 solves, not 14),
    * up to a hard cap `max_burnup` (default 100 MWd/kgHM ~ 10,020 EFPD at
      9.98 W/gHM). A design still above target at the cap is flagged
      `censored=True` and its EFPD is an explicit LOWER BOUND.

Chunks restart through the public OpenMC API: a new CoupledOperator with
prev_results continues from the previous chunk's compositions; each chunk
writes its own depletion_results.h5 in case_NNNN/dep_MM/. Power is fixed in
WATTS from the FRESH heavy-metal mass of chunk 0 and timesteps are given in
DAYS, so restarts cannot drift the power or the burnup bookkeeping (a nominal
"MWd/kg" step on a restarted operator would be normalised by the DEPLETED
mass, silently shrinking late-life steps).

The EOC interpolation uses the LAST DOWNWARD crossing of k through k_target
(core_geometry.eoc_crossing_burnup). The old np.interp(-k_target, -k, bu)
assumed k monotonically decreasing and silently returned garbage for any Gd
design whose k rises through the burnout hump first.

Requires: openmc (with cross sections + a depletion chain), reactor_model.py
and core_geometry.py in the same folder. See run_optimization.py for the
driver and sweep_ktarget.py for the Route B table.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

import openmc
import openmc.deplete

import core_geometry as cg
import reactor_model as rm
from reactor_optimization import Evaluator, ProblemSpec


# BOL block: coarse through the Xe/Sm transient, cumulative 13.5 MWd/kgHM.
DEFAULT_BOL_STEPS = (0.5, 1.0, 2.0, 4.0, 6.0)
DEFAULT_DEP_STEP = 4.0      # MWd/kgHM per marching step (~401 EFPD)
DEFAULT_CHUNK_STEPS = 2     # marching steps per operator restart
DEFAULT_MAX_BURNUP = 100.0  # MWd/kgHM hard cap (~10,020 EFPD; anti-runaway)


class OpenMCEvaluator(Evaluator):
    """Run an OpenMC assembly depletion per design and return the objectives.

    Parameters
    ----------
    spec : ProblemSpec
        Must match example_reactor_problem(): objectives {cycle_length,
        peaking}, constraints {g_kmin, g_kmax, g_enr, g_peak, g_geom}.
    k_target : float | str | dict
        ROUTE A -- a single float: the FROZEN leakage-corrected EOC target,
        the SAME value for every design regardless of refl_thick.
        ROUTE B -- a path to (or an already-loaded dict of) the JSON table
        written by sweep_ktarget.py; k_target is interpolated per design.
        Both table schemas are supported:
          * 1-D (legacy): keys refl_thick_cm, k_target -> np.interp on
            refl_thick (clamped at the grid edges);
          * 2-D (schema 2): keys pitch_cm, refl_thick_cm, k_target[i][j] ->
            bilinear interpolation on (pitch, refl_thick), clamped. The 2-D
            table removes the old nominal-pitch bias, which matters now that
            g_geom couples pitch and refl_thick (thick-reflector designs are
            FORCED to small pitch -- exactly where a pitch-1.26 table is most
            wrong).
        Route B pairs with the bc="reflective" (infinite-medium) depletion
        below -- do not ALSO give the depletion model an explicit reflector,
        or reflector leakage is counted twice.
    chain_file : str | None
        Path to the depletion chain. Falls back to $OPENMC_CHAIN_FILE.
    transport : dict
        Particle/batch settings forwarded to make_assembly_model(). Keep
        modest during exploration (e.g. 4000 particles); re-score the final
        Pareto front at high fidelity afterwards (two-fidelity strategy).
    workdir : str
        Each design runs in workdir/case_NNNN/ so runs never collide.
    op, geo : optional
        Operating() and Geometry17x17() overrides.
    bol_steps, dep_step, chunk_steps, max_burnup :
        The adaptive schedule knobs described in the module docstring.
    """

    def __init__(self, spec: ProblemSpec, *,
                 k_target: float | str | dict,
                 chain_file: str | None = None,
                 transport: dict | None = None,
                 workdir: str = "openmc_runs",
                 op=None, geo=None,
                 bol_steps=DEFAULT_BOL_STEPS,
                 dep_step: float = DEFAULT_DEP_STEP,
                 chunk_steps: int = DEFAULT_CHUNK_STEPS,
                 max_burnup: float = DEFAULT_MAX_BURNUP,
                 verbose: bool = True):
        super().__init__(spec)

        # ROUTE A (float) vs ROUTE B (1-D or 2-D table) -- detected once here;
        # _k_target_for() does the per-design lookup every evaluation.
        self.k_target = None
        self._kt_1d = None          # (refl_grid, k_target_grid)
        self._kt_2d = None          # (pitch_grid, refl_grid, Z[i][j])
        if isinstance(k_target, (str, Path)):
            with open(k_target) as f:
                table = json.load(f)
            self._load_table(table)
        elif isinstance(k_target, dict):
            self._load_table(k_target)
        else:
            self.k_target = float(k_target)
        self.route_b = self.k_target is None

        self.chain_file = chain_file or os.environ.get("OPENMC_CHAIN_FILE")
        if not self.chain_file:
            raise RuntimeError(
                "No depletion chain. Set OPENMC_CHAIN_FILE or pass chain_file=...")
        openmc.config["chain_file"] = self.chain_file

        self.transport = dict(transport or
                              dict(particles=4000, batches=60, inactive=20))
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.op = op or rm.Operating()
        self.geo = geo or rm.Geometry17x17()

        self.bol_steps = [float(s) for s in bol_steps]
        self.dep_step = float(dep_step)
        self.chunk_steps = max(1, int(chunk_steps))
        self.max_burnup = float(max_burnup)
        if sum(self.bol_steps) >= self.max_burnup:
            raise ValueError("max_burnup must exceed the BOL block "
                             f"({sum(self.bol_steps)} MWd/kg).")
        self.verbose = verbose

        # Specific power depends on geometry (active height, pellet radius,
        # assembly count) -- none of which are design variables here -- so it
        # is constant across the search and computed once. [W/gHM]
        self.spec_power = rm.core_specific_power_w_per_g(self.op, self.geo)

    def _load_table(self, table: dict):
        refl = np.array(table["refl_thick_cm"], dtype=float)
        kt = np.array(table["k_target"], dtype=float)
        if "pitch_cm" in table:                      # 2-D schema
            pitch = np.array(table["pitch_cm"], dtype=float)
            if kt.shape != (len(pitch), len(refl)):
                raise ValueError(
                    f"k_target table shape {kt.shape} does not match "
                    f"(pitch {len(pitch)} x refl {len(refl)})")
            self._kt_2d = (pitch, refl, kt)
        else:                                        # 1-D legacy schema
            self._kt_1d = (refl, kt)

    # ------------------------------------------------------------------ #
    # one design -> objectives + constraints                             #
    # ------------------------------------------------------------------ #
    def evaluate_one(self, design: dict) -> dict:
        case = self.workdir / f"case_{self.n_calls:04d}"
        case.mkdir(parents=True, exist_ok=True)

        peaking = self._bol_peaking(design, case)
        (cycle_efpd, k_bol, k_target_used,
         censored, bu_eoc, n_solves) = self._cycle_length(design, case)

        e_in = design["enrich_inner"]
        e_out = design["enrich_outer"]
        res = {
            "cycle_length": cycle_efpd,                 # objective (maximise)
            "peaking":      peaking,                    # objective (minimise)
            "g_kmin":  1.02 - k_bol,                    # need k_bol >= 1.02
            "g_kmax":  k_bol - 1.35,                    # and  k_bol <= 1.35
            "g_enr":   max(e_in, e_out) - 19.75,        # LEU cap
            "g_peak":  peaking - 2.0,                   # peaking <= 2.0
            "g_geom":  cg.geometry_margin(design["pitch"],
                                          design["refl_thick"]),
            "k_bol":   k_bol,                           # carried for plots
            "k_target": k_target_used,                  # carried for analysis
            "censored": bool(censored),   # True -> EFPD is a LOWER BOUND (cap)
            "bu_eoc_mwd_kg": bu_eoc,                    # EOC burnup [MWd/kgHM]
            "n_dep_solves": n_solves,     # transport solves spent on depletion
        }
        if self.verbose:
            print(f"  [case {self.n_calls:04d}] "
                  f"e=({e_in:5.2f}/{e_out:5.2f}) Gd={design['gd_wt']:4.2f} "
                  f"p={design['pitch']:.3f} refl={design['refl_thick']:5.1f} "
                  f"k_target={k_target_used:.4f} "
                  f"-> EFPD={cycle_efpd:7.0f}{'(CEN)' if censored else '     '} "
                  f"F_dh={peaking:.3f} k_bol={k_bol:.4f} "
                  f"[{n_solves} solves]")
        return res

    # ------------------------------------------------------------------ #
    # BOL radial peaking from a fresh-assembly mesh tally                 #
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
    # EOC target for THIS design -- frozen (Route A), refl-interpolated   #
    # (Route B, 1-D legacy table) or (pitch, refl)-interpolated (2-D)     #
    # ------------------------------------------------------------------ #
    def _k_target_for(self, design: dict) -> float:
        if not self.route_b:
            return self.k_target
        if self._kt_2d is not None:
            pitch_g, refl_g, Z = self._kt_2d
            # bilinear, CLAMPED at the grid edges (like np.interp in 1-D)
            return cg.bilinear_clamped(design["pitch"], design["refl_thick"],
                                       pitch_g, refl_g, Z)
        refl_g, kt_g = self._kt_1d
        return float(np.interp(design["refl_thick"], refl_g, kt_g))

    # ------------------------------------------------------------------ #
    # cycle length from ADAPTIVE assembly depletion                       #
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

        k_target = self._k_target_for(design)

        # ---- adaptive chunked depletion ---------------------------------- #
        bu_hist = [0.0]          # cumulative burnup [MWd/kgHM], OUR bookkeeping
        k_hist: list[float] = []
        state = {"prev": None, "power_w": None, "chunk": 0}
        cwd = Path.cwd()

        def run_chunk(steps_mwd_kg: list[float]):
            """Deplete `steps` more, continuing from the previous chunk."""
            op_dep = openmc.deplete.CoupledOperator(
                model, prev_results=state["prev"], diff_burnable_mats=False)
            if state["power_w"] is None:
                # FRESH heavy-metal mass [g] x specific power [W/g] = watts.
                # Captured ONCE and reused: constant thermal power all cycle,
                # immune to the depleted-mass drift a per-chunk "MWd/kg"
                # normalisation would introduce.
                state["power_w"] = self.spec_power * op_dep.heavy_metal
            # our MWd/kgHM steps -> days at constant power: the fresh HM mass
            # cancels, days = bu * 1000 / spec_power  (same conversion as EFPD)
            days = [s * 1000.0 / self.spec_power for s in steps_mwd_kg]
            integrator = openmc.deplete.PredictorIntegrator(
                op_dep, days, power=state["power_w"], timestep_units="d")
            cdir = case / f"dep_{state['chunk']:02d}"
            cdir.mkdir(parents=True, exist_ok=True)
            try:
                os.chdir(cdir)
                integrator.integrate()
                results = openmc.deplete.Results("depletion_results.h5")
            finally:
                os.chdir(cwd)
            state["chunk"] += 1
            state["prev"] = results
            _t, karr = results.get_keff()
            kvals = [float(v) for v in karr[:, 0]]
            # Stitch robustly whichever restart semantics this OpenMC version
            # uses (chunk-local vs cumulative results): the LAST len(steps)
            # entries are always the newly computed end-of-step states.
            if not k_hist:
                if len(kvals) != len(steps_mwd_kg) + 1:
                    raise RuntimeError(
                        f"depletion chunk returned {len(kvals)} k values for "
                        f"{len(steps_mwd_kg)} steps (expected "
                        f"{len(steps_mwd_kg)+1} incl. BOL) -- OpenMC "
                        f"results layout not understood.")
                k_hist.extend(kvals)              # includes the BOL state
            else:
                k_hist.extend(kvals[-len(steps_mwd_kg):])
            for s in steps_mwd_kg:
                bu_hist.append(bu_hist[-1] + s)
            if len(k_hist) != len(bu_hist):
                raise RuntimeError(
                    f"burnup/k bookkeeping out of sync "
                    f"({len(bu_hist)} vs {len(k_hist)})")

        # 1) BOL block
        run_chunk(list(self.bol_steps))
        k_bol = k_hist[0]

        # 2) march until k has PEAKED and fallen through the target, or cap
        censored = False
        while True:
            past_peak = int(np.argmax(k_hist)) < len(k_hist) - 1
            if past_peak and k_hist[-1] <= k_target:
                break                       # crossing bracketed (or dud past
                                            # a sub-target peak -> interp None)
            remaining = self.max_burnup - bu_hist[-1]
            if remaining <= 1e-9:
                censored = k_hist[-1] > k_target   # still above target at cap
                break
            steps = []
            for _ in range(self.chunk_steps):
                s = min(self.dep_step, remaining - sum(steps))
                if s <= 1e-9:
                    break
                steps.append(s)
            run_chunk(steps)

        # 3) EOC burnup at the LAST DOWNWARD crossing (Gd-hump safe)
        bu_eoc = cg.eoc_crossing_burnup(bu_hist, k_hist, k_target)
        if bu_eoc is None:
            if censored:
                bu_eoc = bu_hist[-1]        # explicit LOWER BOUND at the cap
                efpd = bu_eoc * 1000.0 / self.spec_power
            else:
                bu_eoc = 0.0                # never reached the target: the
                efpd = 0.0                  # core cannot be made critical
        else:
            efpd = bu_eoc * 1000.0 / self.spec_power

        return efpd, k_bol, k_target, censored, bu_eoc, len(k_hist)


# Convenience: quick standalone check of a single design (not the optimizer).
if __name__ == "__main__":
    from reactor_optimization import example_reactor_problem

    spec = example_reactor_problem()
    ev = OpenMCEvaluator(spec, k_target=1.0556,
                         transport=dict(particles=2000, batches=40, inactive=10))
    ref = {"enrich_inner": 4.55, "enrich_outer": 4.05, "gd_wt": 0.0,
           "pitch": 1.26, "refl_thick": 12.0}
    print("specific power:", round(ev.spec_power, 2), "W/gHM")
    print(ev.evaluate_one(ref))
