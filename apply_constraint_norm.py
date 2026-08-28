#!/usr/bin/env python3
"""apply_constraint_norm.py -- normalize the OPTIMIZER'S view of G.

WHY
---
pymoo 0.6.2 ranks infeasible designs by CV = sum of max(g_i, 0) over the RAW
constraint values. Our five constraints live in four different units:

    g_kmin, g_kmax : delta-k          (typical violation 0.001 .. 0.10)
    g_enr          : wt% U-235        (0 .. ~18 inside the search box)
    g_peak         : peaking factor   (typical violation 0.001 .. 0.5)
    g_geom         : centimetres      (-17 .. +17 across the NSGA-II box)

Summing those raw numbers weights the constraints by their UNITS, not by
their importance. In an all-infeasible population (C4 and C5 are exactly
that) NSGA-II degenerates to sorting by this sum, so:
  * on the surrogate population, one centimetre of g_geom outweighs
    ten thousand pcm of g_kmax,
  * in the truth archive, g_peak carries 94% of the summed violation and
    the k-window almost none, although the C4 finding is that the
    g_peak / g_kmax TANGENCY is the frontier that matters.

FIX
---
Divide each g by its own limit before it reaches pymoo, the constraint GP
(Gaussian Process), and the CV (constraint violation) ranking:

    g_kmin / k_min      g_kmax / k_max      g_enr / enr_max
    g_peak / f_max      g_geom / (R_VESSEL_INNER - VESSEL_CLEARANCE_CM)

so every constraint is expressed as a FRACTION OF ITS OWN LIMIT.

WHAT DOES NOT CHANGE
--------------------
  * The feasible set. Scales are positive, so g/s <= 0 iff g <= 0.
  * The raw dicts, checkpoints, salvage CSVs, and every rescore_*.py /
    plot script: they keep PHYSICAL g values. Division happens only in
    the three places where the optimizer assembles its G matrix.
  * Old checkpoints resume correctly WITHOUT migration: _seed_from_raw
    re-divides the stored physical values at load time.
  * Objectives, hypervolume, and the frozen HV reference point.

WHAT CHANGES
------------
  * The ranking AMONG infeasible designs (survival selection, and the
    least-infeasible infill fallback). That is the intended change.

Edits (anchor-verified, each anchor must appear EXACTLY once):
  reactor_optimization.py : ProblemSpec field + g_scale helpers,
                            Evaluator.evaluate, _seed_from_raw,
                            _SurrogateProblem exact-column overwrite,
                            example_reactor_problem default scales.
  run_optimization.py     : sync scales from the evaluator's actual limits
                            (k_basis / CLI overrides included).

Originals are backed up as <file>.bak.norm. Re-running is refused once the
CONSTRAINT-NORM marker is present.
"""
from __future__ import annotations
import py_compile
import shutil
import sys
from pathlib import Path

MARKER = "CONSTRAINT-NORM"

RO = Path("reactor_optimization.py")
RU = Path("run_optimization.py")

# ---------------------------------------------------------------------------
# (anchor, replacement) pairs. Anchors are exact strings from commit c995652.
# ---------------------------------------------------------------------------

EDITS_RO = [

# -- A: ProblemSpec: add the scales field --------------------------------
(
"""    exact_constraints: dict = field(default_factory=dict)
""",
"""    exact_constraints: dict = field(default_factory=dict)
    # CONSTRAINT-NORM: positive per-constraint scales for the OPTIMIZER'S
    # view of G. Physical g values (raw dicts, checkpoints, plots) stay in
    # their native units; the division happens only where the optimizer
    # assembles its G matrix. A missing name defaults to 1.0, so an empty
    # dict reproduces the old behaviour bit for bit. Scales are positive,
    # so the feasible set is IDENTICAL; only the ranking AMONG infeasible
    # designs changes.
    constraint_scales: dict = field(default_factory=dict)
""",
),

# -- B: ProblemSpec: add the accessor next to exact_ok -------------------
(
"""    def exact_ok(self, design: dict) -> bool:
        \"\"\"True iff the design satisfies every ANALYTIC constraint (g <= 0).\"\"\"
        return all(f(design) <= 0.0 for f in self.exact_constraints.values())
""",
"""    def exact_ok(self, design: dict) -> bool:
        \"\"\"True iff the design satisfies every ANALYTIC constraint (g <= 0).\"\"\"
        return all(f(design) <= 0.0 for f in self.exact_constraints.values())

    def g_scale(self, name: str) -> float:
        \"\"\"CONSTRAINT-NORM: positive scale dividing constraint `name` in the
        optimizer's G matrix. Defaults to 1.0 (no normalization).\"\"\"
        s = float(self.constraint_scales.get(name, 1.0))
        if not s > 0.0:
            raise ValueError(f"constraint scale for {name} must be > 0, "
                             f"got {s}")
        return s
""",
),

# -- C: Evaluator.evaluate: normalize the truth G ------------------------
(
"""            for j, cname in enumerate(self.spec.constraint_names):
                G[i, j] = res[cname]
""",
"""            for j, cname in enumerate(self.spec.constraint_names):
                # CONSTRAINT-NORM: pymoo sums raw g into CV, so heterogeneous
                # units (delta-k, peaking, wt%, cm) would weight constraints
                # by their units. Divide by each limit: G is dimensionless.
                G[i, j] = res[cname] / self.spec.g_scale(cname)
""",
),

# -- D: _seed_from_raw: same division on checkpoint load -----------------
(
"""        G = (np.array([[float(r[c]) for c in self.spec.constraint_names]
                       for r in raw_list])
             if self.spec.n_constr else np.empty((len(raw_list), 0)))
""",
"""        G = (np.array([[float(r[c]) / self.spec.g_scale(c)   # CONSTRAINT-NORM
                        for c in self.spec.constraint_names]
                       for r in raw_list])
             if self.spec.n_constr else np.empty((len(raw_list), 0)))
""",
),

# -- E1: _SurrogateProblem.__init__: carry the scale of each exact col ---
(
"""        self._exact_cols = [(spec.constraint_names.index(name), fn)
                            for name, fn in spec.exact_constraints.items()]
""",
"""        self._exact_cols = [(spec.constraint_names.index(name), fn,
                             spec.g_scale(name))          # CONSTRAINT-NORM
                            for name, fn in spec.exact_constraints.items()]
""",
),

# -- E2: _SurrogateProblem._evaluate: scale the exact column too ---------
(
"""            for col, fn in self._exact_cols:
                g_mean[:, col] = [fn(self.spec.design_space.as_dict(x))
                                  for x in np.atleast_2d(X)]
""",
"""            for col, fn, scl in self._exact_cols:
                # CONSTRAINT-NORM: the GP columns are trained on normalized G,
                # so the exact overwrite must be divided by the SAME scale or
                # the geometry column would re-enter in centimetres.
                g_mean[:, col] = [fn(self.spec.design_space.as_dict(x)) / scl
                                  for x in np.atleast_2d(X)]
""",
),

# -- F: example_reactor_problem: default scales = the limits -------------
(
"""    constraints = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom"]
    exact = {"g_geom": lambda d: geometry_margin(d["pitch"], d["refl_thick"])}
    return ProblemSpec(ds, objs, constraints, exact_constraints=exact)
""",
"""    constraints = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom"]
    exact = {"g_geom": lambda d: geometry_margin(d["pitch"], d["refl_thick"])}
    # CONSTRAINT-NORM: default scales = each constraint's own limit, so the
    # optimizer compares FRACTIONAL violations (dimensionless). If the run
    # overrides a limit (k_basis table, CLI), run_optimization.py re-syncs
    # these from the evaluator's actual attributes after construction.
    import core_geometry as _cg
    scales = {
        "g_kmin": 1.02,
        "g_kmax": 1.35,
        "g_enr":  19.75,
        "g_peak": 2.0,
        "g_geom": _cg.R_VESSEL_INNER - _cg.VESSEL_CLEARANCE_CM,
    }
    return ProblemSpec(ds, objs, constraints, exact_constraints=exact,
                       constraint_scales=scales)
""",
),
]

EDITS_RU = [

# -- G: sync scales from the evaluator's ACTUAL limits -------------------
(
"""    opt = ActiveLearningMOO(spec, ev, cfg)
""",
"""    # CONSTRAINT-NORM: the spec's default scales assume the default limits.
    # The evaluator is the single source of truth for k_min / k_max (k_basis
    # aware), f_max and enr_max, so overwrite the scales from its attributes.
    import core_geometry as _cg
    spec.constraint_scales.update({
        "g_kmin": ev.k_min,
        "g_kmax": ev.k_max,
        "g_enr":  ev.enr_max,
        "g_peak": ev.f_max,
        "g_geom": _cg.R_VESSEL_INNER - _cg.VESSEL_CLEARANCE_CM,
    })
    opt = ActiveLearningMOO(spec, ev, cfg)
""",
),
]


def apply(path: Path, edits) -> None:
    text = path.read_text()
    if MARKER in text:
        sys.exit(f"REFUSED: {path} already contains the {MARKER} marker. "
                 f"Nothing was changed.")
    for i, (anchor, _) in enumerate(edits, 1):
        n = text.count(anchor)
        if n != 1:
            sys.exit(f"ABORT: anchor {i} for {path} found {n} times "
                     f"(need exactly 1). No file was modified.\n"
                     f"Anchor begins: {anchor.splitlines()[0]!r}")
    backup = path.with_suffix(path.suffix + ".bak.norm")
    shutil.copy2(path, backup)
    for anchor, repl in edits:
        text = text.replace(anchor, repl)
    path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    print(f"[ok] {path}: {len(edits)} edits applied, backup at {backup}, "
          f"py_compile passed")


def main() -> None:
    for p in (RO, RU):
        if not p.exists():
            sys.exit(f"ABORT: {p} not found. Run from the repository root.")
    # verify EVERY anchor in BOTH files before touching either one
    for p, edits in ((RO, EDITS_RO), (RU, EDITS_RU)):
        text = p.read_text()
        if MARKER in text:
            sys.exit(f"REFUSED: {p} already contains the {MARKER} marker.")
        for i, (anchor, _) in enumerate(edits, 1):
            n = text.count(anchor)
            if n != 1:
                sys.exit(f"ABORT: anchor {i} for {p} found {n} times "
                         f"(need exactly 1). No file was modified.\n"
                         f"Anchor begins: {anchor.splitlines()[0]!r}")
    apply(RO, EDITS_RO)
    apply(RU, EDITS_RU)
    print("[done] optimizer-side constraint normalization is in place.")
    print("       Physical g values in raw dicts and checkpoints are "
          "unchanged.")


if __name__ == "__main__":
    main()
