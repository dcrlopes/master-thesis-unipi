#!/usr/bin/env python3
r"""
apply_c9_reformulation.py
=========================
Wires the Campaign 9 objective set into the framework, on branch main,
without changing any Campaign 1 to 8 behaviour: every new path is behind
--objective-set c9, and the default remains c8.

WHAT CHANGES
------------
reactor_optimization.py
  + campaign9_problem(efpd_req, f_max): objectives {peaking, c_max}, both
    minimised; constraints {g_kmin, g_kmax, g_enr, g_peak, g_geom, g_efpd,
    g_ctrl_peak}; scales g_efpd = efpd_req, g_ctrl_peak = 1.0.

openmc_evaluator.py
  + _cycle_length keeps the full k and burnup histories on the instance
    (self._last_k_hist, self._last_bu_hist). Its return is unchanged.
  + evaluate_one, when the evaluator carries c9_efpd_req, runs the
    adaptive boron measurement (boron_objective.py) after the control
    solves, and adds c_bol, c_max, c_max_op, the local worths, the hump,
    g_efpd, g_ctrl_peak, g_boron (recorded, not constrained), the number
    of extra solves and their wall time.
  + _boron_solve: one unrodded zoned core solve at a chosen concentration,
    same fidelity, zoning and deterministic seeding as _ctrl_solve, cached
    per (design, concentration).

run_optimization.py
  + flags --objective-set, --efpd-req, --boron-objective, --boron-step,
    --boron-top, --boron-ceiling, --hump-noise.
  + spec selection, evaluator configuration, EFPD clip disabled under c9
    (the clip acts on objective 0, which is no longer cycle length).
  + campaign9 block in the checkpoint meta.
  + _plot labels its axes from the objective list instead of assuming
    cycle length on the x axis.

REQUIRES boron_objective.py next to the framework files.

USAGE, from the repository root on main
    python apply_c9_reformulation.py --selftest
    python apply_c9_reformulation.py --check
    python apply_c9_reformulation.py
    python apply_c9_reformulation.py --revert
"""

import argparse
import datetime
import pathlib
import sys

MARKER = "c9_efpd_req"

# ============================================================ anchors ==
EDITS = []


def edit(fname, name, old, new):
    EDITS.append((fname, name, old, new))


# --------------------------------------------- reactor_optimization.py --
edit("reactor_optimization.py", "campaign9_problem before the demo section",
     "# =============================================================================\n"
     "# 8.  DEMO  (runs with the analytic evaluator -- no OpenMC needed)\n",
     '''def campaign9_problem(efpd_req: float, f_max: float = 1.65) -> ProblemSpec:
    """CAMPAIGN 9: the reformulated problem.

    Cycle length and peaking become constraints, the objectives become the
    unrodded hot channel factor and the critical boron at the operating
    maximum. Everything else (design space, analytic vessel-fit
    constraint, normalisation) is inherited from example_reactor_problem,
    so the search box is identical to Campaign 8.

        minimise   peaking, c_max
        subject to EFPD >= efpd_req            (g_efpd)
                   F_dH <= f_max               (g_peak)
                   k_min <= k_core <= k_max    (g_kmin, g_kmax)
                   LEU cap, vessel fit         (g_enr, g_geom)
                   four-bank margin at the operating maximum (g_ctrl_peak)

    The two-bank reading and the boron ceiling are recorded on every
    evaluation and never constrained, so the front can be split
    afterwards without shrinking the feasible region in advance.
    """
    spec = example_reactor_problem()
    spec.objectives = [
        Objective("peaking", maximize=False, label="Power peaking factor"),
        Objective("c_max", maximize=False,
                  label="Critical boron at the operating maximum [ppm]"),
    ]
    spec.constraint_names = ["g_kmin", "g_kmax", "g_enr", "g_peak", "g_geom",
                             "g_efpd", "g_ctrl_peak"]
    spec.constraint_scales["g_efpd"] = float(efpd_req)
    spec.constraint_scales["g_ctrl_peak"] = 1.0        # k-units, like g_ctrl
    spec.constraint_scales["g_peak"] = float(f_max)
    return spec


# =============================================================================
# 8.  DEMO  (runs with the analytic evaluator -- no OpenMC needed)
''')

# ------------------------------------------------ openmc_evaluator.py --
edit("openmc_evaluator.py", "keep the depletion history on the instance",
     "        return efpd, k_bol, k_target, censored, bu_eoc, len(k_hist)\n",
     "        # CAMPAIGN 9: the operating maximum needs the whole history. Keep\n"
     "        # it on the instance so the return signature, which other\n"
     "        # scripts unpack, is untouched.\n"
     "        self._last_k_hist = [float(v) for v in k_hist]\n"
     "        self._last_bu_hist = [float(v) for v in bu_hist]\n"
     "        return efpd, k_bol, k_target, censored, bu_eoc, len(k_hist)\n")

edit("openmc_evaluator.py", "boron block after the control solves",
     '        for _tk in ("t_ctrl_s", "t_ctrl12_s"):\n'
     '            if res.get(_tk):\n'
     '                res["t_eval_s"] += float(res[_tk])\n',
     '        for _tk in ("t_ctrl_s", "t_ctrl12_s"):\n'
     '            if res.get(_tk):\n'
     '                res["t_eval_s"] += float(res[_tk])\n'
     '        # CAMPAIGN 9: adaptive boron measurement and the objective at\n'
     '        # the operating maximum. Runs only when run_optimization has\n'
     '        # configured the evaluator, so every earlier campaign is\n'
     '        # bit-for-bit unchanged. Needs the control solve above.\n'
     '        if getattr(self, "c9_efpd_req", None) is not None:\n'
     '            res.update(_c9_boron_block(self, design, res))\n'
     '            res["t_eval_s"] += float(res.get("t_boron_s", 0.0))\n')

edit("openmc_evaluator.py", "boron solve and block helpers at the end",
     '        rodded_map=(set(pos),\n'
     '                    getattr(ev, "ctrl_absorber", "B4C")))\n',
     '        rodded_map=(set(pos),\n'
     '                    getattr(ev, "ctrl_absorber", "B4C")))\n'
     '''

# --------------------------------------------------------------------------- #
# CAMPAIGN 9 helpers (appended by apply_c9_reformulation.py)                   #
# --------------------------------------------------------------------------- #
def _boron_solve(ev, design, ppm):
    """One unrodded zoned core solve at soluble boron `ppm`. Same fidelity,
    zoning path and deterministic seeding as _ctrl_solve; the case
    directory is keyed by the design hash AND the concentration, so each
    concentration reuses its own cache."""
    import dataclasses
    op = dataclasses.replace(ev.op, boron_ppm=float(ppm))
    salt = f"boron{float(ppm):g}"
    tag = _design_seed(design, salt=salt) & 0xFFFFFFFF
    return zn.core_bol_solve(
        design, zn.evaluator_design_map(design), op, ev.geo,
        particles=ev.core_particles, batches=ev.core_batches,
        inactive=ev.core_inactive,
        seed=_design_seed(design, salt=salt),
        case=ev.workdir / f"{salt}_{tag:08x}",
        rodded_map=None)


def _c9_boron_block(ev, design, res):
    """Adaptive measurement of the critical boron and the objective at the
    operating maximum. See boron_objective.py for the arithmetic."""
    import boron_objective as bo
    t0 = time.perf_counter()
    k_hist = getattr(ev, "_last_k_hist", None) or [float(res["k_bol"])]
    hump = bo.hump_from_history(k_hist, float(res["keff_core_bol"]),
                                noise_pcm=float(ev.c9_hump_noise_pcm))
    points = {bo.PPM_REF: bo.rho_pcm(float(res["keff_core_bol"]))}
    targets = [0.0, -hump["hump_core_pcm"], -hump["hump_core_op_pcm"]]
    n_extra = 0
    while True:
        nxt = bo.next_concentration(points, targets,
                                    top=float(ev.c9_ppm_top),
                                    step=float(ev.c9_ppm_step))
        if nxt is None:
            break
        sol = _boron_solve(ev, design, nxt)
        points[float(nxt)] = bo.rho_pcm(float(sol["keff"]))
        n_extra += 1
    obj = bo.boron_objective(points, hump)
    ctrl = bo.ctrl_margin_at_peak(float(res["k_allre"]), hump,
                                  float(ev.ctrl_margin_dk))
    out = {}
    out.update(hump)
    out.update(obj)
    out.update(ctrl)
    out["c_max"] = float(obj["c_max_ppm"] if ev.c9_boron_objective == "floor"
                         else obj["c_max_op_ppm"])
    out["c_bol"] = float(obj["c_bol_ppm"])
    out["g_efpd"] = float(ev.c9_efpd_req) - float(res["cycle_length"])
    # recorded, never constrained: the measured MTC ceiling drawn as a line
    out["g_boron"] = out["c_max"] - float(ev.c9_boron_ceiling_ppm)
    out["n_boron_solves"] = n_extra
    out["t_boron_s"] = time.perf_counter() - t0
    if getattr(ev, "verbose", False):
        print(f"      c9: c_BOL={out['c_bol']:6.0f} ppm ({obj['c_bol_status']}) "
              f"c_max={out['c_max']:6.0f} ppm ({obj['c_max_status']}) "
              f"hump_core={hump['hump_core_pcm']:+6.0f} pcm "
              f"g_efpd={out['g_efpd']:+7.0f} g_ctrl_peak={ctrl['g_ctrl_peak']:+.4f} "
              f"[{n_extra} boron solves, {out['t_boron_s']:.0f} s]")
    return out
''')

# ------------------------------------------------- run_optimization.py --
edit("run_optimization.py", "campaign 9 flags",
     "    args = ap.parse_args()\n",
     '''    # ---- CAMPAIGN 9 ----------------------------------------------------
    ap.add_argument("--objective-set", choices=["c8", "c9"], default="c8",
                    help="c8: maximise cycle length, minimise peaking "
                         "(Campaigns 1 to 8). c9: minimise peaking and the "
                         "critical boron at the operating maximum, with the "
                         "cycle length as a constraint (--efpd-req). c9 "
                         "requires --ctrl-margin.")
    ap.add_argument("--efpd-req", type=float, default=1826.0,
                    help="c9: mission cycle length, EFPD (default 1826, five "
                         "years at capacity factor 1.0). Constraint g_efpd.")
    ap.add_argument("--boron-objective", choices=["floor", "op"],
                    default="floor",
                    help="c9: which critical boron is the objective. floor: "
                         "hump floored at zero, xenon-free reference "
                         "(conservative, the campaign default). op: no floor, "
                         "the xenon credit reduces the requirement. Both are "
                         "recorded on every evaluation.")
    ap.add_argument("--boron-step", type=float, default=2000.0,
                    help="c9: second concentration measured when the core is "
                         "supercritical at the 1000 ppm reference, ppm")
    ap.add_argument("--boron-top", type=float, default=3000.0,
                    help="c9: third and last concentration, measured only when "
                         "a root is not yet bracketed, ppm")
    ap.add_argument("--boron-ceiling", type=float, default=2763.0,
                    help="c9: measured MTC ceiling, ppm (design 47, hardware "
                         "3D, 12.8 MPa). Recorded as g_boron, never "
                         "constrained.")
    ap.add_argument("--hump-noise", type=float, default=400.0,
                    help="c9: gadolinium humps below this are treated as "
                         "unresolved and set to zero, pcm")
    args = ap.parse_args()
''')

edit("run_optimization.py", "import campaign9_problem",
     "from reactor_optimization import (example_reactor_problem, OptimizerConfig,\n",
     "from reactor_optimization import (example_reactor_problem, campaign9_problem,\n"
     "                                      OptimizerConfig,\n")

edit("run_optimization.py", "spec selection",
     "    spec = example_reactor_problem()\n",
     '    if args.objective_set == "c9":\n'
     "        spec = campaign9_problem(args.efpd_req, f_max=args.f_max)\n"
     "    else:\n"
     "        spec = example_reactor_problem()\n")

edit("run_optimization.py", "evaluator configuration for c9",
     '        spec.constraint_scales["g_ctrl"] = 1.0     # k-units: limit is 1.0\n',
     '        spec.constraint_scales["g_ctrl"] = 1.0     # k-units: limit is 1.0\n'
     '    # CAMPAIGN 9: configure the evaluator. The control solve is required\n'
     '    # because the operating-maximum screen starts from k_allre. The EFPD\n'
     '    # clip is disabled because it acts on objective 0, which is now the\n'
     '    # peaking factor, and cycle length is a constraint surrogate.\n'
     '    if args.objective_set == "c9":\n'
     '        if args.ctrl_margin is None:\n'
     '            raise SystemExit("--objective-set c9 requires --ctrl-margin "\n'
     '                             "(the operating-maximum screen starts from "\n'
     '                             "the four-bank solve)")\n'
     '        ev.c9_efpd_req = float(args.efpd_req)\n'
     '        ev.c9_boron_objective = args.boron_objective\n'
     '        ev.c9_ppm_step = float(args.boron_step)\n'
     '        ev.c9_ppm_top = float(args.boron_top)\n'
     '        ev.c9_boron_ceiling_ppm = float(args.boron_ceiling)\n'
     '        ev.c9_hump_noise_pcm = float(args.hump_noise)\n'
     '        args.no_efpd_clip = True\n'
     '        print(f"CAMPAIGN 9: objectives peaking + c_max ({args.boron_objective}) | "\n'
     '              f"EFPD >= {args.efpd_req:g} | boron points 1000, "\n'
     '              f"{args.boron_step:g}, {args.boron_top:g} ppm | ceiling "\n'
     '              f"{args.boron_ceiling:g} ppm recorded | hump noise "\n'
     '              f"{args.hump_noise:g} pcm")\n')

edit("run_optimization.py", "checkpoint meta for c9",
     '                           "geometry": "v2-envelope",\n',
     '                           "geometry": "v2-envelope",\n'
     '                           "objective_set": args.objective_set,\n'
     '                           "campaign9": ({\n'
     '                               "efpd_req": args.efpd_req,\n'
     '                               "boron_objective": args.boron_objective,\n'
     '                               "boron_points_ppm": [1000.0, args.boron_step,\n'
     '                                                    args.boron_top],\n'
     '                               "boron_ceiling_ppm": args.boron_ceiling,\n'
     '                               "hump_noise_pcm": args.hump_noise,\n'
     '                               "hump_reference": "xenon-free k_hist[0], "\n'
     '                                                 "core solve is xenon-free",\n'
     '                               "worth": "local slope of the measured "\n'
     '                                        "rho(c) at the root",\n'
     '                               "g_boron_note": "recorded, not constrained"}\n'
     '                               if args.objective_set == "c9" else None),\n')

edit("run_optimization.py", "plot call passes the objectives",
     "    _plot(res, args.out)\n",
     "    _plot(res, args.out, opt.spec.objectives)\n")

edit("run_optimization.py", "plot axes follow the objective list",
     'def _plot(res, outdir):\n'
     '    import matplotlib\n'
     '    matplotlib.use("Agg")\n'
     '    import matplotlib.pyplot as plt\n'
     '\n'
     '    allF = res["all_F"]\n'
     '    pf = res["pareto_F"]\n'
     '    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))\n'
     '    ax[0].scatter(-allF[:, 0], allF[:, 1], s=18, c="lightgray",\n'
     '                  label="all evaluations")\n'
     '    if len(pf):\n'
     '        ax[0].scatter(-pf[:, 0], pf[:, 1], s=42, c="crimson", zorder=3,\n'
     '                      label="Pareto front")\n'
     '    ax[0].set_xlabel("Cycle length [EFPD]  (maximise \\u2192)")\n'
     '    ax[0].set_ylabel("Power peaking factor  (\\u2190 minimise)")\n',
     'def _plot(res, outdir, objectives=None):\n'
     '    import matplotlib\n'
     '    matplotlib.use("Agg")\n'
     '    import matplotlib.pyplot as plt\n'
     '\n'
     '    allF = res["all_F"]\n'
     '    pf = res["pareto_F"]\n'
     '    # objectives are stored minimised; undo the sign for maximised ones.\n'
     '    # With no objective list the Campaign 1 to 8 layout is reproduced.\n'
     '    if objectives is None:\n'
     '        sx, sy = -1.0, 1.0\n'
     '        lx, ly = "Cycle length [EFPD]  (maximise)", "Power peaking factor  (minimise)"\n'
     '    else:\n'
     '        ox, oy = objectives[0], objectives[1]\n'
     '        sx = -1.0 if ox.maximize else 1.0\n'
     '        sy = -1.0 if oy.maximize else 1.0\n'
     '        lx = f"{ox.label}  ({\'maximise\' if ox.maximize else \'minimise\'})"\n'
     '        ly = f"{oy.label}  ({\'maximise\' if oy.maximize else \'minimise\'})"\n'
     '    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))\n'
     '    ax[0].scatter(sx * allF[:, 0], sy * allF[:, 1], s=18, c="lightgray",\n'
     '                  label="all evaluations")\n'
     '    if len(pf):\n'
     '        ax[0].scatter(sx * pf[:, 0], sy * pf[:, 1], s=42, c="crimson", zorder=3,\n'
     '                      label="Pareto front")\n'
     '    ax[0].set_xlabel(lx)\n'
     '    ax[0].set_ylabel(ly)\n')


# ============================================================== driver ==
def selftest():
    seen = set()
    for f, name, old, new in EDITS:
        assert old and new, name
        assert (f, old) not in seen, f"duplicate anchor {name}"
        seen.add((f, old))
    assert any(MARKER in n for *_, n in EDITS)
    # every edited python block must itself be valid python when compiled
    # in isolation where it is a whole definition
    import ast
    for f, name, old, new in EDITS:
        if name.startswith("campaign9_problem"):
            ast.parse(new.split("# ====")[0])
        if name.startswith("boron solve"):
            body = new.split("# CAMPAIGN 9 helpers")[1].split("#\n", 1)[-1]
            ast.parse("\n".join(l for l in body.splitlines() if not l.startswith("# ---")))
        if name.startswith("plot axes"):
            ast.parse(new)
    print("selftest OK")
    return 0


def run(mode):
    root = pathlib.Path(".")
    need = ["boron_objective.py"]
    if mode == "apply" and not (root / need[0]).is_file():
        sys.exit(f"FAIL: {need[0]} must sit next to the framework files.")
    texts = {}
    for fname in sorted({f for f, *_ in EDITS}):
        p = root / fname
        if not p.is_file():
            sys.exit(f"FAIL: {fname} not found. Run from the repository root.")
        texts[fname] = p.read_text(encoding="utf-8")
    if mode == "apply" and any(MARKER in t for t in texts.values()):
        sys.exit("Already applied (marker found). Nothing to do.")
    bad = 0
    for fname, name, old, new in EDITS:
        n = texts[fname].count(old)
        ok = n == 1
        bad += (not ok)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {fname:24s} {n} match  {name}")
        if ok:
            texts[fname] = texts[fname].replace(old, new, 1)
    if bad:
        sys.exit(f"\n{bad} anchor(s) did not match exactly once. Nothing written.")
    if mode == "check":
        print("\nall anchors unique, nothing written")
        return 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for fname, text in texts.items():
        p = root / fname
        (root / f"{fname}.bak_{stamp}").write_text(p.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
        p.write_text(text, encoding="utf-8")
        print(f"  wrote {fname}  (backup {fname}.bak_{stamp})")
    import py_compile
    for fname in texts:
        py_compile.compile(str(root / fname), doraise=True)
    print("  all three files compile")
    return 0


def revert():
    root = pathlib.Path(".")
    for fname in sorted({f for f, *_ in EDITS}):
        baks = sorted(root.glob(f"{fname}.bak_*"))
        if not baks:
            print(f"  no backup for {fname}")
            continue
        (root / fname).write_text(baks[-1].read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  restored {fname} from {baks[-1].name}")
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
