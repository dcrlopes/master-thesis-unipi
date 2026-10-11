#!/usr/bin/env python3
"""
core3d_subprocess.py -- the eight-layer core depletion of one design in its own
process, for the Campaign 10 hook of openmc_evaluator.py.

Campaign 10 showed why this must be a separate process: after seven complete
evaluations in one Python process, OpenMC crashed with a segmentation fault
in the fourteenth solve of a three-dimensional depletion (11 Oct 2026), and
the driver died with it, losing the evaluations of the design of experiments.
Here the OpenMC state of the depletion is fresh for every design, and a
crash ends only this process: the evaluator keeps the assembly value and
flags the evaluation.

usage:  python core3d_subprocess.py IN.json OUT.json
IN.json  {design, rec, schedule, hump_noise_pcm, transport, layers, case, salt, label}
OUT.json the result of c9_dep_core3d.deplete_design without the material
         rows and the solve times, numpy values converted
"""
import json
import os
import sys


def slim(dep):
    """The fields the evaluator needs, as plain JSON types."""
    out = {k: v for k, v in dep.items() if k not in ("materials", "info", "solve_times")}
    return json.loads(json.dumps(out, default=float))


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: core3d_subprocess.py IN.json OUT.json")
    spec = json.load(open(argv[1]))
    chain = os.environ.get("OPENMC_CHAIN_FILE")
    if not chain:
        sys.exit("OPENMC_CHAIN_FILE is not set")
    import openmc
    openmc.config["chain_file"] = chain
    from c9_dep_core3d import deplete_design
    dep = deplete_design(spec["design"], spec["rec"], spec["schedule"], float(spec["hump_noise_pcm"]),
                         dict(spec["transport"]), int(spec["layers"]), spec["case"],
                         salt=str(spec.get("salt", "core3d-loop")), label=str(spec.get("label", "3D")))
    with open(argv[2], "w") as f:
        json.dump(slim(dep), f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
