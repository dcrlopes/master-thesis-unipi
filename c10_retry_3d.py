#!/usr/bin/env python3
"""
c10_retry_3d.py -- repeat the eight-layer core depletion of the evaluations of
an evaluator cache whose three-dimensional cycle length failed (crash or time
limit of core3d_subprocess.py), and write the result back into the cache.

The campaign part of such an evaluation (assembly depletion, core solves,
control rods, boron) is complete and kept; only the depletion of the hook
is repeated, in a subprocess as in the loop, in <workdir>/retry3d/<n>/.

usage (repository root, openmc-env, about an hour per entry):
    python c10_retry_3d.py --cache openmc_runs_c10/eval_cache.json --workdir openmc_runs_c10 \
        --transport 10000,100,50 --layers 8 --max-burnup 29.5 --hump-noise 400
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--transport", default="10000,100,50", help="particles,batches,inactive")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--max-burnup", type=float, default=29.5)
    ap.add_argument("--hump-noise", type=float, default=400.0)
    ap.add_argument("--bol-steps", default="0.5,1,2,4,6")
    ap.add_argument("--dep-step", type=float, default=4.0)
    ap.add_argument("--chunk-steps", type=int, default=2)
    ap.add_argument("--efpd-req", type=float, default=1826.0)
    ap.add_argument("--timeout-min", type=float, default=180.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    cache_path = Path(a.cache)
    if not cache_path.exists():
        print(f"retry3d: no cache at {cache_path}, nothing to do")
        return 0
    cache = json.loads(cache_path.read_text())
    todo = [k for k, v in cache.items() if v.get("axial_measured") is False]
    print(f"retry3d: {len(todo)} of {len(cache)} cached evaluations have no 3D cycle length")
    if not todo or a.dry_run:
        return 0
    p, b, i = (int(v) for v in a.transport.split(","))
    schedule = dict(bol_steps=[float(s) for s in a.bol_steps.split(",")], dep_step=a.dep_step,
                    chunk_steps=a.chunk_steps, max_burnup=a.max_burnup)
    # the bookkeeping of the hook, loaded without importing openmc
    import ast
    src = (HERE / "openmc_evaluator.py").read_text(encoding="utf-8")
    ns = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "core3d_cycle_fields":
            exec(compile(ast.Module([node], []), "openmc_evaluator.py", "exec"), ns)
    fields = ns["core3d_cycle_fields"]
    for n, key in enumerate(todo):
        res = cache[key]
        design = json.loads(key)
        case = Path(a.workdir) / "retry3d" / f"{n:02d}"
        case.mkdir(parents=True, exist_ok=True)
        spec = dict(design=design, rec={k: res[k] for k in ("k_target", "k_bol", "boron_points", "n_dep_solves") if k in res},
                    schedule=schedule, hump_noise_pcm=a.hump_noise, transport=dict(particles=p, batches=b, inactive=i),
                    layers=a.layers, case=str(case), salt="core3d-loop", label=f"retry {n:02d} 3D")
        fin, fout = case / "core3d_in.json", case / "core3d_out.json"
        fin.write_text(json.dumps(spec, default=float))
        if fout.exists():
            fout.unlink()
        print(f"retry3d: entry {n} ({res.get('core3d_error', '')}), enrich {design.get('enrich', float('nan')):.3f}, "
              f"assembly EFPD {res['cycle_length']:.0f} d", flush=True)
        t0 = time.perf_counter()
        try:
            rc = subprocess.run([sys.executable, "-u", str(HERE / "core3d_subprocess.py"), str(fin), str(fout)],
                                cwd=str(HERE), timeout=60.0 * a.timeout_min).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = time.perf_counter() - t0
        if rc != 0 or not fout.exists():
            print(f"retry3d: entry {n} failed again ({rc}) after {wall / 60:.1f} min, left as it is", flush=True)
            continue
        dep = json.loads(fout.read_text())
        dep["wall_s"] = wall
        out = fields({"cycle_length": res.get("cycle_length_asm", res["cycle_length"])}, dep, a.efpd_req)
        res.update(out)
        res["t_eval_s"] = float(res["t_eval_s"]) + wall
        cache[key] = res
        cache_path.write_text(json.dumps(cache))
        print(f"retry3d: entry {n} done, 3D EFPD {out['efpd_3d']:.0f} d in {wall / 60:.1f} min, cache updated", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
