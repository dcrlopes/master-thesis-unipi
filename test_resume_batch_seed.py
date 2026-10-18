"""A campaign resumed from its checkpoint must propose the same infill batches
as the uninterrupted campaign, so that a relaunch in the middle of a batch
replays its finished evaluations from the cache (Campaign 10, 17 Oct 2026:
launch 4 proposed a different batch 3 and 4 evaluations were not used).
Analytic evaluator, no OpenMC, about a minute.

usage:  python test_resume_batch_seed.py
"""
import os
import tempfile
import warnings

import numpy as np

import reactor_optimization as ro


def make(spec, n_iter):
    ev = ro.AnalyticEvaluator(spec, noise=0.0, seed=1)
    cfg = ro.OptimizerConfig(n_init=12, n_iter=n_iter, n_infill=3, nsga_pop=20, nsga_gen=10,
                             surrogate="gp", seed=1)
    return ro.ActiveLearningMOO(spec, ev, cfg)


def main():
    warnings.filterwarnings("ignore")
    spec = ro.example_reactor_problem()
    full = make(spec, 2)
    full.run(verbose=False)
    part = make(spec, 1)
    part.run(verbose=False)
    path = os.path.join(tempfile.mkdtemp(), "optimization_checkpoint.json")
    part.save_checkpoint(path)
    resumed = make(spec, 1)
    resumed.load_checkpoint(path)
    resumed.run(verbose=False)
    assert full.X.shape == resumed.X.shape, (full.X.shape, resumed.X.shape)
    assert np.allclose(full.X, resumed.X), "the resumed batch 2 differs from the uninterrupted batch 2"
    print("resume reproduces the uninterrupted campaign OK")


if __name__ == "__main__":
    main()
