#!/usr/bin/env python3
"""Repair zoning.read_k_history for chunk-LOCAL depletion results.

This OpenMC version zero-fills the entries a chunk did not compute, in BOTH
the time axis and the k array, so reading only the last chunk yields a mostly
zero history. Real entries always carry k > 0 and their times ARE cumulative
across chunks, so every chunk is read and the real entries are merged.
"""
import shutil, py_compile
from pathlib import Path

P = Path("zoning.py")
src = P.read_text()
start = src.index("def read_k_history(")
end = src.index("def late_slope_pcm_per_mwdkg(")
NEW = '''def read_k_history(case_dir, spec_power_w_per_g: float):
    """(burnup [MWd/kgHM], k_inf) merged across ALL depletion chunks.

    The evaluator chains chunks with prev_results. On this OpenMC version the
    per-chunk results file zero-fills every entry the chunk did not compute,
    in both the time axis and the k array, so reading the last chunk alone
    gives a mostly-zero history. Real entries always have k > 0 and their
    times are cumulative, so all chunks are read and the real entries merged,
    deduplicated at the chunk boundaries where the restart state repeats.
    Burnup follows the evaluator conversion bu = t * spec_power / 1000."""
    import openmc.deplete
    chunks = sorted(glob.glob(str(Path(case_dir) / "dep_*" /
                                  "depletion_results.h5")))
    if not chunks:
        raise FileNotFoundError(f"no dep_*/depletion_results.h5 under "
                                f"{case_dir}")
    pairs = []
    for ch in chunks:
        res = openmc.deplete.Results(ch)
        try:
            t_d, karr = res.get_keff(time_units="d")
        except TypeError:
            t_s, karr = res.get_keff()
            t_d = np.asarray(t_s) / 86400.0
        t_d = np.asarray(t_d, dtype=float)
        kv = np.asarray(karr, dtype=float)[:, 0]
        real = kv > 0.0
        pairs.extend(zip(t_d[real], kv[real]))
    if not pairs:
        raise RuntimeError(f"no non-zero k entries in any chunk under "
                           f"{case_dir}")
    pairs.sort()
    t_out, k_out = [], []
    for t, kk in pairs:
        if t_out and abs(t - t_out[-1]) < 1e-6:
            continue
        t_out.append(t)
        k_out.append(kk)
    t = np.asarray(t_out)
    k = np.asarray(k_out)
    if len(t) < 3:
        raise RuntimeError(f"only {len(t)} real depletion points under "
                           f"{case_dir}: cannot fit a slope")
    if np.any(np.diff(t) <= 0):
        raise RuntimeError(f"non-monotonic merged time axis under {case_dir}")
    bu = t * spec_power_w_per_g / 1000.0
    return bu, k


'''
if "real entries merged" in src:
    raise SystemExit("zoning.py already patched. Nothing to do.")
shutil.copy(P, "zoning.py.khist.bak")
P.write_text(src[:start] + NEW + src[end:])
py_compile.compile(str(P), doraise=True)
print("patched zoning.py (backup zoning.py.khist.bak)")
