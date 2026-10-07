#!/usr/bin/env python3
"""
surrogate_walkforward.py -- walk-forward comparison of the seven surrogates of
surrogate_kernel_cv.py on Campaigns 8 and 9: every infill block is predicted
from the designs evaluated before it, as the loop saw them. No transport.

For each campaign, output and surrogate: the metrics pooled over all infill
designs (RMSE, R^2, Spearman, coverage, s_z), the RMSE per block, and for the
Campaign 9 cycle length the feasibility decisions on the designs that turned
out below 1826 d: how many the surrogate predicted feasible (mean >= 1826)
and margin-feasible (mean - kappa sd >= 1826, kappa 1.5).

usage (repository root, a few minutes):
    python surrogate_walkforward.py
Writes figs_surrogate_cv/walkforward.json and walkforward.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from surrogate_kernel_cv import ARCHIVES, OUTPUTS, VARIANTS, metrics

KAPPA, REQ = 1.5, 1826.0


def blocks_of(D):
    n0, out = 0, []
    for p in D["phase_log"]:
        m = int(p["n_eval"])
        if p["stage"] == "DOE":
            n0 += m
        else:
            out.append((n0, n0 + m)); n0 += m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default="figs_surrogate_cv")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    R, lines = {}, []
    for tag, path in ARCHIVES.items():
        D = json.load(open(path)); A = D["all_raw"]; names = D["design_variables"]
        X = np.array([[r[v] for v in names] for r in A], float)
        blocks = blocks_of(D)
        key = "C9" if "c9" in path else "C8"
        print(f"== {tag}: {len(A)} designs, blocks {blocks}")
        R[tag] = {}
        for oname, olab in OUTPUTS[key]:
            y = np.array([r[oname] for r in A], float)
            R[tag][oname] = {}
            lines.append(f"\n### {tag}, {olab}: walk-forward over {len(blocks)} blocks\n")
            hdr = "| Surrogate | RMSE | R^2 | Spearman | Cov. 1 sigma | Cov. 2 sigma | s_z | RMSE per block |"
            if key == "C9" and oname == "cycle_length":
                hdr += " Below 1826 d: predicted feasible / margin-feasible |"
            lines.append(hdr); lines.append("|---" * (9 if key == "C9" and oname == "cycle_length" else 8) + "|")
            for v in a.variants:
                t0 = time.time()
                mu, sd, idx = [], [], []
                per_block = []
                for (s, e) in blocks:
                    m, sdev = VARIANTS[v]().fit(X[:s], y[:s]).predict(X[s:e])
                    m, sdev = np.asarray(m, float).ravel(), np.asarray(sdev, float).ravel()
                    mu += m.tolist(); sd += sdev.tolist(); idx += list(range(s, e))
                    per_block.append(float(np.sqrt(np.mean((m - y[s:e]) ** 2))))
                mu, sd, idx = np.array(mu), np.array(sd), np.array(idx)
                res = metrics(y[idx], mu, sd); res["per_block_rmse"] = per_block; res["seconds"] = time.time() - t0
                extra = ""
                if key == "C9" and oname == "cycle_length":
                    short = y[idx] < REQ
                    res["decisions"] = dict(n_short=int(short.sum()),
                                            short_pred_feasible=int(np.sum(short & (mu >= REQ))),
                                            short_pred_margin_feasible=int(np.sum(short & (mu - KAPPA * sd >= REQ))),
                                            n_ok=int((~short).sum()),
                                            ok_pred_feasible=int(np.sum(~short & (mu >= REQ))))
                    d = res["decisions"]
                    extra = f" {d['n_short']}: {d['short_pred_feasible']} / {d['short_pred_margin_feasible']} |"
                R[tag][oname][v] = res
                fmt = "{:.4g}" if oname == "peaking" else "{:.0f}"
                lines.append(f"| {v} | {fmt.format(res['rmse'])} | {res['r2']:.3f} | {res['spearman']:.3f} | {res['cov1']:.2f} | "
                             f"{res['cov2']:.2f} | {res['sz']:.2f} | " + ", ".join(fmt.format(b) for b in per_block) + " |" + extra)
                print(f"  {olab:18s} {v:12s} RMSE {res['rmse']:.4g} R2 {res['r2']:.3f} rho {res['spearman']:.3f} s_z {res['sz']:.2f} "
                      f"| blocks " + " ".join(fmt.format(b) for b in per_block) + (f" | short {res['decisions']}" if extra else ""), flush=True)
    (out / "walkforward.json").write_text(json.dumps(R, indent=1))
    (out / "walkforward.md").write_text("# Walk-forward comparison of surrogates\n" + "\n".join(lines) + "\n")
    print(f"wrote {out}/walkforward.json and walkforward.md")


if __name__ == "__main__":
    main()
