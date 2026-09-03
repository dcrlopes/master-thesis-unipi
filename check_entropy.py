#!/usr/bin/env python3
"""
check_entropy.py -- inspect the Shannon entropy trace of confirm3d solves.

confirm3d.py flags entropy_conv = (last batch outside a 3-sigma band of the
late active batches) + 2. Three production solves reported values above the
80 inactive batches: d23 ARO 2D seed 1 (125), d23 ARO 3Dhw seed 1 (123),
d21 ARO 2D seed 0 (108). This prints, for each case directory, the entropy
of the first and last inactive batch, the mean and sd over the active
batches split in two halves, and the batches outside the band, so you can
see whether the flag is a late random excursion (harmless) or a source that
was still drifting when the active batches began (k and F biased).

USAGE (wks720, openmc-env, branch campaign8):
  python check_entropy.py confirm3d_c8/d23/ARO_2D_s1 confirm3d_c8/d23/ARO_3Dhw_s1 \
                          confirm3d_c8/d21/ARO_2D_s0 confirm3d_c8/d47/ARO_2D_s0
The last one is a reference with entropy_conv 8.
"""
import sys, glob
import numpy as np

INACTIVE = 80

def main(paths):
    import openmc
    for case in paths:
        sps = sorted(glob.glob(f"{case}/statepoint.*.h5"))
        if not sps:
            print(f"{case}: no statepoint found"); continue
        with openmc.StatePoint(sps[-1]) as sp:
            H = np.asarray(getattr(sp, "entropy", []), dtype=float)
            k = float(sp.keff.nominal_value); sd = float(sp.keff.std_dev)
        n = len(H); act = H[INACTIVE:]; h1, h2 = act[: len(act) // 2], act[len(act) // 2:]
        tail = H[INACTIVE + (n - INACTIVE) // 2:]; mu, s = tail.mean(), tail.std(ddof=1)
        Hs = np.convolve(H, np.ones(3) / 3.0, mode="same"); Hs[0], Hs[-1] = H[0], H[-1]
        bad = np.where(~((Hs >= mu - 3 * s) & (Hs <= mu + 3 * s)))[0]
        late = [int(b) for b in bad if b >= INACTIVE]
        print(f"\n{case}")
        print(f"  batches {n}, inactive {INACTIVE}, k = {k:.5f} +/- {sd:.5f}")
        print(f"  H[0] {H[0]:.4f}  H[{INACTIVE-1}] {H[INACTIVE-1]:.4f}  H[-1] {H[-1]:.4f}")
        print(f"  active first half  mean {h1.mean():.5f} sd {h1.std(ddof=1):.5f}")
        print(f"  active second half mean {h2.mean():.5f} sd {h2.std(ddof=1):.5f}   "
              f"shift {(h1.mean()-h2.mean())/h2.std(ddof=1):+.2f} sigma")
        print(f"  batches outside the 3-sigma band after inactive: {late if late else 'none'}")
        verdict = ("drift: first-half mean differs from second-half by > 2 sigma, treat k/F with caution"
                   if abs(h1.mean() - h2.mean()) > 2 * h2.std(ddof=1) / np.sqrt(len(h2))
                   else "no drift between active halves, the flag is a late random excursion")
        print(f"  verdict: {verdict}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
