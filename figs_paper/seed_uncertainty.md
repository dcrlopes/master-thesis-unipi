# In-run against between-seed uncertainty

## 1. Fresh-assembly k_bol, campaign solve 4000 x 60

- 21 designs, record minus high statistics: mean -46 pcm, SD 271 pcm.
- High-statistics solves: 64 pcm reported (mean of 21), removed in quadrature.
- Campaign solve: **264 pcm between seeds against 190 pcm reported** (ratio 1.39).

## 2 and 3. Eight-layer core at 20 000 x 160: k at each depletion state, and the cycle length

| Design | Seeds | States | k: between-seed SD [pcm] | k: in-run sd [pcm] | Ratio | EFPD: between-seed SD [d] | EFPD: in-run sigma [d] | Ratio |
|---|---|---|---|---|---|---|---|---|
| C10-31 | 2 | 8 | 68 | 65 | 1.04 | 3.7 | 8.3 | 0.44 |
| C10-42 | 2 | 8 | 102 | 63 | 1.63 | 17.0 | 6.8 | 2.49 |
| C10-48 | 2 | 8 | 89 | 62 | 1.43 | 30.7 | 8.6 | 3.59 |
| C10-56 | 2 | 8 | 37 | 64 | 0.57 | 0.5 | 8.2 | 0.06 |
| C9-27 | 5 | 8 | 74 | 63 | 1.16 | 23.3 | 8.4 | 2.78 |
| C9A-69 | 4 | 8 | 77 | 64 | 1.19 | 19.9 | 6.8 | 2.91 |
| C9A-70 | 4 | 8 | 68 | 64 | 1.07 | 2.1 | 7.8 | 0.26 |
| C9A-88 | 5 | 8 | 75 | 64 | 1.18 | 9.8 | 9.6 | 1.03 |

Pooled over 32 states of the designs with 3 or more seeds: **between-seed k error = 1.15 x the in-run sd**. The cycle-length sigma is propagated from the in-run k sd of the two bracketing states, so it inherits the same factor, and the interpolation between states two to four MWd/kgHM apart adds to it.
