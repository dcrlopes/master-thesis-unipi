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

## 4. Batch-to-batch correlation of k in the statepoints

| Statepoint | Active batches | k combined +/- (OpenMC) [pcm] | Naive SE of the batch mean [pcm] | rho(1) | rho(2) | rho(5) | Corrected SE [pcm] | Corrected / OpenMC |
|---|---|---|---|---|---|---|---|---|
| dep_00/openmc_simulation_n0.h5 | 50 | 134 | 169 | -0.03 | -0.00 | -0.25 | 101 | 0.75 |
| dep_00/openmc_simulation_n1.h5 | 50 | 148 | 181 | +0.13 | +0.05 | +0.16 | 203 | 1.37 |
| dep_00/openmc_simulation_n2.h5 | 50 | 123 | 188 | +0.11 | +0.04 | +0.04 | 267 | 2.17 |
| dep_00/openmc_simulation_n3.h5 | 50 | 122 | 178 | +0.02 | +0.02 | +0.12 | 206 | 1.69 |
| dep_00/openmc_simulation_n4.h5 | 50 | 144 | 178 | +0.04 | -0.14 | +0.02 | 173 | 1.20 |
| dep_00/openmc_simulation_n5.h5 | 50 | 118 | 179 | +0.05 | -0.12 | +0.03 | 145 | 1.22 |
| dep_01/openmc_simulation_n6.h5 | 50 | 138 | 214 | +0.07 | -0.00 | +0.02 | 165 | 1.20 |
| dep_01/openmc_simulation_n7.h5 | 50 | 131 | 182 | -0.07 | -0.12 | -0.16 | 202 | 1.54 |
| dep_02/openmc_simulation_n8.h5 | 50 | 135 | 200 | -0.13 | -0.07 | -0.25 | 47 | 0.34 |
| dep_02/openmc_simulation_n9.h5 | 50 | 128 | 171 | -0.09 | +0.07 | -0.10 | 219 | 1.72 |

Mean corrected / reported over 10 statepoints: **1.32**. OpenMC's combined estimator treats the active batches as independent; a positive lag-1 correlation of the fission source between batches makes its uncertainty too small by about sqrt((1 + rho) / (1 - rho)).

## 4. Batch-to-batch correlation of k in the statepoints

| Statepoint | Active batches | k combined +/- (OpenMC) [pcm] | Naive SE of the batch mean [pcm] | rho(1) | rho(2) | rho(5) | Corrected SE [pcm] | Corrected / OpenMC |
|---|---|---|---|---|---|---|---|---|
| dep_00/openmc_simulation_n0.h5 | 40 | 186 | 326 | -0.07 | -0.05 | -0.18 | 189 | 1.02 |
| dep_00/openmc_simulation_n0.h5 | 40 | 194 | 337 | -0.26 | +0.16 | +0.00 | 290 | 1.50 |

Mean corrected / reported over 2 statepoints: **1.26**. OpenMC's combined estimator treats the active batches as independent; a positive lag-1 correlation of the fission source between batches makes its uncertainty too small by about sqrt((1 + rho) / (1 - rho)).

## 4. Batch-to-batch correlation of k in the statepoints

| Statepoint | Active batches | k combined +/- (OpenMC) [pcm] | Naive SE of the batch mean [pcm] | rho(1) | rho(2) | rho(5) | Corrected SE [pcm] | Corrected / OpenMC |
|---|---|---|---|---|---|---|---|---|
| dep_00/openmc_simulation_n0.h5 | 100 | 55 | 72 | -0.16 | +0.14 | -0.02 | 32 | 0.59 |
| dep_00/openmc_simulation_n0.h5 | 100 | 65 | 111 | +0.32 | +0.01 | +0.10 | 149 | 2.28 |
| dep_00/openmc_simulation_n0.h5 | 100 | 64 | 119 | -0.02 | +0.02 | -0.03 | 136 | 2.13 |
| dep_00/openmc_simulation_n0.h5 | 100 | 63 | 94 | -0.19 | +0.00 | -0.13 | 936 | 14.93 |
| dep_00/openmc_simulation_n0.h5 | 100 | 60 | 75 | -0.11 | -0.04 | +0.00 | 52 | 0.87 |
| dep_00/openmc_simulation_n0.h5 | 100 | 62 | 102 | -0.00 | +0.02 | +0.01 | 85 | 1.38 |
| dep_00/openmc_simulation_n0.h5 | 100 | 57 | 94 | -0.08 | -0.14 | +0.17 | 97 | 1.71 |
| dep_00/openmc_simulation_n0.h5 | 100 | 61 | 117 | -0.19 | +0.04 | +0.01 | 65 | 1.06 |
| dep_00/openmc_simulation_n0.h5 | 100 | 67 | 96 | -0.04 | -0.11 | +0.19 | 93 | 1.39 |
| dep_00/openmc_simulation_n0.h5 | 100 | 67 | 109 | +0.07 | +0.01 | +0.09 | 125 | 1.87 |
| dep_00/openmc_simulation_n0.h5 | 100 | 62 | 82 | +0.13 | +0.08 | +0.04 | 107 | 1.72 |
| dep_00/openmc_simulation_n0.h5 | 100 | 68 | 106 | +0.03 | +0.21 | +0.14 | 152 | 2.22 |
| dep_00/openmc_simulation_n0.h5 | 100 | 65 | 86 | -0.15 | +0.05 | +0.08 | 73 | 1.13 |
| dep_00/openmc_simulation_n0.h5 | 100 | 65 | 100 | +0.05 | -0.19 | +0.02 | 57 | 0.89 |
| dep_00/openmc_simulation_n0.h5 | 100 | 65 | 101 | +0.05 | +0.01 | -0.06 | 147 | 2.26 |
| dep_00/openmc_simulation_n0.h5 | 100 | 63 | 97 | -0.04 | -0.00 | +0.15 | 96 | 1.51 |
| dep_00/openmc_simulation_n0.h5 | 100 | 72 | 91 | -0.14 | +0.13 | -0.02 | 106 | 1.47 |
| dep_00/openmc_simulation_n0.h5 | 100 | 69 | 98 | -0.15 | +0.02 | +0.01 | 75 | 1.08 |
| dep_00/openmc_simulation_n0.h5 | 100 | 68 | 98 | -0.07 | +0.09 | -0.05 | 92 | 1.35 |
| dep_00/openmc_simulation_n0.h5 | 100 | 54 | 89 | -0.01 | -0.02 | +0.05 | 72 | 1.35 |
| dep_00/openmc_simulation_n0.h5 | 100 | 64 | 85 | -0.03 | +0.05 | -0.00 | 103 | 1.60 |
| dep_00/openmc_simulation_n0.h5 | 100 | 66 | 104 | -0.05 | -0.07 | -0.13 | 83 | 1.26 |
| dep_00/openmc_simulation_n0.h5 | 100 | 62 | 96 | -0.08 | -0.10 | -0.14 | 33 | 0.53 |
| dep_00/openmc_simulation_n0.h5 | 100 | 62 | 97 | +0.01 | -0.04 | +0.05 | 112 | 1.81 |
| dep_00/openmc_simulation_n0.h5 | 100 | 60 | 94 | -0.12 | +0.02 | +0.11 | 102 | 1.69 |
| dep_00/openmc_simulation_n0.h5 | 100 | 65 | 108 | +0.10 | -0.01 | -0.06 | 128 | 1.97 |
| dep_00/openmc_simulation_n0.h5 | 100 | 69 | 93 | -0.03 | +0.11 | -0.03 | 117 | 1.69 |
| dep_00/openmc_simulation_n0.h5 | 100 | 64 | 100 | +0.20 | +0.07 | +0.05 | 115 | 1.79 |
| dep_00/openmc_simulation_n0.h5 | 100 | 61 | 93 | +0.05 | -0.00 | +0.02 | 86 | 1.41 |
| dep_00/openmc_simulation_n0.h5 | 100 | 60 | 98 | -0.03 | +0.02 | +0.07 | 91 | 1.51 |

Mean corrected / reported over 30 statepoints: **1.95**. OpenMC's combined estimator treats the active batches as independent; a positive lag-1 correlation of the fission source between batches makes its uncertainty too small by about sqrt((1 + rho) / (1 - rho)).
