# Reactivity margin at the mission burnup as the constraint target

Campaign 10 three-dimensional histories, 99 designs, 7 with a cycle of 0, 96 reaching B_mission = 18.23 MWd/kgHM. k_bol high statistics where available. Leave-one-out, one design out at a time.

| Target | Kernel | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | Verdict accuracy | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| cycle length [d] | Matern 5/2 ARD (campaign) | 158.0 | 0.932 | 0.948 | 0.92 | 1.30 | 0.89 | 5 | 6 |
| margin at B_mission [pcm] | Matern 5/2 ARD (campaign) | 271.8 | 0.998 | 0.950 | 0.97 | 1.00 | 0.85 | 8 | 6 |
| cycle length [d] | Matern 1/2 ARD | 111.5 | 0.966 | 0.932 | 0.94 | 0.99 | 0.86 | 7 | 6 |
| margin at B_mission [pcm] | Matern 1/2 ARD | 731.7 | 0.983 | 0.943 | 0.96 | 0.95 | 0.90 | 7 | 3 |
