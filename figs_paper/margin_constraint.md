# Reactivity margin at the mission burnup as the constraint target

Campaign 10 three-dimensional histories, 57 designs, 7 with a cycle of 0, 54 reaching B_mission = 18.23 MWd/kgHM. k_bol campaign record. Leave-one-out, one design out at a time.

| Target | Kernel | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | Verdict accuracy | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| cycle length [d] | Matern 5/2 ARD (campaign) | 204.4 | 0.932 | 0.961 | 0.89 | 1.50 | 0.94 | 2 | 1 |
| margin at B_mission [pcm] | Matern 5/2 ARD (campaign) | 398.1 | 0.997 | 0.985 | 0.96 | 0.99 | 0.94 | 2 | 1 |
| cycle length [d] | Matern 1/2 ARD | 156.2 | 0.961 | 0.949 | 0.91 | 0.85 | 0.93 | 2 | 2 |
| margin at B_mission [pcm] | Matern 1/2 ARD | 1048.1 | 0.977 | 0.978 | 0.98 | 0.83 | 0.93 | 2 | 2 |
