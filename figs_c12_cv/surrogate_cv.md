# Campaign 12: comparison of surrogates on the constraints and the objectives


### Leave-one-out, 117 designs: g_margin [pcm]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible | Mean rule FP/FN | Margin rule FP/FN | Margin rule recalibrated FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 417 | 0.994 | 0.936 | 0.95 | 1.06 | 280 | 13/7 | 0/27 | 0/28 |
| GP Matern 5/2 iso | 556 | 0.990 | 0.937 | 0.96 | 1.10 | 313 | 12/8 | 0/28 | 0/31 |
| GP Matern 3/2 ARD | 477 | 0.992 | 0.937 | 0.94 | 1.04 | 274 | 14/9 | 0/28 | 0/28 |
| GP Matern 1/2 ARD | 856 | 0.976 | 0.942 | 0.97 | 0.91 | 282 | 7/4 | 0/32 | 0/28 |
| GP SE ARD | 398 | 0.995 | 0.937 | 0.96 | 1.08 | 286 | 13/7 | 0/28 | 0/32 |
| GP rational quadratic iso | 575 | 0.989 | 0.933 | 0.96 | 1.11 | 320 | 14/8 | 0/29 | 0/32 |
| GP Matern 5/2 ARD + Gd inventory | 394 | 0.995 | 0.937 | 0.95 | 1.08 | 270 | 13/8 | 0/27 | 0/30 |
| Quadratic response surface | 795 | 0.979 | 0.897 | 0.92 | 1.29 | 362 | 12/5 | 0/39 | 0/44 |
| Random forest | 1351 | 0.939 | 0.801 | 0.92 | 1.13 | 520 | 19/5 | 0/48 | 0/49 |
| Gradient boosting, quantiles | 1489 | 0.926 | 0.848 | 0.73 | 2.63 | 416 | 15/4 | 4/28 | 0/53 |
| MLP ensemble | 869 | 0.975 | 0.926 | 0.51 | 3.63 | 303 | 15/9 | 6/18 | 1/36 |
| GP classifier | -- | -- | -- | -- | -- | -- | 9/5 | -- | -- |

117 designs, 36 violate the constraint. FP: admitted, measured violated. FN: rejected, measured satisfied.

### Leave-one-out, 117 designs: g_kmin [pcm]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible | Mean rule FP/FN | Margin rule FP/FN | Margin rule recalibrated FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 2192 | 0.935 | 0.928 | 0.95 | 1.44 | 1489 | 2/7 | 0/31 | 0/36 |
| GP Matern 5/2 iso | 2505 | 0.915 | 0.925 | 0.95 | 1.14 | 1588 | 1/8 | 0/27 | 0/29 |
| GP Matern 3/2 ARD | 2290 | 0.929 | 0.928 | 0.93 | 1.71 | 1403 | 2/5 | 0/27 | 0/39 |
| GP Matern 1/2 ARD | 2677 | 0.903 | 0.943 | 0.94 | 1.15 | 1299 | 2/5 | 0/11 | 0/12 |
| GP SE ARD | 2075 | 0.942 | 0.932 | 0.97 | 1.08 | 1526 | 2/8 | 0/31 | 0/32 |
| GP rational quadratic iso | 2659 | 0.904 | 0.923 | 0.94 | 1.65 | 1585 | 2/7 | 0/28 | 0/44 |
| GP Matern 5/2 ARD + Gd inventory | 2230 | 0.933 | 0.922 | 0.93 | 1.45 | 1309 | 1/2 | 0/24 | 0/38 |
| Quadratic response surface | 2875 | 0.888 | 0.917 | 0.97 | 1.09 | 1693 | 0/9 | 0/35 | 0/36 |
| Random forest | 3349 | 0.848 | 0.941 | 0.96 | 0.97 | 966 | 1/1 | 0/16 | 0/14 |
| Gradient boosting, quantiles | 3430 | 0.841 | 0.946 | 0.81 | 3.52 | 965 | 0/0 | 0/4 | 0/29 |
| MLP ensemble | 2492 | 0.916 | 0.916 | 0.23 | 12.44 | 1500 | 1/6 | 0/7 | 0/37 |
| GP classifier | -- | -- | -- | -- | -- | -- | 4/2 | -- | -- |

117 designs, 18 violate the constraint. FP: admitted, measured violated. FN: rejected, measured satisfied.

### Leave-one-out, 117 designs: F_dH [-]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible |
|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 0.0262 | 0.744 | 0.871 | 0.95 | 1.07 | 0.0197 |
| GP Matern 5/2 iso | 0.0271 | 0.725 | 0.851 | 0.96 | 1.04 | 0.0211 |
| GP Matern 3/2 ARD | 0.0254 | 0.759 | 0.873 | 0.95 | 1.04 | 0.0196 |
| GP Matern 1/2 ARD | 0.0269 | 0.730 | 0.877 | 0.94 | 1.04 | 0.0185 |
| GP SE ARD | 0.0277 | 0.712 | 0.855 | 0.95 | 1.13 | 0.0205 |
| GP rational quadratic iso | 0.0279 | 0.708 | 0.847 | 0.96 | 1.06 | 0.0218 |
| GP Matern 5/2 ARD + Gd inventory | 0.0281 | 0.706 | 0.854 | 0.95 | 1.12 | 0.0195 |
| Quadratic response surface | 0.0297 | 0.670 | 0.835 | 0.92 | 1.06 | 0.0204 |
| Random forest | 0.0294 | 0.677 | 0.863 | 0.87 | 1.31 | 0.0225 |
| Gradient boosting, quantiles | 0.0295 | 0.675 | 0.854 | 0.71 | 3.84 | 0.0238 |
| MLP ensemble | 0.0257 | 0.754 | 0.857 | 0.14 | 24.52 | 0.0208 |

### Leave-one-out, 117 designs: c_max [ppm]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible |
|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 212 | 0.975 | 0.957 | 0.93 | 1.27 | 168 |
| GP Matern 5/2 iso | 212 | 0.975 | 0.962 | 0.97 | 1.02 | 175 |
| GP Matern 3/2 ARD | 217 | 0.974 | 0.967 | 0.92 | 1.22 | 137 |
| GP Matern 1/2 ARD | 191 | 0.980 | 0.977 | 0.97 | 0.87 | 144 |
| GP SE ARD | 270 | 0.960 | 0.957 | 0.91 | 1.20 | 181 |
| GP rational quadratic iso | 205 | 0.977 | 0.962 | 0.96 | 1.00 | 183 |
| GP Matern 5/2 ARD + Gd inventory | 215 | 0.975 | 0.953 | 0.95 | 1.18 | 166 |
| Quadratic response surface | 307 | 0.948 | 0.957 | 0.93 | 1.11 | 184 |
| Random forest | 486 | 0.870 | 0.930 | 0.97 | 1.44 | 199 |
| Gradient boosting, quantiles | 447 | 0.890 | 0.956 | 0.82 | 5.45 | 186 |
| MLP ensemble | 308 | 0.948 | 0.958 | 0.33 | 6.17 | 203 |

### Walk-forward, the 18 infill designs: g_margin [pcm]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible | Mean rule FP/FN | Margin rule FP/FN | Margin rule recalibrated FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 510 | 0.259 | 0.366 | 0.78 | 1.73 | 586 | 5/0 | 0/8 | 0/11 |
| GP Matern 5/2 iso | 479 | 0.346 | 0.571 | 0.89 | 1.31 | 565 | 2/1 | 0/10 | 0/11 |
| GP Matern 3/2 ARD | 474 | 0.360 | 0.598 | 0.83 | 1.48 | 559 | 2/2 | 1/8 | 0/10 |
| GP Matern 1/2 ARD | 555 | 0.123 | 0.420 | 1.00 | 0.76 | 686 | 1/8 | 0/11 | 0/11 |
| GP SE ARD | 520 | 0.229 | 0.344 | 0.78 | 1.81 | 616 | 1/5 | 0/9 | 0/11 |
| GP rational quadratic iso | 519 | 0.231 | 0.430 | 0.83 | 1.41 | 600 | 3/1 | 0/10 | 0/11 |
| GP Matern 5/2 ARD + Gd inventory | 459 | 0.401 | 0.651 | 0.78 | 1.65 | 538 | 1/2 | 0/8 | 0/10 |
| Quadratic response surface | 527 | 0.208 | 0.286 | 0.94 | 0.98 | 613 | 2/3 | 0/12 | 0/12 |
| Random forest | 609 | -0.058 | 0.038 | 0.94 | 0.96 | 675 | 5/1 | 0/13 | 0/13 |
| Gradient boosting, quantiles | 1077 | -2.302 | -0.430 | 0.33 | 8.08 | 1084 | 5/4 | 5/8 | 1/13 |
| MLP ensemble | 435 | 0.460 | 0.616 | 0.44 | 2.23 | 483 | 3/1 | 1/7 | 0/11 |
| GP classifier | -- | -- | -- | -- | -- | -- | 2/4 | -- | -- |

18 designs, 5 violate the constraint. FP: admitted, measured violated. FN: rejected, measured satisfied.

### Walk-forward, the 18 infill designs: g_kmin [pcm]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible | Mean rule FP/FN | Margin rule FP/FN | Margin rule recalibrated FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 1924 | 0.706 | 0.847 | 0.94 | 1.03 | 1757 | 4/0 | 0/2 | 0/2 |
| GP Matern 5/2 iso | 2167 | 0.627 | 0.752 | 1.00 | 1.00 | 2001 | 4/0 | 0/3 | 0/3 |
| GP Matern 3/2 ARD | 1844 | 0.730 | 0.851 | 0.94 | 0.94 | 1768 | 0/1 | 0/2 | 0/2 |
| GP Matern 1/2 ARD | 1969 | 0.692 | 0.822 | 1.00 | 0.69 | 1825 | 3/1 | 0/4 | 0/2 |
| GP SE ARD | 1920 | 0.707 | 0.806 | 0.94 | 1.02 | 1751 | 2/1 | 0/2 | 0/2 |
| GP rational quadratic iso | 2275 | 0.589 | 0.676 | 1.00 | 0.93 | 2157 | 3/0 | 0/7 | 0/7 |
| GP Matern 5/2 ARD + Gd inventory | 2241 | 0.601 | 0.814 | 0.94 | 1.24 | 2002 | 3/2 | 0/3 | 0/3 |
| Quadratic response surface | 1886 | 0.718 | 0.903 | 1.00 | 0.74 | 1325 | 3/0 | 0/2 | 0/1 |
| Random forest | 2277 | 0.589 | 0.829 | 1.00 | 0.79 | 2699 | 0/3 | 0/5 | 0/5 |
| Gradient boosting, quantiles | 2385 | 0.549 | 0.903 | 0.56 | 2.78 | 2077 | 0/0 | 0/1 | 0/2 |
| MLP ensemble | 1743 | 0.759 | 0.897 | 0.33 | 6.21 | 1112 | 3/0 | 2/0 | 0/4 |
| GP classifier | -- | -- | -- | -- | -- | -- | 0/1 | -- | -- |

18 designs, 4 violate the constraint. FP: admitted, measured violated. FN: rejected, measured satisfied.

### Walk-forward, the 18 infill designs: F_dH [-]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible |
|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 0.0222 | 0.735 | 0.845 | 0.94 | 1.00 | 0.0208 |
| GP Matern 5/2 iso | 0.0287 | 0.555 | 0.800 | 0.89 | 1.11 | 0.0308 |
| GP Matern 3/2 ARD | 0.0222 | 0.735 | 0.841 | 0.94 | 0.99 | 0.0212 |
| GP Matern 1/2 ARD | 0.0199 | 0.787 | 0.858 | 1.00 | 0.78 | 0.0190 |
| GP SE ARD | 0.0228 | 0.720 | 0.880 | 1.00 | 1.01 | 0.0212 |
| GP rational quadratic iso | 0.0285 | 0.561 | 0.798 | 0.94 | 1.06 | 0.0321 |
| GP Matern 5/2 ARD + Gd inventory | 0.0202 | 0.781 | 0.872 | 1.00 | 0.97 | 0.0197 |
| Quadratic response surface | 0.0309 | 0.485 | 0.806 | 0.94 | 1.16 | 0.0293 |
| Random forest | 0.0169 | 0.846 | 0.907 | 1.00 | 0.75 | 0.0166 |
| Gradient boosting, quantiles | 0.0180 | 0.825 | 0.911 | 0.78 | 5.39 | 0.0176 |
| MLP ensemble | 0.0314 | 0.470 | 0.806 | 0.22 | 10.42 | 0.0324 |

### Walk-forward, the 18 infill designs: c_max [ppm]

| Surrogate | RMSE | R^2 | Spearman | Cov. 2 sigma | s_z | RMSE feasible |
|---|---|---|---|---|---|---|
| GP Matern 5/2 ARD (campaign) | 358 | 0.455 | 0.703 | 0.78 | 2.38 | 350 |
| GP Matern 5/2 iso | 377 | 0.396 | 0.651 | 0.72 | 2.37 | 371 |
| GP Matern 3/2 ARD | 328 | 0.544 | 0.740 | 0.78 | 2.23 | 331 |
| GP Matern 1/2 ARD | 327 | 0.547 | 0.730 | 0.89 | 1.37 | 319 |
| GP SE ARD | 376 | 0.399 | 0.680 | 0.78 | 2.41 | 363 |
| GP rational quadratic iso | 395 | 0.339 | 0.651 | 0.67 | 2.45 | 399 |
| GP Matern 5/2 ARD + Gd inventory | 354 | 0.469 | 0.725 | 0.72 | 2.59 | 338 |
| Quadratic response surface | 319 | 0.570 | 0.825 | 0.89 | 1.29 | 256 |
| Random forest | 398 | 0.326 | 0.598 | 0.89 | 1.21 | 410 |
| Gradient boosting, quantiles | 436 | 0.193 | 0.682 | 0.56 | 7.42 | 390 |
| MLP ensemble | 378 | 0.394 | 0.728 | 0.28 | 3.97 | 346 |
