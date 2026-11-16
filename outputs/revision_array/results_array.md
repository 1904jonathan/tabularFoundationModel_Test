# Array revision analyses (A1-A3)

Scripts: `outputs/revision_array/scripts/` (`array_utils.py`, `a1_crossmodel_synth.py`, `a2_20ng_taxonomies.py`, `a3_20ng_audit.py prepare|score`, `make_results_array.py`). All analyses are post hoc on the saved predictions of `outputs/revision/` (no model re-trained). Logs: `a1_log.txt`, `a2_log.txt`, `a3_prepare_log.txt`, `a3_score_log.txt`.

Definitions. For item i and model m, an *unaligned error* is a misclassification whose predicted class lies outside the (hypothesised) group of the observed label. `n_unal(i)` = number of models with an unaligned error on i. `U_all` = items with n_unal = M (all models), `U_exactly1` = items with n_unal = 1. Detector `unaligned>=k` flags items with n_unal >= k; baseline `misclassified>=k` flags items misclassified by >= k models (consensus filter). Ground truth (testbed only): corrupted = observed label != clean label (generator rebuilt with label_noise = 0).

## A1. Testbed: cross-model consistency

| set | n | M | n_corrupt | n_corrupt_crossgroup | n_corrupt_ingroup | U_all | U_all_corrupt | U_exactly1 | U_exactly1_corrupt | U_any | U_any_corrupt | E_all | E_all_corrupt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed42_7models | 432 | 7 | 15 | 15 | 0 | 15 | 15 | 15 | 0 | 33 | 15 | 49 | 15 |
| seed42_8models | 432 | 8 | 15 | 15 | 0 | 15 | 15 | 13 | 0 | 33 | 15 | 46 | 15 |
| seed0 | 438 | 8 | 22 | 19 | 3 | 19 | 19 | 9 | 0 | 37 | 19 | 53 | 20 |
| seed1 | 463 | 8 | 20 | 13 | 7 | 13 | 13 | 14 | 1 | 33 | 14 | 39 | 19 |
| seed2 | 471 | 8 | 24 | 22 | 2 | 22 | 22 | 8 | 0 | 35 | 22 | 44 | 24 |
| seed3 | 430 | 8 | 22 | 20 | 2 | 20 | 20 | 5 | 0 | 33 | 20 | 54 | 21 |
| seed4 | 465 | 8 | 21 | 20 | 1 | 20 | 20 | 7 | 0 | 32 | 20 | 43 | 21 |
| seed5 | 449 | 8 | 19 | 17 | 2 | 16 | 16 | 7 | 0 | 31 | 17 | 40 | 18 |
| seed6 | 436 | 8 | 25 | 22 | 3 | 22 | 22 | 10 | 1 | 41 | 23 | 51 | 24 |
| seed7 | 435 | 8 | 22 | 19 | 3 | 19 | 19 | 6 | 0 | 32 | 19 | 55 | 21 |
| seed8 | 462 | 8 | 23 | 22 | 1 | 21 | 21 | 8 | 0 | 35 | 22 | 37 | 22 |
| seed9 | 449 | 8 | 24 | 20 | 4 | 20 | 20 | 5 | 2 | 27 | 22 | 40 | 23 |

Seeds 0-9 pooled: n = 4498, corrupted = 222 (cross-group 194, in-group 28 = 0.126); U_all = 192, of which corrupted 192; U_exactly1 = 79, of which corrupted 4; E_all (misclassified by all 8) = 456, of which corrupted 213. Expected in-group share of uniform corruptions = mean beta = 0.083.

### Precision / recall, seed 42 (7 models of Table 2)

| detector | k | flagged | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|---|---|
| unaligned>=k | 1 | 33 | 15 | 18 | 0 | 0.455 | 1.000 |
| misclassified>=k | 1 | 214 | 15 | 199 | 0 | 0.070 | 1.000 |
| unaligned>=k | 2 | 18 | 15 | 3 | 0 | 0.833 | 1.000 |
| misclassified>=k | 2 | 155 | 15 | 140 | 0 | 0.097 | 1.000 |
| unaligned>=k | 3 | 17 | 15 | 2 | 0 | 0.882 | 1.000 |
| misclassified>=k | 3 | 120 | 15 | 105 | 0 | 0.125 | 1.000 |
| unaligned>=k | 4 | 15 | 15 | 0 | 0 | 1.000 | 1.000 |
| misclassified>=k | 4 | 95 | 15 | 80 | 0 | 0.158 | 1.000 |
| unaligned>=k | 5 | 15 | 15 | 0 | 0 | 1.000 | 1.000 |
| misclassified>=k | 5 | 80 | 15 | 65 | 0 | 0.188 | 1.000 |
| unaligned>=k | 6 | 15 | 15 | 0 | 0 | 1.000 | 1.000 |
| misclassified>=k | 6 | 65 | 15 | 50 | 0 | 0.231 | 1.000 |
| unaligned>=k | 7 | 15 | 15 | 0 | 0 | 1.000 | 1.000 |
| misclassified>=k | 7 | 49 | 15 | 34 | 0 | 0.306 | 1.000 |

### Precision / recall, seeds 0-9 (8 models): mean +- sd over seeds and pooled

| detector | k | precision_mean | precision_sd | recall_mean | recall_sd | recall_cross_mean | tp | fp | fn | precision_pooled | recall_pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| misclassified>=k | 1 | 0.096 | 0.009 | 0.991 | 0.019 | 1.000 | 220 | 2078 | 2 | 0.096 | 0.991 |
| misclassified>=k | 2 | 0.127 | 0.015 | 0.991 | 0.019 | 1.000 | 220 | 1521 | 2 | 0.126 | 0.991 |
| misclassified>=k | 3 | 0.156 | 0.021 | 0.991 | 0.019 | 1.000 | 220 | 1197 | 2 | 0.155 | 0.991 |
| misclassified>=k | 4 | 0.191 | 0.027 | 0.987 | 0.022 | 1.000 | 219 | 938 | 3 | 0.189 | 0.986 |
| misclassified>=k | 5 | 0.230 | 0.029 | 0.982 | 0.023 | 1.000 | 218 | 738 | 4 | 0.228 | 0.982 |
| misclassified>=k | 6 | 0.288 | 0.050 | 0.978 | 0.023 | 1.000 | 217 | 550 | 5 | 0.283 | 0.977 |
| misclassified>=k | 7 | 0.351 | 0.057 | 0.974 | 0.023 | 1.000 | 216 | 411 | 6 | 0.344 | 0.973 |
| misclassified>=k | 8 | 0.476 | 0.079 | 0.959 | 0.026 | 0.994 | 213 | 243 | 9 | 0.467 | 0.959 |
| unaligned>=k | 1 | 0.594 | 0.100 | 0.889 | 0.073 | 1.000 | 198 | 138 | 24 | 0.589 | 0.892 |
| unaligned>=k | 2 | 0.756 | 0.075 | 0.872 | 0.087 | 1.000 | 194 | 63 | 28 | 0.755 | 0.874 |
| unaligned>=k | 3 | 0.864 | 0.043 | 0.872 | 0.087 | 1.000 | 194 | 31 | 28 | 0.862 | 0.874 |
| unaligned>=k | 4 | 0.932 | 0.032 | 0.872 | 0.087 | 1.000 | 194 | 14 | 28 | 0.933 | 0.874 |
| unaligned>=k | 5 | 0.976 | 0.034 | 0.872 | 0.087 | 1.000 | 194 | 5 | 28 | 0.975 | 0.874 |
| unaligned>=k | 6 | 0.994 | 0.018 | 0.872 | 0.087 | 1.000 | 194 | 1 | 28 | 0.995 | 0.874 |
| unaligned>=k | 7 | 0.994 | 0.018 | 0.872 | 0.087 | 1.000 | 194 | 1 | 28 | 0.995 | 0.874 |
| unaligned>=k | 8 | 1.000 | 0.000 | 0.862 | 0.083 | 0.990 | 192 | 0 | 30 | 1.000 | 0.865 |

### Items unaligned for exactly one model, by model

| model | n | corrupt | n_seed42 |
|---|---|---|---|
| HistGB | 13 | 0 | 4.000 |
| KNN | 19 | 1 | 3.000 |
| LightGBM | 3 | 0 | 0.000 |
| LogisticRegression | 29 | 2 | 7.000 |
| MLP | 5 | 0 | 0.000 |
| RandomForest | 1 | 0 | 0.000 |
| XGBoost | 9 | 1 | 1.000 |

## A1. Mixture model: dilution and noise-rate estimators

### Seed 42 per-model decomposition (GAER = alpha*E_clean/E + beta_realised*E_corrupt/E)

| model | E | U | GAER | GAER0 | E_clean | E_corrupt | alpha_clean | beta_realised | GAER_pred_alpha1 | eta_hat_plugin |
|---|---|---|---|---|---|---|---|---|---|---|
| TabICL | 73 | 15 | 0.795 | 0.083 | 58 | 15 | 1.000 | 0.000 | 0.795 | 0.038 |
| XGBoost | 103 | 19 | 0.816 | 0.085 | 88 | 15 | 0.955 | 0.000 | 0.854 | 0.048 |
| RandomForest | 103 | 15 | 0.854 | 0.084 | 88 | 15 | 1.000 | 0.000 | 0.854 | 0.038 |
| LightGBM | 109 | 17 | 0.844 | 0.083 | 94 | 15 | 0.979 | 0.000 | 0.862 | 0.043 |
| HistGB | 112 | 21 | 0.812 | 0.085 | 97 | 15 | 0.938 | 0.000 | 0.866 | 0.053 |
| LogisticRegression | 124 | 23 | 0.815 | 0.087 | 109 | 15 | 0.927 | 0.000 | 0.879 | 0.058 |
| KNN | 154 | 18 | 0.883 | 0.087 | 139 | 15 | 0.978 | 0.000 | 0.903 | 0.046 |

Seeds 0-9, 80 model-seed pairs: GAER - (1 - U_corrupt/E) [alpha = 1 prediction]: mean -0.0260, mean |.| 0.0260, max |.| 0.0735; alpha_clean range 0.911-1.000; realised beta on corrupted errors: mean 0.110. Within-seed correlation of GAER with 1/E: mean -0.848 (range -0.977 to -0.733).

### Noise-rate estimators vs truth

| set | n | n_corrupt | eta_true | U_all | lower_bound_ok | eta_hat_Uall | U_min | U_min_model | eta_hat_Umin | eta_hat_plugin_median | eta_hat_plugin_max | reg_R2 | n_hat_reg | eta_hat_reg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed42_7models | 432 | 15 | 0.0347 | 15 | True | 0.0378 | 15 | TabICL | 0.0379 | 0.0456 | 0.0583 | 0.4719 | 11.3701 | 0.0263 |
| seed42_8models | 432 | 15 | 0.0347 | 15 | True | 0.0378 | 15 | TabICL | 0.0379 | 0.0443 | 0.0583 | 0.4946 | 12.0474 | 0.0279 |
| seed0 | 438 | 22 | 0.0502 | 19 | True | 0.0472 | 19 | TabICL | 0.0477 | 0.0576 | 0.0675 | 0.5993 | 18.3810 | 0.0420 |
| seed1 | 463 | 20 | 0.0432 | 13 | True | 0.0306 | 14 | TabICL | 0.0329 | 0.0399 | 0.0495 | 0.5762 | 11.1192 | 0.0240 |
| seed2 | 471 | 24 | 0.0510 | 22 | True | 0.0510 | 22 | TabICL | 0.0509 | 0.0571 | 0.0653 | 0.8309 | 19.3213 | 0.0410 |
| seed3 | 430 | 22 | 0.0512 | 20 | True | 0.0508 | 20 | TabICL | 0.0507 | 0.0585 | 0.0687 | 0.6689 | 18.5964 | 0.0432 |
| seed4 | 465 | 21 | 0.0452 | 20 | True | 0.0468 | 20 | TabICL | 0.0470 | 0.0518 | 0.0683 | 0.6876 | 19.5064 | 0.0419 |
| seed5 | 449 | 19 | 0.0423 | 16 | True | 0.0389 | 17 | TabICL | 0.0415 | 0.0500 | 0.0584 | 0.6810 | 15.1193 | 0.0337 |
| seed6 | 436 | 25 | 0.0573 | 22 | True | 0.0550 | 22 | TabICL | 0.0555 | 0.0680 | 0.0803 | 0.5376 | 17.5097 | 0.0402 |
| seed7 | 435 | 22 | 0.0506 | 19 | True | 0.0476 | 20 | TabICL | 0.0502 | 0.0553 | 0.0629 | 0.8404 | 19.0704 | 0.0438 |
| seed8 | 462 | 23 | 0.0498 | 21 | True | 0.0496 | 22 | TabICL | 0.0517 | 0.0566 | 0.0659 | 0.8693 | 22.8890 | 0.0495 |
| seed9 | 449 | 24 | 0.0535 | 20 | True | 0.0487 | 20 | TabICL | 0.0486 | 0.0535 | 0.0561 | 0.9551 | 21.2420 | 0.0473 |

Seeds 0-9 summary:

| estimator | mean_est | mean_true | mean_bias | mean_abs_err | max_abs_err | rel_bias |
|---|---|---|---|---|---|---|
| eta_hat_Uall | 0.0466 | 0.0494 | -0.0028 | 0.0031 | 0.0126 | -0.0595 |
| eta_hat_Umin | 0.0477 | 0.0494 | -0.0018 | 0.0025 | 0.0103 | -0.0370 |
| eta_hat_plugin_median | 0.0548 | 0.0494 | 0.0054 | 0.0061 | 0.0106 | 0.1078 |
| eta_hat_reg | 0.0407 | 0.0494 | -0.0087 | 0.0087 | 0.0192 | -0.1786 |

U_all <= n_corrupt held in 12 of 12 sets.

## A2. 20 Newsgroups: wrong taxonomies

Taxonomy definitions: `a2_20ng_taxonomy_defs.csv`.

| taxonomy | model | GAER | GAER0 | enrichment | perm_p95 | perm_p | aligned_H_only | aligned_T_only | paired_p_vs_H |
|---|---|---|---|---|---|---|---|---|---|
| H | TabICL | 0.4746 | 0.1515 | 3.132 | 0.2168 | 0.0004998 | 0 | 0 | 1 |
| H | RandomForest | 0.4283 | 0.1544 | 2.774 | 0.2061 | 0.0004998 | 0 | 0 | 1 |
| H | XGBoost | 0.4382 | 0.1515 | 2.893 | 0.206 | 0.0004998 | 0 | 0 | 1 |
| H | LightGBM | 0.4185 | 0.1519 | 2.754 | 0.2074 | 0.0004998 | 0 | 0 | 1 |
| H | HistGB | 0.432 | 0.1527 | 2.829 | 0.2086 | 0.0004998 | 0 | 0 | 1 |
| H | LogisticRegression | 0.4701 | 0.1507 | 3.12 | 0.2119 | 0.0004998 | 0 | 0 | 1 |
| H | KNN | 0.3888 | 0.1551 | 2.506 | 0.202 | 0.0004998 | 0 | 0 | 1 |
| H | MLP | 0.4704 | 0.1526 | 3.082 | 0.2233 | 0.0004998 | 0 | 0 | 1 |
| W1_alpha_full | TabICL | 0.3203 | 0.147 | 2.179 | 0.2168 | 0.0004998 | 100 | 21 | 1.606e-13 |
| W1_alpha_full | RandomForest | 0.2921 | 0.1486 | 1.966 | 0.2061 | 0.0009995 | 95 | 19 | 2.421e-13 |
| W1_alpha_full | XGBoost | 0.2884 | 0.148 | 1.948 | 0.2079 | 0.0009995 | 100 | 20 | 5.508e-14 |
| W1_alpha_full | LightGBM | 0.3 | 0.1484 | 2.021 | 0.2093 | 0.0004998 | 83 | 19 | 1.016e-10 |
| W1_alpha_full | HistGB | 0.2942 | 0.1481 | 1.987 | 0.2086 | 0.0004998 | 93 | 19 | 6.748e-13 |
| W1_alpha_full | LogisticRegression | 0.3064 | 0.1472 | 2.081 | 0.2158 | 0.001499 | 103 | 18 | 1.158e-15 |
| W1_alpha_full | KNN | 0.2462 | 0.151 | 1.63 | 0.2038 | 0.005497 | 103 | 19 | 3.762e-15 |
| W1_alpha_full | MLP | 0.3221 | 0.1494 | 2.157 | 0.2253 | 0.001499 | 98 | 23 | 3.24e-12 |
| W2_alpha_leaf | TabICL | 0.1934 | 0.144 | 1.343 | 0.2148 | 0.1194 | 181 | 37 | 5.633e-24 |
| W2_alpha_leaf | RandomForest | 0.1685 | 0.1457 | 1.156 | 0.2061 | 0.2519 | 193 | 48 | 8.645e-22 |
| W2_alpha_leaf | XGBoost | 0.1685 | 0.146 | 1.155 | 0.206 | 0.2519 | 184 | 40 | 2.907e-23 |
| W2_alpha_leaf | LightGBM | 0.1722 | 0.1462 | 1.178 | 0.2074 | 0.2164 | 172 | 39 | 4.013e-21 |
| W2_alpha_leaf | HistGB | 0.1527 | 0.1452 | 1.052 | 0.2086 | 0.4163 | 187 | 37 | 2.614e-25 |
| W2_alpha_leaf | LogisticRegression | 0.1946 | 0.1454 | 1.338 | 0.2158 | 0.1199 | 191 | 48 | 2.223e-21 |
| W2_alpha_leaf | KNN | 0.1664 | 0.1457 | 1.142 | 0.2037 | 0.2789 | 182 | 51 | 1.889e-18 |
| W2_alpha_leaf | MLP | 0.1917 | 0.1463 | 1.31 | 0.2234 | 0.1494 | 180 | 39 | 7.648e-23 |
| W3_halfswap_draw0 | TabICL | 0.2285 | 0.1537 | 1.487 | 0.2188 | 0.03798 | 159 | 33 | 5.556e-21 |
| W3_halfswap_draw0 | RandomForest | 0.2312 | 0.1532 | 1.509 | 0.2115 | 0.02149 | 159 | 49 | 9.38e-15 |
| W3_halfswap_draw0 | XGBoost | 0.2397 | 0.1518 | 1.579 | 0.2097 | 0.01749 | 152 | 46 | 1.945e-14 |
| W3_halfswap_draw0 | LightGBM | 0.2204 | 0.1522 | 1.448 | 0.2112 | 0.03998 | 153 | 46 | 1.262e-14 |
| W3_halfswap_draw0 | HistGB | 0.2067 | 0.1532 | 1.349 | 0.2123 | 0.06497 | 160 | 39 | 1.343e-18 |
| W3_halfswap_draw0 | LogisticRegression | 0.2524 | 0.1496 | 1.687 | 0.2217 | 0.01599 | 163 | 50 | 3.598e-15 |
| W3_halfswap_draw0 | KNN | 0.2173 | 0.1523 | 1.427 | 0.2071 | 0.02949 | 149 | 48 | 3.081e-13 |
| W3_halfswap_draw0 | MLP | 0.2312 | 0.1542 | 1.5 | 0.2274 | 0.04748 | 162 | 41 | 3.274e-18 |
| W4_crosscut | TabICL | 0.08008 | 0.1498 | 0.5347 | 0.2188 | 0.9775 | 243 | 41 | 4.279e-36 |
| W4_crosscut | RandomForest | 0.09857 | 0.1506 | 0.6544 | 0.2062 | 0.948 | 239 | 55 | 1.669e-28 |
| W4_crosscut | XGBoost | 0.1049 | 0.1505 | 0.6968 | 0.2097 | 0.902 | 234 | 56 | 4.929e-27 |
| W4_crosscut | LightGBM | 0.1111 | 0.1517 | 0.7326 | 0.2094 | 0.8636 | 226 | 60 | 8.018e-24 |
| W4_crosscut | HistGB | 0.1024 | 0.1509 | 0.6786 | 0.2087 | 0.9235 | 232 | 55 | 4.973e-27 |
| W4_crosscut | LogisticRegression | 0.07514 | 0.1498 | 0.5017 | 0.2177 | 0.988 | 244 | 39 | 2.004e-37 |
| W4_crosscut | KNN | 0.1087 | 0.1499 | 0.7251 | 0.2054 | 0.8991 | 229 | 64 | 6.091e-23 |
| W4_crosscut | MLP | 0.09486 | 0.1502 | 0.6316 | 0.2253 | 0.908 | 238 | 48 | 1.988e-31 |

Summary over the 8 models:

| taxonomy | GAER_min | GAER_max | enr_min | enr_max | perm_p_min | perm_p_max | n_models_perm_p_lt_05 | paired_p_max |
|---|---|---|---|---|---|---|---|---|
| H | 0.3888 | 0.4746 | 2.506 | 3.132 | 0.0004998 | 0.0004998 | 8 | 1 |
| W1_alpha_full | 0.2462 | 0.3221 | 1.63 | 2.179 | 0.0004998 | 0.005497 | 8 | 1.016e-10 |
| W2_alpha_leaf | 0.1527 | 0.1946 | 1.052 | 1.343 | 0.1194 | 0.4163 | 0 | 1.889e-18 |
| W3_halfswap_draw0 | 0.2067 | 0.2524 | 1.349 | 1.687 | 0.01599 | 0.06497 | 7 | 3.081e-13 |
| W4_crosscut | 0.07514 | 0.1111 | 0.5017 | 0.7326 | 0.8636 | 0.988 | 0 | 6.091e-23 |

W3, 1,000 random half-swapped taxonomies:

| model | GAER_median | GAER_q05 | GAER_q95 | frac_above_H_null_p95 | frac_sig_worse_than_H |
|---|---|---|---|---|---|
| HistGB | 0.196 | 0.147 | 0.250 | 0.310 | 1.000 |
| KNN | 0.191 | 0.146 | 0.239 | 0.332 | 1.000 |
| LightGBM | 0.194 | 0.144 | 0.246 | 0.315 | 1.000 |
| LogisticRegression | 0.198 | 0.145 | 0.258 | 0.350 | 1.000 |
| MLP | 0.200 | 0.136 | 0.269 | 0.288 | 1.000 |
| RandomForest | 0.197 | 0.154 | 0.249 | 0.375 | 1.000 |
| TabICL | 0.203 | 0.152 | 0.260 | 0.332 | 1.000 |
| XGBoost | 0.195 | 0.150 | 0.245 | 0.348 | 1.000 |

Over all 8,000 model-draw pairs: GAER median 0.196, above the permutation-null 95th percentile in 0.331, significantly worse than H (paired exact test, p < 0.05) in 1.000.


### Cross-model consistency on 20NG under each taxonomy (8 models, 1,000 documents)

| taxonomy | n | M | E_all | U_all | A_all | U_exactly1 | U_any | U_ge1 | U_ge2 | U_ge3 | U_ge4 | U_ge5 | U_ge6 | U_ge7 | U_ge8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 1000 | 8 | 313 | 127 | 67 | 98 | 517 | 517 | 419 | 366 | 315 | 268 | 224 | 174 | 127 |
| W1_alpha_full | 1000 | 8 | 313 | 173 | 38 | 92 | 609 | 609 | 517 | 456 | 398 | 347 | 291 | 236 | 173 |
| W2_alpha_leaf | 1000 | 8 | 313 | 202 | 11 | 93 | 690 | 690 | 597 | 524 | 467 | 418 | 356 | 287 | 202 |
| W3_halfswap_draw0 | 1000 | 8 | 313 | 187 | 27 | 97 | 657 | 657 | 560 | 496 | 440 | 380 | 324 | 271 | 187 |
| W4_crosscut | 1000 | 8 | 313 | 225 | 4 | 82 | 734 | 734 | 652 | 577 | 519 | 456 | 396 | 318 | 225 |

Items unaligned for exactly one model under H, by model: KNN 48, RandomForest 20, LogisticRegression 7, HistGB 6, MLP 6, LightGBM 4, XGBoost 4, TabICL 3


20NG defect-rate estimate: beta_bar = 0.1493, U_all = 127, eta_hat_Uall = U_all/((1-beta_bar) n) = 0.1493; per-model plug-in U_m/((1-GAER0_m) n) ranges 0.316-0.426 (alpha_m far from 1 on real data, Lemma 2).

## A3. 20 Newsgroups item audit

Length (words) of U_all (unaligned for all 8 models), A_all (misclassified by all 8, every error in-group), all test documents and documents correct for all 8:

| set | n | words_median | words_q25 | words_q75 | frac_le_20_words |
|---|---|---|---|---|---|
| U_all | 127 | 44.000 | 20.000 | 88.500 | 0.268 |
| A_all | 67 | 106.000 | 40.500 | 167.500 | 0.045 |
| all_test | 1000 | 87.000 | 42.750 | 167.000 | 0.104 |
| correct_by_all | 252 | 119.000 | 60.000 | 239.250 | 0.060 |

Blind audit (15 random U_all + 15 random A_all documents, shuffled; judged by the first author from a 400-character excerpt and the observed newsgroup, without set membership or predictions):

| set | ambiguous | off-topic | on-topic | uninformative |
|---|---|---|---|---|
| A_all | 0 | 0 | 15 | 0 |
| U_all | 5 | 4 | 3 | 3 |

On-topic: U_all 3/15, A_all 15/15; Fisher exact two-sided p = 1.1e-05.


Per-document judgements (unblinded):

| audit_id | set | observed_newsgroup | majority_pred | judgement | note |
|---|---|---|---|---|---|
| 1 | A_all | comp.windows.x | comp.os.ms-windows.misc | on-topic | X window manager config |
| 2 | A_all | talk.politics.misc | talk.politics.guns | on-topic | BATF/Waco raid: politics |
| 3 | A_all | sci.electronics | sci.crypt | on-topic | antenna market/specs |
| 4 | U_all | talk.politics.mideast | rec.motorcycles | on-topic | US military aid to Israel |
| 5 | U_all | sci.space | rec.autos | off-topic | espionage remark, nothing about space |
| 6 | A_all | talk.religion.misc | soc.religion.christian | on-topic | hell in the KJV Bible |
| 7 | U_all | talk.politics.guns | rec.motorcycles | ambiguous | shareholder activism against media; no gun content in text |
| 8 | U_all | sci.med | comp.sys.mac.hardware | on-topic | sciatica relief |
| 9 | U_all | comp.sys.mac.hardware | sci.electronics | uninformative | company address and phone numbers only |
| 10 | A_all | comp.graphics | comp.os.ms-windows.misc | on-topic | DOS screen capture, VESA |
| 11 | U_all | talk.politics.misc | sci.med | ambiguous | MDs vs RNs on a (policy) list: medicine vocabulary, political context |
| 12 | U_all | comp.sys.mac.hardware | sci.space | ambiguous | computer purchase discounts: comp vs forsale |
| 13 | U_all | sci.electronics | rec.sport.baseball | uninformative | reply-to address only |
| 14 | A_all | comp.windows.x | comp.graphics | on-topic | displaying a gif in an X window |
| 15 | A_all | comp.sys.ibm.pc.hardware | comp.sys.mac.hardware | on-topic | RAM over 16 MB in ISA PC |
| 16 | U_all | rec.sport.hockey | alt.atheism | uninformative | meta-discussion about a letter; no hockey content |
| 17 | U_all | talk.politics.misc | alt.atheism | on-topic | Rodney King trial: politics |
| 18 | U_all | sci.space | alt.atheism | off-topic | native American cultures 1400 vs 1800; nothing about space |
| 19 | A_all | sci.electronics | sci.space | on-topic | Atari 2600 chips: electronics |
| 20 | A_all | talk.religion.misc | soc.religion.christian | on-topic | biblical claims about Satan |
| 21 | A_all | comp.sys.mac.hardware | comp.sys.ibm.pc.hardware | on-topic | Mac IIvi with CD-ROM |
| 22 | U_all | talk.politics.mideast | rec.sport.baseball | off-topic | personal remark about a UVA student; no Middle-East content |
| 23 | U_all | talk.politics.mideast | sci.med | off-topic | university party ranking; no Middle-East content |
| 24 | A_all | alt.atheism | soc.religion.christian | on-topic | God and science |
| 25 | A_all | comp.graphics | comp.os.ms-windows.misc | on-topic | ATM fonts in Windows |
| 26 | A_all | talk.politics.mideast | talk.politics.guns | on-topic | Bosnia arms embargo: politics |
| 27 | A_all | sci.electronics | sci.med | on-topic | pink noise in semiconductors |
| 28 | U_all | alt.atheism | sci.electronics | ambiguous | advert for Darwin-fish emblem: religion vs forsale |
| 29 | A_all | alt.atheism | soc.religion.christian | on-topic | creationism vs science |
| 30 | U_all | misc.forsale | soc.religion.christian | ambiguous | advert for a psychic phone line (spam): forsale only by format |
