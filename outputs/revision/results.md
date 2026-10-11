# Revision experiments (Pattern Recognition resubmission)

All scripts: `outputs/revision/scripts/` (shared helper `revision_utils.py`). Run sequentially in one background process (`run_all.sh`), log in `outputs/revision/log.txt`.

## Environment

- CPU: AMD Ryzen 5 Pro 7535U with Radeon Graphics; logical cores: 12; torch threads: 6; CPU only (no GPU)
- OS: Windows-11-10.0.26300-SP0; Python 3.13.5
- Packages: scikit-learn 1.6.1, xgboost 3.3.0, lightgbm 4.6.0, torch 2.12.1, tabicl 2.1.1, numpy 2.1.3, scipy 1.15.3, pandas 2.2.3
- TabICL checkpoint: C:\Users\mehes/.cache/huggingface/hub/models--jingang--TabICL/snapshots\4dcd344ece2c00be9e831fdd35bed57b5ad83e19\tabicl-classifier-v2-20260212.ckpt (HF repo `jingang/TabICL`)

## Key findings (numbers from the tables below)

1. **E1 (10 seeds, d=64)**: TabICL macro-F1 0.815 ± 0.024 vs best baseline RandomForest 0.728 ± 0.038 (XGBoost 0.724 ± 0.033). TabICL beats every baseline on 10/10 seeds (Wilcoxon exact p = 0.00195 for each, the minimum attainable with n=10; Holm-adjusted 0.0137); Friedman χ² = 50.8, p = 1.0e-8; average rank 1.0.
2. **E2 (tuning, seed 42)**: tuning barely moves the baselines on the test set (best tuned: RandomForest 0.737, HistGB 0.735, XGBoost 0.723; LR/KNN grid optima equal the fixed settings). TabICL (0.809) beats every tuned baseline: McNemar p ≤ 2.7e-4, bootstrap 95% CIs of the macro-F1 gap all exclude 0 (smallest: vs tuned RF, +0.073 [0.038, 0.110]). TabICL with n_estimators=1: macro-F1 0.799 in 6.3 s predict vs 0.809 in 48 s for 8 members.
3. **E3 (LSA dimension)**: d is the dominant factor. d=64 is where TabICL's lead is largest (+0.086 macro-F1 over XGBoost); at d=128 the lead shrinks (0.878 vs LR 0.858) and at d=256 **LogisticRegression ties TabICL** (0.8865 vs 0.8869) while every model gains 0.07–0.20 macro-F1 over d=64. The d=64 setting of the paper is therefore a bottleneck for all models, and the TabICL advantage is largely specific to the low-dimensional regime. GAER decreases as d grows (more of the residual errors leave the group) but stays ~8–9× chance.
4. **E4 (20 Newsgroups, real data)**: the hypothesised taxonomy is strongly recovered by every model (GAER 0.39–0.47 vs permutation-null 95th pct ≈ 0.20–0.22, max ≈ 0.28–0.32; p = 1/2001 for all; enrichment 2.5–3.1×). TabICL is top on macro-F1 (0.477) but statistically tied with MLP (0.475) and LogisticRegression (0.471) (McNemar p = 0.69 / 0.60; bootstrap CIs include 0); it significantly beats RandomForest and KNN.
5. **Clean labels**: ≈5% of test labels are corrupted by `build_dataset`. Against clean labels TabICL accuracy is 0.870 ± 0.019 and its clean-label GAER is 0.997 ± 0.007, i.e. virtually all of its genuine errors fall within the confusable group; most of its out-of-group 'errors' are corrupted test labels.

## Definitions

- **GAER** = among misclassified test rows, fraction whose predicted class is in the same (hypothesised) group as the true class. **GAER0** (chance) = mean over errors of (|G_g(y)|−1)/(K−1). **Enrichment** = GAER/GAER0.
- **Permutation null**: 2000 random reassignments of classes to groups preserving the multiset of group sizes (seed 0); reported: null mean, 95th percentile, max, p = (1+#{null ≥ obs})/(B+1). With B=2000 the minimum p is 1/2001 ≈ 5.0e-4.
- **Super-class accuracy** = mean(group(ŷ) == group(y)).
- **Clean-label metrics** (synthetic only): the dataset is rebuilt with `label_noise=0.0` and the same seed; the text column was verified identical (row by row) for every seed used, so the uncorrupted label of each test row is recovered. Note: label noise in `build_dataset` is applied to the whole dataset before splitting, so ≈5% of TEST labels are also corrupted; clean-label metrics score predictions against the uncorrupted labels.

## E1 — Multi-seed rerun (main result)

Protocol: for seed s ∈ {0..9}: `build_dataset(n_per_class=60, label_noise=0.05, imbalance=True, seed=s)` → `train_test_split(test_size=0.25, random_state=s, stratify=category)` → `transform_texts(n_components=64, seed=s)` → the 7 models of `build_models(s, 30)` (fixed settings, `random_state=s`) + MLP (`MLPClassifier(hidden_layer_sizes=(256,128), early_stopping=True, max_iter=500, random_state=s)`). Dataset size varies with the seed (imbalance draw). Per-seed predictions: `predictions_seed{s}/predictions.csv`.

### Predictive metrics, mean ± std over 10 seeds

| model | accuracy | macro-F1 | weighted-F1 | log-loss | fit s | predict s |
|---|---|---|---|---|---|---|
| TabICL | 0.8284 ± 0.0202 | 0.8151 ± 0.0240 | 0.8263 ± 0.0214 | 0.7650 ± 0.0592 | 0.92 ± 0.29 | 51.33 ± 5.99 |
| RandomForest | 0.7576 ± 0.0316 | 0.7285 ± 0.0384 | 0.7487 ± 0.0351 | 1.4586 ± 0.2077 | 1.48 ± 0.22 | 0.13 ± 0.04 |
| XGBoost | 0.7459 ± 0.0280 | 0.7242 ± 0.0332 | 0.7418 ± 0.0300 | 1.0001 ± 0.1031 | 9.93 ± 2.07 | 0.03 ± 0.00 |
| LightGBM | 0.7419 ± 0.0340 | 0.7205 ± 0.0383 | 0.7372 ± 0.0358 | 1.3891 ± 0.1923 | 16.59 ± 6.13 | 0.12 ± 0.04 |
| LogisticRegression | 0.7374 ± 0.0235 | 0.7154 ± 0.0288 | 0.7342 ± 0.0249 | 1.0023 ± 0.0634 | 0.08 ± 0.03 | 0.00 ± 0.00 |
| HistGB | 0.7319 ± 0.0366 | 0.7071 ± 0.0420 | 0.7262 ± 0.0391 | 1.4612 ± 0.2031 | 18.88 ± 4.79 | 0.40 ± 0.16 |
| MLP | 0.7201 ± 0.0294 | 0.6910 ± 0.0349 | 0.7125 ± 0.0334 | 1.0230 ± 0.0844 | 0.92 ± 0.13 | 0.00 ± 0.00 |
| KNN | 0.6378 ± 0.0255 | 0.5967 ± 0.0326 | 0.6220 ± 0.0287 | 2.8047 ± 0.2818 | 0.00 ± 0.00 | 0.07 ± 0.13 |

### Error-structure metrics, mean ± std over seeds

| model | GAER | GAER0 | enrichment | super-class acc | clean acc | clean GAER | n_err | max perm p |
|---|---|---|---|---|---|---|---|---|
| TabICL | 0.7430 ± 0.0440 | 0.0845 ± 0.0037 | 8.8028 ± 0.6193 | 0.9564 ± 0.0058 | 0.8700 ± 0.0194 | 0.9968 ± 0.0067 | 77.0 ± 7.1 | 0.0005 |
| RandomForest | 0.8074 ± 0.0312 | 0.0856 ± 0.0027 | 9.4387 ± 0.4307 | 0.9537 ± 0.0067 | 0.7959 ± 0.0298 | 0.9848 ± 0.0076 | 108.7 ± 11.4 | 0.0005 |
| XGBoost | 0.7882 ± 0.0347 | 0.0851 ± 0.0028 | 9.2704 ± 0.5539 | 0.9465 ± 0.0085 | 0.7838 ± 0.0253 | 0.9526 ± 0.0205 | 114.0 ± 9.8 | 0.0005 |
| LightGBM | 0.7968 ± 0.0256 | 0.0848 ± 0.0021 | 9.4034 ± 0.5170 | 0.9476 ± 0.0094 | 0.7782 ± 0.0310 | 0.9565 ± 0.0184 | 115.7 ± 12.1 | 0.0005 |
| LogisticRegression | 0.7883 ± 0.0313 | 0.0852 ± 0.0029 | 9.2621 ± 0.4916 | 0.9448 ± 0.0063 | 0.7762 ± 0.0225 | 0.9434 ± 0.0169 | 117.9 ± 8.5 | 0.0005 |
| HistGB | 0.7984 ± 0.0290 | 0.0852 ± 0.0033 | 9.3910 ± 0.6034 | 0.9463 ± 0.0081 | 0.7671 ± 0.0338 | 0.9542 ± 0.0137 | 120.2 ± 13.4 | 0.0005 |
| MLP | 0.8289 ± 0.0314 | 0.0850 ± 0.0028 | 9.7676 ± 0.5717 | 0.9525 ± 0.0078 | 0.7590 ± 0.0298 | 0.9821 ± 0.0221 | 125.7 ± 11.5 | 0.0005 |
| KNN | 0.8599 ± 0.0251 | 0.0871 ± 0.0027 | 9.8814 ± 0.4873 | 0.9497 ± 0.0061 | 0.6712 ± 0.0263 | 0.9767 ± 0.0117 | 162.7 ± 9.0 | 0.0005 |

### Paired tests on macro-F1 across seeds (TabICL − baseline)

Two-sided Wilcoxon signed-rank (n = 10 seeds; smallest attainable exact p = 0.00195); Holm-adjusted over the 7 comparisons.

| baseline | mean_diff | std_diff | wins | losses | wilcoxon_stat | wilcoxon_p | holm_p |
|---|---|---|---|---|---|---|---|
| HistGB | 0.108 | 0.02638 | 10 | 0 | 0 | 0.001953 | 0.01367 |
| KNN | 0.2184 | 0.02022 | 10 | 0 | 0 | 0.001953 | 0.01367 |
| LightGBM | 0.09457 | 0.02297 | 10 | 0 | 0 | 0.001953 | 0.01367 |
| LogisticRegression | 0.09966 | 0.01886 | 10 | 0 | 0 | 0.001953 | 0.01367 |
| MLP | 0.124 | 0.0394 | 10 | 0 | 0 | 0.001953 | 0.01367 |
| RandomForest | 0.08657 | 0.02304 | 10 | 0 | 0 | 0.001953 | 0.01367 |
| XGBoost | 0.09084 | 0.01741 | 10 | 0 | 0 | 0.001953 | 0.01367 |

### Average rank on macro-F1 (1 = best); Friedman χ² = 50.767, p = 1.02e-08 (n = 10 seeds, 8 models). TabICL ranks first on 10/10 seeds.

| model | avg_rank |
|---|---|
| TabICL | 1.00 |
| RandomForest | 3.00 |
| XGBoost | 3.90 |
| LightGBM | 4.00 |
| LogisticRegression | 4.70 |
| HistGB | 5.40 |
| MLP | 6.00 |
| KNN | 8.00 |

### Per-seed macro-F1

| seed | HistGB | KNN | LightGBM | LogisticRegression | MLP | RandomForest | TabICL | XGBoost |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7031 | 0.5592 | 0.7060 | 0.6989 | 0.6728 | 0.7103 | 0.7930 | 0.7012 |
| 1 | 0.7434 | 0.5851 | 0.7553 | 0.7064 | 0.6901 | 0.7720 | 0.8198 | 0.7425 |
| 2 | 0.7610 | 0.6498 | 0.7702 | 0.7611 | 0.7429 | 0.7793 | 0.8365 | 0.7848 |
| 3 | 0.6501 | 0.5621 | 0.6637 | 0.7083 | 0.6891 | 0.6900 | 0.7893 | 0.6982 |
| 4 | 0.7087 | 0.6048 | 0.7270 | 0.7105 | 0.7044 | 0.7193 | 0.8232 | 0.7191 |
| 5 | 0.7286 | 0.5861 | 0.7511 | 0.7295 | 0.7521 | 0.7539 | 0.8217 | 0.7437 |
| 6 | 0.6412 | 0.6209 | 0.6691 | 0.7080 | 0.6834 | 0.7105 | 0.7991 | 0.6954 |
| 7 | 0.6659 | 0.5532 | 0.6809 | 0.6590 | 0.6600 | 0.6561 | 0.7784 | 0.6718 |
| 8 | 0.7197 | 0.6236 | 0.7348 | 0.7539 | 0.6777 | 0.7443 | 0.8380 | 0.7376 |
| 9 | 0.7491 | 0.6219 | 0.7468 | 0.7185 | 0.6379 | 0.7492 | 0.8516 | 0.7479 |

## E2 — Tuned baselines (seed-42 original protocol)

Protocol: identical data/split/features to `run_benchmark.py` (seed 42; 1296 train / 432 test; d = 64). Hyper-parameter search on the TRAINING split only: 3-fold `StratifiedKFold(shuffle=True, random_state=42)`, `scoring='f1_macro'`, `n_jobs=-1` at the search level (inner estimators `n_jobs=1` during search to avoid oversubscription; refit of the best configuration on the full training split with the original thread settings, which is the timed fit). `RandomizedSearchCV(n_iter=25, random_state=42)` for XGBoost, LightGBM, HistGB, RandomForest, MLP (the MLP space has only 24 points, so it is exhaustively enumerated; HistGB has 108, RF 36); `GridSearchCV` for LogisticRegression (C ∈ {0.01,0.1,1,10,100}) and KNN (k ∈ {3,5,7,11,15,21,31,51} × {uniform, distance}). Search spaces as specified in the revision plan. Note: LightGBM's `subsample` has no effect without `subsample_freq>0` (left at the default 0, as in the fixed setting). TabICL is run with default settings (n_estimators=8, as in the paper) and with n_estimators=1. Predictions: `e2_predictions_seed42.csv`.

| model | best_cv_f1_macro | accuracy | f1_macro | log_loss | GAER | enrichment | superclass_acc | clean_accuracy | fit_time_s | predict_time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| TabICL_n8 | nan | 0.8310 | 0.8094 | 0.6826 | 0.7945 | 9.5568 | 0.9653 | 0.8588 | 0.6537 | 48.4003 |
| TabICL_n1 | nan | 0.8218 | 0.7988 | 0.7280 | 0.8052 | 9.3161 | 0.9653 | 0.8519 | 0.5158 | 6.3116 |
| RandomForest_fixed | nan | 0.7616 | 0.7145 | 1.3676 | 0.8544 | 10.2080 | 0.9653 | 0.7870 | 1.2309 | 0.1089 |
| XGBoost_fixed | nan | 0.7616 | 0.7233 | 0.9659 | 0.8155 | 9.5529 | 0.9560 | 0.7870 | 7.7127 | 0.0291 |
| LightGBM_fixed | nan | 0.7477 | 0.7105 | 1.4083 | 0.8440 | 10.1832 | 0.9606 | 0.7731 | 11.7836 | 0.0978 |
| HistGB_fixed | nan | 0.7407 | 0.7077 | 1.4831 | 0.8125 | 9.5964 | 0.9514 | 0.7639 | 12.0548 | 0.2924 |
| LogisticRegression_fixed | nan | 0.7130 | 0.6824 | 0.9783 | 0.8145 | 9.3280 | 0.9468 | 0.7361 | 0.0705 | 0.0015 |
| KNN_fixed | nan | 0.6435 | 0.5799 | 2.3862 | 0.8831 | 10.1912 | 0.9583 | 0.6644 | 0.0025 | 0.4338 |
| MLP_fixed | nan | 0.7153 | 0.6822 | 0.9595 | 0.8618 | 10.1452 | 0.9606 | 0.7384 | 0.5490 | 0.0029 |
| XGBoost_tuned | 0.7374 | 0.7593 | 0.7226 | 0.9776 | 0.8077 | 9.5529 | 0.9537 | 0.7847 | 11.0569 | 0.0354 |
| LightGBM_tuned | 0.7404 | 0.7477 | 0.7091 | 1.2732 | 0.8532 | 10.4131 | 0.9630 | 0.7731 | 22.2621 | 0.1490 |
| HistGB_tuned | 0.7261 | 0.7639 | 0.7353 | 1.4797 | 0.8137 | 9.5516 | 0.9560 | 0.7870 | 13.6226 | 0.2604 |
| RandomForest_tuned | 0.7308 | 0.7755 | 0.7368 | 1.3334 | 0.8351 | 10.1250 | 0.9630 | 0.8009 | 3.6907 | 0.2163 |
| MLP_tuned | 0.7365 | 0.7176 | 0.6855 | 0.9575 | 0.8607 | 10.0828 | 0.9606 | 0.7431 | 0.1966 | 0.0023 |
| LogisticRegression_tuned | 0.7490 | 0.7130 | 0.6824 | 0.9783 | 0.8145 | 9.3280 | 0.9468 | 0.7361 | 0.0618 | 0.0031 |
| KNN_tuned | 0.6532 | 0.6435 | 0.5799 | 2.3862 | 0.8831 | 10.1912 | 0.9583 | 0.6644 | 0.0021 | 0.0256 |

Best hyper-parameters (and search wall-clock):

- **XGBoost**: `{"subsample": 0.8, "n_estimators": 800, "min_child_weight": 1, "max_depth": 8, "learning_rate": 0.1, "colsample_bytree": 0.8}` — 25 candidates × 3 folds, 167 s
- **LightGBM**: `{"subsample": 0.8, "num_leaves": 127, "n_estimators": 800, "min_child_samples": 10, "learning_rate": 0.03, "colsample_bytree": 0.5}` — 25 candidates × 3 folds, 354 s
- **HistGB**: `{"max_leaf_nodes": 15, "max_iter": 400, "learning_rate": 0.05, "l2_regularization": 0.0}` — 25 candidates × 3 folds, 384 s
- **RandomForest**: `{"n_estimators": 800, "min_samples_leaf": 1, "max_features": 0.3}` — 25 candidates × 3 folds, 184 s
- **MLP**: `{"learning_rate_init": 0.003, "hidden_layer_sizes": [128], "alpha": 0.001}` — 24 candidates × 3 folds, 20 s
- **LogisticRegression**: `{"C": 10}` — 5 candidates × 3 folds, 0 s
- **KNN**: `{"n_neighbors": 15, "weights": "distance"}` — 16 candidates × 3 folds, 1 s

### TabICL (n_estimators=8) vs each TUNED baseline on the seed-42 test set (n = 432)

McNemar exact test = two-sided binomial test on discordant pairs. Paired bootstrap: 10,000 resamples of test rows (seed 0), 95% percentile CI of macro-F1(TabICL) − macro-F1(baseline).

| baseline | tabicl_right_baseline_wrong | tabicl_wrong_baseline_right | mcnemar_exact_p | f1_macro_diff | boot_ci_low | boot_ci_high | boot_frac_le0 |
|---|---|---|---|---|---|---|---|
| XGBoost_tuned | 42 | 11 | 2.248e-05 | 0.08679 | 0.04875 | 0.129 | 0 |
| LightGBM_tuned | 47 | 11 | 2.032e-06 | 0.1003 | 0.06189 | 0.1423 | 0 |
| HistGB_tuned | 43 | 14 | 0.0001539 | 0.07406 | 0.03594 | 0.1156 | 0 |
| RandomForest_tuned | 33 | 9 | 0.0002715 | 0.07252 | 0.03844 | 0.1098 | 0 |
| MLP_tuned | 58 | 9 | 6.811e-10 | 0.1238 | 0.08486 | 0.1677 | 0 |
| LogisticRegression_tuned | 62 | 11 | 9.089e-10 | 0.127 | 0.08668 | 0.1717 | 0 |
| KNN_tuned | 93 | 12 | 1.101e-16 | 0.2295 | 0.18 | 0.28 | 0 |

## E3 — Sensitivity to the LSA dimension d

Protocol: seed-42 protocol with `transform_texts(n_components=d)` for d ∈ {16, 32, 64, 128, 256} (guard d_eff = min(d, |TF-IDF vocabulary| − 1)); TabICL (n_estimators=8), XGBoost, LogisticRegression, RandomForest with the fixed settings of `build_models(42, 30)`.

TF-IDF vocabulary on train: 2630; effective d: [16, 32, 64, 128, 256].

**accuracy**

| d_requested | LogisticRegression | RandomForest | TabICL | XGBoost |
|---|---|---|---|---|
| 16 | 0.3958 | 0.4375 | 0.4583 | 0.3796 |
| 32 | 0.4537 | 0.4468 | 0.5116 | 0.4259 |
| 64 | 0.7130 | 0.7616 | 0.8310 | 0.7616 |
| 128 | 0.8681 | 0.8542 | 0.8843 | 0.8542 |
| 256 | 0.8935 | 0.8611 | 0.8912 | 0.8565 |

**f1_macro**

| d_requested | LogisticRegression | RandomForest | TabICL | XGBoost |
|---|---|---|---|---|
| 16 | 0.3330 | 0.3882 | 0.3917 | 0.3488 |
| 32 | 0.4267 | 0.3802 | 0.4680 | 0.3940 |
| 64 | 0.6824 | 0.7145 | 0.8094 | 0.7233 |
| 128 | 0.8575 | 0.8388 | 0.8784 | 0.8436 |
| 256 | 0.8865 | 0.8493 | 0.8869 | 0.8446 |

**GAER**

| d_requested | LogisticRegression | RandomForest | TabICL | XGBoost |
|---|---|---|---|---|
| 16 | 0.8851 | 0.9218 | 0.9316 | 0.9067 |
| 32 | 0.9153 | 0.9372 | 0.9289 | 0.9234 |
| 64 | 0.8145 | 0.8544 | 0.7945 | 0.8155 |
| 128 | 0.6667 | 0.7460 | 0.7000 | 0.7302 |
| 256 | 0.6739 | 0.7333 | 0.6809 | 0.6774 |

**fit + predict time (s)**

| d_requested | LogisticRegression | RandomForest | TabICL | XGBoost |
|---|---|---|---|---|
| 16 | 0.06 | 1.01 | 23.60 | 4.86 |
| 32 | 0.06 | 1.05 | 28.21 | 6.04 |
| 64 | 0.06 | 1.27 | 40.88 | 7.70 |
| 128 | 0.15 | 1.82 | 74.71 | 14.26 |
| 256 | 0.08 | 2.18 | 127.48 | 21.84 |

d = 256 note: with TabICL's default `batch_size=8` the d=256 run exhausted the 15 GB RAM and thrashed (killed after ~16 min); the d=256 row was re-run in `e3b_lsa_dim256.py` with TabICL `batch_size=1` (ensemble members processed sequentially; same 8 members). Control at d=64 with `batch_size=1`: accuracy 0.8310 / macro-F1 0.8094 vs 0.8310 / 0.8094 with `batch_size=8` (predict time 43.8 s vs 40.3 s).

## E4 — Real-data check: 20 Newsgroups with a hypothesised taxonomy

Protocol: `fetch_20newsgroups(subset='train'/'test', remove=('headers','footers','quotes'))`; empty documents (after stripping) dropped; stratified subsample of 2,400 train and 1,000 test documents (`train_test_split(train_size=..., stratify=label, random_state=42)` on the official train and test subsets respectively); `transform_texts(n_components=64, seed=42)`; the 7 models of `build_models(42, 20)` (fixed settings) + MLP. Taxonomy fixed before any model was run: comp (5), rec (4), sci (4), politics (3), religion (3), misc = {misc.forsale} (singleton, contributes 0 to GAER0). Permutation null: 2000 size-preserving reassignments. Predictions: `e4_predictions.csv`.

Data status: `{"status": "ok", "train_full": 11314, "train_nonempty": 11014, "test_full": 7532, "test_nonempty": 7317, "n_train": 2400, "n_test": 1000, "n_classes": 20}`

| model | accuracy | f1_macro | log_loss | GAER | GAER0 | enrichment | perm_null_mean | perm_null_p95 | perm_null_max | perm_p | superclass_acc | n_err | fit_time_s | predict_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TabICL | 0.4880 | 0.4767 | 1.6033 | 0.4746 | 0.1515 | 3.1323 | 0.1467 | 0.2168 | 0.3086 | 0.0005 | 0.7310 | 512 | 0.7191 | 92.7979 |
| MLP | 0.4940 | 0.4749 | 1.7148 | 0.4704 | 0.1526 | 3.0825 | 0.1460 | 0.2233 | 0.3182 | 0.0005 | 0.7320 | 506 | 0.5916 | 0.0108 |
| LogisticRegression | 0.4810 | 0.4712 | 1.6519 | 0.4701 | 0.1507 | 3.1198 | 0.1460 | 0.2119 | 0.3044 | 0.0005 | 0.7250 | 519 | 0.1670 | 0.0025 |
| XGBoost | 0.4660 | 0.4585 | 1.8957 | 0.4382 | 0.1515 | 2.8926 | 0.1464 | 0.2060 | 0.2828 | 0.0005 | 0.7000 | 534 | 15.9168 | 0.0561 |
| HistGB | 0.4630 | 0.4529 | 2.6182 | 0.4320 | 0.1527 | 2.8293 | 0.1465 | 0.2086 | 0.2849 | 0.0005 | 0.6950 | 537 | 13.4618 | 0.3266 |
| LightGBM | 0.4600 | 0.4510 | 2.4092 | 0.4185 | 0.1519 | 2.7543 | 0.1466 | 0.2074 | 0.2778 | 0.0005 | 0.6860 | 540 | 12.8653 | 0.2009 |
| RandomForest | 0.4420 | 0.4219 | 2.0175 | 0.4283 | 0.1544 | 2.7740 | 0.1469 | 0.2061 | 0.2760 | 0.0005 | 0.6810 | 558 | 2.2465 | 0.1703 |
| KNN | 0.4110 | 0.3986 | 6.4804 | 0.3888 | 0.1551 | 2.5063 | 0.1459 | 0.2020 | 0.3090 | 0.0005 | 0.6400 | 589 | 0.0107 | 0.8585 |

### 20NG: TabICL vs each baseline (n = 1000 test documents)

Exact McNemar and paired bootstrap (10,000 resamples, seed 0) of the macro-F1 difference, as in E2.

| baseline | tabicl_right_baseline_wrong | tabicl_wrong_baseline_right | mcnemar_exact_p | f1_macro_diff | boot_ci_low | boot_ci_high |
|---|---|---|---|---|---|---|
| RandomForest | 116 | 70 | 0.0009181 | 0.05474 | 0.02826 | 0.08127 |
| XGBoost | 104 | 82 | 0.1234 | 0.01819 | -0.008408 | 0.04556 |
| LightGBM | 103 | 75 | 0.04269 | 0.02571 | -0.001142 | 0.05314 |
| HistGB | 102 | 77 | 0.07254 | 0.02381 | -0.003 | 0.05153 |
| LogisticRegression | 69 | 62 | 0.6003 | 0.005484 | -0.01827 | 0.02921 |
| KNN | 134 | 57 | 2.494e-08 | 0.07809 | 0.05077 | 0.1056 |
| MLP | 74 | 80 | 0.6871 | 0.001752 | -0.02392 | 0.0266 |

## Execution notes and failures

- MLP: sklearn 1.6.1 `MLPClassifier(early_stopping=True)` crashes on string labels (`TypeError: ufunc 'isnan'` in the validation scorer). The first launch was aborted; the MLP is wrapped in the repository's `LabelEncodedClassifier` (integer-encodes y only; hyper-parameters exposed as `base__*`).
- After that fix, a stale copy of the first runner kept running concurrently for a few minutes; all its outputs were deleted and the whole sequence was re-run from scratch on an otherwise idle machine, so every reported timing comes from a non-contended run.
- E3 d=256: out-of-memory thrashing with TabICL `batch_size=8`; re-run with `batch_size=1` (see E3).
- E4 (20NG): the TabICL predict time was measured with high memory pressure (<1 GB free RAM), so it may be inflated.
- Seed-42 reproduction check: E2 fixed-setting and TabICL numbers match `outputs/benchmark_summary.csv` exactly (accuracy / macro-F1 / log-loss identical to 4+ decimals).

Lines flagged in the log:

- `===== END e3_lsa_dim exit=1 Sat Oct 10 23:42:24 JDT 2026 =====`
