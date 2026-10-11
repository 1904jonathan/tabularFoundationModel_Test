"""Assemble outputs/revision/results.md from the experiment CSVs."""
import glob
import importlib.metadata as md
import json
import os
import platform
import subprocess

import pandas as pd

import revision_utils as u

R = u.REV
L = []
add = L.append


def read(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def pm(df, col):
    return df[col + "_mean"].map("{:.4f}".format) + " ± " + df[col + "_std"].map("{:.4f}".format)


# ---------------- environment ----------------
try:
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True, timeout=60).stdout.strip()
except Exception:  # noqa: BLE001
    cpu = platform.processor()
import torch  # noqa: E402
vers = {p: md.version(p) for p in ["scikit-learn", "xgboost", "lightgbm", "torch", "tabicl", "numpy", "scipy", "pandas"]}
ckpts = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--jingang--TabICL/snapshots/*/*.ckpt"))

add("# Revision experiments (Pattern Recognition resubmission)\n")
add("All scripts: `outputs/revision/scripts/` (shared helper `revision_utils.py`). Run sequentially in one "
    "background process (`run_all.sh`), log in `outputs/revision/log.txt`.\n")
add("## Environment\n")
add(f"- CPU: {cpu}; logical cores: {os.cpu_count()}; torch threads: {torch.get_num_threads()}; CPU only (no GPU)")
add(f"- OS: {platform.platform()}; Python {platform.python_version()}")
add("- Packages: " + ", ".join(f"{k} {v}" for k, v in vers.items()))
add(f"- TabICL checkpoint: {', '.join(ckpts) if ckpts else 'not found'} (HF repo `jingang/TabICL`)")
add("")

add("## Key findings (numbers from the tables below)\n")
add("1. **E1 (10 seeds, d=64)**: TabICL macro-F1 0.815 ± 0.024 vs best baseline RandomForest 0.728 ± 0.038 "
    "(XGBoost 0.724 ± 0.033). TabICL beats every baseline on 10/10 seeds (Wilcoxon exact p = 0.00195 for each, "
    "the minimum attainable with n=10; Holm-adjusted 0.0137); Friedman χ² = 50.8, p = 1.0e-8; average rank 1.0.")
add("2. **E2 (tuning, seed 42)**: tuning barely moves the baselines on the test set (best tuned: RandomForest "
    "0.737, HistGB 0.735, XGBoost 0.723; LR/KNN grid optima equal the fixed settings). TabICL (0.809) beats every "
    "tuned baseline: McNemar p ≤ 2.7e-4, bootstrap 95% CIs of the macro-F1 gap all exclude 0 (smallest: vs tuned RF, "
    "+0.073 [0.038, 0.110]). TabICL with n_estimators=1: macro-F1 0.799 in 6.3 s predict vs 0.809 in 48 s for 8 members.")
add("3. **E3 (LSA dimension)**: d is the dominant factor. d=64 is where TabICL's lead is largest (+0.086 macro-F1 "
    "over XGBoost); at d=128 the lead shrinks (0.878 vs LR 0.858) and at d=256 **LogisticRegression ties TabICL** "
    "(0.8865 vs 0.8869) while every model gains 0.07–0.20 macro-F1 over d=64. The d=64 setting of the paper is "
    "therefore a bottleneck for all models, and the TabICL advantage is largely specific to the low-dimensional regime. "
    "GAER decreases as d grows (more of the residual errors leave the group) but stays ~8–9× chance.")
add("4. **E4 (20 Newsgroups, real data)**: the hypothesised taxonomy is strongly recovered by every model "
    "(GAER 0.39–0.47 vs permutation-null 95th pct ≈ 0.20–0.22, max ≈ 0.28–0.32; p = 1/2001 for all; enrichment "
    "2.5–3.1×). TabICL is top on macro-F1 (0.477) but statistically tied with MLP (0.475) and LogisticRegression "
    "(0.471) (McNemar p = 0.69 / 0.60; bootstrap CIs include 0); it significantly beats RandomForest and KNN.")
add("5. **Clean labels**: ≈5% of test labels are corrupted by `build_dataset`. Against clean labels TabICL accuracy "
    "is 0.870 ± 0.019 and its clean-label GAER is 0.997 ± 0.007, i.e. virtually all of its genuine errors fall "
    "within the confusable group; most of its out-of-group 'errors' are corrupted test labels.")
add("")
add("## Definitions\n")
add("- **GAER** = among misclassified test rows, fraction whose predicted class is in the same (hypothesised) group "
    "as the true class. **GAER0** (chance) = mean over errors of (|G_g(y)|−1)/(K−1). **Enrichment** = GAER/GAER0.")
add("- **Permutation null**: 2000 random reassignments of classes to groups preserving the multiset of group sizes "
    "(seed 0); reported: null mean, 95th percentile, max, p = (1+#{null ≥ obs})/(B+1). With B=2000 the minimum p is 1/2001 ≈ 5.0e-4.")
add("- **Super-class accuracy** = mean(group(ŷ) == group(y)).")
add("- **Clean-label metrics** (synthetic only): the dataset is rebuilt with `label_noise=0.0` and the same seed; "
    "the text column was verified identical (row by row) for every seed used, so the uncorrupted label of each test "
    "row is recovered. Note: label noise in `build_dataset` is applied to the whole dataset before splitting, so "
    "≈5% of TEST labels are also corrupted; clean-label metrics score predictions against the uncorrupted labels.")
add("")

# ---------------- E1 ----------------
add("## E1 — Multi-seed rerun (main result)\n")
add("Protocol: for seed s ∈ {0..9}: `build_dataset(n_per_class=60, label_noise=0.05, imbalance=True, seed=s)` → "
    "`train_test_split(test_size=0.25, random_state=s, stratify=category)` → `transform_texts(n_components=64, seed=s)` "
    "→ the 7 models of `build_models(s, 30)` (fixed settings, `random_state=s`) + MLP "
    "(`MLPClassifier(hidden_layer_sizes=(256,128), early_stopping=True, max_iter=500, random_state=s)`). "
    "Dataset size varies with the seed (imbalance draw). Per-seed predictions: `predictions_seed{s}/predictions.csv`.\n")
s = read("e1_multiseed_summary.csv")
raw = read("e1_multiseed_raw.csv")
if s is not None:
    t = pd.DataFrame({"model": s["model"], "accuracy": pm(s, "accuracy"), "macro-F1": pm(s, "f1_macro"),
                      "weighted-F1": pm(s, "f1_weighted"), "log-loss": pm(s, "log_loss"),
                      "fit s": pm(s, "fit_time_s").str.replace(r"(\d+\.\d{2})\d+", r"\1", regex=True),
                      "predict s": pm(s, "predict_time_s").str.replace(r"(\d+\.\d{2})\d+", r"\1", regex=True)})
    add(f"### Predictive metrics, mean ± std over {raw['seed'].nunique()} seeds\n")
    add(u.md_table(t)); add("")
    t2 = pd.DataFrame({"model": s["model"], "GAER": pm(s, "GAER"), "GAER0": pm(s, "GAER0"),
                       "enrichment": pm(s, "enrichment"), "super-class acc": pm(s, "superclass_acc"),
                       "clean acc": pm(s, "clean_accuracy"), "clean GAER": pm(s, "clean_GAER"),
                       "n_err": pm(s, "n_err").str.replace(r"(\d+\.\d)\d+", r"\1", regex=True),
                       "max perm p": raw.groupby("model")["perm_p"].max().reindex(s["model"]).map("{:.2g}".format).values})
    add("### Error-structure metrics, mean ± std over seeds\n")
    add(u.md_table(t2)); add("")
    w = read("e1_wilcoxon.csv"); rk = read("e1_ranks_friedman.csv")
    if w is not None:
        add("### Paired tests on macro-F1 across seeds (TabICL − baseline)\n")
        add("Two-sided Wilcoxon signed-rank (n = 10 seeds; smallest attainable exact p = 0.00195); Holm-adjusted over the 7 comparisons.\n")
        add(u.md_table(w, ".4g")); add("")
    if rk is not None:
        rk = rk.rename(columns={rk.columns[0]: "model"})
        add(f"### Average rank on macro-F1 (1 = best); Friedman χ² = {rk['friedman_chi2'].iloc[0]:.3f}, "
            f"p = {rk['friedman_p'].iloc[0]:.3g} (n = {int(rk['n_seeds'].iloc[0])} seeds, {len(rk)} models). "
            f"TabICL ranks first on {int(rk['tabicl_rank1_count'].iloc[0])}/{int(rk['n_seeds'].iloc[0])} seeds.\n")
        add(u.md_table(rk[["model", "avg_rank"]], ".2f")); add("")
    piv = raw.pivot(index="seed", columns="model", values="f1_macro")
    add("### Per-seed macro-F1\n")
    add(u.md_table(piv.reset_index(), ".4f")); add("")
else:
    add("E1 did not produce results (see log.txt).\n")

# ---------------- E2 ----------------
add("## E2 — Tuned baselines (seed-42 original protocol)\n")
add("Protocol: identical data/split/features to `run_benchmark.py` (seed 42; 1296 train / 432 test; d = 64). "
    "Hyper-parameter search on the TRAINING split only: 3-fold `StratifiedKFold(shuffle=True, random_state=42)`, "
    "`scoring='f1_macro'`, `n_jobs=-1` at the search level (inner estimators `n_jobs=1` during search to avoid "
    "oversubscription; refit of the best configuration on the full training split with the original thread settings, "
    "which is the timed fit). `RandomizedSearchCV(n_iter=25, random_state=42)` for XGBoost, LightGBM, HistGB, "
    "RandomForest, MLP (the MLP space has only 24 points, so it is exhaustively enumerated; HistGB has 108, RF 36); "
    "`GridSearchCV` for LogisticRegression (C ∈ {0.01,0.1,1,10,100}) and KNN (k ∈ {3,5,7,11,15,21,31,51} × "
    "{uniform, distance}). Search spaces as specified in the revision plan. Note: LightGBM's `subsample` has no effect "
    "without `subsample_freq>0` (left at the default 0, as in the fixed setting). TabICL is run with default settings "
    "(n_estimators=8, as in the paper) and with n_estimators=1. Predictions: `e2_predictions_seed42.csv`.\n")
e2 = read("e2_tuned.csv")
if e2 is not None:
    cols = ["model", "best_cv_f1_macro", "accuracy", "f1_macro", "log_loss", "GAER", "enrichment", "superclass_acc",
            "clean_accuracy", "fit_time_s", "predict_time_s"]
    for c in cols:
        if c not in e2:
            e2[c] = float("nan")
    add(u.md_table(e2[cols], ".4f")); add("")
    if "best_params" in e2:
        add("Best hyper-parameters (and search wall-clock):\n")
        for _, r in e2[e2["setting"] == "tuned"].iterrows():
            add(f"- **{r['base_model']}**: `{r['best_params']}` — {int(r['n_candidates'])} candidates × 3 folds, "
                f"{r['search_time_s']:.0f} s")
        add("")
    st = read("e2_tests.csv")
    if st is not None:
        add("### TabICL (n_estimators=8) vs each TUNED baseline on the seed-42 test set (n = 432)\n")
        add("McNemar exact test = two-sided binomial test on discordant pairs. Paired bootstrap: 10,000 resamples of "
            "test rows (seed 0), 95% percentile CI of macro-F1(TabICL) − macro-F1(baseline).\n")
        add(u.md_table(st, ".4g")); add("")

# ---------------- E3 ----------------
add("## E3 — Sensitivity to the LSA dimension d\n")
add("Protocol: seed-42 protocol with `transform_texts(n_components=d)` for d ∈ {16, 32, 64, 128, 256} "
    "(guard d_eff = min(d, |TF-IDF vocabulary| − 1)); TabICL (n_estimators=8), XGBoost, LogisticRegression, "
    "RandomForest with the fixed settings of `build_models(42, 30)`.\n")
e3 = read("e3_lsa_dim.csv")
if e3 is not None:
    e3["d_requested"] = e3["d_requested"].astype(int); e3["d_effective"] = e3["d_effective"].astype(int)
    add(f"TF-IDF vocabulary on train: {int(e3['tfidf_vocab'].iloc[0])}; effective d: "
        f"{sorted(set(e3['d_effective'].astype(int)))}.\n")
    for met in ["accuracy", "f1_macro", "GAER"]:
        p = e3.pivot(index="d_requested", columns="model", values=met).reset_index()
        add(f"**{met}**\n"); add(u.md_table(p, ".4f")); add("")
    e3 = e3.assign(total_time_s=e3["fit_time_s"] + e3["predict_time_s"])
    p = e3.pivot(index="d_requested", columns="model", values="total_time_s").reset_index()
    add("**fit + predict time (s)**\n"); add(u.md_table(p, ".2f")); add("")
    ctl = read("e3b_batchsize_control.csv")
    if ctl is not None:
        ref = e3[(e3["model"] == "TabICL") & (e3["d_requested"] == 64)].iloc[0]
        c = ctl.iloc[0]
        add(f"d = 256 note: with TabICL's default `batch_size=8` the d=256 run exhausted the 15 GB RAM and thrashed "
            f"(killed after ~16 min); the d=256 row was re-run in `e3b_lsa_dim256.py` with TabICL `batch_size=1` "
            f"(ensemble members processed sequentially; same 8 members). Control at d=64 with `batch_size=1`: "
            f"accuracy {c['accuracy']:.4f} / macro-F1 {c['f1_macro']:.4f} vs {ref['accuracy']:.4f} / "
            f"{ref['f1_macro']:.4f} with `batch_size=8` (predict time {c['predict_time_s']:.1f} s vs "
            f"{ref['predict_time_s']:.1f} s).\n")

# ---------------- E4 ----------------
add("## E4 — Real-data check: 20 Newsgroups with a hypothesised taxonomy\n")
stat = os.path.join(R, "e4_status.json")
info = json.load(open(stat)) if os.path.exists(stat) else {"status": "not run"}
add("Protocol: `fetch_20newsgroups(subset='train'/'test', remove=('headers','footers','quotes'))`; empty documents "
    "(after stripping) dropped; stratified subsample of 2,400 train and 1,000 test documents "
    "(`train_test_split(train_size=..., stratify=label, random_state=42)` on the official train and test subsets "
    "respectively); `transform_texts(n_components=64, seed=42)`; the 7 models of `build_models(42, 20)` (fixed settings) "
    "+ MLP. Taxonomy fixed before any model was run: comp (5), rec (4), sci (4), politics (3), religion (3), "
    "misc = {misc.forsale} (singleton, contributes 0 to GAER0). Permutation null: 2000 size-preserving reassignments. "
    "Predictions: `e4_predictions.csv`.\n")
add(f"Data status: `{json.dumps(info)}`\n")
e4 = read("e4_20newsgroups.csv")
if e4 is not None:
    cols = ["model", "accuracy", "f1_macro", "log_loss", "GAER", "GAER0", "enrichment", "perm_null_mean",
            "perm_null_p95", "perm_null_max", "perm_p", "superclass_acc", "n_err", "fit_time_s", "predict_time_s"]
    add(u.md_table(e4.sort_values("f1_macro", ascending=False)[cols], ".4f")); add("")
    pr = read("e4_predictions.csv")
    if pr is not None and "pred_TabICL" in pr:
        st = []
        for c in [c for c in pr.columns if c.startswith("pred_") and c != "pred_TabICL"]:
            n01, n10, p = u.mcnemar_exact(pr["true"], pr["pred_TabICL"], pr[c])
            obs, lo, hi, frac = u.paired_bootstrap_f1(pr["true"], pr["pred_TabICL"], pr[c], B=10000, seed=0)
            st.append(dict(baseline=c[5:], tabicl_right_baseline_wrong=n01, tabicl_wrong_baseline_right=n10,
                           mcnemar_exact_p=p, f1_macro_diff=obs, boot_ci_low=lo, boot_ci_high=hi))
        st = pd.DataFrame(st)
        st.to_csv(os.path.join(R, "e4_tests.csv"), index=False)
        add("### 20NG: TabICL vs each baseline (n = 1000 test documents)\n")
        add("Exact McNemar and paired bootstrap (10,000 resamples, seed 0) of the macro-F1 difference, as in E2.\n")
        add(u.md_table(st, ".4g")); add("")

# ---------------- failures ----------------
add("## Execution notes and failures\n")
add("- MLP: sklearn 1.6.1 `MLPClassifier(early_stopping=True)` crashes on string labels (`TypeError: ufunc 'isnan'` "
    "in the validation scorer). The first launch was aborted; the MLP is wrapped in the repository's "
    "`LabelEncodedClassifier` (integer-encodes y only; hyper-parameters exposed as `base__*`).")
add("- After that fix, a stale copy of the first runner kept running concurrently for a few minutes; all its "
    "outputs were deleted and the whole sequence was re-run from scratch on an otherwise idle machine, so every "
    "reported timing comes from a non-contended run.")
add("- E3 d=256: out-of-memory thrashing with TabICL `batch_size=8`; re-run with `batch_size=1` (see E3).")
add("- E4 (20NG): the TabICL predict time was measured with high memory pressure (<1 GB free RAM), so it may be "
    "inflated.")
add("- Seed-42 reproduction check: E2 fixed-setting and TabICL numbers match `outputs/benchmark_summary.csv` "
    "exactly (accuracy / macro-F1 / log-loss identical to 4+ decimals).")
add("")
add("Lines flagged in the log:\n")
logp = os.path.join(R, "log.txt")
bad = []
if os.path.exists(logp):
    for line in open(logp, encoding="utf-8", errors="replace"):
        if any(k in line for k in ("FAILED", "WARNING seed", "exit=1", "exit=2", "download failed", "Error:")):
            bad.append(line.rstrip())
add("\n".join(f"- `{b}`" for b in bad) if bad else "- None.")
add("")

open(os.path.join(R, "results.md"), "w", encoding="utf-8").write("\n".join(L))
print("results.md written")
