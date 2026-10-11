# Tabular Foundation Model — TabICL vs classical models (text → 30 classes)

A complete, reproducible example around **[TabICL](https://github.com/soda-inria/tabicl)**
(the *tabular foundation model* from soda-inria, ICML 2025/2026) applied to a
**30-class text classification** task, compared against the usual baselines
(XGBoost, LightGBM, Random Forest, HistGradientBoosting, logistic regression, KNN).

The raw *features* are **text**; they are converted to numbers with a clean, shared
method (**TF-IDF → TruncatedSVD/LSA**), then given identically to every model → a fair
comparison.

---

## 1. Idea

| Step | Choice |
|------|--------|
| Data | **Synthetic** hard text: shared vocabulary, *confusable* categories, noise, label noise, imbalance |
| Target | 30 thematic categories (football, astronomy, medicine, robotics…) |
| Text → numbers | `TfidfVectorizer` (1-2 grams) → `TruncatedSVD` (64 dims) → `Normalizer` |
| Data leakage | Vectorizer fitted **on the train only**, applied as-is to the test set |
| Flagship model | `TabICLClassifier` (in-context learning, `support_many_classes=True`) |
| Comparison | XGBoost, LightGBM, RandomForest, HistGB, LogisticRegression, KNN |

> TabICL natively handles more than 10 classes (`support_many_classes=True`), so 30
> categories are not a problem. `fit` + `predict` happen in a single forward pass.

---

## 2. Installation

The project uses the **Anaconda base** environment.

```bash
pip install tabicl            # pulls PyTorch + the checkpoint (downloaded on first use)
pip install xgboost lightgbm  # comparison models
# scikit-learn, matplotlib, seaborn, pandas, numpy are already provided by Anaconda
```

> On the first `fit`, TabICL automatically downloads its pre-trained checkpoint
> (`tabicl-classifier-v2-20260212.ckpt`) from the Hugging Face Hub (`jingang/TabICL`).

---

## 3. Usage

```bash
# Minimal end-to-end example (generation → TF-IDF/SVD → TabICL → predictions)
python tabicl_text_classification_example.py

# Full benchmark: all models + stats + scientific figures
python run_benchmark.py
```

---

## 4. Project structure

```
tabicl_text_classification_example.py   # standalone TabICL example (quick to read)
run_benchmark.py                         # full benchmark orchestrator
tfm_bench/
├── data_synth.py      # hard synthetic text generation (30 classes)
├── text_features.py   # text → numbers conversion (TF-IDF + SVD)
├── models.py          # model registry (+ label-encoding wrapper for XGBoost)
├── sci_style.py       # scientific matplotlib style + per-model colors
└── curves.py          # learning curves, ROC curves, per-class F1
outputs/
├── benchmark_summary.csv      # all metrics per model
├── learning_curves.csv        # scores as a function of training size
├── predictions/<model>.csv    # detailed per-model predictions
└── plots/                     # all figures (PNG, 200 dpi)
```

---

## 5. Hard synthetic data

To avoid a trivial task (the first version reached 100% accuracy), the generator
([tfm_bench/data_synth.py](tfm_bench/data_synth.py)) deliberately introduces ambiguity:

- **Shared vocabulary**: a pool of words common to every class (noise).
- **Confusable categories**: close groups (e.g. *football/basketball*,
  *physics/chemistry/biology/astronomy*) borrow vocabulary from each other.
- **Noisy sentences**: few discriminative words, lots of filler.
- **Label noise**: ~5% of training labels are corrupted.
- **Imbalance**: ±40% samples depending on the class.

---

## 6. Text → numbers conversion (method)

```text
raw text → TfidfVectorizer(min_df=2, ngram_range=(1,2), sublinear_tf=True)
         → TruncatedSVD(n_components=64)   (LSA: dimensionality reduction)
         → Normalizer()                    (unit-norm vectors)
```

This dense 64-dimensional representation is given **identically** to every model,
which isolates the effect of the classifier from the choice of features.

---

## 7. Results

Metrics on the test set (432 samples, 30 classes), sorted by F1-macro:

| Model | Accuracy | F1-macro | F1-weighted | Log-loss |
|-------|:--------:|:--------:|:-----------:|:--------:|
| **TabICL** | **0.831** | **0.809** | **0.823** | **0.683** |
| XGBoost | 0.762 | 0.723 | 0.748 | 0.966 |
| RandomForest | 0.762 | 0.715 | 0.742 | 1.368 |
| LightGBM | 0.748 | 0.710 | 0.736 | 1.408 |
| HistGB | 0.741 | 0.708 | 0.732 | 1.483 |
| LogisticRegression | 0.713 | 0.682 | 0.706 | 0.978 |
| KNN | 0.644 | 0.580 | 0.615 | 2.386 |

> *(reproducible with `random_state=42`; see `outputs/benchmark_summary.csv`.)*

**Reading:** TabICL clearly dominates (≈ +9 F1-macro points over the best
challenger) **without any hyperparameter tuning**, and achieves the best log-loss
(best-calibrated probabilities). The learning curve shows that its advantage is
largest at **low data volume**.

### Generated figures (`outputs/plots/`)

| Figure | Description |
|--------|-------------|
| `comparison_metrics.png` | Accuracy & F1-macro bars per model |
| `comparison_timing.png` | Fit/predict time (log scale) |
| `metrics_heatmap.png` | Color metrics table (green = best) |
| `learning_curve_accuracy.png` / `learning_curve_f1_macro.png` | Learning curves |
| `roc_curves.png` | Micro-average one-vs-rest ROC curves (+ AUC) |
| `per_class_f1.png` | Per-category F1, one curve per model |
| `confusion_<model>.png` | 30×30 confusion matrix per model |

---

## 8. Notes

- **Speed trade-off**: TabICL runs inference on CPU here (no GPU). Its `predict` is
  slower than tree models (in-context learning over the whole train set), but stays
  around ten seconds for this dataset. On GPU it would be much faster.
- **Fairness**: every model shares the exact same numeric features.
- **Reproducibility**: `random_state=42` everywhere; outputs are versioned via the
  CSV files in `outputs/` (the PNG figures are ignored by `.gitignore`).

---

## 8b. Revision experiments (Pattern Recognition resubmission)

`outputs/revision/` contains the additional experiments run for the journal revision,
all reproducible from the repository root with `outputs/revision/scripts/run_all.sh`:

- `e1_*` : 10-seed rerun (new corpus, split and encoder per seed) of the 7 models plus an
  MLP baseline, with Wilcoxon / Friedman tests and the GAER taxonomy-permutation null.
- `e2_*` : baselines tuned by 3-fold randomised search on the training split (seed 42),
  plus TabICL with a single ensemble member.
- `e3_*` : sensitivity to the LSA dimension d in {16, 32, 64, 128, 256}.
- `e4_*` : 20 Newsgroups (2,400 train / 1,000 test) with a hypothesised 6-group taxonomy.
- `results.md` : every table, protocol, hardware and package versions; `log.txt` : run log.

Note: the 5 % label noise is applied to the whole corpus **before** the train/test split,
so about 3.5 % of test labels are corrupted (accuracy ceiling ~0.965); see `results.md`.

---

## 9. References

- TabICL — repository: <https://github.com/soda-inria/tabicl>
- TabICL (ICML 2025): <https://arxiv.org/abs/2502.05564>
- TabICLv2 (ICML 2026): <https://arxiv.org/abs/2602.11139>
- TabArena benchmark: <https://tabarena.ai>
