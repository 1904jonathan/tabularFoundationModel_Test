"""
Benchmark: TabICL vs XGBoost / RandomForest / LightGBM / HistGB / LogReg / KNN
on a *hard* synthetic TEXT classification task with 30 categories.

Pipeline shared by all models:
    raw text  ->  TF-IDF + TruncatedSVD (numeric features)  ->  model

For EACH model we compute: accuracy, F1-macro, F1-weighted, log-loss,
fit time and predict time, then we save:
    - predictions -> outputs/predictions/<model>.csv
    - confusion matrix (plot) -> outputs/plots/confusion_<model>.png
And as a global COMPARISON:
    - outputs/benchmark_summary.csv
    - outputs/plots/comparison_metrics.png   (accuracy / F1)
    - outputs/plots/comparison_timing.png    (fit/predict time)
    - outputs/plots/metrics_heatmap.png      (visual table)
    - outputs/plots/learning_curve_*.png     (learning curves)
    - outputs/plots/roc_curves.png           (ROC, micro-average OvR)
    - outputs/plots/per_class_f1.png         (per-class F1)

Run:  python run_benchmark.py
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (save to files)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

from tfm_bench.data_synth import build_dataset, CATEGORIES
from tfm_bench.text_features import transform_texts
from tfm_bench.models import build_models
from tfm_bench.sci_style import set_scientific_style
from tfm_bench.curves import (
    compute_learning_curves,
    plot_learning_curves,
    plot_roc_curves,
    plot_per_class_f1,
)

warnings.filterwarnings("ignore", category=UserWarning)
set_scientific_style()  # global scientific plotting style

SEED = 42
OUT = "outputs"
PLOTS = os.path.join(OUT, "plots")
PREDS = os.path.join(OUT, "predictions")
for d in (OUT, PLOTS, PREDS):
    os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1) Data + features (shared by all models)
# --------------------------------------------------------------------------- #
def prepare_data():
    df = build_dataset(n_per_class=60, label_noise=0.05, imbalance=True, seed=SEED)

    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=SEED, stratify=df["category"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # The test set is used WITHOUT labels to predict; the true label is for scoring.
    X_train, X_test, _ = transform_texts(
        train_df["text"], test_df["text"], n_components=64, seed=SEED
    )
    y_train = train_df["category"].to_numpy()
    y_test = test_df["category"].to_numpy()

    print(f"Train: {len(X_train)} | Test: {len(X_test)} | "
          f"Features: {X_train.shape[1]} | Classes: {len(CATEGORIES)}")
    return X_train, y_train, X_test, y_test, test_df["text"].to_numpy()


# --------------------------------------------------------------------------- #
# 2) Train + evaluate one model
# --------------------------------------------------------------------------- #
def evaluate_model(name, model, X_train, y_train, X_test, y_test, test_text):
    print(f"\n>>> {name}")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    # Probabilities (for log-loss), aligned with the model's class order.
    proba_ordered = None  # reordered to CATEGORIES, for ROC curves
    try:
        proba = model.predict_proba(X_test)
        ll = log_loss(y_test, proba, labels=model.classes_)
        confidence = proba.max(axis=1)
        # Reorder proba columns to the canonical CATEGORIES order.
        col = {c: i for i, c in enumerate(model.classes_)}
        proba_ordered = proba[:, [col[c] for c in CATEGORIES]]
    except Exception:
        ll = np.nan
        confidence = np.full(len(y_pred), np.nan)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "log_loss": ll,
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }
    print(f"    accuracy={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}  "
          f"log_loss={ll:.3f}  fit={fit_time:.1f}s  predict={predict_time:.1f}s")

    # Save this model's predictions.
    pd.DataFrame({
        "text": test_text,
        "true_category": y_test,
        "predicted_category": y_pred,
        "confidence": confidence,
        "correct": (y_pred == y_test),
    }).to_csv(os.path.join(PREDS, f"{name}.csv"), index=False, encoding="utf-8-sig")

    # Confusion matrix (per-model plot).
    plot_confusion(name, y_test, y_pred)

    return metrics, y_pred, proba_ordered


# --------------------------------------------------------------------------- #
# 3) Plots
# --------------------------------------------------------------------------- #
def plot_confusion(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm, annot=False, cmap="viridis",
        xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax, cbar=True,
    )
    ax.set_title(f"Confusion matrix -- {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"confusion_{name}.png"), dpi=130)
    plt.close(fig)


def plot_comparison_metrics(summary: pd.DataFrame):
    df = summary.sort_values("f1_macro", ascending=False)
    x = np.arange(len(df))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, df["accuracy"], w, label="Accuracy", color="#4C72B0")
    ax.bar(x + w / 2, df["f1_macro"], w, label="F1-macro", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison -- Accuracy & F1-macro (30-class test set)")
    for i, (a, f) in enumerate(zip(df["accuracy"], df["f1_macro"])):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "comparison_metrics.png"), dpi=130)
    plt.close(fig)


def plot_comparison_timing(summary: pd.DataFrame):
    df = summary.sort_values("fit_time_s", ascending=True)
    x = np.arange(len(df))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, df["fit_time_s"], w, label="Fit (s)", color="#55A868")
    ax.bar(x + w / 2, df["predict_time_s"], w, label="Predict (s)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_yscale("log")
    ax.set_title("Model comparison -- Compute time (log scale)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "comparison_timing.png"), dpi=130)
    plt.close(fig)


def plot_metrics_heatmap(summary: pd.DataFrame):
    cols = ["accuracy", "f1_macro", "f1_weighted", "log_loss"]
    df = summary.set_index("model")[cols].sort_values("f1_macro", ascending=False)

    # Color = per-column normalized "quality" (green = best), annotation = raw value.
    # log_loss: lower = better -> invert for color.
    color = df.copy()
    for c in cols:
        col = df[c]
        rng = col.max() - col.min()
        norm = (col - col.min()) / rng if rng > 0 else col * 0 + 0.5
        color[c] = (1 - norm) if c == "log_loss" else norm

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 2))
    sns.heatmap(color, annot=df.values, fmt=".3f", cmap="RdYlGn",
                cbar=False, ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_title("Metrics summary table (green = best)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "metrics_heatmap.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4) Orchestration
# --------------------------------------------------------------------------- #
def main():
    X_train, y_train, X_test, y_test, test_text = prepare_data()
    models = build_models(seed=SEED, n_classes=len(CATEGORIES))

    rows, preds, probas = [], {}, {}
    for name, model in models.items():
        try:
            metrics, y_pred, proba = evaluate_model(
                name, model, X_train, y_train, X_test, y_test, test_text
            )
            rows.append(metrics)
            preds[name] = y_pred
            if proba is not None:
                probas[name] = proba
        except Exception as e:
            print(f"    [FAILED] {name}: {e}")

    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    summary.to_csv(os.path.join(OUT, "benchmark_summary.csv"), index=False, encoding="utf-8-sig")

    # Comparison plots (bars + heatmap).
    plot_comparison_metrics(summary)
    plot_comparison_timing(summary)
    plot_metrics_heatmap(summary)

    # CURVE plots (scientific style).
    print("\nGenerating ROC curves and per-class F1...")
    plot_roc_curves(probas, y_test, CATEGORIES, PLOTS)
    plot_per_class_f1(preds, y_test, CATEGORIES, PLOTS)

    print("Computing learning curves (retraining at several sizes)...")
    lc = compute_learning_curves(
        build_models, X_train, y_train, X_test, y_test,
        n_classes=len(CATEGORIES), seed=SEED,
    )
    lc.to_csv(os.path.join(OUT, "learning_curves.csv"), index=False, encoding="utf-8-sig")
    plot_learning_curves(lc, PLOTS, metric="accuracy")
    plot_learning_curves(lc, PLOTS, metric="f1_macro")

    print("\n" + "=" * 70)
    print("FINAL RANKING (sorted by F1-macro)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nGenerated files:")
    print(f"  - {os.path.join(OUT, 'benchmark_summary.csv')} / learning_curves.csv")
    print(f"  - {PREDS}\\<model>.csv        (per-model predictions)")
    print(f"  - {PLOTS}\\confusion_<model>.png")
    print(f"  - {PLOTS}\\comparison_metrics.png / comparison_timing.png / metrics_heatmap.png")
    print(f"  - {PLOTS}\\roc_curves.png / per_class_f1.png")
    print(f"  - {PLOTS}\\learning_curve_accuracy.png / learning_curve_f1_macro.png")


if __name__ == "__main__":
    main()
