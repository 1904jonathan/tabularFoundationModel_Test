"""
Scientific-style CURVE plots to compare the models:

  1) Learning curves     : accuracy / F1-macro as a function of training set size.
                           -> highlights TabICL's data efficiency.
  2) ROC curves          : micro-average one-vs-rest (multiclass), one curve/model.
  3) Per-class F1        : F1 per category (30 points), one curve/model.

Every figure reuses the style and colors from sci_style.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .sci_style import color_for, marker_for


# --------------------------------------------------------------------------- #
# 1) Learning curves
# --------------------------------------------------------------------------- #
def compute_learning_curves(
    build_models_fn, X_train, y_train, X_test, y_test,
    n_classes, fractions=(0.15, 0.35, 0.6, 1.0), seed=42, test_cap=300,
):
    """
    For each training fraction, retrain FRESH models and evaluate on a fixed test
    set (capped at test_cap to limit TabICL's cost).
    Returns a long DataFrame: [model, n_train, accuracy, f1_macro].
    """
    # Fixed test subsample (stratified) to speed things up.
    if len(X_test) > test_cap:
        X_eval, _, y_eval, _ = train_test_split(
            X_test, y_test, train_size=test_cap, stratify=y_test, random_state=seed
        )
    else:
        X_eval, y_eval = X_test, y_test

    rows = []
    for frac in fractions:
        if frac >= 1.0:
            Xs, ys = X_train, y_train
        else:
            Xs, _, ys, _ = train_test_split(
                X_train, y_train, train_size=frac,
                stratify=y_train, random_state=seed,
            )
        n_train = len(Xs)
        models = build_models_fn(seed, n_classes)
        for name, model in models.items():
            try:
                model.fit(Xs, ys)
                pred = model.predict(X_eval)
                rows.append({
                    "model": name,
                    "n_train": n_train,
                    "accuracy": accuracy_score(y_eval, pred),
                    "f1_macro": f1_score(y_eval, pred, average="macro", zero_division=0),
                })
            except Exception as e:
                print(f"    [learning curve] {name}@{n_train} failed: {e}")
        print(f"    learning curve: fraction={frac:.2f} (n_train={n_train}) done")
    return pd.DataFrame(rows)


def plot_learning_curves(lc_df: pd.DataFrame, out_dir: str, metric="accuracy"):
    fig, ax = plt.subplots(figsize=(9, 6))
    # Order the legend by final (largest train size) score, descending.
    order = (lc_df.sort_values("n_train").groupby("model")[metric].last()
             .sort_values(ascending=False).index)
    for name in order:
        d = lc_df[lc_df["model"] == name].sort_values("n_train")
        ax.plot(d["n_train"], d[metric], marker=marker_for(name),
                color=color_for(name), label=name)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel(metric.replace("_", "-").capitalize())
    ax.set_title(f"Learning curves -- {metric.replace('_', '-')} (30-class test set)")
    ax.legend(title="Model", ncols=2)
    fig.savefig(os.path.join(out_dir, f"learning_curve_{metric}.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2) ROC curves (micro-average OvR)
# --------------------------------------------------------------------------- #
def plot_roc_curves(probas: dict, y_test, categories, out_dir: str):
    """
    probas[name] : (n_test, n_classes) matrix REORDERED to match `categories`.
    Plots the micro-average ROC curve (all classes aggregated) for each model.
    """
    Yt = label_binarize(y_test, classes=categories)  # (n_test, n_classes)
    fig, ax = plt.subplots(figsize=(8, 7))

    aucs = {n: auc(*roc_curve(Yt.ravel(), p.ravel())[:2]) for n, p in probas.items()}
    for name in sorted(aucs, key=aucs.get, reverse=True):
        fpr, tpr, _ = roc_curve(Yt.ravel(), probas[name].ravel())
        ax.plot(fpr, tpr, color=color_for(name), lw=1.8,
                label=f"{name} (AUC = {aucs[name]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random (AUC = 0.5)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves -- micro-average one-vs-rest (30 classes)")
    ax.legend(title="Model", loc="lower right")
    fig.savefig(os.path.join(out_dir, "roc_curves.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3) Per-class F1
# --------------------------------------------------------------------------- #
def plot_per_class_f1(preds: dict, y_test, categories, out_dir: str):
    """preds[name] = prediction vector. Plots per-class F1, one curve/model."""
    per_class = {
        name: f1_score(y_test, pred, average=None, labels=categories, zero_division=0)
        for name, pred in preds.items()
    }
    # Order classes by mean F1 (readability).
    mean_f1 = np.mean(list(per_class.values()), axis=0)
    order = np.argsort(mean_f1)[::-1]
    cats_sorted = [categories[i] for i in order]
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    rank = sorted(per_class, key=lambda n: per_class[n].mean(), reverse=True)
    for name in rank:
        ax.plot(x, per_class[name][order], marker=marker_for(name), ms=4,
                color=color_for(name), label=name, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats_sorted, rotation=90, fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("Category (sorted by descending mean F1)")
    ax.set_title("Per-category F1 by model (30 classes)")
    ax.legend(title="Model", ncols=2, loc="lower left")
    ax.set_ylim(0, 1.02)
    fig.savefig(os.path.join(out_dir, "per_class_f1.png"))
    plt.close(fig)
