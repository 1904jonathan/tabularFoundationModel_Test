"""Shared helpers for the Array revision analyses (cross-model consistency, mixture-model estimators).

All analyses here are post hoc: they read saved predictions only (no model is re-trained), except
a3_20ng_audit.py, which re-downloads the 20 Newsgroups texts to recover the test documents.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "outputs", "revision_array")
REV = os.path.join(REPO, "outputs", "revision")
os.makedirs(OUT, exist_ok=True)

from tfm_bench.data_synth import CAT2GROUP, CATEGORIES  # noqa: E402,F401

# 20 Newsgroups taxonomy, identical to outputs/revision/scripts/e4_20newsgroups.py (fixed before any model).
TAX20 = {
    "comp": ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
             "comp.windows.x"],
    "rec": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
    "sci": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "politics": ["talk.politics.misc", "talk.politics.guns", "talk.politics.mideast"],
    "religion": ["alt.atheism", "soc.religion.christian", "talk.religion.misc"],
    "misc": ["misc.forsale"],
}
GMAP20 = {c: g for g, cs in TAX20.items() for c in cs}

SEED42_MODELS = ["TabICL_n8", "XGBoost_fixed", "RandomForest_fixed", "LightGBM_fixed", "HistGB_fixed",
                 "LogisticRegression_fixed", "KNN_fixed"]  # the seven models of Table 2 (seed 42)
SEED_MODELS = ["TabICL", "RandomForest", "XGBoost", "LightGBM", "HistGB", "LogisticRegression", "KNN", "MLP"]


def chance_ingroup(labels, gmap):
    """Per-item probability that a uniformly random wrong label lies in the item's group: (|G|-1)/(K-1)."""
    K = len(gmap)
    sizes = pd.Series(gmap).value_counts().to_dict()
    return np.array([(sizes[gmap[c]] - 1) / (K - 1) for c in labels])


def error_matrices(y, preds: dict, gmap):
    """Return (err, unal) boolean arrays of shape (n_items, n_models).
    err[i,m]  = model m misclassifies item i against the observed label y[i];
    unal[i,m] = err[i,m] and the predicted class lies outside the group of y[i] (taxonomy-unaligned error)."""
    y = np.asarray(y)
    names = list(preds)
    P = np.column_stack([np.asarray(preds[m]) for m in names])
    err = P != y[:, None]
    gy = np.array([gmap[c] for c in y])
    gp = np.vectorize(lambda c: gmap[c])(P)
    unal = err & (gp != gy[:, None])
    return err, unal, names


def gaer_from(err_col, unal_col):
    e = int(err_col.sum())
    return (1 - unal_col.sum() / e) if e else np.nan, e, int(unal_col.sum())


def pr(flag, truth):
    tp = int((flag & truth).sum()); fp = int((flag & ~truth).sum()); fn = int((~flag & truth).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return tp, fp, fn, prec, rec


def md_table(df, floatfmt=".3f"):
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i in range(len(df)):
        lines.append("| " + " | ".join(fmt(df[c].iloc[i]) for c in cols) + " |")
    return "\n".join(lines)
