"""
TEXT -> NUMBERS conversion, shared by ALL models in the benchmark.

Method: TF-IDF (weighted bag of words, 1-2 grams) -> TruncatedSVD (LSA) -> Normalizer.
The pipeline is fitted ONLY on the training set and then applied to the test set
(no leakage). The same numeric features are given to every model => fair comparison.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def build_vectorizer(n_components: int = 64, seed: int = 42):
    """scikit-learn pipeline: text -> dense numeric vector."""
    return make_pipeline(
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
        TruncatedSVD(n_components=n_components, random_state=seed),
        Normalizer(copy=False),
    )


def transform_texts(train_text, test_text, n_components: int = 64, seed: int = 42):
    """
    Fit the vectorizer on train_text, transform train + test.
    Returns (X_train, X_test, fitted_vectorizer) as named-column DataFrames.
    """
    vec = build_vectorizer(n_components=n_components, seed=seed)
    Xtr = vec.fit_transform(train_text)
    Xte = vec.transform(test_text)
    cols = [f"lsa_{i:02d}" for i in range(Xtr.shape[1])]
    return (
        pd.DataFrame(Xtr, columns=cols).reset_index(drop=True),
        pd.DataFrame(Xte, columns=cols).reset_index(drop=True),
        vec,
    )
