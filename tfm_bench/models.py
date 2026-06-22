"""
Registry of the models to compare.

Each entry is a scikit-learn compliant estimator exposing fit / predict /
predict_proba. They all receive the SAME numeric features (TF-IDF + SVD), which
makes the comparison fair.

Models:
  - TabICL             : tabular foundation model (in-context learning, soda-inria)
  - RandomForest       : random forest (bagging of trees)
  - XGBoost            : gradient boosting (XGBoost)
  - LightGBM           : gradient boosting (LightGBM)
  - HistGB             : scikit-learn native gradient boosting
  - LogisticRegression : linear baseline
  - KNN                : k-nearest neighbors (baseline)
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps an estimator that requires integer labels (e.g. XGBoost 3.x).

    Encodes y (strings) into integers for fitting, and exposes a standard
    interface: predict returns the original labels, predict_proba keeps the
    classes_ ordering.
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_           # order = predict_proba columns
        self.base_ = clone(self.base)
        self.base_.fit(X, y_enc)
        return self

    def predict(self, X):
        return self._le.inverse_transform(self.base_.predict(X))

    def predict_proba(self, X):
        return self.base_.predict_proba(X)


def _tabicl(seed: int):
    from tabicl import TabICLClassifier
    return TabICLClassifier(
        n_estimators=8,
        support_many_classes=True,   # required for 30 classes (>10)
        random_state=seed,
        device=None,
    )


def _xgboost(seed: int, n_classes: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


def _lightgbm(seed: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_models(seed: int, n_classes: int) -> dict:
    """Return {name: estimator} ready to fit/predict. Heavy imports are lazy."""
    models = {
        "TabICL": _tabicl(seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=seed
        ),
        "XGBoost": LabelEncodedClassifier(_xgboost(seed, n_classes)),
        "LightGBM": _lightgbm(seed),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.1, random_state=seed
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=10.0, random_state=seed
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }
    return models
