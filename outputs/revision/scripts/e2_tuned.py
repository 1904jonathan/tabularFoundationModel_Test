"""E2 - Tuned baselines on the original seed-42 protocol + TabICL (n_estimators 8 and 1)."""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform  # noqa: F401
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

import revision_utils as u
from tfm_bench.models import LabelEncodedClassifier, build_models

SEED = 42
K = len(u.CATEGORIES)
d = u.prepare_synth(SEED)
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
u.log(f"E2 start: train={Xtr.shape} test={Xte.shape}")


def xgb_base():
    from xgboost import XGBClassifier
    return LabelEncodedClassifier(XGBClassifier(objective="multi:softprob", num_class=K, tree_method="hist",
                                                random_state=SEED, n_jobs=1, eval_metric="mlogloss"))


def lgbm_base():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)


# Inner estimators use n_jobs=1; the search itself parallelises over candidates x folds (n_jobs=-1).
SEARCHES = {
    "XGBoost": ("random", xgb_base(), {
        "base__n_estimators": [200, 400, 800], "base__max_depth": [3, 4, 6, 8],
        "base__learning_rate": [0.03, 0.05, 0.1, 0.2], "base__subsample": [0.6, 0.8, 1.0],
        "base__colsample_bytree": [0.5, 0.8, 1.0], "base__min_child_weight": [1, 3, 5]}),
    "LightGBM": ("random", lgbm_base(), {
        "n_estimators": [200, 400, 800], "num_leaves": [15, 31, 63, 127], "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0], "min_child_samples": [5, 10, 20]}),
    "HistGB": ("random", HistGradientBoostingClassifier(random_state=SEED), {
        "max_iter": [200, 400, 800], "learning_rate": [0.03, 0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63],
        "l2_regularization": [0.0, 0.1, 1.0]}),
    "RandomForest": ("random", RandomForestClassifier(random_state=SEED, n_jobs=1), {
        "n_estimators": [200, 400, 800], "max_features": ["sqrt", 0.3, 0.5, None], "min_samples_leaf": [1, 2, 4]}),
    "MLP": ("random", u.mlp(SEED), {
        "base__hidden_layer_sizes": [(128,), (256,), (256, 128), (512, 256)], "base__alpha": [1e-4, 1e-3, 1e-2],
        "base__learning_rate_init": [1e-3, 3e-3]}),
    "LogisticRegression": ("grid", LogisticRegression(max_iter=2000, random_state=SEED),
                           {"C": [0.01, 0.1, 1, 10, 100]}),
    "KNN": ("grid", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 11, 15, 21, 31, 51],
                                             "weights": ["uniform", "distance"]}),
}

rows, preds = [], {"true": yte, "clean_true": d["yte_clean"]}

# ---- TabICL (8 members and 1 member) ----
from tabicl import TabICLClassifier  # noqa: E402
for ne in (8, 1):
    m = TabICLClassifier(n_estimators=ne, support_many_classes=True, random_state=SEED, device=None)
    r, yhat, _ = u.fit_eval(f"TabICL_n{ne}", m, Xtr, ytr, Xte, yte, u.CAT2GROUP, d["yte_clean"])
    r.update(setting="default", n_estimators=ne); rows.append(r); preds[f"pred_TabICL_n{ne}"] = yhat

# ---- fixed-setting baselines (as in the paper) ----
fixed = build_models(SEED, K); fixed.pop("TabICL"); fixed["MLP"] = u.mlp(SEED)
for name, m in fixed.items():
    r, yhat, _ = u.fit_eval(name + "_fixed", m, Xtr, ytr, Xte, yte, u.CAT2GROUP, d["yte_clean"])
    r.update(setting="fixed", base_model=name); rows.append(r); preds[f"pred_{name}_fixed"] = yhat
pd.DataFrame(rows).to_csv(os.path.join(u.REV, "e2_tuned.csv"), index=False)

# ---- tuned baselines ----
for name, (kind, est, space) in SEARCHES.items():
    u.log(f"E2 tuning {name}")
    t0 = time.perf_counter()
    try:
        if kind == "random":
            n_space = int(np.prod([len(v) for v in space.values()]))
            s = RandomizedSearchCV(est, space, n_iter=min(25, n_space), cv=cv, scoring="f1_macro",
                                   random_state=SEED, n_jobs=-1, refit=True)
        else:
            s = GridSearchCV(est, space, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
        s.fit(Xtr, ytr)
    except Exception as e:  # noqa: BLE001
        u.log(f"  FAILED tuning {name}: {e!r}"); continue
    search_t = time.perf_counter() - t0
    best = s.best_estimator_
    # re-fit best config on full train (timed, consistent with other rows); restore n_jobs=-1 where applicable
    from sklearn.base import clone
    best = clone(best)
    if name == "XGBoost":
        best.set_params(base__n_jobs=-1)
    elif name in ("LightGBM", "RandomForest"):
        best.set_params(n_jobs=-1)
    r, yhat, _ = u.fit_eval(name + "_tuned", best, Xtr, ytr, Xte, yte, u.CAT2GROUP, d["yte_clean"])
    bp = {k.replace("base__", ""): (list(v) if isinstance(v, tuple) else v) for k, v in s.best_params_.items()}
    r.update(setting="tuned", base_model=name, best_cv_f1_macro=s.best_score_, best_params=json.dumps(bp),
             n_candidates=len(s.cv_results_["params"]), search_time_s=search_t)
    rows.append(r); preds[f"pred_{name}_tuned"] = yhat
    u.log(f"  best CV f1M={s.best_score_:.4f} params={bp} ({search_t:.0f}s)")
    pd.DataFrame(rows).to_csv(os.path.join(u.REV, "e2_tuned.csv"), index=False)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(u.REV, "e2_tuned.csv"), index=False)
pd.DataFrame({"text": d["test_text"], **preds}).to_csv(os.path.join(u.REV, "e2_predictions_seed42.csv"), index=False)

# ---- McNemar + paired bootstrap: TabICL (n=8) vs each tuned baseline ----
tab = preds["pred_TabICL_n8"]
st = []
for name in SEARCHES:
    key = f"pred_{name}_tuned"
    if key not in preds:
        continue
    b = preds[key]
    n01, n10, p = u.mcnemar_exact(yte, tab, b)
    obs, lo, hi, frac = u.paired_bootstrap_f1(yte, tab, b, B=10000, seed=0)
    st.append(dict(baseline=name + "_tuned", tabicl_right_baseline_wrong=n01, tabicl_wrong_baseline_right=n10,
                   mcnemar_exact_p=p, f1_macro_diff=obs, boot_ci_low=lo, boot_ci_high=hi, boot_frac_le0=frac))
st = pd.DataFrame(st)
st.to_csv(os.path.join(u.REV, "e2_tests.csv"), index=False)
u.log("E2 done")
print(df[["model", "accuracy", "f1_macro", "log_loss", "GAER", "superclass_acc", "fit_time_s"]].to_string())
print(st.to_string())
