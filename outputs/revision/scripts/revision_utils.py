"""Shared helpers for the revision experiments (GAER, permutation null, data prep, evaluation)."""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # avoid loky's wmic probe (wmic absent)

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
REV = os.path.join(REPO, "outputs", "revision")
os.makedirs(REV, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier

from tfm_bench.data_synth import build_dataset, CATEGORIES, CAT2GROUP, CONFUSABLES  # noqa: F401
from tfm_bench.text_features import transform_texts
from tfm_bench.models import build_models

warnings.filterwarnings("ignore")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ GAER ---
def gaer(y, yhat, group_map):
    """Return (GAER, GAER0, enrichment, n_err).
    GAER  = fraction of errors whose predicted class lies in the true class's group.
    GAER0 = chance level = mean over errors of (|G_g(y)|-1)/(K-1)."""
    y = np.asarray(y); yhat = np.asarray(yhat)
    K = len(group_map)
    sizes = pd.Series(group_map).value_counts().to_dict()
    err = y != yhat
    n_err = int(err.sum())
    if n_err == 0:
        return np.nan, np.nan, np.nan, 0
    ye, pe = y[err], yhat[err]
    same = np.array([group_map[a] == group_map[b] for a, b in zip(ye, pe)])
    g = float(same.mean())
    g0 = float(np.mean([(sizes[group_map[a]] - 1) / (K - 1) for a in ye]))
    return g, g0, (g / g0 if g0 > 0 else np.nan), n_err


def gaer_perm_null(y, yhat, group_map, B=2000, seed=0):
    """Permutation null: random class->group reassignment preserving group sizes.
    p = (1 + #{null >= obs}) / (B + 1)."""
    y = np.asarray(y); yhat = np.asarray(yhat)
    classes = sorted(group_map)
    idx = {c: i for i, c in enumerate(classes)}
    groups = np.array([group_map[c] for c in classes])
    err = y != yhat
    if err.sum() == 0:
        return dict(obs=np.nan, null_mean=np.nan, null_p95=np.nan, null_max=np.nan, p=np.nan)
    yi = np.array([idx[c] for c in y[err]]); pi = np.array([idx[c] for c in yhat[err]])
    obs = float(np.mean(groups[yi] == groups[pi]))
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        perm = rng.permutation(groups)  # class i -> group perm[i]; multiset of sizes preserved
        null[b] = np.mean(perm[yi] == perm[pi])
    p = (1 + np.sum(null >= obs - 1e-12)) / (B + 1)
    return dict(obs=obs, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                null_max=float(null.max()), p=float(p))


def superclass_acc(y, yhat, group_map):
    return float(np.mean([group_map[a] == group_map[b] for a, b in zip(y, yhat)]))


# ------------------------------------------------------------------ data ---
def prepare_synth(seed, n_components=64, check_clean=True):
    """Exact run_benchmark.py protocol for a given seed (dataset seed = split seed = SVD seed)."""
    df = build_dataset(n_per_class=60, label_noise=0.05, imbalance=True, seed=seed)
    clean = None
    if check_clean:
        dfc = build_dataset(n_per_class=60, label_noise=0.0, imbalance=True, seed=seed)
        if len(dfc) == len(df) and (dfc["text"].values == df["text"].values).all():
            clean = dfc["category"].to_numpy()
        else:
            log(f"WARNING seed={seed}: clean rebuild text mismatch -> clean-label metrics skipped")
    tr, te = train_test_split(df, test_size=0.25, random_state=seed, stratify=df["category"])
    yte_clean = clean[te.index.values] if clean is not None else None
    ytr_clean = clean[tr.index.values] if clean is not None else None
    tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
    Xtr, Xte, _ = transform_texts(tr["text"], te["text"], n_components=n_components, seed=seed)
    return dict(Xtr=Xtr, Xte=Xte, ytr=tr["category"].to_numpy(), yte=te["category"].to_numpy(),
                yte_clean=yte_clean, ytr_clean=ytr_clean, test_text=te["text"].to_numpy(),
                train_text=tr["text"], test_text_s=te["text"])


def mlp(seed, **kw):
    """MLP baseline. Wrapped in LabelEncodedClassifier because sklearn 1.6.1's MLPClassifier with
    early_stopping=True crashes on string labels (isnan on the validation targets); the wrapper only
    integer-encodes y and is otherwise transparent (params are exposed as base__*)."""
    from tfm_bench.models import LabelEncodedClassifier
    params = dict(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=500, random_state=seed)
    params.update(kw)
    return LabelEncodedClassifier(MLPClassifier(**params))


def all_models(seed, n_classes):
    m = build_models(seed, n_classes)
    m["MLP"] = mlp(seed)
    return m


# ------------------------------------------------------------- evaluate ---
def fit_eval(name, model, Xtr, ytr, Xte, yte, group_map, yte_clean=None, perm_B=0):
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); yhat = np.asarray(model.predict(Xte)); pred_t = time.perf_counter() - t0
    try:
        proba = model.predict_proba(Xte)
        ll = log_loss(yte, proba, labels=model.classes_)
    except Exception as e:  # noqa: BLE001
        log(f"  log_loss failed for {name}: {e}")
        proba, ll = None, np.nan
    g, g0, enr, nerr = gaer(yte, yhat, group_map)
    r = dict(model=name, accuracy=accuracy_score(yte, yhat),
             f1_macro=f1_score(yte, yhat, average="macro", zero_division=0),
             f1_weighted=f1_score(yte, yhat, average="weighted", zero_division=0),
             log_loss=ll, fit_time_s=fit_t, predict_time_s=pred_t,
             GAER=g, GAER0=g0, enrichment=enr, superclass_acc=superclass_acc(yte, yhat, group_map),
             n_err=nerr, n_test=len(yte))
    if yte_clean is not None:
        gc, g0c, enrc, nerrc = gaer(yte_clean, yhat, group_map)
        r.update(clean_accuracy=accuracy_score(yte_clean, yhat),
                 clean_f1_macro=f1_score(yte_clean, yhat, average="macro", zero_division=0),
                 clean_GAER=gc, clean_GAER0=g0c, clean_enrichment=enrc, clean_n_err=nerrc)
    if perm_B:
        pn = gaer_perm_null(yte, yhat, group_map, B=perm_B, seed=0)
        r.update(perm_null_mean=pn["null_mean"], perm_null_p95=pn["null_p95"],
                 perm_null_max=pn["null_max"], perm_p=pn["p"])
    log(f"  {name:20s} acc={r['accuracy']:.4f} f1M={r['f1_macro']:.4f} ll={ll:.3f} "
        f"GAER={g:.3f} (x{enr:.2f}) fit={fit_t:.1f}s pred={pred_t:.1f}s")
    return r, yhat, proba


# ------------------------------------------------------------ stats ---
def mcnemar_exact(y, a, b):
    """Exact McNemar: binomial test on discordant pairs. Returns (a right & b wrong, a wrong & b right, p)."""
    from scipy.stats import binomtest
    y = np.asarray(y)
    ca, cb = (np.asarray(a) == y), (np.asarray(b) == y)
    n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, float(p)


def paired_bootstrap_f1(y, a, b, B=10000, seed=0):
    """Paired bootstrap (resample test rows) of macro-F1(a) - macro-F1(b).
    Returns (observed diff, CI low, CI high, fraction of resamples with diff <= 0)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); a = np.asarray(a); b = np.asarray(b); n = len(y)
    labs = np.unique(np.concatenate([y, a, b]))
    m = {c: i for i, c in enumerate(labs)}
    yi = np.array([m[c] for c in y]); ai = np.array([m[c] for c in a]); bi = np.array([m[c] for c in b])
    L = len(labs)

    def f1m(yy, pp):
        # identical to sklearn macro-F1 (labels = union of present true/pred labels, zero_division=0)
        tp = np.bincount(yy[yy == pp], minlength=L).astype(float)
        nt = np.bincount(yy, minlength=L).astype(float); npred = np.bincount(pp, minlength=L).astype(float)
        present = (nt + npred) > 0
        denom = nt + npred
        f = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
        return f[present].mean()

    obs = (f1_score(y, a, average="macro", zero_division=0)
           - f1_score(y, b, average="macro", zero_division=0))
    assert abs(obs - (f1m(yi, ai) - f1m(yi, bi))) < 1e-9
    diffs = np.empty(B)
    for i in range(B):
        ix = rng.integers(0, n, n)
        diffs[i] = f1m(yi[ix], ai[ix]) - f1m(yi[ix], bi[ix])
    return float(obs), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(diffs <= 0))


def md_table(df, floatfmt=".4f"):
    """DataFrame -> markdown table without tabulate dependency."""
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i in range(len(df)):  # column-wise access keeps integer dtypes (iterrows upcasts to float)
        lines.append("| " + " | ".join(fmt(df[c].iloc[i]) for c in cols) + " |")
    return "\n".join(lines)
