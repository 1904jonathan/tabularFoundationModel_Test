"""E3 - Sensitivity to the LSA dimension d (seed-42 protocol)."""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import revision_utils as u
from tfm_bench.models import build_models

SEED = 42
K = len(u.CATEGORIES)
DIMS = [16, 32, 64, 128, 256]
KEEP = ["TabICL", "XGBoost", "LogisticRegression", "RandomForest"]
out = os.path.join(u.REV, "e3_lsa_dim.csv")

df = u.build_dataset(n_per_class=60, label_noise=0.05, imbalance=True, seed=SEED)
tr, te = train_test_split(df, test_size=0.25, random_state=SEED, stratify=df["category"])
n_feat = len(TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(tr["text"]).vocabulary_)
u.log(f"E3: TF-IDF vocabulary size on train = {n_feat}")

rows = []
for dim in DIMS:
    d_eff = min(dim, n_feat - 1)
    u.log(f"E3 d={dim} (effective {d_eff})")
    dd = u.prepare_synth(SEED, n_components=d_eff)
    models = build_models(SEED, K)
    for name in KEEP:
        try:
            r, _, _ = u.fit_eval(name, models[name], dd["Xtr"], dd["ytr"], dd["Xte"], dd["yte"], u.CAT2GROUP,
                                 dd["yte_clean"])
        except Exception as e:  # noqa: BLE001
            u.log(f"  FAILED {name} d={dim}: {e!r}"); continue
        r.update(d_requested=dim, d_effective=d_eff, tfidf_vocab=n_feat)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
res = pd.DataFrame(rows)
res.to_csv(out, index=False)
u.log("E3 done")
print(res.pivot(index="d_requested", columns="model", values="f1_macro").to_string())
