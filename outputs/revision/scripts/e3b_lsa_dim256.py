"""E3b - d=256 point of E3, rerun separately: with the default TabICL batch_size=8 (all 8 ensemble members
batched together) the d=256 run exhausted the 15 GB of RAM and thrashed. Here TabICL uses batch_size=1
(ensemble members processed one at a time; same model, same members, only memory/batching differs).
A d=64 TabICL run with batch_size=1 is also included as a control that batch_size does not change results."""
import os

import pandas as pd
from tabicl import TabICLClassifier

import revision_utils as u
from tfm_bench.models import build_models

SEED = 42
K = len(u.CATEGORIES)
out = os.path.join(u.REV, "e3_lsa_dim.csv")
prev = pd.read_csv(out)
vocab = int(prev["tfidf_vocab"].iloc[0])
rows = prev[prev["d_requested"] != 256].to_dict("records")

# control: d=64, batch_size=1
dd = u.prepare_synth(SEED, n_components=64)
r, _, _ = u.fit_eval("TabICL", TabICLClassifier(n_estimators=8, support_many_classes=True, random_state=SEED,
                                                device=None, batch_size=1),
                     dd["Xtr"], dd["ytr"], dd["Xte"], dd["yte"], u.CAT2GROUP, dd["yte_clean"])
r.update(d_requested=64, d_effective=64, tfidf_vocab=vocab, tabicl_batch_size=1, note="control (batch_size=1)")
pd.DataFrame([r]).to_csv(os.path.join(u.REV, "e3b_batchsize_control.csv"), index=False)

dim = 256
d_eff = min(dim, vocab - 1)
u.log(f"E3b d={dim} (effective {d_eff})")
dd = u.prepare_synth(SEED, n_components=d_eff)
models = build_models(SEED, K)
models["TabICL"] = TabICLClassifier(n_estimators=8, support_many_classes=True, random_state=SEED, device=None,
                                    batch_size=1)
for name in ["TabICL", "XGBoost", "LogisticRegression", "RandomForest"]:
    try:
        r, _, _ = u.fit_eval(name, models[name], dd["Xtr"], dd["ytr"], dd["Xte"], dd["yte"], u.CAT2GROUP,
                             dd["yte_clean"])
    except Exception as e:  # noqa: BLE001
        u.log(f"  FAILED {name} d={dim}: {e!r}"); continue
    r.update(d_requested=dim, d_effective=d_eff, tfidf_vocab=vocab,
             tabicl_batch_size=1 if name == "TabICL" else None)
    rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
u.log("E3b done")
