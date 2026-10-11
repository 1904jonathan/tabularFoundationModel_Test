"""E4 - Real-data check: 20 Newsgroups with a hypothesised (pre-registered) taxonomy."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import revision_utils as u

SEED = 42
STATUS = os.path.join(u.REV, "e4_status.json")

# Taxonomy fixed BEFORE running any model.
TAXONOMY = {
    "comp": ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
             "comp.windows.x"],
    "rec": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
    "sci": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "politics": ["talk.politics.misc", "talk.politics.guns", "talk.politics.mideast"],
    "religion": ["alt.atheism", "soc.religion.christian", "talk.religion.misc"],
    "misc": ["misc.forsale"],
}
GMAP = {c: g for g, cs in TAXONOMY.items() for c in cs}
assert len(GMAP) == 20

try:
    from sklearn.datasets import fetch_20newsgroups
    rm = ("headers", "footers", "quotes")
    trb = fetch_20newsgroups(subset="train", remove=rm)
    teb = fetch_20newsgroups(subset="test", remove=rm)
except Exception as e:  # noqa: BLE001
    json.dump({"status": "download_failed", "error": repr(e)}, open(STATUS, "w"))
    u.log(f"E4 download failed: {e!r}")
    raise SystemExit(0)

assert set(trb.target_names) == set(GMAP), "taxonomy/target_names mismatch"


def frame(b):
    df = pd.DataFrame({"text": b.data, "category": np.array(b.target_names)[b.target]})
    n0 = len(df)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df, n0


tr_all, ntr0 = frame(trb)
te_all, nte0 = frame(teb)
tr, _ = train_test_split(tr_all, train_size=2400, stratify=tr_all["category"], random_state=SEED)
te, _ = train_test_split(te_all, train_size=1000, stratify=te_all["category"], random_state=SEED)
tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
info = dict(status="ok", train_full=ntr0, train_nonempty=len(tr_all), test_full=nte0, test_nonempty=len(te_all),
            n_train=len(tr), n_test=len(te), n_classes=20)
u.log(f"E4 data: {info}")

Xtr, Xte, _ = u.transform_texts(tr["text"], te["text"], n_components=64, seed=SEED)
ytr, yte = tr["category"].to_numpy(), te["category"].to_numpy()

rows, preds = [], {"true": yte}
for name, model in u.all_models(SEED, 20).items():
    try:
        r, yhat, _ = u.fit_eval(name, model, Xtr, ytr, Xte, yte, GMAP, None, perm_B=2000)
    except Exception as e:  # noqa: BLE001
        u.log(f"  FAILED {name}: {e!r}"); continue
    rows.append(r); preds[f"pred_{name}"] = yhat
    pd.DataFrame(rows).to_csv(os.path.join(u.REV, "e4_20newsgroups.csv"), index=False)

pd.DataFrame(rows).to_csv(os.path.join(u.REV, "e4_20newsgroups.csv"), index=False)
pd.DataFrame(preds).to_csv(os.path.join(u.REV, "e4_predictions.csv"), index=False)
json.dump(info, open(STATUS, "w"), indent=1)
u.log("E4 done")
print(pd.DataFrame(rows)[["model", "accuracy", "f1_macro", "GAER", "GAER0", "enrichment", "perm_p",
                          "superclass_acc"]].to_string())
