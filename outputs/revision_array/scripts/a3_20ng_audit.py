"""A3 - Item-level audit on 20 Newsgroups.

Stage 'prepare': re-creates the 1,000-document test subsample of E4 (same calls as
outputs/revision/scripts/e4_20newsgroups.py), checks that its labels equal the `true` column of
e4_predictions.csv row by row, and builds two sets:
  U_all : documents whose error is taxonomy-unaligned for all 8 models (hypothesised taxonomy H);
  A_all : control, documents misclassified by all 8 models with every error aligned (in-group).
It writes length statistics for U_all, A_all and all test documents, and a BLIND audit sheet with 15
documents drawn at random from each set (seed 0), shuffled together, showing only an id, the
observed newsgroup and an excerpt (first 400 characters, whitespace collapsed), without set membership
or predictions.
Stage 'score': reads the annotated sheet (column `judgement`, filled by hand by the first author)
and the hidden key, and counts judgements per set.
Judgement codes (judged at the level of the hypothesised group, blind to set membership):
  on-topic       the text is clearly about the topic of its observed newsgroup
  off-topic      the text is clearly about something other than its observed newsgroup's topic
                 (another group's topic or no group's topic: possible mislabel / cross-post / drift)
  uninformative  too short or generic to identify any topic (e.g. one-line replies, signatures, code)
  ambiguous      topical content present but compatible with more than one group
Usage: python a3_20ng_audit.py prepare | score
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

import array_utils as u

stage = sys.argv[1] if len(sys.argv) > 1 else "prepare"
SHEET = os.path.join(u.OUT, "a3_audit_sheet_blind.csv")
KEY = os.path.join(u.OUT, "a3_audit_key.csv")
ANN = os.path.join(u.OUT, "a3_audit_sheet_annotated.csv")

if stage == "prepare":
    teb = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    te_all = pd.DataFrame({"text": teb.data, "category": np.array(teb.target_names)[teb.target]})
    te_all = te_all[te_all["text"].str.strip().str.len() > 0].reset_index(drop=True)
    te, _ = train_test_split(te_all, train_size=1000, stratify=te_all["category"], random_state=42)
    te = te.reset_index(drop=True)
    d = pd.read_csv(os.path.join(u.REV, "e4_predictions.csv"))
    assert (te["category"].to_numpy() == d["true"].to_numpy()).all(), "subsample does not match e4_predictions"
    models = [c[5:] for c in d.columns if c.startswith("pred_")]
    preds = {m: d[f"pred_{m}"].to_numpy() for m in models}
    err, unal, names = u.error_matrices(d["true"].to_numpy(), preds, u.GMAP20)
    M = len(names)
    nu, ne = unal.sum(1), err.sum(1)
    U = np.where(nu == M)[0]; A = np.where((ne == M) & (nu == 0))[0]
    words = te["text"].str.split().str.len().to_numpy()
    lens = []
    for nm, idx in (("U_all", U), ("A_all", A), ("all_test", np.arange(len(te))),
                    ("correct_by_all", np.where(ne == 0)[0])):
        w = words[idx]
        lens.append(dict(set=nm, n=len(idx), words_median=float(np.median(w)), words_q25=float(np.percentile(w, 25)),
                         words_q75=float(np.percentile(w, 75)), frac_le_20_words=float(np.mean(w <= 20))))
    L = pd.DataFrame(lens); L.to_csv(os.path.join(u.OUT, "a3_length_stats.csv"), index=False)
    print(L.to_string())
    # majority prediction (group) for U_all items, to see where the unaligned errors go
    maj = []
    for i in U:
        pg = pd.Series([u.GMAP20[preds[m][i]] for m in names]).value_counts()
        maj.append(dict(item=int(i), true=d["true"][i], true_group=u.GMAP20[d["true"][i]],
                        majority_pred_group=pg.index[0], n_majority=int(pg.iloc[0]), words=int(words[i])))
    MJ = pd.DataFrame(maj); MJ.to_csv(os.path.join(u.OUT, "a3_uall_items.csv"), index=False)
    print(pd.crosstab(MJ.true_group, MJ.majority_pred_group).to_string())
    rng = np.random.default_rng(0)
    su = rng.choice(U, size=min(15, len(U)), replace=False); sa = rng.choice(A, size=min(15, len(A)), replace=False)
    key = pd.DataFrame({"item": np.concatenate([su, sa]), "set": ["U_all"] * len(su) + ["A_all"] * len(sa)})
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    key["audit_id"] = np.arange(1, len(key) + 1)
    key["majority_pred"] = [pd.Series([preds[m][i] for m in names]).value_counts().index[0] for i in key["item"]]
    key.to_csv(KEY, index=False)

    def excerpt(t, n=400):
        t = re.sub(r"\s+", " ", t).strip()
        return t[:n] + (" [...]" if len(t) > n else "")
    sheet = pd.DataFrame({"audit_id": key["audit_id"], "observed_newsgroup": te["category"].to_numpy()[key["item"]],
                          "n_words": words[key["item"]], "excerpt": [excerpt(te["text"][i]) for i in key["item"]],
                          "judgement": "", "note": ""})
    sheet.to_csv(SHEET, index=False)
    for _, r in sheet.iterrows():
        print(f"\n#{r.audit_id} [{r.observed_newsgroup}] ({r.n_words} words): {r.excerpt}")

elif stage == "score":
    key = pd.read_csv(KEY); ann = pd.read_csv(ANN)
    m = key.merge(ann[["audit_id", "observed_newsgroup", "judgement", "note"]], on="audit_id")
    m.to_csv(os.path.join(u.OUT, "a3_audit_unblinded.csv"), index=False)
    T = pd.crosstab(m["set"], m["judgement"], margins=True)
    T.to_csv(os.path.join(u.OUT, "a3_audit_counts.csv"))
    print(T.to_string())
    print(m[["audit_id", "set", "observed_newsgroup", "majority_pred", "judgement", "note"]].to_string())
