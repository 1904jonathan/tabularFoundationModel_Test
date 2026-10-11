"""E1 - Multi-seed rerun (seeds 0..9) of the full benchmark (7 models + MLP)."""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

import revision_utils as u

SEEDS = list(range(10))
out_csv = os.path.join(u.REV, "e1_multiseed_raw.csv")

rows = []
if os.path.exists(out_csv):  # resume support
    rows = pd.read_csv(out_csv).to_dict("records")
done = {r["seed"] for r in rows}

for s in SEEDS:
    if s in done:
        u.log(f"E1 seed {s} already done, skipping"); continue
    u.log(f"E1 seed {s}")
    d = u.prepare_synth(s)
    pdir = os.path.join(u.REV, f"predictions_seed{s}")
    os.makedirs(pdir, exist_ok=True)
    preds = {"true": d["yte"], "clean_true": d["yte_clean"] if d["yte_clean"] is not None else [None] * len(d["yte"])}
    for name, model in u.all_models(s, len(u.CATEGORIES)).items():
        try:
            r, yhat, _ = u.fit_eval(name, model, d["Xtr"], d["ytr"], d["Xte"], d["yte"], u.CAT2GROUP,
                                    d["yte_clean"], perm_B=2000)
        except Exception as e:  # noqa: BLE001
            u.log(f"  FAILED {name}: {e!r}"); continue
        r["seed"] = s
        r["n_train"] = len(d["ytr"])
        rows.append(r)
        preds[f"pred_{name}"] = yhat
    pd.DataFrame({"text": d["test_text"], **preds}).to_csv(os.path.join(pdir, "predictions.csv"), index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)

# ---- summary mean +- std ----
metrics = ["accuracy", "f1_macro", "f1_weighted", "log_loss", "fit_time_s", "predict_time_s", "GAER", "GAER0",
           "enrichment", "superclass_acc", "clean_accuracy", "clean_GAER", "clean_enrichment", "n_err", "perm_p"]
g = df.groupby("model")[metrics]
summ = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
summ = summ.sort_values("f1_macro_mean", ascending=False)
summ.to_csv(os.path.join(u.REV, "e1_multiseed_summary.csv"))

# ---- ranks / tests on macro-F1 ----
piv = df.pivot(index="seed", columns="model", values="f1_macro").dropna(axis=0)
ranks = piv.rank(axis=1, ascending=False)
avg_rank = ranks.mean().sort_values()
fr = friedmanchisquare(*[piv[c].values for c in piv.columns])
tabicl_first = int((ranks["TabICL"] == 1).sum())
tests = []
for c in piv.columns:
    if c == "TabICL":
        continue
    diff = piv["TabICL"] - piv[c]
    w = wilcoxon(piv["TabICL"], piv[c], alternative="two-sided")
    tests.append(dict(baseline=c, mean_diff=diff.mean(), std_diff=diff.std(ddof=1), wins=int((diff > 0).sum()),
                      losses=int((diff < 0).sum()), wilcoxon_stat=w.statistic, wilcoxon_p=w.pvalue))
tests = pd.DataFrame(tests)
tests["holm_p"] = np.nan
order = np.argsort(tests["wilcoxon_p"].values)
m = len(tests); prev = 0
for k, i in enumerate(order):
    val = min(1.0, max(prev, (m - k) * tests["wilcoxon_p"].values[i])); tests.loc[i, "holm_p"] = val; prev = val
tests.to_csv(os.path.join(u.REV, "e1_wilcoxon.csv"), index=False)
rk = pd.DataFrame({"avg_rank": avg_rank})
rk["friedman_chi2"] = fr.statistic; rk["friedman_p"] = fr.pvalue; rk["n_seeds"] = len(piv)
rk["tabicl_rank1_count"] = tabicl_first
rk.to_csv(os.path.join(u.REV, "e1_ranks_friedman.csv"))
u.log(f"E1 done. Friedman chi2={fr.statistic:.3f} p={fr.pvalue:.3g}; TabICL first on {tabicl_first}/{len(piv)} seeds")
print(summ[["accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std", "GAER_mean", "clean_accuracy_mean"]].to_string())
print(tests.to_string())
print(rk.to_string())
