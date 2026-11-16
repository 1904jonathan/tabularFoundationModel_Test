"""A1 - Cross-model consistency of taxonomy-unaligned errors on the designed testbed, and the
mixture-model estimators of the label-noise rate.

Inputs (saved predictions, nothing is re-trained):
  - seed 42: outputs/revision/e2_predictions_seed42.csv, the seven fixed-setting models of Table 2
    (TabICL with 8 members, XGBoost, RandomForest, LightGBM, HistGB, LogReg, kNN); MLP_fixed added as a
    supplementary 8-model variant;
  - seeds 0-9: outputs/revision/predictions_seed{s}/predictions.csv, eight models (incl. MLP).
Ground truth: the column clean_true (generator rebuilt with label_noise=0, text verified identical),
so corrupted = (true != clean_true).

Outputs (outputs/revision_array/):
  a1_summary.csv        one row per data set (seed / model set)
  a1_pr_curve.csv       precision/recall of the two detectors for every k
  a1_per_model.csv      per-model GAER decomposition (clean / corrupted errors) and plug-in noise estimate
  a1_estimators.csv     noise-rate estimators per data set vs truth
  a1_exactly_one.csv    items unaligned for exactly one model: which model, clean or corrupted
  a1_uall_clean_items.csv  clean items that are unaligned for every model (inspection)
"""
import os

import numpy as np
import pandas as pd

import array_utils as u

GM = u.CAT2GROUP


def load_sets():
    sets = []
    d = pd.read_csv(os.path.join(u.REV, "e2_predictions_seed42.csv"))
    preds7 = {m.replace("_fixed", "").replace("_n8", ""): d[f"pred_{m}"].to_numpy() for m in u.SEED42_MODELS}
    sets.append(("seed42_7models", 42, d, preds7))
    preds8 = dict(preds7); preds8["MLP"] = d["pred_MLP_fixed"].to_numpy()
    sets.append(("seed42_8models", 42, d, preds8))
    for s in range(10):
        p = pd.read_csv(os.path.join(u.REV, f"predictions_seed{s}", "predictions.csv"))
        sets.append((f"seed{s}", s, p, {m: p[f"pred_{m}"].to_numpy() for m in u.SEED_MODELS}))
    return sets


summ, prc, permodel, est, ex1, uclean = [], [], [], [], [], []
for name, seed, df, preds in load_sets():
    y = df["true"].to_numpy(); yc = df["clean_true"].to_numpy()
    n = len(y)
    corrupt = y != yc
    cross = corrupt & (np.array([GM[a] for a in y]) != np.array([GM[b] for b in yc]))
    err, unal, names = u.error_matrices(y, preds, GM)
    M = len(names)
    n_unal = unal.sum(1); n_err = err.sum(1)
    beta_items = u.chance_ingroup(y, GM)   # chance that a random wrong label is in-group, per item
    beta_bar = float(beta_items.mean())

    uall = n_unal == M
    ex1_mask = n_unal == 1
    summ.append(dict(
        set=name, seed=seed, n=n, M=M, n_corrupt=int(corrupt.sum()), n_corrupt_crossgroup=int(cross.sum()),
        n_corrupt_ingroup=int((corrupt & ~cross).sum()),
        U_all=int(uall.sum()), U_all_corrupt=int((uall & corrupt).sum()), U_all_clean=int((uall & ~corrupt).sum()),
        U_exactly1=int(ex1_mask.sum()), U_exactly1_corrupt=int((ex1_mask & corrupt).sum()),
        U_any=int((n_unal >= 1).sum()), U_any_corrupt=int(((n_unal >= 1) & corrupt).sum()),
        E_all=int((n_err == M).sum()), E_all_corrupt=int(((n_err == M) & corrupt).sum()),
        corrupt_err_by_all=int(((n_err == M) & corrupt).sum()),
        beta_bar=beta_bar))

    for k in range(1, M + 1):
        for det, cnt in (("unaligned>=k", n_unal), ("misclassified>=k", n_err)):
            flag = cnt >= k
            tp, fp, fn, p_, r_ = u.pr(flag, corrupt)
            _, _, _, _, rc = u.pr(flag, cross)
            prc.append(dict(set=name, seed=seed, M=M, detector=det, k=k, flagged=int(flag.sum()), tp=tp, fp=fp,
                            fn=fn, precision=p_, recall=r_, recall_crossgroup=rc))

    # per-model decomposition and plug-in estimate
    rows_m = []
    for j, m in enumerate(names):
        e = err[:, j]; uu = unal[:, j]
        g, E, U = u.gaer_from(e, uu)
        g0 = float(beta_items[e].mean())
        Ec = int((e & ~corrupt).sum()); En = int((e & corrupt).sum())
        Uc = int((uu & ~corrupt).sum()); Un = int((uu & corrupt).sum())
        alpha = 1 - Uc / Ec if Ec else np.nan              # GAER on clean items (observed label = clean label)
        beta_real = 1 - Un / En if En else np.nan          # aligned share of errors on corrupted items
        pred_alpha1 = 1 - Un / E                            # dilution prediction with alpha = 1
        pred_full = (alpha * Ec + beta_real * En) / E       # identity of the mixture decomposition
        eta_hat = U / ((1 - g0) * n)
        rows_m.append(dict(set=name, seed=seed, model=m, n=n, E=E, U=U, GAER=g, GAER0=g0, E_clean=Ec, E_corrupt=En,
                           U_clean=Uc, U_corrupt=Un, alpha_clean=alpha, beta_realised=beta_real,
                           GAER_pred_alpha1=pred_alpha1, GAER_pred_full=pred_full, eta_hat_plugin=eta_hat,
                           eta_true=corrupt.mean()))
    permodel += rows_m
    pm = pd.DataFrame(rows_m)

    # regression estimator: GAER_m = a + b / E_m  (common alpha, beta) -> n_eta = -b / (a - beta)
    X = np.column_stack([np.ones(M), 1 / pm["E"].to_numpy()])
    a, b = np.linalg.lstsq(X, pm["GAER"].to_numpy(), rcond=None)[0]
    resid = pm["GAER"].to_numpy() - X @ np.array([a, b])
    r2 = 1 - (resid ** 2).sum() / ((pm["GAER"] - pm["GAER"].mean()) ** 2).sum()
    n_reg = -b / (a - beta_bar)
    imin = int(pm["U"].idxmin())
    est.append(dict(set=name, seed=seed, n=n, M=M, n_corrupt=int(corrupt.sum()), eta_true=corrupt.mean(),
                    n_corrupt_crossgroup=int(cross.sum()), beta_bar=beta_bar,
                    U_all=int(uall.sum()), lower_bound_ok=bool(uall.sum() <= corrupt.sum()),
                    eta_hat_Uall=uall.sum() / ((1 - beta_bar) * n),
                    U_min=int(pm["U"].min()), U_min_model=pm.loc[imin, "model"],
                    eta_hat_Umin=pm.loc[imin, "eta_hat_plugin"],
                    eta_hat_plugin_median=float(pm["eta_hat_plugin"].median()),
                    eta_hat_plugin_max=float(pm["eta_hat_plugin"].max()),
                    reg_intercept=a, reg_slope=b, reg_R2=r2, n_hat_reg=n_reg, eta_hat_reg=n_reg / n,
                    corr_GAER_invE=float(np.corrcoef(pm["GAER"], 1 / pm["E"])[0, 1])))

    for i in np.where(ex1_mask)[0]:
        j = int(np.where(unal[i])[0][0])
        ex1.append(dict(set=name, seed=seed, item=int(i), model=names[j], corrupt=bool(corrupt[i]),
                        n_models_err=int(n_err[i]), true=y[i], pred=preds[names[j]][i]))
    for i in np.where(uall & ~corrupt)[0]:
        uclean.append(dict(set=name, seed=seed, item=int(i), text=df["text"].iloc[i], true=y[i],
                           preds=";".join(str(preds[m][i]) for m in names)))

S = pd.DataFrame(summ); P = pd.DataFrame(prc); PM = pd.DataFrame(permodel); E = pd.DataFrame(est)
X1 = pd.DataFrame(ex1); UC = pd.DataFrame(uclean)
S.to_csv(os.path.join(u.OUT, "a1_summary.csv"), index=False)
P.to_csv(os.path.join(u.OUT, "a1_pr_curve.csv"), index=False)
PM.to_csv(os.path.join(u.OUT, "a1_per_model.csv"), index=False)
E.to_csv(os.path.join(u.OUT, "a1_estimators.csv"), index=False)
X1.to_csv(os.path.join(u.OUT, "a1_exactly_one.csv"), index=False)
UC.to_csv(os.path.join(u.OUT, "a1_uall_clean_items.csv"), index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(S.to_string())
seeds = P[P.set.str.match(r"seed\d$")]
agg = seeds.groupby(["detector", "k"]).agg(precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
                                           recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                                           recall_cross_mean=("recall_crossgroup", "mean"),
                                           tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum")).reset_index()
agg["precision_pooled"] = agg.tp / (agg.tp + agg.fp); agg["recall_pooled"] = agg.tp / (agg.tp + agg.fn)
agg.to_csv(os.path.join(u.OUT, "a1_pr_seeds_aggregate.csv"), index=False)
print(P[P.set == "seed42_7models"].to_string())
print(agg.to_string())
print(PM[PM.set == "seed42_7models"].to_string())
print(E.to_string())
print(X1.groupby(["set"]).agg(n=("item", "size"), corrupt=("corrupt", "sum")).to_string())
print(X1[X1.set.str.match(r"seed\d$")].groupby("model").agg(n=("item", "size"), corrupt=("corrupt", "sum")).to_string())
print(UC.to_string())
