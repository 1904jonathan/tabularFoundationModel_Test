"""Collect the A1-A3 outputs into outputs/revision_array/results_array.md (every number quoted in the
Array manuscript that is new relative to outputs/revision/results.md comes from this file)."""
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon

import array_utils as u

O = u.OUT
rd = lambda f: pd.read_csv(os.path.join(O, f))  # noqa: E731
S, P, PM, E = rd("a1_summary.csv"), rd("a1_pr_curve.csv"), rd("a1_per_model.csv"), rd("a1_estimators.csv")
AG, X1 = rd("a1_pr_seeds_aggregate.csv"), rd("a1_exactly_one.csv")
T, D, C, X20 = rd("a2_20ng_taxonomies.csv"), rd("a2_20ng_halfswap_draws.csv"), rd("a2_20ng_consistency.csv"), rd(
    "a2_20ng_exactly_one.csv")
L, AU = rd("a3_length_stats.csv"), rd("a3_audit_unblinded.csv")
seeds = E[E.set.str.match(r"seed\d$")]
out = []
w = out.append

w("# Array revision analyses (A1-A3)\n")
w("Scripts: `outputs/revision_array/scripts/` (`array_utils.py`, `a1_crossmodel_synth.py`, `a2_20ng_taxonomies.py`, "
  "`a3_20ng_audit.py prepare|score`, `make_results_array.py`). All analyses are post hoc on the saved predictions of "
  "`outputs/revision/` (no model re-trained). Logs: `a1_log.txt`, `a2_log.txt`, `a3_prepare_log.txt`, `a3_score_log.txt`.\n")
w("Definitions. For item i and model m, an *unaligned error* is a misclassification whose predicted class lies outside "
  "the (hypothesised) group of the observed label. `n_unal(i)` = number of models with an unaligned error on i. "
  "`U_all` = items with n_unal = M (all models), `U_exactly1` = items with n_unal = 1. Detector `unaligned>=k` flags "
  "items with n_unal >= k; baseline `misclassified>=k` flags items misclassified by >= k models (consensus filter). "
  "Ground truth (testbed only): corrupted = observed label != clean label (generator rebuilt with label_noise = 0).\n")

w("## A1. Testbed: cross-model consistency\n")
w(u.md_table(S[["set", "n", "M", "n_corrupt", "n_corrupt_crossgroup", "n_corrupt_ingroup", "U_all", "U_all_corrupt",
                "U_exactly1", "U_exactly1_corrupt", "U_any", "U_any_corrupt", "E_all", "E_all_corrupt"]]))
tot = seeds[["n", "n_corrupt", "n_corrupt_crossgroup", "U_all"]].sum()
s10 = S[S.set.str.match(r"seed\d$")]
w(f"\nSeeds 0-9 pooled: n = {int(tot.n)}, corrupted = {int(tot.n_corrupt)} (cross-group {int(tot.n_corrupt_crossgroup)}, "
  f"in-group {int(s10.n_corrupt_ingroup.sum())} = {s10.n_corrupt_ingroup.sum() / tot.n_corrupt:.3f}); "
  f"U_all = {int(s10.U_all.sum())}, of which corrupted {int(s10.U_all_corrupt.sum())}; "
  f"U_exactly1 = {int(s10.U_exactly1.sum())}, of which corrupted {int(s10.U_exactly1_corrupt.sum())}; "
  f"E_all (misclassified by all 8) = {int(s10.E_all.sum())}, of which corrupted {int(s10.E_all_corrupt.sum())}. "
  f"Expected in-group share of uniform corruptions = mean beta = {s10.beta_bar.mean():.3f}.\n")
w("### Precision / recall, seed 42 (7 models of Table 2)\n")
w(u.md_table(P[P.set == "seed42_7models"][["detector", "k", "flagged", "tp", "fp", "fn", "precision", "recall"]]))
w("\n### Precision / recall, seeds 0-9 (8 models): mean +- sd over seeds and pooled\n")
w(u.md_table(AG))
w("\n### Items unaligned for exactly one model, by model\n")
x = X1[X1.set.str.match(r"seed\d$")].groupby("model").agg(n=("item", "size"), corrupt=("corrupt", "sum")).reset_index()
x42 = X1[X1.set == "seed42_7models"].groupby("model").agg(n_seed42=("item", "size")).reset_index()
w(u.md_table(x.merge(x42, on="model", how="left").fillna(0)))

w("\n## A1. Mixture model: dilution and noise-rate estimators\n")
w("### Seed 42 per-model decomposition (GAER = alpha*E_clean/E + beta_realised*E_corrupt/E)\n")
w(u.md_table(PM[PM.set == "seed42_7models"][["model", "E", "U", "GAER", "GAER0", "E_clean", "E_corrupt", "alpha_clean",
                                             "beta_realised", "GAER_pred_alpha1", "eta_hat_plugin"]]))
pm10 = PM[PM.set.str.match(r"seed\d$")]
dev = pm10.GAER - pm10.GAER_pred_alpha1
w(f"\nSeeds 0-9, 80 model-seed pairs: GAER - (1 - U_corrupt/E) [alpha = 1 prediction]: mean {dev.mean():.4f}, "
  f"mean |.| {dev.abs().mean():.4f}, max |.| {dev.abs().max():.4f}; alpha_clean range "
  f"{pm10.alpha_clean.min():.3f}-{pm10.alpha_clean.max():.3f}; realised beta on corrupted errors: mean "
  f"{pm10.beta_realised.mean():.3f}. Within-seed correlation of GAER with 1/E: mean {seeds.corr_GAER_invE.mean():.3f} "
  f"(range {seeds.corr_GAER_invE.min():.3f} to {seeds.corr_GAER_invE.max():.3f}).\n")
w("### Noise-rate estimators vs truth\n")
cols = ["set", "n", "n_corrupt", "eta_true", "U_all", "lower_bound_ok", "eta_hat_Uall", "U_min", "U_min_model",
        "eta_hat_Umin", "eta_hat_plugin_median", "eta_hat_plugin_max", "reg_R2", "n_hat_reg", "eta_hat_reg"]
w(u.md_table(E[cols], ".4f"))
summ = []
for c in ["eta_hat_Uall", "eta_hat_Umin", "eta_hat_plugin_median", "eta_hat_reg"]:
    dd = seeds[c] - seeds.eta_true
    summ.append(dict(estimator=c, mean_est=seeds[c].mean(), mean_true=seeds.eta_true.mean(), mean_bias=dd.mean(),
                     mean_abs_err=dd.abs().mean(), max_abs_err=dd.abs().max(),
                     rel_bias=(dd / seeds.eta_true).mean()))
SU = pd.DataFrame(summ); SU.to_csv(os.path.join(O, "a1_estimator_summary.csv"), index=False)
w("\nSeeds 0-9 summary:\n")
w(u.md_table(SU, ".4f"))
w(f"\nU_all <= n_corrupt held in {int(E.lower_bound_ok.sum())} of {len(E)} sets.\n")

w("## A2. 20 Newsgroups: wrong taxonomies\n")
w("Taxonomy definitions: `a2_20ng_taxonomy_defs.csv`.\n")
w(u.md_table(T[["taxonomy", "model", "GAER", "GAER0", "enrichment", "perm_p95", "perm_p", "aligned_H_only",
                "aligned_T_only", "paired_p_vs_H"]], ".4g"))
g = T.groupby("taxonomy").agg(GAER_min=("GAER", "min"), GAER_max=("GAER", "max"), enr_min=("enrichment", "min"),
                              enr_max=("enrichment", "max"), perm_p_min=("perm_p", "min"), perm_p_max=("perm_p", "max"),
                              n_models_perm_p_lt_05=("perm_p", lambda s: int((s < 0.05).sum())),
                              paired_p_max=("paired_p_vs_H", "max")).reset_index()
g.to_csv(os.path.join(O, "a2_20ng_taxonomies_summary.csv"), index=False)
w("\nSummary over the 8 models:\n")
w(u.md_table(g, ".4g"))
dg = D.groupby("model").agg(GAER_median=("GAER", "median"), GAER_q05=("GAER", lambda s: s.quantile(.05)),
                            GAER_q95=("GAER", lambda s: s.quantile(.95)), frac_above_H_null_p95=("above_null_p95", "mean"),
                            frac_sig_worse_than_H=("worse_than_H", "mean")).reset_index()
dg.to_csv(os.path.join(O, "a2_20ng_halfswap_summary.csv"), index=False)
w("\nW3, 1,000 random half-swapped taxonomies:\n")
w(u.md_table(dg))
w(f"\nOver all 8,000 model-draw pairs: GAER median {D.GAER.median():.3f}, above the permutation-null 95th percentile in "
  f"{D.above_null_p95.mean():.3f}, significantly worse than H (paired exact test, p < 0.05) in {D.worse_than_H.mean():.3f}.\n")
w("\n### Cross-model consistency on 20NG under each taxonomy (8 models, 1,000 documents)\n")
w(u.md_table(C))
w("\nItems unaligned for exactly one model under H, by model: " +
  ", ".join(f"{k} {v}" for k, v in X20.groupby('model').size().sort_values(ascending=False).items()) + "\n")

# noise / defect-rate estimate on 20NG (no ground truth): eta_hat = U_all / ((1 - beta_bar) n)
d20 = pd.read_csv(os.path.join(u.REV, "e4_predictions.csv"))
beta20 = float(u.chance_ingroup(d20["true"].to_numpy(), u.GMAP20).mean())
UH = int(C[C.taxonomy == "H"].U_all.iloc[0])
TH = T[T.taxonomy == "H"].copy()
TH["U"] = ((1 - TH.GAER) * TH.n_err).round().astype(int)
TH["eta_hat_plugin"] = TH.U / ((1 - TH.GAER0) * len(d20))
pd.DataFrame([dict(beta_bar=beta20, U_all=UH, n=len(d20), eta_hat_Uall=UH / ((1 - beta20) * len(d20)),
                   eta_hat_plugin_min=TH.eta_hat_plugin.min(), eta_hat_plugin_max=TH.eta_hat_plugin.max())]).to_csv(
    os.path.join(O, "a2_20ng_eta_hat.csv"), index=False)
w(f"\n20NG defect-rate estimate: beta_bar = {beta20:.4f}, U_all = {UH}, eta_hat_Uall = U_all/((1-beta_bar) n) = "
  f"{UH / ((1 - beta20) * len(d20)):.4f}; per-model plug-in U_m/((1-GAER0_m) n) ranges "
  f"{TH.eta_hat_plugin.min():.3f}-{TH.eta_hat_plugin.max():.3f} (alpha_m far from 1 on real data, Lemma 2).\n")

w("## A3. 20 Newsgroups item audit\n")
w("Length (words) of U_all (unaligned for all 8 models), A_all (misclassified by all 8, every error in-group), all test "
  "documents and documents correct for all 8:\n")
w(u.md_table(L))
ct = pd.crosstab(AU["set"], AU["judgement"])
w("\nBlind audit (15 random U_all + 15 random A_all documents, shuffled; judged by the first author from a 400-character "
  "excerpt and the observed newsgroup, without set membership or predictions):\n")
w(u.md_table(ct.reset_index()))
a = int((AU[AU.set == "U_all"].judgement == "on-topic").sum()); b = int((AU[AU.set == "A_all"].judgement == "on-topic").sum())
nU = int((AU.set == "U_all").sum()); nA = int((AU.set == "A_all").sum())
_, pf = fisher_exact([[a, nU - a], [b, nA - b]])
w(f"\nOn-topic: U_all {a}/{nU}, A_all {b}/{nA}; Fisher exact two-sided p = {pf:.2g}.\n")
w("\nPer-document judgements (unblinded):\n")
w(u.md_table(AU[["audit_id", "set", "observed_newsgroup", "majority_pred", "judgement", "note"]]))
open(os.path.join(O, "results_array.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
