"""Taxonomy-aligned error analysis from a prediction file (stand-alone scorer).

Input
  --pred   CSV with a column `true` (observed labels) and one column `pred_<model>` per classifier
  --tax    JSON {group: [class, ...]} (the hypothesised taxonomy, fixed before looking at errors)
  --alt    optional JSON files with alternative taxonomies (same format) to compare with --tax
  --out    output prefix (default: scorer)
Output
  <out>_models.csv   per classifier: accuracy, errors, GAER, chance level GAER0, enrichment, unaligned
                     errors U, permuted-taxonomy p-value (size-preserving, --B draws), plug-in eta_hat_m
  <out>_items.csv    per item: number of classifiers that err (e) and that make an unaligned error (u),
                     flags u == M (data-set defect: label or representation) and u == 1 (classifier
                     limitation, with the classifier's name)
  <out>_summary.txt  |U_all|, |U_1| by classifier, eta_hat_all = |U_all| / ((1 - beta_bar) n), and the
                     paired exact test of --tax against every --alt taxonomy
Example (20 Newsgroups predictions of the paper):
  python taxonomy_error_scorer.py --pred outputs/revision/e4_predictions.csv --tax tax20.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def load_tax(path):
    t = json.load(open(path, encoding="utf-8"))
    return {c: g for g, cs in t.items() for c in cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--tax", required=True)
    ap.add_argument("--alt", nargs="*", default=[]); ap.add_argument("--out", default="scorer")
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = pd.read_csv(a.pred)
    gm = load_tax(a.tax)
    y = d["true"].astype(str).to_numpy()
    models = [c[5:] for c in d.columns if c.startswith("pred_")]
    P = np.column_stack([d[f"pred_{m}"].astype(str).to_numpy() for m in models])
    classes = sorted(gm); K = len(classes); idx = {c: i for i, c in enumerate(classes)}
    missing = set(y) | set(P.ravel())
    missing -= set(gm)
    if missing:
        raise SystemExit(f"classes missing from the taxonomy: {sorted(missing)}")
    sizes = pd.Series(gm).value_counts().to_dict()
    beta = np.array([(sizes[gm[c]] - 1) / (K - 1) for c in y])
    g = np.array([gm[c] for c in classes])
    yi = np.array([idx[c] for c in y]); Pi = np.vectorize(idx.get)(P)
    err = Pi != yi[:, None]
    unal = err & (g[Pi] != g[yi][:, None])
    n, M = err.shape
    rng = np.random.default_rng(a.seed)
    perms = [rng.permutation(g) for _ in range(a.B)]
    rows = []
    for j, m in enumerate(models):
        e = err[:, j]; E = int(e.sum())
        G = 1 - unal[:, j].sum() / E if E else np.nan
        G0 = beta[e].mean() if E else np.nan
        null = np.array([np.mean(pg[yi[e]] == pg[Pi[e, j]]) for pg in perms]) if E else np.array([np.nan])
        rows.append(dict(model=m, accuracy=1 - E / n, errors=E, GAER=G, GAER0=G0, enrichment=G / G0,
                         unaligned=int(unal[:, j].sum()), perm_p=(1 + np.sum(null >= G - 1e-12)) / (a.B + 1),
                         eta_hat_plugin=unal[:, j].sum() / ((1 - G0) * n)))
    pd.DataFrame(rows).to_csv(f"{a.out}_models.csv", index=False)
    u = unal.sum(1)
    only = [models[int(np.where(unal[i])[0][0])] if u[i] == 1 else "" for i in range(n)]
    items = pd.DataFrame(dict(item=np.arange(n), true=y, e=err.sum(1), u=u, data_set_defect=u == M,
                              classifier_limitation_of=only))
    items.to_csv(f"{a.out}_items.csv", index=False)
    lines = [f"n = {n}, classifiers = {M}, beta_bar = {beta.mean():.4f}",
             f"|U_all| = {int((u == M).sum())}  (eta_hat_all = {(u == M).sum() / ((1 - beta.mean()) * n):.4f})",
             f"|U_1| = {int((u == 1).sum())}: " + ", ".join(f"{k} {v}" for k, v in
                                                          pd.Series([o for o in only if o]).value_counts().items())]
    for alt in a.alt:
        ga = load_tax(alt); gA = np.array([ga[c] for c in classes])
        for j, m in enumerate(models):
            e = err[:, j]
            alH = g[yi[e]] == g[Pi[e, j]]; alT = gA[yi[e]] == gA[Pi[e, j]]
            x, z = int((alH & ~alT).sum()), int((~alH & alT).sum())
            p = binomtest(x, x + z).pvalue if x + z else 1.0
            lines.append(f"{alt} vs --tax, {m}: aligned only under --tax {x}, only under alt {z}, exact p = {p:.3g}")
    open(f"{a.out}_summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
