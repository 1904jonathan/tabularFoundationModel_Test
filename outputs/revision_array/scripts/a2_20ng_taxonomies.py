"""A2 - 20 Newsgroups: cross-model consistency under the hypothesised taxonomy, and plausible-but-wrong
taxonomies as negative controls.

Input: outputs/revision/e4_predictions.csv (1,000 test documents, 8 models, fixed settings, d = 64).
Taxonomies (all with the group sizes 5,4,4,3,3,1 of the hypothesised one, so the size-preserving
permutation null is the same for all of them):
  H   hypothesised (newsgroup hierarchy, fixed before any model was run; as in E4)
  W1  alphabetical blocks of the full newsgroup names (sizes 5,4,4,3,3,1 in this order)
  W2  alphabetical blocks of the leaf names (last dotted component; ties broken by full name)
  W3  half-swapped: in every group of size >= 2, floor(size/2) members are moved to other groups by a
      random size-preserving derangement (1,000 draws; draw 0 is also reported as a single taxonomy)
  W4  cross-cutting: every new group takes at most one class from each hypothesised group
For each taxonomy and model: GAER, chance level GAER0, enrichment, size-preserving permutation p
(B = 2,000), and a paired exact test against H on the same errors (errors aligned under H only vs
under T only; two-sided binomial).
Outputs: a2_20ng_consistency.csv, a2_20ng_exactly_one.csv, a2_20ng_taxonomies.csv,
         a2_20ng_halfswap_draws.csv, a2_20ng_taxonomy_defs.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import array_utils as u

B = 2000
d = pd.read_csv(os.path.join(u.REV, "e4_predictions.csv"))
y = d["true"].to_numpy()
models = [c[5:] for c in d.columns if c.startswith("pred_")]
preds = {m: d[f"pred_{m}"].to_numpy() for m in models}
H = u.GMAP20
classes = sorted(H)
sizes = [5, 4, 4, 3, 3, 1]


def blocks(order, names=("G1", "G2", "G3", "G4", "G5", "G6")):
    g, i = {}, 0
    for nm, s in zip(names, sizes):
        for c in order[i:i + s]:
            g[c] = nm
        i += s
    return g


W1 = blocks(sorted(classes))
W2 = blocks(sorted(classes, key=lambda c: (c.split(".")[-1], c)))


def halfswap(rng):
    groups = {g: list(cs) for g, cs in u.TAX20.items()}
    while True:
        moved, slots = [], []
        for g, cs in groups.items():
            k = len(cs) // 2
            pick = list(rng.choice(cs, size=k, replace=False)) if k else []
            moved += [(c, g) for c in pick]; slots += [g] * k
        perm = rng.permutation(len(slots))
        if all(slots[perm[i]] != moved[i][1] for i in range(len(moved))):
            T = dict(H)
            for i, (c, _) in enumerate(moved):
                T[c] = slots[perm[i]]
            return T


def crosscut():
    # new groups of sizes 5,4,4,3,3,1 taking at most one class from each hypothesised group:
    # deal the hypothesised groups round-robin (largest first) into the new groups.
    order = [c for g in ["comp", "rec", "sci", "politics", "religion", "misc"] for c in u.TAX20[g]]
    new = {f"X{j}": [] for j in range(6)}
    cap = dict(zip(new, sizes))
    for c in order:
        for gname in new:  # first new group with room and no member of c's hypothesised group
            if len(new[gname]) < cap[gname] and all(H[o] != H[c] for o in new[gname]):
                new[gname].append(c); break
        else:
            raise RuntimeError("cross-cutting assignment failed")
    return {c: g for g, cs in new.items() for c in cs}


W4 = crosscut()
rng = np.random.default_rng(0)
W3draws = [halfswap(rng) for _ in range(1000)]
TAX = {"H": H, "W1_alpha_full": W1, "W2_alpha_leaf": W2, "W3_halfswap_draw0": W3draws[0], "W4_crosscut": W4}
for nm, T in TAX.items():
    assert sorted(pd.Series(T).value_counts().tolist()) == sorted(sizes), nm
pd.DataFrame([dict(taxonomy=nm, group=g, classes=", ".join(sorted(c for c in T if T[c] == g)))
              for nm, T in TAX.items() for g in sorted(set(T.values()))]).to_csv(
    os.path.join(u.OUT, "a2_20ng_taxonomy_defs.csv"), index=False)

cidx = {c: i for i, c in enumerate(classes)}
yi = np.array([cidx[c] for c in y])
Pi = {m: np.array([cidx[c] for c in preds[m]]) for m in models}
null_rng_seed = 0


def eval_tax(T, m, perm=True):
    g = np.array([T[c] for c in classes])
    e = yi != Pi[m]
    al = g[yi[e]] == g[Pi[m][e]]
    G = float(al.mean())
    G0 = float(u.chance_ingroup(np.array(classes)[yi[e]], T).mean())
    out = dict(GAER=G, GAER0=G0, enrichment=G / G0, n_err=int(e.sum()))
    if perm:
        r = np.random.default_rng(null_rng_seed)
        null = np.array([np.mean((pg := r.permutation(g))[yi[e]] == pg[Pi[m][e]]) for _ in range(B)])
        out.update(perm_p95=float(np.percentile(null, 95)), perm_p=float((1 + np.sum(null >= G - 1e-12)) / (B + 1)))
    gh = np.array([H[c] for c in classes])
    alH = gh[yi[e]] == gh[Pi[m][e]]
    a, b = int((alH & ~al).sum()), int((~alH & al).sum())
    out.update(aligned_H_only=a, aligned_T_only=b, paired_p_vs_H=float(binomtest(a, a + b).pvalue) if a + b else 1.0)
    return out, al


rows = []
for nm, T in TAX.items():
    for m in models:
        r, _ = eval_tax(T, m)
        rows.append(dict(taxonomy=nm, model=m, **r))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(u.OUT, "a2_20ng_taxonomies.csv"), index=False)

# all 1,000 half-swapped draws (no permutation null per draw; compare with the H null's 95th percentile)
p95H = R[R.taxonomy == "H"].set_index("model")["perm_p95"]
drows = []
for t, T in enumerate(W3draws):
    for m in models:
        r, _ = eval_tax(T, m, perm=False)
        drows.append(dict(draw=t, model=m, GAER=r["GAER"], enrichment=r["enrichment"],
                          above_null_p95=r["GAER"] > p95H[m], paired_p_vs_H=r["paired_p_vs_H"],
                          worse_than_H=(r["aligned_H_only"] > r["aligned_T_only"]) and r["paired_p_vs_H"] < 0.05))
D = pd.DataFrame(drows)
D.to_csv(os.path.join(u.OUT, "a2_20ng_halfswap_draws.csv"), index=False)

# cross-model consistency under each taxonomy
crows, ex1 = [], []
for nm, T in TAX.items():
    err, unal, names = u.error_matrices(y, preds, T)
    nu = unal.sum(1); ne = err.sum(1); M = len(names)
    crows.append(dict(taxonomy=nm, n=len(y), M=M, E_all=int((ne == M).sum()), U_all=int((nu == M).sum()),
                      A_all=int(((ne == M) & (unal.sum(1) == 0)).sum()), U_exactly1=int((nu == 1).sum()),
                      U_any=int((nu >= 1).sum()),
                      **{f"U_ge{k}": int((nu >= k).sum()) for k in range(1, M + 1)}))
    if nm == "H":
        for i in np.where(nu == 1)[0]:
            j = int(np.where(unal[i])[0][0])
            ex1.append(dict(item=int(i), model=names[j], n_models_err=int(ne[i])))
C = pd.DataFrame(crows)
C.to_csv(os.path.join(u.OUT, "a2_20ng_consistency.csv"), index=False)
X1 = pd.DataFrame(ex1); X1.to_csv(os.path.join(u.OUT, "a2_20ng_exactly_one.csv"), index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(pd.read_csv(os.path.join(u.OUT, "a2_20ng_taxonomy_defs.csv")).to_string())
print(R.to_string())
print(R.groupby("taxonomy").agg(GAER_min=("GAER", "min"), GAER_max=("GAER", "max"), enr_min=("enrichment", "min"),
                                enr_max=("enrichment", "max"), p_max=("perm_p", "max"),
                                paired_p_max=("paired_p_vs_H", "max")).to_string())
print(D.groupby("model").agg(GAER_median=("GAER", "median"), GAER_q05=("GAER", lambda s: s.quantile(.05)),
                             GAER_q95=("GAER", lambda s: s.quantile(.95)), frac_above_null=("above_null_p95", "mean"),
                             frac_worse_than_H=("worse_than_H", "mean")).to_string())
print(C.to_string())
print(X1.groupby("model").size().to_string())
