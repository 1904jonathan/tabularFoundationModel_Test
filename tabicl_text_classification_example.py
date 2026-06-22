"""
Example: 30-class text classification with TabICL (tabular foundation model).

Pipeline:
  1) Generation of SYNTHETIC data whose raw features are TEXT
     (each sample = a short sentence) split into 30 categories.
  2) Text -> numbers conversion with a clean, standard method:
        TF-IDF (weighted bag of words)  ->  TruncatedSVD (dimensionality reduction, "LSA").
     The vectorizer is fitted ONLY on the train set, then applied to test (no leakage).
  3) A set of UNLABELED rows is held out for testing (inference).
  4) TabICL (TabICLClassifier) learns + predicts in a single forward (in-context learning).
  5) Results (predictions + probabilities) are saved to CSV files.

TabICL natively handles >10 classes (support_many_classes=True), so 30 categories are OK.
Model repo: https://github.com/soda-inria/tabicl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from tabicl import TabICLClassifier


# --------------------------------------------------------------------------- #
# 1) Generate synthetic data: raw features = TEXT, 30 classes
# --------------------------------------------------------------------------- #
RNG = np.random.default_rng(42)

# 30 thematic categories, each with a distinctive vocabulary.
TOPICS: dict[str, list[str]] = {
    "football":     ["match", "scorer", "goalkeeper", "stadium", "championship", "penalty", "referee"],
    "basketball":   ["basket", "dunk", "rebound", "court", "freethrow", "forward", "playoffs"],
    "cooking":      ["recipe", "oven", "spices", "sauce", "baking", "dessert", "ingredient"],
    "astronomy":    ["galaxy", "star", "telescope", "orbit", "planet", "nebula", "comet"],
    "computing":    ["server", "algorithm", "compiler", "network", "processor", "software", "byte"],
    "finance":      ["stock", "market", "dividend", "bond", "portfolio", "interest", "trading"],
    "medicine":     ["patient", "diagnosis", "symptom", "treatment", "surgery", "vaccine", "prescription"],
    "music":        ["melody", "concert", "guitar", "rhythm", "score", "orchestra", "singer"],
    "cinema":       ["film", "director", "screenplay", "actor", "camera", "festival", "editing"],
    "gardening":    ["seed", "watering", "compost", "vegetable", "flower", "fertilizer", "pruning"],
    "automotive":   ["engine", "fuel", "tire", "brakes", "bodywork", "gearbox", "track"],
    "travel":       ["airport", "hotel", "suitcase", "customs", "itinerary", "passport", "excursion"],
    "fashion":      ["fabric", "catwalk", "designer", "trend", "collection", "accessory", "tailor"],
    "ecology":      ["climate", "recycling", "biodiversity", "carbon", "pollution", "renewable", "ecosystem"],
    "history":      ["empire", "revolution", "dynasty", "battle", "archive", "monarch", "treaty"],
    "law":          ["court", "lawyer", "verdict", "contract", "law", "appeal", "plaintiff"],
    "physics":      ["particle", "energy", "field", "quantum", "force", "mass", "acceleration"],
    "chemistry":    ["molecule", "reaction", "solvent", "acid", "catalyst", "atom", "bond"],
    "biology":      ["cell", "gene", "protein", "enzyme", "membrane", "mutation", "organism"],
    "photography":  ["lens", "aperture", "exposure", "framing", "pixel", "tripod", "light"],
    "agriculture":  ["harvest", "tractor", "seed", "irrigation", "livestock", "field", "crop"],
    "education":    ["student", "teacher", "exam", "textbook", "diploma", "classroom", "lesson"],
    "realestate":   ["rent", "apartment", "mortgage", "agency", "lease", "land", "notary"],
    "videogames":   ["controller", "console", "level", "character", "score", "graphics", "player"],
    "aviation":     ["pilot", "altitude", "turbine", "runway", "cabin", "takeoff", "radar"],
    "marine":       ["ship", "port", "sail", "anchor", "crew", "cargo", "tide"],
    "robotics":     ["sensor", "actuator", "motor", "autonomous", "arm", "trajectory", "calibration"],
    "wine":         ["grape", "vintage", "barrel", "tasting", "cellar", "aroma", "vineyard"],
    "weather":      ["cloud", "storm", "anticyclone", "precipitation", "humidity", "front", "gust"],
    "archaeology":  ["excavation", "remains", "burial", "artifact", "stratigraphy", "dating", "site"],
}

CATEGORIES = list(TOPICS.keys())
assert len(CATEGORIES) == 30, f"expected 30 categories, found {len(CATEGORIES)}"

# Common "stop" words to make sentences realistic and noisy.
FILLER = ["the", "a", "an", "and", "with", "for", "in", "on", "very",
          "today", "according", "new", "big", "small", "this", "that", "these"]


def make_sentence(topic: str) -> str:
    """Build a synthetic sentence mixing topic keywords + noise."""
    keywords = TOPICS[topic]
    n_kw = RNG.integers(3, 6)             # 3 to 5 topic keywords
    n_noise = RNG.integers(2, 5)          # 2 to 4 filler words
    words = list(RNG.choice(keywords, size=n_kw, replace=True))
    words += list(RNG.choice(FILLER, size=n_noise, replace=True))
    RNG.shuffle(words)
    return " ".join(words)


def build_dataset(n_per_class: int) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_class):
            rows.append({"text": make_sentence(cat), "category": cat})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffle


# Full labeled dataset, then hold out an "unlabeled" set for inference.
full_df = build_dataset(n_per_class=50)          # 30 * 50 = 1500 samples
train_df, unlabeled_df = train_test_split(
    full_df, test_size=0.2, random_state=42, stratify=full_df["category"]
)
train_df = train_df.reset_index(drop=True)
unlabeled_df = unlabeled_df.reset_index(drop=True)

# Keep the true labels of the test set ONLY to evaluate at the end.
true_test_labels = unlabeled_df["category"].copy()

# Test rows are given to the prediction pipeline WITHOUT labels.
unlabeled_texts = unlabeled_df.drop(columns=["category"])

print(f"Train (labeled)   : {len(train_df)} rows")
print(f"Test  (unlabeled) : {len(unlabeled_texts)} rows")
print(f"Number of categories : {len(CATEGORIES)}")


# --------------------------------------------------------------------------- #
# 2) TEXT -> NUMBERS conversion: TF-IDF + TruncatedSVD (LSA)
#    Fitted on the TRAIN only, then applied to the TEST -> no leakage.
# --------------------------------------------------------------------------- #
N_COMPONENTS = 50  # dimension of the numeric space given to TabICL

text2num = make_pipeline(
    TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
    TruncatedSVD(n_components=N_COMPONENTS, random_state=42),
    Normalizer(copy=False),  # normalize the LSA vectors (good practice)
)

X_train = text2num.fit_transform(train_df["text"])
y_train = train_df["category"].to_numpy()
X_test = text2num.transform(unlabeled_texts["text"])

# Give readable feature names (useful for interpretation / saving).
feat_names = [f"lsa_{i:02d}" for i in range(N_COMPONENTS)]
X_train = pd.DataFrame(X_train, columns=feat_names)
X_test = pd.DataFrame(X_test, columns=feat_names)

print(f"\nNumeric features : {X_train.shape[1]} (TF-IDF -> SVD)")
print(f"X_train shape : {X_train.shape} | X_test shape : {X_test.shape}")


# --------------------------------------------------------------------------- #
# 3 & 4) TabICL: fit + predict (in-context learning, single forward)
# --------------------------------------------------------------------------- #
clf = TabICLClassifier(
    n_estimators=8,
    support_many_classes=True,   # required for >10 classes (here 30)
    random_state=42,
    device=None,                 # auto: CUDA if available, else CPU
)

print("\nTraining (fit) TabICL...")
clf.fit(X_train, y_train)        # downloads the checkpoint on first use

print("Predicting (predict) on UNLABELED rows...")
pred_labels = clf.predict(X_test)
pred_proba = clf.predict_proba(X_test)   # (n_test, 30)


# --------------------------------------------------------------------------- #
# 5) Save the results
# --------------------------------------------------------------------------- #
# a) Readable predictions: original text + predicted category + confidence.
results = unlabeled_texts.copy()
results["predicted_category"] = pred_labels
results["confidence"] = pred_proba.max(axis=1)
results.to_csv("tabicl_predictions.csv", index=False, encoding="utf-8-sig")

# b) Full per-class probabilities (n_test x 30 matrix).
proba_df = pd.DataFrame(pred_proba, columns=clf.classes_)
proba_df.insert(0, "text", unlabeled_texts["text"].to_numpy())
proba_df.to_csv("tabicl_probabilities.csv", index=False, encoding="utf-8-sig")

print("\nResults saved:")
print("  - tabicl_predictions.csv     (text + predicted category + confidence)")
print("  - tabicl_probabilities.csv   (probabilities for the 30 classes)")


# --------------------------------------------------------------------------- #
# Evaluation (the true test labels were held out, never seen by the model)
# --------------------------------------------------------------------------- #
acc = accuracy_score(true_test_labels, pred_labels)
f1 = f1_score(true_test_labels, pred_labels, average="macro")
print(f"\n=== Evaluation on the test set ===")
print(f"Accuracy        : {acc:.3f}")
print(f"F1-macro        : {f1:.3f}")
print("\nDetailed report:")
print(classification_report(true_test_labels, pred_labels, zero_division=0))

# Preview a few predictions
print("Preview (first 5 predictions):")
print(results[["text", "predicted_category", "confidence"]].head().to_string(index=False))
