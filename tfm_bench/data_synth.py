"""
Generation of *hard* synthetic TEXT data for 30-class classification.

To make the task realistic and NON-trivial (vs the first version that reached 100%
accuracy), we deliberately introduce ambiguity:

  - SHARED VOCAB   : a pool of words common to every category (weak signal).
  - CONFUSABLES    : groups of close categories that borrow vocabulary from each
                     other (e.g. football/basketball, chemistry/biology/physics...).
  - NOISE          : a high proportion of filler words + variable length.
  - LABEL NOISE    : a small fraction of training labels are corrupted.
  - IMBALANCE      : a mild per-class imbalance in the number of samples.

Result: models must actually learn -> accuracy < 100%, comparison is meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Category-specific vocabulary (deliberately small -> less signal than before).
TOPICS: dict[str, list[str]] = {
    "football":     ["scorer", "penalty", "championship"],
    "basketball":   ["dunk", "rebound", "court"],
    "cooking":      ["recipe", "spices", "baking"],
    "astronomy":    ["galaxy", "telescope", "nebula"],
    "computing":    ["algorithm", "compiler", "byte"],
    "finance":      ["dividend", "bond", "portfolio"],
    "medicine":     ["diagnosis", "symptom", "prescription"],
    "music":        ["melody", "score", "orchestra"],
    "cinema":       ["director", "screenplay", "editing"],
    "gardening":    ["compost", "vegetable", "pruning"],
    "automotive":   ["fuel", "bodywork", "brakes"],
    "travel":       ["customs", "itinerary", "excursion"],
    "fashion":      ["designer", "catwalk", "tailoring"],
    "ecology":      ["biodiversity", "carbon", "renewable"],
    "history":      ["dynasty", "archive", "monarch"],
    "law":          ["court", "verdict", "plaintiff"],
    "physics":      ["quantum", "acceleration", "field"],
    "chemistry":    ["solvent", "catalyst", "bond"],
    "biology":      ["enzyme", "membrane", "mutation"],
    "photography":  ["aperture", "framing", "tripod"],
    "agriculture":  ["seed", "irrigation", "harvest"],
    "education":    ["textbook", "diploma", "lesson"],
    "realestate":   ["mortgage", "lease", "notary"],
    "videogames":   ["controller", "console", "graphics"],
    "aviation":     ["turbine", "takeoff", "altitude"],
    "marine":       ["anchor", "cargo", "tide"],
    "robotics":     ["actuator", "trajectory", "calibration"],
    "wine":         ["grape", "vintage", "vineyard"],
    "weather":      ["anticyclone", "precipitation", "gust"],
    "archaeology":  ["excavation", "stratigraphy", "dating"],
}

CATEGORIES = list(TOPICS.keys())

# Groups of "confusable" categories: they share extra vocabulary.
CONFUSABLES: dict[str, list[str]] = {
    "sport":     ["football", "basketball"],
    "science":   ["physics", "chemistry", "biology", "astronomy"],
    "tech":      ["computing", "robotics", "videogames"],
    "nature":    ["gardening", "agriculture", "ecology", "weather"],
    "transport": ["automotive", "aviation", "marine", "travel"],
    "culture":   ["music", "cinema", "fashion", "photography"],
    "knowledge": ["history", "archaeology", "education", "law"],
    "money":     ["finance", "realestate"],
    "health":    ["medicine"],
    "food":      ["cooking", "wine"],
}
# Words shared within a confusable group (ambiguous signal).
GROUP_WORDS = {
    "sport":     ["team", "score", "coach", "competition", "victory"],
    "science":   ["measurement", "theory", "experiment", "data", "model"],
    "tech":      ["system", "screen", "data", "speed", "version"],
    "nature":    ["season", "water", "soil", "growth", "species"],
    "transport": ["trip", "speed", "engine", "passenger", "safety"],
    "culture":   ["audience", "style", "creation", "stage", "image"],
    "knowledge": ["document", "period", "society", "study", "text"],
    "money":     ["value", "contract", "risk", "investment", "rate"],
    "health":    ["care", "exam", "health", "study", "result"],
    "food":      ["taste", "flavor", "table", "preparation", "quality"],
}
# Reverse index: category -> its group.
CAT2GROUP = {cat: g for g, cats in CONFUSABLES.items() for cat in cats}

# Pool of VERY common words (noise, present everywhere, no signal).
SHARED_POOL = [
    "the", "a", "an", "and", "with", "for", "in", "on", "very",
    "today", "according", "new", "big", "small", "this", "that",
    "thing", "moment", "part", "way", "type", "example", "case", "point",
    "level", "group", "set", "element", "question", "answer",
]


def _make_sentence(rng: np.random.Generator, cat: str) -> str:
    """Ambiguous sentence: few topic keywords, lots of noise + group words."""
    group = CAT2GROUP[cat]
    own = TOPICS[cat]
    grp = GROUP_WORDS[group]

    n_own = rng.integers(1, 3)      # only 1 to 2 truly discriminative words
    n_grp = rng.integers(1, 4)      # 1 to 3 ambiguous group words
    n_noise = rng.integers(4, 9)    # 4 to 8 noise words (dominates the sentence)

    words = list(rng.choice(own, size=n_own, replace=True))
    words += list(rng.choice(grp, size=n_grp, replace=True))
    words += list(rng.choice(SHARED_POOL, size=n_noise, replace=True))

    # 25% of the time: borrow a keyword from a sibling category (a real trap).
    siblings = [c for c in CONFUSABLES[group] if c != cat]
    if siblings and rng.random() < 0.25:
        intruder = rng.choice(siblings)
        words.append(rng.choice(TOPICS[intruder]))

    rng.shuffle(words)
    return " ".join(words)


def build_dataset(
    n_per_class: int = 60,
    label_noise: float = 0.05,
    imbalance: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a hard ['text', 'category'] DataFrame.

    n_per_class : average number of samples per class.
    label_noise : fraction of corrupted (random) training labels.
    imbalance   : if True, apply a mild per-class imbalance.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cat in CATEGORIES:
        n = n_per_class
        if imbalance:
            n = int(n_per_class * rng.uniform(0.6, 1.4))  # imbalance of +/-40%
        for _ in range(n):
            rows.append({"text": _make_sentence(rng, cat), "category": cat})

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Label noise: corrupt a fraction of the labels (at random).
    if label_noise > 0:
        n_noisy = int(len(df) * label_noise)
        idx = rng.choice(len(df), size=n_noisy, replace=False)
        df.loc[idx, "category"] = rng.choice(CATEGORIES, size=n_noisy)

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(f"{len(df)} samples, {df['category'].nunique()} classes")
    print(df["category"].value_counts().describe())
    print(df.head(10).to_string(index=False))
