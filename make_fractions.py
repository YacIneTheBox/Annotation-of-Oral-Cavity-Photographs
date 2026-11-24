"""Task B4: Create stratified label-fraction subsets (1%, 10%, 25%) from splits/train.csv.

Guarantees at least 1 image per class even at 1% so all 4 classes can be trained and evaluated.
"""

import os
import math
import pandas as pd

TRAIN_CSV = "splits/train.csv"
FRACTIONS = {
    "1pct": 0.01,
    "10pct": 0.10,
    "25pct": 0.25,
}
SEED = 42


def stratified_subset(df: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    target_total = max(len(df["label"].unique()), round(len(df) * frac))
    groups = df.groupby("label")

    # Allocate at least 1 sample per class, rest proportional to class frequency
    counts = {}
    for label, grp in groups:
        counts[label] = max(1, round(len(grp) * frac))

    # Adjust largest class if rounding pushed total slightly off target_total
    diff = target_total - sum(counts.values())
    largest_label = max(counts, key=lambda k: len(groups.get_group(k)))
    counts[largest_label] = max(1, counts[largest_label] + diff)

    parts = []
    for label, grp in groups:
        n = min(counts[label], len(grp))
        parts.append(grp.sample(n=n, random_state=seed))

    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Full train (100%): {len(train_df)} images")
    print(train_df["label"].value_counts().sort_index().to_dict())

    for name, frac in FRACTIONS.items():
        sub = stratified_subset(train_df, frac, seed=SEED)
        out_path = os.path.join("splits", f"train_{name}.csv")
        sub.to_csv(out_path, index=False)
        dist = sub["label"].value_counts().sort_index().to_dict()
        print(f"Saved {out_path}: {len(sub)} images | class counts: {dist}")


if __name__ == "__main__":
    main()
