import os
import pandas as pd

BASELINE_PATH = (
    "tests/generated/complete_baseline_training_pairs.tsv"
)

APPROX_PATH = (
    "tests/generated/approximate_training_pairs.tsv"
)

OUTPUT_PATH = (
    "tests/generated/combined_training_pairs.tsv"
)


def main():

    print("=" * 70)
    print("MERGING TRAINING DATASETS")
    print("=" * 70)

    print("\nLoading baseline dataset...")
    baseline = pd.read_csv(
        BASELINE_PATH,
        sep="\t",
    )

    print(
        f"Baseline pairs: {len(baseline):,}"
    )

    print("\nLoading approximate dataset...")
    approx = pd.read_csv(
        APPROX_PATH,
        sep="\t",
    )

    print(
        f"Approx pairs: {len(approx):,}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [baseline, approx],
        ignore_index=True,
    )

    before = len(combined)

    # Same S1 + candidate pair should only occur once
    combined = combined.drop_duplicates(
        subset=[
            "s1_entity_id",
            "candidate_entity_id",
        ],
        keep="first",
    )

    after = len(combined)

    print(
        f"\nDuplicates removed: "
        f"{before - after:,}"
    )

    # Shuffle
    combined = combined.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    positives = (
        combined["label"] == 1
    ).sum()

    negatives = (
        combined["label"] == 0
    ).sum()

    print("\n" + "=" * 70)
    print("COMBINED DATASET")
    print("=" * 70)

    print(
        f"Total pairs    : {len(combined):,}"
    )

    print(
        f"Positive pairs : {positives:,}"
    )

    print(
        f"Negative pairs : {negatives:,}"
    )

    print(
        f"Positive ratio : "
        f"{combined['label'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()