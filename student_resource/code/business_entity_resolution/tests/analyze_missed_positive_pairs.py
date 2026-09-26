import pandas as pd
import numpy as np

PATH = "tests/generated/complete_baseline_training_pairs.tsv"

FEATURES = [
    "name_sequence_similarity",
    "core_name_sequence_similarity",
    "name_token_jaccard",
    "name_token_overlap",
    "address_sequence_similarity",
    "address_token_jaccard",
    "address_token_overlap",
    "address_number_overlap",
    "address_numeric_component_overlap",
    "address_location_overlap",
    "country_match",
]


def main():
    print("Loading training pairs...")

    df = pd.read_csv(PATH, sep="\t")

    # Only positive pairs
    positives = df[df["label"] == 1].copy()

    # A positive pair was recovered by our exact blocking if
    # ANY exact blocking rule matched.
    positives["exact_recovered"] = (
        (positives["name_exact"] == 1)
        | (positives["core_name_exact"] == 1)
        | (positives["address_exact"] == 1)
    )

    recovered = positives[positives["exact_recovered"]]
    missed = positives[~positives["exact_recovered"]]

    print("\n" + "=" * 70)
    print("MISSED POSITIVE PAIR ANALYSIS")
    print("=" * 70)

    print(f"\nTotal positive pairs : {len(positives):,}")
    print(f"Exact recovered      : {len(recovered):,}")
    print(f"Exact missed         : {len(missed):,}")

    if len(positives) > 0:
        print(
            f"Exact recovery rate  : "
            f"{len(recovered) / len(positives) * 100:.2f}%"
        )
        print(
            f"Missed rate           : "
            f"{len(missed) / len(positives) * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Feature comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE MEANS")
    print("=" * 70)

    print(
        f"\n{'Feature':45s}"
        f"{'Recovered':>12s}"
        f"{'Missed':>12s}"
        f"{'Difference':>12s}"
    )

    print("-" * 82)

    for feature in FEATURES:
        if feature not in df.columns:
            print(f"{feature:45s} MISSING")
            continue

        r = recovered[feature].mean()
        m = missed[feature].mean()

        print(
            f"{feature:45s}"
            f"{r:12.4f}"
            f"{m:12.4f}"
            f"{(m-r):12.4f}"
        )

    # ---------------------------------------------------------
    # Threshold analysis on missed pairs
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("WHAT DO MISSED PAIRS LOOK LIKE?")
    print("=" * 70)

    if len(missed) == 0:
        print("\nNo missed positive pairs.")
        return

    checks = [
        ("name_sequence_similarity", 0.80),
        ("name_sequence_similarity", 0.70),
        ("name_sequence_similarity", 0.60),
        ("core_name_sequence_similarity", 0.80),
        ("core_name_sequence_similarity", 0.70),
        ("core_name_sequence_similarity", 0.60),
        ("address_sequence_similarity", 0.80),
        ("address_sequence_similarity", 0.70),
        ("address_sequence_similarity", 0.60),
    ]

    print(
        f"\n{'Condition':45s}"
        f"{'Count':>10s}"
        f"{'Percent':>12s}"
    )

    print("-" * 70)

    for feature, threshold in checks:
        count = (missed[feature] >= threshold).sum()
        pct = count / len(missed) * 100

        print(
            f"{feature} >= {threshold:<5}"
            f"{count:>10,}"
            f"{pct:>11.2f}%"
        )

    # ---------------------------------------------------------
    # Address evidence
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ADDRESS EVIDENCE IN MISSED PAIRS")
    print("=" * 70)

    address_checks = [
        ("address_number_overlap", 0),
        ("address_numeric_component_overlap", 0),
        ("address_location_overlap", 0),
    ]

    for feature, threshold in address_checks:
        count = (missed[feature] > threshold).sum()
        pct = count / len(missed) * 100

        print(
            f"{feature} > {threshold}: "
            f"{count:,} / {len(missed):,} "
            f"({pct:.2f}%)"
        )

    # ---------------------------------------------------------
    # Country
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("COUNTRY MATCH")
    print("=" * 70)

    count = (missed["country_match"] == 1).sum()
    pct = count / len(missed) * 100

    print(
        f"Country match = 1: "
        f"{count:,} / {len(missed):,} "
        f"({pct:.2f}%)"
    )

    # ---------------------------------------------------------
    # Strong combined evidence
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINED EVIDENCE")
    print("=" * 70)

    conditions = {
        "name >= .70 AND address >= .70": (
            (missed["name_sequence_similarity"] >= 0.70)
            & (missed["address_sequence_similarity"] >= 0.70)
        ),
        "name >= .60 AND address >= .70": (
            (missed["name_sequence_similarity"] >= 0.60)
            & (missed["address_sequence_similarity"] >= 0.70)
        ),
        "name >= .70 AND address number overlap": (
            (missed["name_sequence_similarity"] >= 0.70)
            & (missed["address_number_overlap"] > 0)
        ),
        "address >= .80 AND number overlap": (
            (missed["address_sequence_similarity"] >= 0.80)
            & (missed["address_number_overlap"] > 0)
        ),
    }

    for label, condition in conditions.items():
        count = condition.sum()
        pct = count / len(missed) * 100

        print(
            f"{label:45s}"
            f"{count:>8,} "
            f"({pct:6.2f}%)"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()