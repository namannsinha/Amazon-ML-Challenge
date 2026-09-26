import joblib
import numpy as np
import pandas as pd


DATA_PATH = (
    "tests/generated/"
    "complete_baseline_training_pairs.tsv"
)

MODEL_PATH = (
    "tests/generated/"
    "baseline_logistic_model.joblib"
)

RANDOM_SEED = 42
THRESHOLD = 0.80


def main():

    print("Loading data and model...")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
    )

    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    X = df[feature_columns].astype(float)
    y = df["label"].astype(int)

    # --------------------------------------------------
    # Reproduce the exact S1-level validation split
    # --------------------------------------------------

    unique_s1 = (
        df["s1_entity_id"]
        .drop_duplicates()
        .sample(
            frac=1.0,
            random_state=RANDOM_SEED,
        )
        .tolist()
    )

    split_index = int(
        len(unique_s1) * 0.80
    )

    validation_s1 = set(
        unique_s1[split_index:]
    )

    validation_mask = (
        df["s1_entity_id"]
        .isin(validation_s1)
    )

    val_df = df[validation_mask].copy()

    X_val = X[validation_mask]
    y_val = y[validation_mask]

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    val_df["probability"] = (
        model.predict_proba(X_val)[:, 1]
    )

    val_df["prediction"] = (
        val_df["probability"] >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------
    # False negatives
    # --------------------------------------------------

    false_negatives = val_df[
        (val_df["label"] == 1)
        & (val_df["prediction"] == 0)
    ].copy()

    # --------------------------------------------------
    # False positives
    # --------------------------------------------------

    false_positives = val_df[
        (val_df["label"] == 0)
        & (val_df["prediction"] == 1)
    ].copy()

    print("\n" + "=" * 80)
    print("MODEL ERROR SUMMARY")
    print("=" * 80)

    print(
        f"Validation pairs: "
        f"{len(val_df):,}"
    )

    print(
        f"False negatives: "
        f"{len(false_negatives):,}"
    )

    print(
        f"False positives: "
        f"{len(false_positives):,}"
    )

    # --------------------------------------------------
    # False negative probability distribution
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FALSE NEGATIVE PROBABILITY DISTRIBUTION")
    print("=" * 80)

    print(
        false_negatives["probability"]
        .describe()
    )

    # --------------------------------------------------
    # Hardest false negatives
    #
    # Highest predicted probability among missed
    # positives.
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("HARDEST FALSE NEGATIVES")
    print("=" * 80)

    hard_fn = false_negatives.sort_values(
        "probability",
        ascending=False,
    )

    display_columns = [
        "s1_entity_id",
        "candidate_entity_id",
        "probability",
        "name_exact",
        "core_name_exact",
        "name_sequence_similarity",
        "core_name_sequence_similarity",
        "name_token_jaccard",
        "name_token_overlap",
        "address_exact",
        "address_sequence_similarity",
        "address_token_jaccard",
        "address_token_overlap",
        "address_number_overlap",
        "address_numeric_component_overlap",
        "address_location_overlap",
        "country_match",
    ]

    for _, row in hard_fn.head(25).iterrows():

        print("\n----------------------------------------")

        print(
            f"S1:        {row['s1_entity_id']}"
        )

        print(
            f"Candidate: {row['candidate_entity_id']}"
        )

        print(
            f"Probability: {row['probability']:.4f}"
        )

        print(
            f"Name exact: {row['name_exact']}"
        )

        print(
            f"Core exact: {row['core_name_exact']}"
        )

        print(
            f"Name similarity: "
            f"{row['name_sequence_similarity']:.4f}"
        )

        print(
            f"Core similarity: "
            f"{row['core_name_sequence_similarity']:.4f}"
        )

        print(
            f"Name token overlap: "
            f"{row['name_token_overlap']:.4f}"
        )

        print(
            f"Address similarity: "
            f"{row['address_sequence_similarity']:.4f}"
        )

        print(
            f"Address token overlap: "
            f"{row['address_token_overlap']:.4f}"
        )

        print(
            f"Number overlap: "
            f"{row['address_number_overlap']:.4f}"
        )

        print(
            f"Numeric component overlap: "
            f"{row['address_numeric_component_overlap']:.4f}"
        )

        print(
            f"Location overlap: "
            f"{row['address_location_overlap']:.4f}"
        )

        print(
            f"Country match: "
            f"{row['country_match']:.0f}"
        )

    # --------------------------------------------------
    # False positives
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("MOST DANGEROUS FALSE POSITIVES")
    print("=" * 80)

    hard_fp = false_positives.sort_values(
        "probability",
        ascending=False,
    )

    for _, row in hard_fp.head(25).iterrows():

        print("\n----------------------------------------")

        print(
            f"S1:        {row['s1_entity_id']}"
        )

        print(
            f"Candidate: {row['candidate_entity_id']}"
        )

        print(
            f"Probability: {row['probability']:.4f}"
        )

        print(
            f"Name exact: {row['name_exact']}"
        )

        print(
            f"Core exact: {row['core_name_exact']}"
        )

        print(
            f"Name similarity: "
            f"{row['name_sequence_similarity']:.4f}"
        )

        print(
            f"Core similarity: "
            f"{row['core_name_sequence_similarity']:.4f}"
        )

        print(
            f"Address similarity: "
            f"{row['address_sequence_similarity']:.4f}"
        )

        print(
            f"Address token overlap: "
            f"{row['address_token_overlap']:.4f}"
        )

        print(
            f"Number overlap: "
            f"{row['address_number_overlap']:.4f}"
        )

        print(
            f"Location overlap: "
            f"{row['address_location_overlap']:.4f}"
        )

        print(
            f"Country match: "
            f"{row['country_match']:.0f}"
        )

    # --------------------------------------------------
    # Feature statistics: FN vs TP
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("TRUE POSITIVE vs FALSE NEGATIVE FEATURE MEANS")
    print("=" * 80)

    true_positive = val_df[
        (val_df["label"] == 1)
        & (val_df["prediction"] == 1)
    ]

    comparison_features = [
        "name_exact",
        "core_name_exact",
        "name_sequence_similarity",
        "core_name_sequence_similarity",
        "name_token_jaccard",
        "name_token_overlap",
        "address_exact",
        "address_sequence_similarity",
        "address_token_jaccard",
        "address_token_overlap",
        "address_number_overlap",
        "address_numeric_component_overlap",
        "address_location_overlap",
        "country_match",
    ]

    print(
        f"\n{'Feature':40s}"
        f"{'TP':>12s}"
        f"{'FN':>12s}"
    )

    print("-" * 65)

    for feature in comparison_features:

        tp_mean = true_positive[
            feature
        ].mean()

        fn_mean = false_negatives[
            feature
        ].mean()

        print(
            f"{feature:40s}"
            f"{tp_mean:12.4f}"
            f"{fn_mean:12.4f}"
        )

    # --------------------------------------------------
    # Save errors
    # --------------------------------------------------

    false_negatives.to_csv(
        "tests/generated/"
        "validation_false_negatives.tsv",
        sep="\t",
        index=False,
    )

    false_positives.to_csv(
        "tests/generated/"
        "validation_false_positives.tsv",
        sep="\t",
        index=False,
    )

    print(
        "\nSaved:"
        "\n  validation_false_negatives.tsv"
        "\n  validation_false_positives.tsv"
    )


if __name__ == "__main__":
    main()