import os
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score


DATA_PATH = (
    "tests/generated/combined_training_pairs.tsv"
)

MODEL_PATH = (
    "tests/generated/combined_gradient_boosting_model.joblib"
)


FEATURE_COLUMNS = [
    "name_exact",
    "core_name_exact",
    "name_sequence_similarity",
    "core_name_sequence_similarity",
    "name_token_jaccard",
    "name_token_overlap",
    "name_length_difference",
    "address_exact",
    "address_sequence_similarity",
    "address_token_jaccard",
    "address_token_overlap",
    "address_length_difference",
    "address_number_overlap",
    "address_numeric_component_overlap",
    "address_location_overlap",
    "country_match",
]


def f05(precision, recall):
    beta2 = 0.25

    if precision == 0 and recall == 0:
        return 0.0

    return (
        (1 + beta2)
        * precision
        * recall
        / (
            beta2 * precision
            + recall
        )
    )


def main():

    print("=" * 70)
    print("TRAINING COMBINED GRADIENT BOOSTING MODEL")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
    )

    print(
        f"Dataset size: {len(df):,}"
    )

    # --------------------------------------------------------
    # Group split by S1
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["s1_entity_id"],
        )
    )

    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    print(
        f"\nTrain pairs: {len(train):,}"
    )

    print(
        f"Validation pairs: {len(val):,}"
    )

    print(
        f"Train S1: "
        f"{train['s1_entity_id'].nunique():,}"
    )

    print(
        f"Validation S1: "
        f"{val['s1_entity_id'].nunique():,}"
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train["label"]

    X_val = val[FEATURE_COLUMNS]
    y_val = val["label"]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nTraining model...")

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    best = None

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.675,
        0.70,
        0.725,
        0.75,
        0.775,
        0.80,
        0.825,
        0.85,
        0.875,
        0.90,
        0.925,
        0.95,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        score = f05(
            precision,
            recall,
        )

        print(
            f"{threshold:>7.3f} "
            f"{score:>10.4f} "
            f"{precision:>12.4f} "
            f"{recall:>10.4f}"
        )

        if best is None or score > best[0]:

            best = (
                score,
                threshold,
                precision,
                recall,
            )

    print("\n" + "=" * 70)
    print("BEST VALIDATION RESULT")
    print("=" * 70)

    print(
        f"F0.5      : {best[0]:.4f}"
    )

    print(
        f"Threshold : {best[1]:.3f}"
    )

    print(
        f"Precision : {best[2]:.4f}"
    )

    print(
        f"Recall    : {best[3]:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best[1],
    }

    joblib.dump(
        bundle,
        MODEL_PATH,
    )

    print(
        f"\nSaved model:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()