import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
)


DATA_PATH = (
    "tests/generated/"
    "complete_baseline_training_pairs.tsv"
)

MODEL_PATH = (
    "tests/generated/"
    "gradient_boosting_model.joblib"
)

RANDOM_SEED = 42


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
    )

    excluded = {
        "label",
        "s1_entity_id",
        "candidate_entity_id",
    }

    feature_columns = [
        c for c in df.columns
        if c not in excluded
    ]

    X = df[feature_columns].astype(float)
    y = df["label"].astype(int)

    # --------------------------------------------------
    # SAME S1-LEVEL SPLIT AS LOGISTIC REGRESSION
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

    train_s1 = set(
        unique_s1[:split_index]
    )

    validation_s1 = set(
        unique_s1[split_index:]
    )

    train_mask = (
        df["s1_entity_id"]
        .isin(train_s1)
    )

    val_mask = (
        df["s1_entity_id"]
        .isin(validation_s1)
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    print(
        f"Training pairs: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation pairs: "
        f"{len(X_val):,}"
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    print(
        "\nTraining HistGradientBoosting..."
    )

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD EVALUATION")
    print("=" * 70)

    print(
        f"\n{'Threshold':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F0.5':>12}"
        f"{'Matches':>12}"
    )

    print("-" * 60)

    thresholds = np.arange(
        0.10,
        0.996,
        0.025,
    )

    best_threshold = None
    best_f05 = -1

    for threshold in thresholds:

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

        f05 = fbeta_score(
            y_val,
            predictions,
            beta=0.5,
            zero_division=0,
        )

        matches = predictions.sum()

        print(
            f"{threshold:12.3f}"
            f"{precision:12.4f}"
            f"{recall:12.4f}"
            f"{f05:12.4f}"
            f"{matches:12,}"
        )

        if f05 > best_f05:

            best_f05 = f05
            best_threshold = threshold

    # --------------------------------------------------
    # BEST RESULT
    # --------------------------------------------------

    predictions = (
        probabilities >= best_threshold
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

    print("\n" + "=" * 70)
    print("BEST GRADIENT BOOSTING RESULT")
    print("=" * 70)

    print(
        f"Threshold: {best_threshold:.3f}"
    )

    print(
        f"F0.5:      {best_f05:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"Matches:   {predictions.sum():,}"
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_val,
            predictions,
        )
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": best_threshold,
        },
        MODEL_PATH,
    )

    print(
        f"\nSaved model:"
        f"\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()