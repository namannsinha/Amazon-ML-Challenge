import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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

MODEL_DIR = "tests/generated"

RANDOM_SEED = 42


def main():

    print("Loading training dataset...")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
    )

    print(
        f"Pairs: {len(df):,}"
    )

    # --------------------------------------------------
    # 1. Feature columns
    # --------------------------------------------------

    excluded_columns = {
        "label",
        "s1_entity_id",
        "candidate_entity_id",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    print("\nFeatures:")

    for feature in feature_columns:
        print(f"  {feature}")

    X = df[feature_columns].astype(float)
    y = df["label"].astype(int)

    # --------------------------------------------------
    # 2. Split by S1 entity
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

    validation_mask = (
        df["s1_entity_id"]
        .isin(validation_s1)
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[validation_mask]
    y_val = y[validation_mask]

    print("\n" + "=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Training S1 entities: "
        f"{len(train_s1):,}"
    )

    print(
        f"Validation S1 entities: "
        f"{len(validation_s1):,}"
    )

    print(
        f"Training pairs: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation pairs: "
        f"{len(X_val):,}"
    )

    print(
        f"Training positives: "
        f"{y_train.sum():,}"
    )

    print(
        f"Validation positives: "
        f"{y_val.sum():,}"
    )

    # --------------------------------------------------
    # 3. Train Logistic Regression
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------
    # 4. Predictions
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------
    # 5. Threshold evaluation
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
        0.96,
        0.05,
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
            f"{threshold:12.2f}"
            f"{precision:12.4f}"
            f"{recall:12.4f}"
            f"{f05:12.4f}"
            f"{matches:12,}"
        )

        if f05 > best_f05:

            best_f05 = f05
            best_threshold = threshold

    # --------------------------------------------------
    # 6. Best validation result
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BEST BASELINE RESULT")
    print("=" * 70)

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"F0.5:      {best_f05:.4f}"
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print(
        f"Precision: "
        f"{precision_score(y_val, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_val, predictions, zero_division=0):.4f}"
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_val,
            predictions,
        )
    )

    # --------------------------------------------------
    # 7. Feature coefficients
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COEFFICIENTS")
    print("=" * 70)

    coefficients = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_[0],
    })

    coefficients["abs_coefficient"] = (
        coefficients["coefficient"]
        .abs()
    )

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False,
    )

    for _, row in coefficients.iterrows():

        print(
            f"{row['feature']:40s}"
            f"{row['coefficient']:12.4f}"
        )

    # --------------------------------------------------
    # 8. Save model
    # --------------------------------------------------

    import joblib

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = (
        f"{MODEL_DIR}/"
        "baseline_logistic_model.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": best_threshold,
        },
        model_path,
    )

    print(
        f"\nSaved model:"
        f"\n{model_path}"
    )


if __name__ == "__main__":
    main()