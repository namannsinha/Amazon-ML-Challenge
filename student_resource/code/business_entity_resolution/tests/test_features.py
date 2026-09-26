import pandas as pd

from src.preprocessing import preprocess_record
from src.features import compute_pair_features


def load_first_two_records(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    records = []

    for _, row in df.head(2).iterrows():

        records.append(
            preprocess_record(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"]
            )
        )

    return records


def main():

    path = "../../samples/sample_train_source1.tsv"

    records = load_first_two_records(path)

    record_a = records[0]
    record_b = records[1]

    print("RECORD A")
    print(record_a)

    print("\nRECORD B")
    print(record_b)

    print("\nPAIR FEATURES")
    print("-" * 40)

    features = compute_pair_features(
        record_a,
        record_b
    )

    for name, value in features.items():
        print(f"{name:35} {value:.4f}")


if __name__ == "__main__":
    main()