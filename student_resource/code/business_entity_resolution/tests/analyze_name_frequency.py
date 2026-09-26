import pandas as pd
from collections import Counter

from src.preprocessing import preprocess_record


# ---------------------------------------------------------
# Load sample datasets
# ---------------------------------------------------------

S1_PATH = "../../samples/sample_train_source1.tsv"
S2_PATH = "../../samples/sample_train_source2.tsv"
S3_PATH = "../../samples/sample_train_source3.tsv"


def load_and_preprocess(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    records = []

    for _, row in df.iterrows():
        records.append(
            preprocess_record(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"]
            )
        )

    return records


# ---------------------------------------------------------
# Analyze frequency distribution
# ---------------------------------------------------------

def analyze(records, source_name):

    names = [
        r["name_normalized"]
        for r in records
        if r["name_normalized"]
    ]

    core_names = [
        r["name_core"]
        for r in records
        if r["name_core"]
    ]

    name_counts = Counter(names)
    core_counts = Counter(core_names)

    print(f"\n{'=' * 60}")
    print(f"{source_name}")
    print(f"{'=' * 60}")

    # Basic statistics
    print("\nNAME STATISTICS")
    print("----------------")

    print(f"Total records:              {len(records)}")
    print(f"Non-empty names:            {len(names)}")
    print(f"Unique normalized names:    {len(name_counts)}")
    print(f"Unique core names:          {len(core_counts)}")

    # Frequency buckets
    print("\nNORMALIZED NAME FREQUENCY")
    print("-------------------------")

    print(f"Names occurring once:       {sum(v == 1 for v in name_counts.values())}")
    print(f"Names occurring >= 2:       {sum(v >= 2 for v in name_counts.values())}")
    print(f"Names occurring >= 10:      {sum(v >= 10 for v in name_counts.values())}")
    print(f"Names occurring >= 100:     {sum(v >= 100 for v in name_counts.values())}")
    print(f"Names occurring >= 1000:    {sum(v >= 1000 for v in name_counts.values())}")

    # Most common names
    print("\nTOP 20 NORMALIZED NAMES")
    print("-----------------------")

    for name, count in name_counts.most_common(20):
        print(f"{count:5}  {name}")

    # Core name frequency
    print("\nCORE NAME FREQUENCY")
    print("-------------------")

    print(f"Core names occurring once:       {sum(v == 1 for v in core_counts.values())}")
    print(f"Core names occurring >= 2:       {sum(v >= 2 for v in core_counts.values())}")
    print(f"Core names occurring >= 10:      {sum(v >= 10 for v in core_counts.values())}")
    print(f"Core names occurring >= 100:     {sum(v >= 100 for v in core_counts.values())}")
    print(f"Core names occurring >= 1000:    {sum(v >= 1000 for v in core_counts.values())}")

    print("\nTOP 20 CORE NAMES")
    print("-----------------")

    for name, count in core_counts.most_common(20):
        print(f"{count:5}  {name}")

    # Name length
    lengths = [len(name) for name in names]

    if lengths:
        print("\nNAME LENGTH")
        print("-----------")

        print(f"Minimum:    {min(lengths)}")
        print(f"Maximum:    {max(lengths)}")
        print(f"Average:    {sum(lengths) / len(lengths):.2f}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading sample datasets...")

    s1 = load_and_preprocess(S1_PATH)
    s2 = load_and_preprocess(S2_PATH)
    s3 = load_and_preprocess(S3_PATH)

    print(f"S1 records: {len(s1)}")
    print(f"S2 records: {len(s2)}")
    print(f"S3 records: {len(s3)}")

    analyze(s1, "SOURCE 1")
    analyze(s2, "SOURCE 2")
    analyze(s3, "SOURCE 3")


if __name__ == "__main__":
    main()