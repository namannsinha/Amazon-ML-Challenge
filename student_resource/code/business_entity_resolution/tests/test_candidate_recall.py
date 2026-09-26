import pandas as pd

from src.preprocessing import preprocess_record
from src.blocking import build_blocking_indexes, generate_candidates


TRAIN_DIR = "../../dataset/train"

S1_PATH = f"{TRAIN_DIR}/train_source1.tsv"
S2_PATH = f"{TRAIN_DIR}/train_source2.tsv"
S3_PATH = f"{TRAIN_DIR}/train_source3.tsv"
GT_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"

SAMPLE_SIZE = 5000
CHUNK_SIZE = 100_000
RANDOM_SEED = 42


def load_sample():

    print("Loading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt = gt.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    )

    s1_ids = set(
        gt["source1_entity_id"]
    )

    ground_truth = {}

    all_true_ids = set()

    for _, row in gt.iterrows():

        value = row["matched_entity_ids"]

        if pd.isna(value) or not str(value).strip():

            matches = set()

        else:

            matches = {
                x.strip()
                for x in str(value).split(",")
                if x.strip()
            }

        ground_truth[
            row["source1_entity_id"]
        ] = matches

        all_true_ids.update(matches)

    # Load S1

    s1_df = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s1_df = s1_df[
        s1_df["entity_id"].isin(s1_ids)
    ]

    s1_records = {}

    for _, row in s1_df.iterrows():

        name = (
            ""
            if pd.isna(row["business_name"])
            else row["business_name"]
        )

        address = (
            ""
            if pd.isna(row["business_address"])
            else row["business_address"]
        )

        country = (
            ""
            if pd.isna(row["country"])
            else row["country"]
        )

        record = preprocess_record(
            row["entity_id"],
            name,
            address,
            country,
        )

        s1_records[
            record["entity_id"]
        ] = record

    return (
        s1_records,
        ground_truth,
        all_true_ids,
    )


def build_exact_candidates(
    s1_records,
):

    name_keys = set()
    core_keys = set()
    address_keys = set()

    for record in s1_records.values():

        if record["name_normalized"]:
            name_keys.add(
                record["name_normalized"]
            )

        if record["name_core"]:
            core_keys.add(
                record["name_core"]
            )

        if record["address_normalized"]:
            address_keys.add(
                record["address_normalized"]
            )

    print("\nS1 blocking keys:")

    print(
        f"  names:      {len(name_keys):,}"
    )

    print(
        f"  core names: {len(core_keys):,}"
    )

    print(
        f"  addresses:  {len(address_keys):,}"
    )

    candidate_records = []

    for path, source_name in [
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        print(
            f"\nScanning {source_name}..."
        )

        scanned = 0
        retained = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
            chunksize=CHUNK_SIZE,
        ):

            for _, row in chunk.iterrows():

                name = (
                    ""
                    if pd.isna(row["business_name"])
                    else row["business_name"]
                )

                address = (
                    ""
                    if pd.isna(row["business_address"])
                    else row["business_address"]
                )

                country = (
                    ""
                    if pd.isna(row["country"])
                    else row["country"]
                )

                record = preprocess_record(
                    row["entity_id"],
                    name,
                    address,
                    country,
                )

                if (
                    record["name_normalized"] in name_keys
                    or record["name_core"] in core_keys
                    or record["address_normalized"] in address_keys
                ):

                    candidate_records.append(
                        record
                    )

                    retained += 1

            scanned += len(chunk)

            if scanned % 500_000 == 0:
                print(
                    f"  scanned {scanned:,}, "
                    f"retained {retained:,}"
                )

        print(
            f"{source_name}: "
            f"{retained:,} candidates"
        )

    return candidate_records


def evaluate_exact(
    s1_records,
    ground_truth,
    candidate_records,
):

    print(
        "\nBuilding indexes..."
    )

    indexes = build_blocking_indexes(
        candidate_records,
        [],
    )

    total_true_pairs = 0
    recovered_pairs = 0

    s1_with_candidates = 0

    candidate_counts = []

    # Per-block recovery

    name_recovered = 0
    core_recovered = 0
    address_recovered = 0

    for s1_id, true_matches in ground_truth.items():

        s1 = s1_records[s1_id]

        blocks = generate_candidates(
            s1,
            indexes,
        )

        name_candidates = blocks["name"]
        core_candidates = blocks["core_name"]
        address_candidates = blocks["address"]

        all_candidates = (
            name_candidates
            | core_candidates
            | address_candidates
        )

        if all_candidates:
            s1_with_candidates += 1

        candidate_counts.append(
            len(all_candidates)
        )

        for true_id in true_matches:

            total_true_pairs += 1

            if true_id in all_candidates:

                recovered_pairs += 1

            if true_id in name_candidates:
                name_recovered += 1

            if true_id in core_candidates:
                core_recovered += 1

            if true_id in address_candidates:
                address_recovered += 1

    recall = (
        recovered_pairs / total_true_pairs
        if total_true_pairs
        else 0
    )

    name_recall = (
        name_recovered / total_true_pairs
    )

    core_recall = (
        core_recovered / total_true_pairs
    )

    address_recall = (
        address_recovered / total_true_pairs
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXACT BLOCKING RECALL"
    )

    print(
        "=" * 70
    )

    print(
        f"True pairs:              "
        f"{total_true_pairs:,}"
    )

    print(
        f"Recovered pairs:         "
        f"{recovered_pairs:,}"
    )

    print(
        f"Overall recall:          "
        f"{recall:.4%}"
    )

    print(
        f"\nName exact recall:       "
        f"{name_recall:.4%}"
    )

    print(
        f"Core name exact recall:  "
        f"{core_recall:.4%}"
    )

    print(
        f"Address exact recall:    "
        f"{address_recall:.4%}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CANDIDATE VOLUME"
    )

    print(
        "=" * 70
    )

    print(
        f"S1 with candidates:     "
        f"{s1_with_candidates:,} / "
        f"{len(s1_records):,}"
    )

    print(
        f"Average candidates/S1:  "
        f"{sum(candidate_counts) / len(candidate_counts):.2f}"
    )

    print(
        f"Median candidates/S1:   "
        f"{pd.Series(candidate_counts).median():.0f}"
    )

    print(
        f"Max candidates/S1:      "
        f"{max(candidate_counts):,}"
    )


def main():

    (
        s1_records,
        ground_truth,
        all_true_ids,
    ) = load_sample()

    print(
        f"\nSampled S1: "
        f"{len(s1_records):,}"
    )

    print(
        f"True matched IDs: "
        f"{len(all_true_ids):,}"
    )

    candidate_records = build_exact_candidates(
        s1_records
    )

    print(
        f"\nTotal candidate records: "
        f"{len(candidate_records):,}"
    )

    evaluate_exact(
        s1_records,
        ground_truth,
        candidate_records,
    )


if __name__ == "__main__":
    main()