import os
import random
import time

import pandas as pd

from src.preprocessing import preprocess_record
from src.blocking import build_blocking_indexes, generate_candidates
from src.features import compute_pair_features


TRAIN_DIR = "../../dataset/train"
OUTPUT_DIR = "tests/generated"

S1_PATH = f"{TRAIN_DIR}/train_source1.tsv"
S2_PATH = f"{TRAIN_DIR}/train_source2.tsv"
S3_PATH = f"{TRAIN_DIR}/train_source3.tsv"
GT_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"

SAMPLE_SIZE = 5000
MAX_HARD_NEGATIVES_PER_S1 = 5
CHUNK_SIZE = 100_000

RANDOM_SEED = 42


def load_sample_s1_and_ground_truth():

    print("Sampling ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt_sample = gt.sample(
        n=min(SAMPLE_SIZE, len(gt)),
        random_state=RANDOM_SEED,
    )

    s1_ids = set(gt_sample["source1_entity_id"])

    print(
        f"Sampled S1 entities: {len(s1_ids)}"
    )

    s1 = pd.read_csv(
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

    s1 = s1[
        s1["entity_id"].isin(s1_ids)
    ]

    s1_records = {}

    for _, row in s1.iterrows():

        name = row["business_name"]
        address = row["business_address"]
        country = row["country"]

        if pd.isna(name):
            name = ""

        if pd.isna(address):
            address = ""

        if pd.isna(country):
            country = ""

        record = preprocess_record(
            row["entity_id"],
            name,
            address,
            country,
        )

        s1_records[
            record["entity_id"]
        ] = record

    ground_truth = {}

    for _, row in gt_sample.iterrows():

        matched = row["matched_entity_ids"]

        if pd.isna(matched) or not str(matched).strip():

            ground_truth[
                row["source1_entity_id"]
            ] = set()

        else:

            ground_truth[
                row["source1_entity_id"]
            ] = {
                x.strip()
                for x in str(matched).split(",")
                if x.strip()
            }

    return s1_records, ground_truth


def build_sample_s1_key_sets(s1_records):

    """
    Build the exact blocking keys occurring in the
    sampled S1 records.

    We use these keys while scanning S2/S3 so that
    irrelevant records are never retained.
    """

    name_keys = set()
    core_name_keys = set()
    address_keys = set()

    for record in s1_records.values():

        if record["name_normalized"]:
            name_keys.add(
                record["name_normalized"]
            )

        if record["name_core"]:
            core_name_keys.add(
                record["name_core"]
            )

        if record["address_normalized"]:
            address_keys.add(
                record["address_normalized"]
            )

    print("\nSample S1 blocking keys:")

    print(
        f"  names:       {len(name_keys):,}"
    )

    print(
        f"  core names:  {len(core_name_keys):,}"
    )

    print(
        f"  addresses:   {len(address_keys):,}"
    )

    return (
        name_keys,
        core_name_keys,
        address_keys,
    )


def scan_candidate_source(
    path,
    source_name,
    name_keys,
    core_name_keys,
    address_keys,
):

    """
    Scan one full source but retain only records that
    can participate in exact blocking with sampled S1.
    """

    retained = []

    total_rows = 0
    matched_rows = 0

    print(
        f"\nScanning {source_name}..."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
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
        ),
        start=1,
    ):

        for _, row in chunk.iterrows():

            name = row["business_name"]
            address = row["business_address"]
            country = row["country"]

            if pd.isna(name):
                name = ""

            if pd.isna(address):
                address = ""

            if pd.isna(country):
                country = ""

            record = preprocess_record(
                row["entity_id"],
                name,
                address,
                country,
            )

            matches_block = (
                record["name_normalized"] in name_keys
                or record["name_core"] in core_name_keys
                or record["address_normalized"] in address_keys
            )

            if matches_block:

                retained.append(record)
                matched_rows += 1

        total_rows += len(chunk)

        print(
            f"  chunk {chunk_number}: "
            f"{total_rows:,} scanned, "
            f"{matched_rows:,} retained"
        )

    print(
        f"{source_name} retained: "
        f"{len(retained):,}"
    )

    return retained


def load_exact_block_candidates(s1_records):

    (
        name_keys,
        core_name_keys,
        address_keys,
    ) = build_sample_s1_key_sets(
        s1_records
    )

    s2_records = scan_candidate_source(
        S2_PATH,
        "S2",
        name_keys,
        core_name_keys,
        address_keys,
    )

    s3_records = scan_candidate_source(
        S3_PATH,
        "S3",
        name_keys,
        core_name_keys,
        address_keys,
    )

    candidates = (
        s2_records + s3_records
    )

    print(
        f"\nTotal exact-block candidate records: "
        f"{len(candidates):,}"
    )

    return candidates


def build_training_dataset():

    start = time.time()

    random.seed(RANDOM_SEED)

    # --------------------------------------------------
    # 1. Sample S1 + ground truth
    # --------------------------------------------------

    (
        s1_records,
        ground_truth,
    ) = load_sample_s1_and_ground_truth()

    # --------------------------------------------------
    # 2. Load only exact-block candidates
    # --------------------------------------------------

    candidate_records = (
        load_exact_block_candidates(
            s1_records
        )
    )

    candidate_lookup = {
        record["entity_id"]: record
        for record in candidate_records
    }

    # --------------------------------------------------
    # 3. Build exact indexes
    # --------------------------------------------------

    print(
        "\nBuilding exact blocking indexes..."
    )

    indexes = build_blocking_indexes(
        candidate_records,
        [],
    )

    # --------------------------------------------------
    # 4. Positive pairs
    # --------------------------------------------------

    print(
        "\nGenerating positive pairs..."
    )

    positive_pairs = []

    positives_missing_from_candidates = 0

    for s1_id, true_matches in ground_truth.items():

        s1 = s1_records[s1_id]

        for candidate_id in true_matches:

            candidate = candidate_lookup.get(
                candidate_id
            )

            if candidate is None:

                positives_missing_from_candidates += 1
                continue

            features = compute_pair_features(
                s1,
                candidate,
            )

            features["label"] = 1
            features["s1_entity_id"] = s1_id
            features["candidate_entity_id"] = candidate_id

            positive_pairs.append(
                features
            )

    print(
        f"Positive pairs: "
        f"{len(positive_pairs):,}"
    )

    if positives_missing_from_candidates:

        print(
            f"WARNING: "
            f"{positives_missing_from_candidates:,} "
            f"true matches were not found by exact blocking."
        )

    # --------------------------------------------------
    # 5. Exact hard negatives
    # --------------------------------------------------

    print(
        "\nGenerating exact hard negatives..."
    )

    hard_negative_pairs = []

    for counter, (
        s1_id,
        true_matches,
    ) in enumerate(
        ground_truth.items(),
        start=1,
    ):

        s1 = s1_records[s1_id]

        blocks = generate_candidates(
            s1,
            indexes,
        )

        candidate_ids = (
            blocks["name"]
            | blocks["core_name"]
            | blocks["address"]
        )

        # Remove true matches.
        candidate_ids -= true_matches

        candidate_ids = list(
            candidate_ids
        )

        if len(candidate_ids) > (
            MAX_HARD_NEGATIVES_PER_S1
        ):

            candidate_ids = random.sample(
                candidate_ids,
                MAX_HARD_NEGATIVES_PER_S1,
            )

        for candidate_id in candidate_ids:

            candidate = candidate_lookup[
                candidate_id
            ]

            features = compute_pair_features(
                s1,
                candidate,
            )

            features["label"] = 0
            features["s1_entity_id"] = s1_id
            features["candidate_entity_id"] = candidate_id

            hard_negative_pairs.append(
                features
            )

        if counter % 1000 == 0:

            print(
                f"  processed "
                f"{counter:,} S1 entities"
            )

    print(
        f"Exact hard negatives: "
        f"{len(hard_negative_pairs):,}"
    )

    # --------------------------------------------------
    # 6. Combine
    # --------------------------------------------------

    df = pd.DataFrame(
        positive_pairs
        + hard_negative_pairs
    )

    df = df.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 7. Statistics
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE TRAINING DATASET"
    )

    print(
        "=" * 70
    )

    print(
        f"Total pairs: "
        f"{len(df):,}"
    )

    print(
        f"Positive pairs: "
        f"{(df['label'] == 1).sum():,}"
    )

    print(
        f"Negative pairs: "
        f"{(df['label'] == 0).sum():,}"
    )

    print(
        f"Positive ratio: "
        f"{df['label'].mean():.4f}"
    )

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = (
        f"{OUTPUT_DIR}/"
        "baseline_training_pairs.tsv"
    )

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved to:"
        f"\n{output_path}"
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    build_training_dataset()