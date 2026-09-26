import random
import time

import pandas as pd

from src.preprocessing import preprocess_record
from src.features import compute_pair_features
from src.approximate_blocking import (
    ApproximateBlocker,
    process_file_in_chunks,
)


TRAIN_DIR = "../../dataset/train"

S1_PATH = f"{TRAIN_DIR}/train_source1.tsv"
S2_PATH = f"{TRAIN_DIR}/train_source2.tsv"
S3_PATH = f"{TRAIN_DIR}/train_source3.tsv"
GT_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"

SAMPLE_SIZE = 5000

TOP_K_NAME = 10
TOP_K_ADDRESS = 10

CHUNK_SIZE = 100_000
QUERY_BATCH_SIZE = 250

MAX_NEGATIVES_PER_S1 = 5

RANDOM_SEED = 42


def load_sample_s1():

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

    s1_ids = set(
        gt_sample["source1_entity_id"]
    )

    print(
        f"Sampled S1 entities: "
        f"{len(s1_ids)}"
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

    records = {}

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

        records[
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

    return records, ground_truth


def load_candidate_records():

    """
    Load S2/S3 only once.

    We need the records later to calculate features
    for the retrieved hard negatives.

    This dictionary contains roughly 10.3M records,
    so if memory becomes a problem we will change this
    to an on-demand lookup strategy.
    """

    records = {}

    for path, source_name in [
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        print(
            f"\nLoading {source_name} "
            f"for feature lookup..."
        )

        total = 0

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

                records[
                    record["entity_id"]
                ] = record

            total += len(chunk)

            print(
                f"  {source_name}: "
                f"{total:,} rows"
            )

    print(
        f"\nTotal candidate records loaded: "
        f"{len(records):,}"
    )

    return records


def main():

    start_time = time.time()

    random.seed(RANDOM_SEED)

    # --------------------------------------------------
    # 1. Sample S1
    # --------------------------------------------------

    s1_records, ground_truth = load_sample_s1()

    # --------------------------------------------------
    # 2. Approximate name retrieval
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("APPROXIMATE NAME RETRIEVAL")
    print("=" * 80)

    name_blocker = ApproximateBlocker(
        query_records=s1_records,
        field="name_normalized",
        top_k=TOP_K_NAME,
        chunk_size=CHUNK_SIZE,
        query_batch_size=QUERY_BATCH_SIZE,
    )

    name_results_heap = {}

    print("\nScanning S2...")

    process_file_in_chunks(
        S2_PATH,
        name_blocker,
        name_results_heap,
        preprocess_record,
        chunk_size=CHUNK_SIZE,
    )

    print("\nScanning S3...")

    process_file_in_chunks(
        S3_PATH,
        name_blocker,
        name_results_heap,
        preprocess_record,
        chunk_size=CHUNK_SIZE,
    )

    name_results = name_blocker.get_results(
        name_results_heap
    )

    # --------------------------------------------------
    # 3. Approximate address retrieval
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("APPROXIMATE ADDRESS RETRIEVAL")
    print("=" * 80)

    address_blocker = ApproximateBlocker(
        query_records=s1_records,
        field="address_normalized",
        top_k=TOP_K_ADDRESS,
        chunk_size=CHUNK_SIZE,
        query_batch_size=QUERY_BATCH_SIZE,
    )

    address_results_heap = {}

    print("\nScanning S2...")

    process_file_in_chunks(
        S2_PATH,
        address_blocker,
        address_results_heap,
        preprocess_record,
        chunk_size=CHUNK_SIZE,
    )

    print("\nScanning S3...")

    process_file_in_chunks(
        S3_PATH,
        address_blocker,
        address_results_heap,
        preprocess_record,
        chunk_size=CHUNK_SIZE,
    )

    address_results = address_blocker.get_results(
        address_results_heap
    )

    # --------------------------------------------------
    # 4. Collect unique approximate candidates
    # --------------------------------------------------

    print("\nCollecting approximate candidates...")

    candidate_ids = set()

    for results in name_results.values():

        for candidate_id, _ in results:

            candidate_ids.add(candidate_id)

    for results in address_results.values():

        for candidate_id, _ in results:

            candidate_ids.add(candidate_id)

    print(
        f"Unique retrieved candidates: "
        f"{len(candidate_ids):,}"
    )

    # --------------------------------------------------
    # 5. Load only retrieved candidate records
    # --------------------------------------------------

    #
    # IMPORTANT:
    #
    # We DO NOT load all 10.3M records here.
    #
    # We scan the files again and retain only the
    # candidates actually retrieved above.
    #

    candidate_lookup = {}

    for path, source_name in [
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        print(
            f"\nFinding retrieved records in {source_name}..."
        )

        found = 0

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

                entity_id = row["entity_id"]

                if entity_id not in candidate_ids:
                    continue

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
                    entity_id,
                    name,
                    address,
                    country,
                )

                candidate_lookup[
                    entity_id
                ] = record

                found += 1

        print(
            f"  Retrieved records found: "
            f"{found:,}"
        )

    # --------------------------------------------------
    # 6. Generate hard negatives
    # --------------------------------------------------

    hard_negatives = []

    for s1_id, s1_record in s1_records.items():

        true_matches = ground_truth.get(
            s1_id,
            set(),
        )

        combined = {}

        # Name candidates
        for candidate_id, similarity in (
            name_results.get(s1_id, [])
        ):

            if candidate_id not in true_matches:

                combined[candidate_id] = max(
                    combined.get(
                        candidate_id,
                        0.0,
                    ),
                    similarity,
                )

        # Address candidates
        for candidate_id, similarity in (
            address_results.get(s1_id, [])
        ):

            if candidate_id not in true_matches:

                combined[candidate_id] = max(
                    combined.get(
                        candidate_id,
                        0.0,
                    ),
                    similarity,
                )

        # Strongest approximate candidates first
        ranked = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        ranked = ranked[
            :MAX_NEGATIVES_PER_S1
        ]

        for candidate_id, retrieval_similarity in ranked:

            candidate = candidate_lookup.get(
                candidate_id
            )

            if candidate is None:
                continue

            features = compute_pair_features(
                s1_record,
                candidate,
            )

            features[
                "s1_entity_id"
            ] = s1_id

            features[
                "candidate_entity_id"
            ] = candidate_id

            features[
                "retrieval_similarity"
            ] = retrieval_similarity

            hard_negatives.append(
                features
            )

    # --------------------------------------------------
    # 7. Statistics
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("APPROXIMATE HARD NEGATIVE FEATURE STATISTICS")
    print("=" * 80)

    print(
        f"\nApproximate hard negatives generated: "
        f"{len(hard_negatives):,}"
    )

    if not hard_negatives:
        print("No hard negatives generated.")
        return

    df = pd.DataFrame(
        hard_negatives
    )

    excluded = {
        "s1_entity_id",
        "candidate_entity_id",
    }

    feature_columns = [
        c for c in df.columns
        if c not in excluded
    ]

    print(
        f"\n{'Feature':40s}"
        f"{'Mean':>10s}"
        f"{'Min':>11s}"
        f"{'Max':>11s}"
    )

    print("-" * 75)

    for feature in feature_columns:

        values = df[feature]

        print(
            f"{feature:40s}"
            f"{values.mean():10.4f}"
            f"{values.min():11.4f}"
            f"{values.max():11.4f}"
        )

    # --------------------------------------------------
    # 8. Examples
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("APPROXIMATE HARD NEGATIVE EXAMPLES")
    print("=" * 80)

    for _, row in df.head(15).iterrows():

        s1 = s1_records[
            row["s1_entity_id"]
        ]

        candidate = candidate_lookup[
            row["candidate_entity_id"]
        ]

        print("\nS1:")
        print(
            f"  {s1['entity_id']} | "
            f"{s1['name_normalized']} | "
            f"{s1['address_normalized']}"
        )

        print("Candidate:")
        print(
            f"  {candidate['entity_id']} | "
            f"{candidate['name_normalized']} | "
            f"{candidate['address_normalized']}"
        )

        print(
            f"  retrieval similarity: "
            f"{row['retrieval_similarity']:.3f}"
        )

        print(
            f"  name similarity:    "
            f"{row['name_sequence_similarity']:.3f}"
        )

        print(
            f"  core similarity:    "
            f"{row['core_name_sequence_similarity']:.3f}"
        )

        print(
            f"  address similarity: "
            f"{row['address_sequence_similarity']:.3f}"
        )

        print(
            f"  address overlap:    "
            f"{row['address_token_overlap']:.3f}"
        )

        print(
            f"  number overlap:     "
            f"{row['address_number_overlap']:.3f}"
        )

        print(
            f"  country match:      "
            f"{row['country_match']:.0f}"
        )

    print(
        f"\nTotal runtime: "
        f"{time.time() - start_time:.1f}s"
    )


if __name__ == "__main__":
    main()