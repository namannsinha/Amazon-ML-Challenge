import random
import pandas as pd

from src.preprocessing import preprocess_record
from src.blocking import (
    build_blocking_indexes,
    generate_candidates,
)
from src.features import compute_pair_features


S1_PATH = "../../dataset/train/train_source1.tsv"
S2_PATH = "../../dataset/train/train_source2.tsv"
S3_PATH = "../../dataset/train/train_source3.tsv"
GROUND_TRUTH_PATH = "../../dataset/train/train_ground_truth.tsv"

RANDOM_SEED = 42
SAMPLE_SIZE = 5000

# Maximum hard negatives collected per S1
MAX_NEGATIVES_PER_S1 = 5


# ---------------------------------------------------------
# Load sampled ground truth
# ---------------------------------------------------------

def load_ground_truth_sample():

    df = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_SEED,
    )

    ground_truth = {}

    for _, row in sample.iterrows():

        matched_ids = {
            x.strip()
            for x in row["matched_entity_ids"].split(",")
            if x.strip()
        }

        ground_truth[row["source1_entity_id"]] = matched_ids

    return ground_truth


# ---------------------------------------------------------
# Load only required records
# ---------------------------------------------------------

def load_required_records(path, required_ids):

    required_ids = set(required_ids)
    records = {}

    print(f"\nScanning {path}")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):

        matches = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in matches.iterrows():

            records[row["entity_id"]] = preprocess_record(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"],
            )

        if len(records) == len(required_ids):
            break

    print(
        f"Loaded {len(records)}/{len(required_ids)}"
    )

    return records


# ---------------------------------------------------------
# Load S2/S3 records needed for exact blocking
#
# IMPORTANT:
# For this experiment we need to scan S2/S3 once,
# but we only retain records that participate in an
# exact block with our sampled S1 records.
# ---------------------------------------------------------

def collect_blocking_candidates(
    s1_records,
    path,
    source_name,
):

    print(
        f"\nBuilding exact blocking candidates from {source_name}"
    )

    # Build lookup values from S1
    names = set()
    core_names = set()
    addresses = set()

    for record in s1_records:

        if record["name_normalized"]:
            names.add(
                record["name_normalized"]
            )

        if record["name_core"]:
            core_names.add(
                record["name_core"]
            )

        if record["address_normalized"]:
            addresses.add(
                record["address_normalized"]
            )

    matched_records = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):

        for _, row in chunk.iterrows():

            name = str(
                row["business_name"]
            )

            address = str(
                row["business_address"]
            )

            record = preprocess_record(
                entity_id=row["entity_id"],
                business_name=name,
                business_address=address,
                country=row["country"],
            )

            if (
                record["name_normalized"] in names
                or
                record["name_core"] in core_names
                or
                record["address_normalized"] in addresses
            ):

                matched_records[
                    record["entity_id"]
                ] = record

    print(
        f"{source_name} exact-block candidates: "
        f"{len(matched_records)}"
    )

    return matched_records


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    random.seed(RANDOM_SEED)

    # =====================================================
    # 1. Ground truth sample
    # =====================================================

    print("Sampling ground truth...")

    ground_truth = (
        load_ground_truth_sample()
    )

    print(
        f"Sampled S1 entities: "
        f"{len(ground_truth)}"
    )

    # =====================================================
    # 2. Load sampled S1 records
    # =====================================================

    s1_map = load_required_records(
        S1_PATH,
        ground_truth.keys(),
    )

    s1_records = list(
        s1_map.values()
    )

    # =====================================================
    # 3. Find exact-block candidates in S2/S3
    # =====================================================

    s2_candidates = collect_blocking_candidates(
        s1_records,
        S2_PATH,
        "S2",
    )

    s3_candidates = collect_blocking_candidates(
        s1_records,
        S3_PATH,
        "S3",
    )

    candidate_records = {
        **s2_candidates,
        **s3_candidates,
    }

    print(
        f"\nTotal unique exact-block candidates: "
        f"{len(candidate_records)}"
    )

    # =====================================================
    # 4. Build blocking indexes
    # =====================================================

    indexes = build_blocking_indexes(
        s2_candidates.values(),
        s3_candidates.values(),
    )

    # =====================================================
    # 5. Generate hard negatives
    # =====================================================

    hard_negatives = []

    for s1_id, true_matches in ground_truth.items():

        s1 = s1_map.get(s1_id)

        if s1 is None:
            continue

        blocks = generate_candidates(
            s1,
            indexes,
        )

        candidate_ids = (
            blocks["name"]
            |
            blocks["core_name"]
            |
            blocks["address"]
        )

        # Remove actual ground-truth matches.
        candidate_ids -= true_matches

        # Randomize candidate order so we don't
        # systematically select the same block.
        candidate_ids = list(candidate_ids)

        random.shuffle(candidate_ids)

        candidate_ids = candidate_ids[
            :MAX_NEGATIVES_PER_S1
        ]

        for candidate_id in candidate_ids:

            candidate = candidate_records.get(
                candidate_id
            )

            if candidate is None:
                continue

            features = compute_pair_features(
                s1,
                candidate,
            )

            hard_negatives.append(
                (
                    s1,
                    candidate,
                    features,
                )
            )

    # =====================================================
    # 6. Results
    # =====================================================

    print(
        f"\nHard negatives generated: "
        f"{len(hard_negatives)}"
    )

    # =====================================================
    # 7. Feature statistics
    # =====================================================

    if hard_negatives:

        feature_names = list(
            hard_negatives[0][2].keys()
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "HARD NEGATIVE FEATURE STATISTICS"
        )

        print(
            "=" * 80
        )

        print(
            f"\n{'Feature':40} "
            f"{'Mean':>10} "
            f"{'Min':>10} "
            f"{'Max':>10}"
        )

        print(
            "-" * 75
        )

        for feature in feature_names:

            values = [
                pair[2][feature]
                for pair in hard_negatives
            ]

            print(
                f"{feature:40} "
                f"{sum(values) / len(values):10.4f} "
                f"{min(values):10.4f} "
                f"{max(values):10.4f}"
            )

    # =====================================================
    # 8. Examples
    # =====================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "HARD NEGATIVE EXAMPLES"
    )

    print(
        "=" * 80
    )

    for s1, candidate, features in (
        hard_negatives[:15]
    ):

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
            f"  name similarity:    "
            f"{features['name_sequence_similarity']:.3f}"
        )

        print(
            f"  core similarity:    "
            f"{features['core_name_sequence_similarity']:.3f}"
        )

        print(
            f"  address similarity: "
            f"{features['address_sequence_similarity']:.3f}"
        )

        print(
            f"  address overlap:    "
            f"{features['address_token_overlap']:.3f}"
        )

        print(
            f"  number overlap:     "
            f"{features['address_number_overlap']:.3f}"
        )

        print(
            f"  country match:      "
            f"{features['country_match']}"
        )


if __name__ == "__main__":
    main()