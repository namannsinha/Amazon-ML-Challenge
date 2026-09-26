import random

import pandas as pd

from src.preprocessing import preprocess_record
from src.features import compute_pair_features


S1_PATH = "../../dataset/train/train_source1.tsv"
S2_PATH = "../../dataset/train/train_source2.tsv"
S3_PATH = "../../dataset/train/train_source3.tsv"
GROUND_TRUTH_PATH = "../../dataset/train/train_ground_truth.tsv"

RANDOM_SEED = 42

# Number of S1 entities to sample
SAMPLE_SIZE = 5000

# Number of negative candidates to collect per S1
NEGATIVES_PER_S1 = 3


# ---------------------------------------------------------
# Load ground truth sample
# ---------------------------------------------------------

def load_ground_truth_sample(path, sample_size):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Ground truth rows available: {len(df)}"
    )

    sample_size = min(
        sample_size,
        len(df)
    )

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )

    ground_truth = {}

    for _, row in sample.iterrows():

        s1_id = row["source1_entity_id"]

        matched_ids = {
            x.strip()
            for x in row["matched_entity_ids"].split(",")
            if x.strip()
        }

        ground_truth[s1_id] = matched_ids

    return ground_truth


# ---------------------------------------------------------
# Load only required records
# ---------------------------------------------------------

def load_required_records(
    path,
    required_ids,
    chunksize=100_000
):

    required_ids = set(required_ids)

    records = {}

    print(
        f"\nScanning {path}"
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):

        matches = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in matches.iterrows():

            records[row["entity_id"]] = (
                preprocess_record(
                    entity_id=row["entity_id"],
                    business_name=row["business_name"],
                    business_address=row["business_address"],
                    country=row["country"],
                )
            )

        if len(records) == len(required_ids):
            break

    print(
        f"Loaded required records: "
        f"{len(records)}/{len(required_ids)}"
    )

    return records


# ---------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------

def summarize_features(
    pairs,
    title
):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not pairs:
        print("No pairs.")
        return

    feature_names = list(
        pairs[0][2].keys()
    )

    print(
        f"\nNumber of pairs: {len(pairs)}"
    )

    print(
        f"\n{'Feature':40} "
        f"{'Mean':>10} "
        f"{'Min':>10} "
        f"{'Max':>10}"
    )

    print("-" * 75)

    for feature in feature_names:

        values = [
            pair[2][feature]
            for pair in pairs
        ]

        print(
            f"{feature:40} "
            f"{sum(values) / len(values):10.4f} "
            f"{min(values):10.4f} "
            f"{max(values):10.4f}"
        )


# ---------------------------------------------------------
# Print examples
# ---------------------------------------------------------

def print_examples(
    pairs,
    title,
    count=10
):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for s1, candidate, features in pairs[:count]:

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

        print("Features:")

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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    random.seed(RANDOM_SEED)

    # =====================================================
    # 1. Sample ground truth
    # =====================================================

    print("Sampling ground truth...")

    ground_truth = load_ground_truth_sample(
        GROUND_TRUTH_PATH,
        SAMPLE_SIZE
    )

    print(
        f"Sampled S1 entities: "
        f"{len(ground_truth)}"
    )

    # =====================================================
    # 2. Collect required IDs
    # =====================================================

    s1_ids = set(
        ground_truth.keys()
    )

    matched_ids = set()

    for ids in ground_truth.values():
        matched_ids.update(ids)

    s2_ids = {
        x
        for x in matched_ids
        if x.startswith("S2-")
    }

    s3_ids = {
        x
        for x in matched_ids
        if x.startswith("S3-")
    }

    print(
        f"Required S1 records: "
        f"{len(s1_ids)}"
    )

    print(
        f"Required S2 records: "
        f"{len(s2_ids)}"
    )

    print(
        f"Required S3 records: "
        f"{len(s3_ids)}"
    )

    # =====================================================
    # 3. Load only required records
    # =====================================================

    s1_map = load_required_records(
        S1_PATH,
        s1_ids
    )

    s2_map = load_required_records(
        S2_PATH,
        s2_ids
    )

    s3_map = load_required_records(
        S3_PATH,
        s3_ids
    )

    candidate_map = {
        **s2_map,
        **s3_map
    }

    # =====================================================
    # 4. Build positive pairs
    # =====================================================

    print("\nBuilding positive pairs...")

    positive_pairs = []

    for s1_id, matched in ground_truth.items():

        s1 = s1_map.get(s1_id)

        if s1 is None:
            continue

        for candidate_id in matched:

            candidate = candidate_map.get(
                candidate_id
            )

            if candidate is None:
                continue

            features = compute_pair_features(
                s1,
                candidate
            )

            positive_pairs.append(
                (
                    s1,
                    candidate,
                    features
                )
            )

    print(
        f"Positive pairs: "
        f"{len(positive_pairs)}"
    )

    # =====================================================
    # 5. Create negative pairs
    #
    # IMPORTANT:
    # These are random negatives for now.
    # Hard negatives will come later after our
    # candidate generator is integrated.
    # =====================================================

    print("\nBuilding negative pairs...")

    negative_pairs = []

    all_candidate_ids = list(
        candidate_map.keys()
    )

    for s1_id, matched in ground_truth.items():

        s1 = s1_map.get(s1_id)

        if s1 is None:
            continue

        true_matches = set(matched)

        attempts = 0

        while (
            attempts < NEGATIVES_PER_S1
            and all_candidate_ids
        ):

            candidate_id = random.choice(
                all_candidate_ids
            )

            if candidate_id in true_matches:
                continue

            candidate = candidate_map.get(
                candidate_id
            )

            if candidate is None:
                continue

            features = compute_pair_features(
                s1,
                candidate
            )

            negative_pairs.append(
                (
                    s1,
                    candidate,
                    features
                )
            )

            attempts += 1

    print(
        f"Negative pairs: "
        f"{len(negative_pairs)}"
    )

    # =====================================================
    # 6. Analyze
    # =====================================================

    summarize_features(
        positive_pairs,
        "POSITIVE PAIR FEATURES"
    )

    summarize_features(
        negative_pairs,
        "NEGATIVE PAIR FEATURES"
    )

    # =====================================================
    # 7. Examples
    # =====================================================

    print_examples(
        positive_pairs,
        "POSITIVE PAIR EXAMPLES"
    )

    print_examples(
        negative_pairs,
        "NEGATIVE PAIR EXAMPLES"
    )


if __name__ == "__main__":
    main()