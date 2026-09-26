import os
import random
import time
import pickle
import re
import sys

import pandas as pd

# Allow imports from business_entity_resolution/
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

from src.preprocessing import preprocess_record
from src.features import compute_pair_features


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = "../../dataset/train"
OUTPUT_DIR = "tests/generated"

S1_PATH = f"{TRAIN_DIR}/train_source1.tsv"
S2_PATH = f"{TRAIN_DIR}/train_source2.tsv"
S3_PATH = f"{TRAIN_DIR}/train_source3.tsv"
GT_PATH = f"{TRAIN_DIR}/train_ground_truth.tsv"

CACHE_DIR = "tests/generated/blocking_cache"

NAME_CACHE = f"{CACHE_DIR}/name_index.pkl"
ADDRESS_CACHE = f"{CACHE_DIR}/address_index.pkl"


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 5000
RANDOM_SEED = 42

# Match the existing approximate blocker
NAME_TOP_K = 50
ADDRESS_TOP_K = 50

# Keep training dataset manageable.
# We take the top approximate candidates, but cap negatives
# per S1 after combining all channels.
MAX_APPROX_NEGATIVES_PER_S1 = 20

CHUNK_SIZE = 100_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).casefold()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(r"\s+", " ", value).strip()

    return value


def ngrams(text):
    text = normalize_text(text)

    if not text:
        return set()

    text = f"^{text}$"

    grams = set()

    for n in (3, 4):
        if len(text) < n:
            continue

        for i in range(len(text) - n + 1):
            grams.add(text[i:i + n])

    return grams


# ============================================================
# APPROXIMATE INDEX
# ============================================================

class ApproximateIndex:

    def __init__(self, postings, entity_values, top_k):
        self.postings = postings
        self.entity_values = entity_values
        self.top_k = top_k

    def retrieve(self, query):

        query = normalize_text(query)

        if not query:
            return []

        query_grams = ngrams(query)

        if not query_grams:
            return []

        overlap = {}

        for gram in query_grams:

            posting = self.postings.get(gram)

            if not posting:
                continue

            for entity_id in posting:

                overlap[entity_id] = (
                    overlap.get(entity_id, 0) + 1
                )

        if not overlap:
            return []

        # Same general first-stage ranking as the existing blocker
        candidates = sorted(
            overlap.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3000]

        scored = []

        from difflib import SequenceMatcher

        for entity_id, shared in candidates:

            value = self.entity_values[entity_id]

            similarity = SequenceMatcher(
                None,
                query,
                value,
            ).ratio()

            scored.append(
                (
                    similarity,
                    shared,
                    entity_id,
                )
            )

        scored.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        return scored[:self.top_k]


# ============================================================
# LOAD CACHED INDEX
# ============================================================

def load_index(path, top_k):

    print(f"Loading {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    return ApproximateIndex(
        data["postings"],
        data["entity_values"],
        top_k,
    )


# ============================================================
# LOAD SAMPLE + GT
# ============================================================

def load_sample():

    print("Loading ground truth...")

    gt_df = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt_sample = gt_df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    )

    s1_ids = set(
        gt_sample["source1_entity_id"]
    )

    ground_truth = {}

    for _, row in gt_sample.iterrows():

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

    print(
        f"Sampled S1 entities: {len(s1_ids):,}"
    )

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

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

    return s1_records, ground_truth


# ============================================================
# LOAD ONLY TRUE CANDIDATE RECORDS
# ============================================================

def load_true_candidates(ground_truth):

    true_ids = set()

    for matches in ground_truth.values():
        true_ids.update(matches)

    records = {}

    print(
        f"Loading {len(true_ids):,} true candidate IDs..."
    )

    for path in [S2_PATH, S3_PATH]:

        with open(
            path,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            import csv

            reader = csv.DictReader(
                f,
                delimiter="\t",
            )

            for row in reader:

                entity_id = row["entity_id"]

                if entity_id not in true_ids:
                    continue

                name = row.get(
                    "business_name",
                    "",
                )

                address = row.get(
                    "business_address",
                    "",
                )

                country = row.get(
                    "country",
                    "",
                )

                if name is None:
                    name = ""

                if address is None:
                    address = ""

                if country is None:
                    country = ""

                records[entity_id] = preprocess_record(
                    entity_id,
                    name,
                    address,
                    country,
                )

    print(
        f"Loaded true candidates: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# SCAN SOURCES FOR APPROXIMATE NEGATIVE RECORDS
# ============================================================

def scan_approximate_candidates(
    s1_records,
    ground_truth,
    name_index,
    address_index,
):

    # candidate IDs that survive approximate retrieval
    candidate_ids = set()

    per_s1_candidates = {}

    print("\nGenerating approximate candidates...")

    for i, (s1_id, s1) in enumerate(
        s1_records.items(),
        start=1,
    ):

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        candidates = set()

        # ----------------------------------------------------
        # Approximate name
        # ----------------------------------------------------

        name_results = name_index.retrieve(
            s1["name_normalized"]
        )

        for item in name_results:

            entity_id = item[2]

            candidates.add(entity_id)

        # ----------------------------------------------------
        # Approximate address
        # ----------------------------------------------------

        address_results = address_index.retrieve(
            s1["address_normalized"]
        )

        for item in address_results:

            entity_id = item[2]

            candidates.add(entity_id)

        # Remove positives
        candidates -= true_ids

        # Cap per S1
        candidates = list(candidates)

        if len(candidates) > MAX_APPROX_NEGATIVES_PER_S1:

            # Deterministic random sampling
            rng = random.Random(
                RANDOM_SEED + i
            )

            candidates = rng.sample(
                candidates,
                MAX_APPROX_NEGATIVES_PER_S1,
            )

        per_s1_candidates[s1_id] = candidates

        candidate_ids.update(candidates)

        if i % 500 == 0:

            print(
                f"  processed "
                f"{i:,}/{len(s1_records):,}"
            )

    print(
        f"\nUnique approximate negative IDs: "
        f"{len(candidate_ids):,}"
    )

    return per_s1_candidates, candidate_ids


# ============================================================
# LOAD APPROX NEGATIVE RECORDS
# ============================================================

def load_candidate_records(candidate_ids):

    records = {}

    print(
        f"\nScanning S2/S3 for "
        f"{len(candidate_ids):,} candidate records..."
    )

    import csv

    for path in [S2_PATH, S3_PATH]:

        with open(
            path,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            reader = csv.DictReader(
                f,
                delimiter="\t",
            )

            for row in reader:

                entity_id = row["entity_id"]

                if entity_id not in candidate_ids:
                    continue

                name = row.get(
                    "business_name",
                    "",
                )

                address = row.get(
                    "business_address",
                    "",
                )

                country = row.get(
                    "country",
                    "",
                )

                if name is None:
                    name = ""

                if address is None:
                    address = ""

                if country is None:
                    country = ""

                records[entity_id] = preprocess_record(
                    entity_id,
                    name,
                    address,
                    country,
                )

    print(
        f"Loaded candidate records: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# BUILD POSITIVES
# ============================================================

def build_positive_pairs(
    s1_records,
    ground_truth,
    true_records,
):

    positives = []

    missing = 0

    for s1_id, true_ids in ground_truth.items():

        s1 = s1_records[s1_id]

        for candidate_id in true_ids:

            candidate = true_records.get(
                candidate_id
            )

            if candidate is None:

                missing += 1
                continue

            features = compute_pair_features(
                s1,
                candidate,
            )

            features["label"] = 1
            features["s1_entity_id"] = s1_id
            features["candidate_entity_id"] = candidate_id

            positives.append(features)

    print(
        f"\nPositive pairs: {len(positives):,}"
    )

    print(
        f"Missing positives: {missing:,}"
    )

    return positives


# ============================================================
# BUILD APPROXIMATE HARD NEGATIVES
# ============================================================

def build_negative_pairs(
    s1_records,
    ground_truth,
    per_s1_candidates,
    candidate_records,
):

    negatives = []

    for i, (
        s1_id,
        candidate_ids,
    ) in enumerate(
        per_s1_candidates.items(),
        start=1,
    ):

        s1 = s1_records[s1_id]

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        for candidate_id in candidate_ids:

            # Safety check
            if candidate_id in true_ids:
                continue

            candidate = candidate_records.get(
                candidate_id
            )

            if candidate is None:
                continue

            features = compute_pair_features(
                s1,
                candidate,
            )

            features["label"] = 0
            features["s1_entity_id"] = s1_id
            features["candidate_entity_id"] = candidate_id

            negatives.append(features)

        if i % 500 == 0:

            print(
                f"  feature pairs for "
                f"{i:,}/{len(per_s1_candidates):,}"
            )

    print(
        f"\nApproximate hard negatives: "
        f"{len(negatives):,}"
    )

    return negatives


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 70)
    print("APPROXIMATE HARD-NEGATIVE TRAINING DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load sample + GT
    # --------------------------------------------------------

    s1_records, ground_truth = load_sample()

    # --------------------------------------------------------
    # 2. Load approximate indexes
    # --------------------------------------------------------

    name_index = load_index(
        NAME_CACHE,
        NAME_TOP_K,
    )

    address_index = load_index(
        ADDRESS_CACHE,
        ADDRESS_TOP_K,
    )

    # --------------------------------------------------------
    # 3. Generate approximate candidates
    # --------------------------------------------------------

    (
        per_s1_candidates,
        candidate_ids,
    ) = scan_approximate_candidates(
        s1_records,
        ground_truth,
        name_index,
        address_index,
    )

    # --------------------------------------------------------
    # 4. Load actual candidate records
    # --------------------------------------------------------

    candidate_records = load_candidate_records(
        candidate_ids
    )

    # --------------------------------------------------------
    # 5. Load true records
    # --------------------------------------------------------

    true_records = load_true_candidates(
        ground_truth
    )

    # --------------------------------------------------------
    # 6. Positives
    # --------------------------------------------------------

    positive_pairs = build_positive_pairs(
        s1_records,
        ground_truth,
        true_records,
    )

    # --------------------------------------------------------
    # 7. Approximate hard negatives
    # --------------------------------------------------------

    negative_pairs = build_negative_pairs(
        s1_records,
        ground_truth,
        per_s1_candidates,
        candidate_records,
    )

    # --------------------------------------------------------
    # 8. Combine
    # --------------------------------------------------------

    df = pd.DataFrame(
        positive_pairs + negative_pairs
    )

    df = df.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 9. Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("APPROXIMATE TRAINING DATASET")
    print("=" * 70)

    print(
        f"Total pairs    : {len(df):,}"
    )

    print(
        f"Positive pairs : "
        f"{(df['label'] == 1).sum():,}"
    )

    print(
        f"Negative pairs : "
        f"{(df['label'] == 0).sum():,}"
    )

    print(
        f"Positive ratio : "
        f"{df['label'].mean():.4f}"
    )

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = (
        f"{OUTPUT_DIR}/"
        "approximate_training_pairs.tsv"
    )

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved:\n{output_path}"
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()