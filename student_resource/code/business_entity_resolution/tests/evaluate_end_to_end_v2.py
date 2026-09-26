import csv
import os
import pickle
import re
import time
from collections import defaultdict, Counter
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..")
)

from src.preprocessing import preprocess_record
from src.features import compute_pair_features


# ============================================================
# PATHS
# ============================================================

S1_PATH = "../../dataset/train/train_source1.tsv"
S2_PATH = "../../dataset/train/train_source2.tsv"
S3_PATH = "../../dataset/train/train_source3.tsv"
GT_PATH = "../../dataset/train/train_ground_truth.tsv"

MODEL_PATH = "tests/generated/combined_gradient_boosting_model.joblib"

CACHE_DIR = "tests/generated/blocking_cache"

SAMPLE_S1 = 5000


# ============================================================
# BLOCKING CONFIG
# ============================================================

NGRAM_SIZES = (3, 4)

NAME_MAX_POSTING = 5000
ADDRESS_MAX_POSTING = 2000

NAME_MAX_RAW = 3000
ADDRESS_MAX_RAW = 3000

NAME_TOP_K = 50
ADDRESS_TOP_K = 50


# ============================================================
# THRESHOLDS
# ============================================================

# THRESHOLDS = [
#     0.70,
#     0.725,
#     0.75,
#     0.775,
#     0.80,
#     0.825,
#     0.85,
#     0.875,
#     0.90,
# ]
THRESHOLDS = 0.875

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

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def normalize_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)


def get_core_name(name):

    name = normalize_name(name)

    name = re.sub(
        r"\b("
        r"incorporated|inc|"
        r"corporation|corp|"
        r"limited|ltd|"
        r"private|pvt|"
        r"company|co|"
        r"llc|llp"
        r")\b",
        "",
        name,
    )

    return re.sub(
        r"\s+",
        " ",
        name,
    ).strip()


def ngrams(text):

    text = normalize_text(text)

    if not text:
        return set()

    text = f"^{text}$"

    grams = set()

    for n in NGRAM_SIZES:

        if len(text) < n:
            continue

        for i in range(len(text) - n + 1):
            grams.add(text[i:i+n])

    return grams


# ============================================================
# APPROXIMATE INDEX
# ============================================================

class ApproximateIndex:

    def __init__(
        self,
        max_posting,
        max_raw,
        top_k,
        label,
    ):

        self.max_posting = max_posting
        self.max_raw = max_raw
        self.top_k = top_k
        self.label = label

        self.postings = {}
        self.entity_values = {}

    @classmethod
    def load(
        cls,
        path,
        max_posting,
        max_raw,
        top_k,
        label,
    ):

        print(
            f"Loading {label} index..."
        )

        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            max_posting,
            max_raw,
            top_k,
            label,
        )

        obj.postings = data["postings"]
        obj.entity_values = data["entity_values"]

        print(
            f"  entities: "
            f"{len(obj.entity_values):,}"
        )

        print(
            f"  ngrams: "
            f"{len(obj.postings):,}"
        )

        return obj

    def retrieve(self, query):

        query = normalize_text(query)

        if not query:
            return []

        query_grams = ngrams(query)

        if not query_grams:
            return []

        overlap = Counter()

        for gram in query_grams:

            posting = self.postings.get(gram)

            if not posting:
                continue

            for entity_id in posting:
                overlap[entity_id] += 1

        if not overlap:
            return []

        candidates = [
            entity_id
            for entity_id, _
            in overlap.most_common(
                self.max_raw
            )
        ]

        scored = []

        query_len = max(
            len(query_grams),
            1,
        )

        for entity_id in candidates:

            candidate = self.entity_values[
                entity_id
            ]

            candidate_grams = ngrams(
                candidate
            )

            shared = overlap[
                entity_id
            ]

            candidate_len = max(
                len(candidate_grams),
                1,
            )

            # Character n-gram Jaccard-like score
            union = (
                query_len
                + candidate_len
                - shared
            )

            jaccard = (
                shared / union
                if union
                else 0.0
            )

            overlap_score = (
                shared / query_len
            )

            sequence = SequenceMatcher(
                None,
                query,
                candidate,
            ).ratio()

            # Retrieval-only score.
            score = (
                0.50 * sequence
                + 0.30 * jaccard
                + 0.20 * overlap_score
            )

            scored.append(
                (
                    score,
                    entity_id,
                )
            )

        scored.sort(
            reverse=True
        )

        return [
            entity_id
            for _, entity_id
            in scored[:self.top_k]
        ]


# ============================================================
# LOAD S1
# ============================================================

def load_s1():

    records = []

    with open(
        S1_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            records.append(row)

            if len(records) >= SAMPLE_S1:
                break

    return records


# ============================================================
# LOAD SOURCE 2 + SOURCE 3
# ============================================================

def load_candidates():

    print(
        "\nLoading S2/S3 candidate records..."
    )

    records = {}

    for path in [S2_PATH, S3_PATH]:

        print(
            f"  Reading {path}"
        )

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

                entity_id = row[
                    "entity_id"
                ]

                records[entity_id] = preprocess_record(
                    entity_id,
                    row.get("business_name", ""),
                    row.get("business_address", ""),
                    row.get("country", ""),
                )

    print(
        f"Candidate records loaded: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(ids):

    ids = set(ids)

    gt = defaultdict(set)

    with open(
        GT_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            s1_id = row[
                "source1_entity_id"
            ]

            if s1_id not in ids:
                continue

            matches = row.get(
                "matched_entity_ids",
                "",
            )

            if matches:

                for entity_id in matches.split(","):

                    entity_id = entity_id.strip()

                    if entity_id:
                        gt[s1_id].add(
                            entity_id
                        )

    return gt


# ============================================================
# EXACT INDEX
# ============================================================

def build_exact_indexes():

    print(
        "\nBuilding exact indexes..."
    )

    name = defaultdict(set)
    core = defaultdict(set)
    address = defaultdict(set)

    for path in [S2_PATH, S3_PATH]:

        print(
            f"  Reading {path}"
        )

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

                entity_id = row[
                    "entity_id"
                ]

                n = normalize_name(
                    row.get(
                        "business_name",
                        "",
                    )
                )

                c = get_core_name(n)

                a = normalize_address(
                    row.get(
                        "business_address",
                        "",
                    )
                )

                if n:
                    name[n].add(
                        entity_id
                    )

                if c:
                    core[c].add(
                        entity_id
                    )

                if a:
                    address[a].add(
                        entity_id
                    )

    return name, core, address


def exact_candidates(
    record,
    indexes,
):

    name_index, core_index, address_index = (
        indexes
    )

    candidates = set()

    name = normalize_name(
        record.get(
            "business_name",
            "",
        )
    )

    core = get_core_name(
        name
    )

    address = normalize_address(
        record.get(
            "business_address",
            "",
        )
    )

    if name:
        candidates.update(
            name_index.get(
                name,
                set(),
            )
        )

    if core:
        candidates.update(
            core_index.get(
                core,
                set(),
            )
        )

    if address:
        candidates.update(
            address_index.get(
                address,
                set(),
            )
        )

    return candidates



# ============================================================
# ENTITY-LEVEL METRICS
# ============================================================

def entity_f05(
    predictions,
    ground_truth,
    s1_ids,
):

    total_f05 = 0.0

    total_precision = 0.0
    total_recall = 0.0

    for s1_id in s1_ids:

        predicted = set(
            predictions.get(
                s1_id,
                set(),
            )
        )

        actual = set(
            ground_truth.get(
                s1_id,
                set(),
            )
        )

        if not predicted and not actual:

            precision = 1.0
            recall = 1.0

        elif not predicted and actual:

            precision = 1.0
            recall = 0.0

        elif predicted and not actual:

            precision = 0.0
            recall = 1.0

        else:

            tp = len(
                predicted & actual
            )

            precision = (
                tp / len(predicted)
            )

            recall = (
                tp / len(actual)
            )

        beta2 = 0.25

        denominator = (
            beta2 * precision
            + recall
        )

        if denominator == 0:

            f05 = 0.0

        else:

            f05 = (
                (1 + beta2)
                * precision
                * recall
                / denominator
            )

        total_f05 += f05
        total_precision += precision
        total_recall += recall

    n = len(s1_ids)

    return (
        total_f05 / n,
        total_precision / n,
        total_recall / n,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print(
        "=" * 70
    )

    print(
        "END-TO-END V2 EVALUATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    s1_records = load_s1()
    processed_s1_records = []

    for record in s1_records:
        processed_s1_records.append(
            preprocess_record(
                record["entity_id"],
                record.get("business_name", ""),
                record.get("business_address", ""),
                record.get("country", ""),
            )
        )

    s1_ids = [
        x["entity_id"]
        for x in s1_records
    ]

    print(
        f"\nS1 records: "
        f"{len(s1_records):,}"
    )

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        s1_ids
    )

    print(
        f"GT positive pairs: "
        f"{sum(len(x) for x in ground_truth.values()):,}"
    )

    # --------------------------------------------------------
    # Candidate records
    # --------------------------------------------------------

    candidate_records = load_candidates()

    # --------------------------------------------------------
    # Exact indexes
    # --------------------------------------------------------

    exact_indexes = build_exact_indexes()

    # --------------------------------------------------------
    # Approx indexes
    # --------------------------------------------------------

    name_cache = os.path.join(
        CACHE_DIR,
        "name_index.pkl",
    )

    address_cache = os.path.join(
        CACHE_DIR,
        "address_index.pkl",
    )

    name_index = ApproximateIndex.load(
        name_cache,
        NAME_MAX_POSTING,
        NAME_MAX_RAW,
        NAME_TOP_K,
        "name",
    )

    address_index = ApproximateIndex.load(
        address_cache,
        ADDRESS_MAX_POSTING,
        ADDRESS_MAX_RAW,
        ADDRESS_TOP_K,
        "address",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading Gradient Boosting model..."
    )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    print(
        f"Model loaded: "
        f"{type(model).__name__}"
    )

    print(
        f"Saved feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"Saved threshold: "
        f"{model_bundle['threshold']}"
    )

    # --------------------------------------------------------
    # Generate candidate sets
    # --------------------------------------------------------

    print(
        "\nGenerating candidates..."
    )

    candidate_sets = {}

    total_candidates = 0

    for i, s1 in enumerate(
        s1_records,
        start=1,
    ):

        s1_id = s1[
            "entity_id"
        ]

        exact = exact_candidates(
            s1,
            exact_indexes,
        )

        approx_name = set(
            name_index.retrieve(
                s1.get(
                    "business_name",
                    "",
                )
            )
        )

        approx_address = set(
            address_index.retrieve(
                s1.get(
                    "business_address",
                    "",
                )
            )
        )

        combined = (
            exact
            | approx_name
            | approx_address
        )

        candidate_sets[
            s1_id
        ] = combined

        total_candidates += len(
            combined
        )

        if i % 500 == 0:

            print(
                f"  generated "
                f"{i:,}/{len(s1_records):,}"
            )

    print(
        f"\nTotal candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{total_candidates / len(s1_records):.2f}"
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print(
        "\nComputing features..."
    )

    feature_rows = []
    pair_keys = []

    processed = 0

    for raw_s1, processed_s1 in zip(
        s1_records,
        processed_s1_records,
    ):

        s1_id = raw_s1["entity_id"]

        for candidate_id in candidate_sets[s1_id]:

            candidate = candidate_records.get(candidate_id)

            if candidate is None:
                continue

            features = compute_pair_features(
                processed_s1,
                candidate,
            )

            feature_rows.append(
                [features[column] for column in feature_columns]
            )

            pair_keys.append(
                (
                    s1_id,
                    candidate_id,
                )
            )

        processed += 1

        if processed % 500 == 0:
            print(
                f"  processed "
                f"{processed:,}/{len(s1_records):,}"
            )

    X = pd.DataFrame(feature_rows, columns=feature_columns)

    print(f"Feature matrix: {X.shape}")

    if list(X.columns) != list(feature_columns):
        raise ValueError(
            f"Feature column mismatch!\n"
            f"Expected: {feature_columns}\n"
            f"Actual:   {list(X.columns)}"
        )

    print("\nFeature means:")
    print(X.mean())

    print(
        f"\nFeature matrix: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        "\nRunning model..."
    )

    if X.shape[1] != len(feature_columns):
        raise ValueError(
            f"Feature mismatch: X has {X.shape[1]} columns "
            f"but model expects {len(feature_columns)}"
        )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    print("\nProbability distribution:")
    print(f"min : {probabilities.min():.6f}")
    print(f"max : {probabilities.max():.6f}")
    print(f"mean: {probabilities.mean():.6f}")
    print(f"p50 : {np.percentile(probabilities, 50):.6f}")
    print(f"p90 : {np.percentile(probabilities, 90):.6f}")
    print(f"p95 : {np.percentile(probabilities, 95):.6f}")
    print(f"p99 : {np.percentile(probabilities, 99):.6f}")

    # --------------------------------------------------------
    # Threshold evaluation
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ENTITY-LEVEL THRESHOLD RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\n{'Threshold':>10}"
        f"{'F0.5':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'Matches':>12}"
    )

    print(
        "-" * 60
    )

    best = None

    for threshold in THRESHOLDS:

        predictions = defaultdict(set)

        for (
            (s1_id, candidate_id),
            probability,
        ) in zip(
            pair_keys,
            probabilities,
        ):

            if probability >= threshold:

                predictions[
                    s1_id
                ].add(
                    candidate_id
                )

        f05, precision, recall = (
            entity_f05(
                predictions,
                ground_truth,
                s1_ids,
            )
        )

        matches = sum(
            len(x)
            for x in predictions.values()
        )

        print(
            f"{threshold:10.3f}"
            f"{f05:12.4f}"
            f"{precision:12.4f}"
            f"{recall:12.4f}"
            f"{matches:12,}"
        )

        if (
            best is None
            or f05 > best[0]
        ):

            best = (
                f05,
                threshold,
                precision,
                recall,
                matches,
            )

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST ENTITY-LEVEL RESULT"
    )

    print(
        "=" * 70
    )

    if best:

        print(
            f"\nF0.5       : "
            f"{best[0]:.4f}"
        )

        print(
            f"Threshold  : "
            f"{best[1]:.3f}"
        )

        print(
            f"Precision  : "
            f"{best[2]:.4f}"
        )

        print(
            f"Recall     : "
            f"{best[3]:.4f}"
        )

        print(
            f"Matches    : "
            f"{best[4]:,}"
        )

    elapsed = (
        time.time() - start
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()