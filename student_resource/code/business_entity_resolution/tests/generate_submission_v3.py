import csv
import os
import sys
import time
import argparse
import pickle
import re
from collections import defaultdict

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import preprocess_record
from src.features import compute_pair_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

TEST_S1 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source1.tsv")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "generated",
    "combined_gradient_boosting_model.joblib",
)

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    "test_index_v2",
)

RECORDS_PATH = os.path.join(INDEX_DIR, "records.pkl")
PREPROCESSED_PATH = os.path.join(INDEX_DIR, "preprocessed_records.pkl")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

THRESHOLD = 0.875
BATCH_SIZE = 5000


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_preprocessed_cache():
    """
    Build once from the existing V2 records.pkl.

    V2 records.pkl contains compact normalized records:
        entity_id -> (name, core_name, address, country)

    We convert them once to the representation expected by
    compute_pair_features().
    """
    print("=" * 70)
    print("BUILDING PREPROCESSED CANDIDATE CACHE")
    print("=" * 70)

    if not os.path.exists(RECORDS_PATH):
        raise FileNotFoundError(
            f"Missing V2 records cache: {RECORDS_PATH}"
        )

    started = time.time()

    print(f"Loading {RECORDS_PATH}", flush=True)
    records = load_pickle(RECORDS_PATH)

    print(
        f"Loaded {len(records):,} candidate records",
        flush=True,
    )

    preprocessed = {}

    total = len(records)

    for i, (entity_id, record) in enumerate(
        records.items(),
        start=1,
    ):
        # V2 record tuple:
        # (normalized_name, normalized_core_name,
        #  normalized_address, normalized_country)
        name = record[0]
        address = record[2]
        country = record[3]

        preprocessed[entity_id] = preprocess_record(
            entity_id,
            name,
            address,
            country,
        )

        if i % 500000 == 0:
            print(
                f"  {i:,}/{total:,} | "
                f"{(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    print("Saving preprocessed cache...", flush=True)

    tmp_path = PREPROCESSED_PATH + ".tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(
            preprocessed,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        tmp_path,
        PREPROCESSED_PATH,
    )

    size_gb = os.path.getsize(
        PREPROCESSED_PATH
    ) / (1024 ** 3)

    print(
        f"Saved: {PREPROCESSED_PATH}"
    )
    print(
        f"Size: {size_gb:.2f} GB"
    )
    print(
        f"Build time: {(time.time() - started) / 60:.1f} min"
    )

    return preprocessed


def load_preprocessed_cache():
    print(
        f"Loading preprocessed cache: "
        f"{PREPROCESSED_PATH}",
        flush=True,
    )

    started = time.time()

    preprocessed = load_pickle(
        PREPROCESSED_PATH
    )

    print(
        f"Loaded {len(preprocessed):,} "
        f"preprocessed records in "
        f"{(time.time() - started):.1f}s",
        flush=True,
    )

    return preprocessed


def load_v2_indexes():
    """
    Load the existing V2 index.

    We intentionally reuse the proven blocking implementation rather
    than rebuilding it.
    """
    exact = load_pickle(
        os.path.join(INDEX_DIR, "exact.pkl")
    )

    name_index = load_pickle(
        os.path.join(INDEX_DIR, "name_index.pkl")
    )

    address_index = load_pickle(
        os.path.join(INDEX_DIR, "address_index.pkl")
    )

    return exact, name_index, address_index


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

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def core_name(value):
    value = normalize_text(value)

    value = re.sub(
        r"\b(incorporated|inc|corporation|corp|limited|ltd|private|pvt|company|co|llc|llp)\b",
        "",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def ngrams(text):
    text = normalize_text(text)

    if not text:
        return set()

    text = f"^{text}$"

    grams = set()

    for n in (3, 4):
        if len(text) >= n:
            grams.update(
                text[i:i + n]
                for i in range(len(text) - n + 1)
            )

    return grams


def approximate_retrieve(index, query, top_k):
    grams = ngrams(query)

    if not grams:
        return []

    overlap = defaultdict(int)

    for gram in grams:
        for entity_id in index.get(gram, ()):
            overlap[entity_id] += 1

    return [
        entity_id
        for entity_id, _ in sorted(
            overlap.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
    ]


def exact_candidates(
    row,
    exact,
):
    name = normalize_text(
        row.get("business_name", "")
    )

    address = normalize_text(
        row.get("business_address", "")
    )

    core = core_name(name)

    result = set()

    if name:
        result.update(
            exact[0].get(name, ())
        )

    if core:
        result.update(
            exact[1].get(core, ())
        )

    if address:
        result.update(
            exact[2].get(address, ())
        )

    return result


def generate_candidates(
    row,
    exact,
    name_index,
    address_index,
):
    result = exact_candidates(
        row,
        exact,
    )

    result.update(
        approximate_retrieve(
            name_index,
            row.get("business_name", ""),
            10,
        )
    )

    result.update(
        approximate_retrieve(
            address_index,
            row.get("business_address", ""),
            10,
        )
    )

    return result


def read_s1(sample_s1=None):
    rows = []

    with open(
        TEST_S1,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            rows.append(row)

            if (
                sample_s1 is not None
                and len(rows) >= sample_s1
            ):
                break

    return rows


def process_batch(
    batch,
    exact,
    name_index,
    address_index,
    preprocessed_records,
    model,
    feature_columns,
):
    candidate_map = {}

    feature_rows = []
    pair_keys = []

    candidate_start = time.time()

    for row in batch:

        source1_id = row["entity_id"]

        candidates = generate_candidates(
            row,
            exact,
            name_index,
            address_index,
        )

        candidate_map[source1_id] = candidates

    candidate_time = time.time() - candidate_start

    feature_start = time.time()

    for row in batch:

        source1_id = row["entity_id"]

        source1_processed = preprocess_record(
            source1_id,
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", ""),
        )

        for candidate_id in candidate_map[source1_id]:

            candidate = preprocessed_records.get(
                candidate_id
            )

            if candidate is None:
                continue

            features = compute_pair_features(
                source1_processed,
                candidate,
            )

            feature_rows.append(
                [
                    features[column]
                    for column in feature_columns
                ]
            )

            pair_keys.append(
                (
                    source1_id,
                    candidate_id,
                )
            )

    feature_time = time.time() - feature_start

    model_start = time.time()

    if feature_rows:

        X = pd.DataFrame(
            feature_rows,
            columns=feature_columns,
        )

        probabilities = model.predict_proba(X)[:, 1]

    else:
        probabilities = []

    model_time = time.time() - model_start

    predictions = defaultdict(list)

    for (
        source1_id,
        candidate_id,
    ), probability in zip(
        pair_keys,
        probabilities,
    ):

        if probability >= THRESHOLD:
            predictions[source1_id].append(
                candidate_id
            )

    return (
        candidate_map,
        predictions,
        candidate_time,
        feature_time,
        model_time,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-s1",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--rebuild-preprocessed-cache",
        action="store_true",
    )

    args = parser.parse_args()

    started = time.time()

    print("=" * 70)
    print("V3 TEST SUBMISSION GENERATOR")
    print("=" * 70)

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    if (
        args.rebuild_preprocessed_cache
        or not os.path.exists(PREPROCESSED_PATH)
    ):
        preprocessed_records = (
            build_preprocessed_cache()
        )
    else:
        preprocessed_records = (
            load_preprocessed_cache()
        )

    print("Loading V2 blocking indexes...", flush=True)

    index_start = time.time()

    exact, name_index, address_index = (
        load_v2_indexes()
    )

    print(
        f"Blocking indexes loaded in "
        f"{time.time() - index_start:.1f}s",
        flush=True,
    )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]
    feature_columns = model_bundle[
        "feature_columns"
    ]

    s1_rows = read_s1(
        args.sample_s1
    )

    print(
        f"S1 records: {len(s1_rows):,}"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    candidate_path = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv",
    )

    matching_path = os.path.join(
        OUTPUT_DIR,
        "matching_results.tsv",
    )

    total_candidates = 0
    total_matches = 0
    processed = 0

    with open(
        candidate_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as candidate_file, open(
        matching_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as matching_file:

        candidate_writer = csv.writer(
            candidate_file,
            delimiter="\t",
            lineterminator="\n",
        )

        matching_writer = csv.writer(
            matching_file,
            delimiter="\t",
            lineterminator="\n",
        )

        candidate_writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

        matching_writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        for start_idx in range(
            0,
            len(s1_rows),
            args.batch_size,
        ):

            batch = s1_rows[
                start_idx:start_idx + args.batch_size
            ]

            (
                candidate_map,
                predictions,
                candidate_time,
                feature_time,
                model_time,
            ) = process_batch(
                batch,
                exact,
                name_index,
                address_index,
                preprocessed_records,
                model,
                feature_columns,
            )

            batch_candidates = 0
            batch_matches = 0

            for row in batch:

                source1_id = row["entity_id"]

                candidates = sorted(
                    candidate_map.get(
                        source1_id,
                        (),
                    )
                )

                matches = sorted(
                    set(
                        predictions.get(
                            source1_id,
                            (),
                        )
                    )
                )

                candidate_writer.writerow(
                    [
                        source1_id,
                        ",".join(candidates),
                    ]
                )

                matching_writer.writerow(
                    [
                        source1_id,
                        ",".join(matches),
                    ]
                )

                batch_candidates += len(candidates)
                batch_matches += len(matches)

            total_candidates += batch_candidates
            total_matches += batch_matches
            processed += len(batch)

            print(
                f"[PROFILE] "
                f"candidates={candidate_time:.2f}s | "
                f"features={feature_time:.2f}s | "
                f"model={model_time:.2f}s | "
                f"total="
                f"{candidate_time + feature_time + model_time:.2f}s",
                flush=True,
            )

            print(
                f"Completed "
                f"{processed:,}/"
                f"{len(s1_rows):,} S1 | "
                f"candidates="
                f"{total_candidates:,} | "
                f"matches="
                f"{total_matches:,} | "
                f"elapsed="
                f"{(time.time() - started) / 60:.1f} min",
                flush=True,
            )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"S1: {len(s1_rows):,}"
    )

    print(
        f"Candidates: {total_candidates:,}"
    )

    print(
        f"Avg candidates/S1: "
        f"{total_candidates / max(1, len(s1_rows)):.2f}"
    )

    print(
        f"Matches: {total_matches:,}"
    )

    print(
        f"Runtime: "
        f"{(time.time() - started) / 60:.2f} minutes"
    )

    print(candidate_path)
    print(matching_path)


if __name__ == "__main__":
    main()
