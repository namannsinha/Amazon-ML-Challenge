import csv
import re
import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from difflib import SequenceMatcher


# ============================================================
# CONFIG
# ============================================================

S1_PATH = "../../dataset/train/train_source1.tsv"
S2_PATH = "../../dataset/train/train_source2.tsv"
S3_PATH = "../../dataset/train/train_source3.tsv"
GT_PATH = "../../dataset/train/train_ground_truth.tsv"
SAMPLE_S1 = 5000

CHUNK_SIZE = 100_000

# Character n-grams used for retrieval.
NGRAM_SIZES = (3, 4)

# Ignore extremely common n-grams.
# This prevents things like "ing", "the", etc. from creating
# enormous candidate lists.
MAX_POSTING_SIZE = 5000

# Number of candidates retained after inverted-index retrieval
TOP_K = 50

# Maximum number of raw candidates generated for one query
MAX_RAW_CANDIDATES = 5000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(value):
    if value is None:
        return ""

    value = str(value).casefold()

    # Keep letters/numbers/spaces.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def name_ngrams(name):
    """
    Generate character n-grams with boundary markers.

    Example:
        "abc ltd"

    becomes things such as:
        "^ab"
        "abc"
        "bc "
        "c l"
        ...
        "td$"
    """

    name = normalize_name(name)

    if not name:
        return set()

    # Boundary markers help distinguish beginnings/endings.
    text = f"^{name}$"

    grams = set()

    for n in NGRAM_SIZES:
        if len(text) < n:
            continue

        for i in range(len(text) - n + 1):
            grams.add(text[i:i + n])

    return grams


# ============================================================
# FILE HELPERS
# ============================================================

def read_sample_source1(path, sample_size):
    """
    Read a deterministic sample of S1.

    We use the first N rows here so the experiment is reproducible.
    """

    records = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            records.append(row)

            if len(records) >= sample_size:
                break

    return records


def read_ground_truth_for_ids(path, target_ids):
    """
    Load GT only for sampled S1 IDs.
    """

    target_ids = set(target_ids)

    gt = defaultdict(set)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"]

            if s1_id not in target_ids:
                continue

            matched = row.get("matched_entity_ids", "")

            if matched:
                for entity_id in matched.split(","):
                    entity_id = entity_id.strip()

                    if entity_id:
                        gt[s1_id].add(entity_id)

    return gt


# ============================================================
# EXACT BLOCKING
# ============================================================

def build_exact_indexes():
    print("\nBuilding exact indexes...")

    name_index = defaultdict(set)
    core_name_index = defaultdict(set)
    address_index = defaultdict(set)

    for path in [S2_PATH, S3_PATH]:

        print(f"  Reading {path}")

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:

                entity_id = row["entity_id"]

                name = normalize_name(row.get("business_name", ""))

                if name:
                    name_index[name].add(entity_id)

                # Core name:
                # Remove common legal suffixes.
                core = re.sub(
                    r"\b(incorporated|inc|corporation|corp|limited|ltd|"
                    r"private|pvt|company|co|llc|llp)\b",
                    "",
                    name,
                )

                core = re.sub(r"\s+", " ", core).strip()

                if core:
                    core_name_index[core].add(entity_id)

                address = normalize_name(
                    row.get("business_address", "")
                )

                if address:
                    address_index[address].add(entity_id)

    return name_index, core_name_index, address_index


def exact_candidates(record, indexes):
    name_index, core_name_index, address_index = indexes

    candidates = set()

    name = normalize_name(record.get("business_name", ""))

    if name:
        candidates.update(name_index.get(name, set()))

        core = re.sub(
            r"\b(incorporated|inc|corporation|corp|limited|ltd|"
            r"private|pvt|company|co|llc|llp)\b",
            "",
            name,
        )

        core = re.sub(r"\s+", " ", core).strip()

        if core:
            candidates.update(core_name_index.get(core, set()))

    address = normalize_name(
        record.get("business_address", "")
    )

    if address:
        candidates.update(address_index.get(address, set()))

    return candidates


# ============================================================
# APPROXIMATE NAME INDEX
# ============================================================

class ApproximateNameIndex:

    def __init__(self):
        self.postings = defaultdict(list)

        # We keep the actual normalized name for each entity.
        self.entity_names = {}

        # Number of names processed.
        self.entity_count = 0

    def add_record(self, entity_id, name):

        name = normalize_name(name)

        if not name:
            return

        self.entity_names[entity_id] = name

        grams = name_ngrams(name)

        for gram in grams:
            self.postings[gram].append(entity_id)

        self.entity_count += 1

    def prune_postings(self):

        print("\nPruning common n-grams...")

        before = len(self.postings)

        self.postings = {
            gram: ids
            for gram, ids in self.postings.items()
            if len(ids) <= MAX_POSTING_SIZE
        }

        after = len(self.postings)

        print(f"  n-grams before: {before:,}")
        print(f"  n-grams after : {after:,}")

    def retrieve(self, query_name):

        query_name = normalize_name(query_name)

        if not query_name:
            return []

        query_grams = name_ngrams(query_name)

        if not query_grams:
            return []

        # Candidate → number of shared n-grams
        overlap = Counter()

        for gram in query_grams:

            posting = self.postings.get(gram)

            if not posting:
                continue

            for entity_id in posting:
                overlap[entity_id] += 1

        if not overlap:
            return []

        # Keep only a bounded number of candidates before
        # expensive SequenceMatcher calculations.
        if len(overlap) > MAX_RAW_CANDIDATES:

            candidates = [
                entity_id
                for entity_id, _ in
                overlap.most_common(MAX_RAW_CANDIDATES)
            ]

        else:
            candidates = list(overlap.keys())

        # Rerank using actual character similarity.
        scored = []

        for entity_id in candidates:

            candidate_name = self.entity_names[entity_id]

            similarity = SequenceMatcher(
                None,
                query_name,
                candidate_name,
            ).ratio()

            shared = overlap[entity_id]

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

        return scored[:TOP_K]


# ============================================================
# BUILD APPROXIMATE INDEX
# ============================================================

def build_approximate_index():

    index = ApproximateNameIndex()

    total = 0

    for path in [S2_PATH, S3_PATH]:

        print(f"\nBuilding approximate index from {path}")

        with open(path, "r", encoding="utf-8", newline="") as f:

            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:

                index.add_record(
                    row["entity_id"],
                    row.get("business_name", ""),
                )

                total += 1

                if total % 500_000 == 0:
                    print(
                        f"  processed: {total:,}"
                    )

    print(
        f"\nTotal candidate records indexed: "
        f"{total:,}"
    )

    index.prune_postings()

    return index


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    start = time.time()

    print("=" * 70)
    print("APPROXIMATE NAME CANDIDATE RECALL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load S1 sample
    # --------------------------------------------------------

    print("\nLoading S1 sample...")

    s1_records = read_sample_source1(
        S1_PATH,
        SAMPLE_S1,
    )

    print(
        f"S1 records loaded: "
        f"{len(s1_records):,}"
    )

    s1_ids = [
        row["entity_id"]
        for row in s1_records
    ]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print("\nLoading ground truth...")

    gt = read_ground_truth_for_ids(
        GT_PATH,
        set(s1_ids),
    )

    total_true_pairs = sum(
        len(v)
        for v in gt.values()
    )

    print(
        f"True positive pairs: "
        f"{total_true_pairs:,}"
    )

    # --------------------------------------------------------
    # Exact indexes
    # --------------------------------------------------------

    exact_indexes = build_exact_indexes()

    # --------------------------------------------------------
    # Approximate index
    # --------------------------------------------------------

    approximate_index = build_approximate_index()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\nEvaluating candidate recall...")

    exact_recovered_pairs = 0
    approximate_recovered_pairs = 0

    s1_with_exact = 0
    s1_with_approx = 0

    candidate_counts = []

    missed_examples = []

    for i, record in enumerate(s1_records, start=1):

        s1_id = record["entity_id"]

        true_ids = gt.get(s1_id, set())

        if not true_ids:
            continue

        # -------------------------
        # Exact candidates
        # -------------------------

        exact = exact_candidates(
            record,
            exact_indexes,
        )

        exact_hits = true_ids.intersection(exact)

        if exact_hits:
            s1_with_exact += 1
            exact_recovered_pairs += len(exact_hits)

        # -------------------------
        # Approximate name
        # -------------------------

        approx_results = approximate_index.retrieve(
            record.get("business_name", "")
        )

        approx_ids = {
            entity_id
            for _, _, entity_id in approx_results
        }

        # Combine exact + approximate.
        combined = exact | approx_ids

        approx_hits = true_ids.intersection(combined)

        approximate_recovered_pairs += len(approx_hits)

        if approx_hits:
            s1_with_approx += 1

        candidate_counts.append(len(combined))

        # Save examples where approximate retrieval
        # recovered something exact blocking missed.
        newly_recovered = (
            approx_hits - exact_hits
        )

        if newly_recovered and len(missed_examples) < 20:

            missed_examples.append(
                (
                    s1_id,
                    record.get("business_name", ""),
                    newly_recovered,
                    approx_results[:5],
                )
            )

        if i % 500 == 0:
            print(
                f"  evaluated: {i:,}/{len(s1_records):,}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    exact_recall = (
        exact_recovered_pairs /
        total_true_pairs
        if total_true_pairs
        else 0
    )

    approximate_recall = (
        approximate_recovered_pairs /
        total_true_pairs
        if total_true_pairs
        else 0
    )

    print(
        f"\nTrue pairs                  : "
        f"{total_true_pairs:,}"
    )

    print(
        f"Exact recovered             : "
        f"{exact_recovered_pairs:,}"
    )

    print(
        f"Exact recall                : "
        f"{exact_recall * 100:.2f}%"
    )

    print(
        f"\nExact + approximate recovered: "
        f"{approximate_recovered_pairs:,}"
    )

    print(
        f"Combined recall             : "
        f"{approximate_recall * 100:.2f}%"
    )

    print(
        f"\nS1 with exact candidates    : "
        f"{s1_with_exact:,}"
    )

    print(
        f"S1 with approx candidates  : "
        f"{s1_with_approx:,}"
    )

    if candidate_counts:

        print(
            f"\nAverage candidates/S1      : "
            f"{np.mean(candidate_counts):.2f}"
        )

        print(
            f"Median candidates/S1       : "
            f"{np.median(candidate_counts):.0f}"
        )

        print(
            f"Max candidates/S1          : "
            f"{max(candidate_counts):,}"
        )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXAMPLES RECOVERED BY APPROXIMATE NAME")
    print("=" * 70)

    for (
        s1_id,
        s1_name,
        newly_recovered,
        results,
    ) in missed_examples:

        print(f"\n{s1_id}")
        print(f"  S1 name: {s1_name}")

        print(
            f"  Newly recovered: "
            f"{len(newly_recovered)}"
        )

        print("  Top approximate candidates:")

        for similarity, shared, entity_id in results:

            print(
                f"    {entity_id}: "
                f"similarity={similarity:.4f}, "
                f"shared_ngrams={shared}"
            )

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print(
        f"Runtime: {elapsed / 60:.2f} minutes"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()