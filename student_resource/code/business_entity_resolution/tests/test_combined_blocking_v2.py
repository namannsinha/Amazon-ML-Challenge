import csv
import os
import pickle
import re
import time
from collections import defaultdict, Counter
from difflib import SequenceMatcher

import numpy as np


# ============================================================
# PATHS
# ============================================================

S1_PATH = "../../dataset/train/train_source1.tsv"
S2_PATH = "../../dataset/train/train_source2.tsv"
S3_PATH = "../../dataset/train/train_source3.tsv"
GT_PATH = "../../dataset/train/train_ground_truth.tsv"

CACHE_DIR = "tests/generated/blocking_cache"

SAMPLE_S1 = 5000


# ============================================================
# CONFIG
# ============================================================

NGRAM_SIZES = (3, 4)

# Name and address posting limits.
NAME_MAX_POSTING = 5000
ADDRESS_MAX_POSTING = 2000

# Number of raw candidates we allow from each channel.
NAME_MAX_RAW = 3000
ADDRESS_MAX_RAW = 3000

# Final candidates retained from each approximate channel.
NAME_TOP_K = 50
ADDRESS_TOP_K = 50


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

    return re.sub(r"\s+", " ", name).strip()


def ngrams(text):
    """
    Character n-grams with boundary markers.
    """

    text = normalize_text(text)

    if not text:
        return set()

    text = f"^{text}$"

    grams = set()

    for n in NGRAM_SIZES:

        if len(text) < n:
            continue

        for i in range(len(text) - n + 1):
            grams.add(text[i:i + n])

    return grams


# ============================================================
# GENERIC APPROXIMATE INDEX
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

        self.postings = defaultdict(list)
        self.entity_values = {}

    def add(self, entity_id, value):

        value = normalize_text(value)

        if not value:
            return

        self.entity_values[entity_id] = value

        for gram in ngrams(value):
            self.postings[gram].append(entity_id)

    def prune(self):

        print(
            f"\nPruning {self.label} index..."
        )

        before = len(self.postings)

        self.postings = {
            gram: ids
            for gram, ids in self.postings.items()
            if len(ids) <= self.max_posting
        }

        after = len(self.postings)

        print(
            f"  ngrams before: {before:,}"
        )

        print(
            f"  ngrams after : {after:,}"
        )

    def save(self, path):

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "postings": dict(self.postings),
                    "entity_values": self.entity_values,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(
        cls,
        path,
        max_posting,
        max_raw,
        top_k,
        label,
    ):

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

        # ----------------------------------------------------
        # First-stage ranking:
        # shared n-gram count
        # ----------------------------------------------------

        candidates = [
            entity_id
            for entity_id, _ in
            overlap.most_common(self.max_raw)
        ]

        # ----------------------------------------------------
        # Second-stage ranking:
        # SequenceMatcher
        #
        # This is only applied to the bounded candidate set.
        # ----------------------------------------------------

        scored = []

        for entity_id in candidates:

            value = self.entity_values[entity_id]

            similarity = SequenceMatcher(
                None,
                query,
                value,
            ).ratio()

            shared = overlap[entity_id]

            # Slightly prefer candidates that share
            # more n-grams when similarity is similar.
            score = (
                similarity,
                shared,
            )

            scored.append(
                (
                    score,
                    entity_id,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            (
                entity_id,
                score[0],
                score[1],
            )
            for score, entity_id in
            scored[:self.top_k]
        ]


# ============================================================
# LOAD S1
# ============================================================

def load_s1_sample():

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
# LOAD GROUND TRUTH
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

            s1_id = row["source1_entity_id"]

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
                        gt[s1_id].add(entity_id)

    return gt


# ============================================================
# EXACT INDEX
# ============================================================

def build_exact_indexes():

    print("\nBuilding exact indexes...")

    name = defaultdict(set)
    core = defaultdict(set)
    address = defaultdict(set)

    for path in [S2_PATH, S3_PATH]:

        print(f"  Reading {path}")

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

                n = normalize_name(
                    row.get(
                        "business_name",
                        "",
                    )
                )

                a = normalize_address(
                    row.get(
                        "business_address",
                        "",
                    )
                )

                c = get_core_name(n)

                if n:
                    name[n].add(entity_id)

                if c:
                    core[c].add(entity_id)

                if a:
                    address[a].add(entity_id)

    return name, core, address


def exact_candidates(record, indexes):

    name_index, core_index, address_index = indexes

    candidates = set()

    name = normalize_name(
        record.get(
            "business_name",
            "",
        )
    )

    address = normalize_address(
        record.get(
            "business_address",
            "",
        )
    )

    core = get_core_name(name)

    if name:
        candidates.update(
            name_index.get(name, set())
        )

    if core:
        candidates.update(
            core_index.get(core, set())
        )

    if address:
        candidates.update(
            address_index.get(address, set())
        )

    return candidates


# ============================================================
# BUILD / LOAD APPROXIMATE INDEXES
# ============================================================

def get_approx_indexes():

    os.makedirs(
        CACHE_DIR,
        exist_ok=True,
    )

    name_cache = os.path.join(
        CACHE_DIR,
        "name_index.pkl",
    )

    address_cache = os.path.join(
        CACHE_DIR,
        "address_index.pkl",
    )

    # --------------------------------------------------------
    # Already cached?
    # --------------------------------------------------------

    if (
        os.path.exists(name_cache)
        and os.path.exists(address_cache)
    ):

        print("\nLoading cached approximate indexes...")

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

        print(
            f"  Name entries   : "
            f"{len(name_index.entity_values):,}"
        )

        print(
            f"  Address entries: "
            f"{len(address_index.entity_values):,}"
        )

        return name_index, address_index

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    print(
        "\nNo cached indexes found."
    )

    print(
        "Building name + address indexes..."
    )

    name_index = ApproximateIndex(
        NAME_MAX_POSTING,
        NAME_MAX_RAW,
        NAME_TOP_K,
        "name",
    )

    address_index = ApproximateIndex(
        ADDRESS_MAX_POSTING,
        ADDRESS_MAX_RAW,
        ADDRESS_TOP_K,
        "address",
    )

    total = 0

    for path in [S2_PATH, S3_PATH]:

        print(
            f"\nReading {path}"
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

                entity_id = row["entity_id"]

                name_index.add(
                    entity_id,
                    row.get(
                        "business_name",
                        "",
                    ),
                )

                address_index.add(
                    entity_id,
                    row.get(
                        "business_address",
                        "",
                    ),
                )

                total += 1

                if total % 500_000 == 0:

                    print(
                        f"  processed: "
                        f"{total:,}"
                    )

    print(
        f"\nTotal records: "
        f"{total:,}"
    )

    name_index.prune()
    address_index.prune()

    print("\nSaving indexes...")

    name_index.save(name_cache)
    address_index.save(address_cache)

    print("Indexes saved.")

    return name_index, address_index


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 70)
    print("COMBINED CANDIDATE BLOCKING V2")
    print("=" * 70)

    # --------------------------------------------------------
    # S1
    # --------------------------------------------------------

    s1 = load_s1_sample()

    print(
        f"\nS1 sample: "
        f"{len(s1):,}"
    )

    s1_ids = [
        row["entity_id"]
        for row in s1
    ]

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    gt = load_ground_truth(
        s1_ids
    )

    total_true_pairs = sum(
        len(x)
        for x in gt.values()
    )

    print(
        f"True positive pairs: "
        f"{total_true_pairs:,}"
    )

    # --------------------------------------------------------
    # Exact
    # --------------------------------------------------------

    exact_indexes = build_exact_indexes()

    # --------------------------------------------------------
    # Approx
    # --------------------------------------------------------

    name_index, address_index = (
        get_approx_indexes()
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    exact_pairs = 0
    name_pairs = 0
    address_pairs = 0
    combined_pairs = 0

    candidate_counts = []

    name_candidate_counts = []
    address_candidate_counts = []

    recovered_by_name = 0
    recovered_by_address = 0

    examples = []

    print(
        "\nEvaluating..."
    )

    for i, record in enumerate(
        s1,
        start=1,
    ):

        s1_id = record["entity_id"]

        true_ids = gt.get(
            s1_id,
            set(),
        )

        if not true_ids:
            continue

        # ----------------------------------------------------
        # Exact
        # ----------------------------------------------------

        exact = exact_candidates(
            record,
            exact_indexes,
        )

        exact_hits = (
            true_ids & exact
        )

        exact_pairs += len(
            exact_hits
        )

        # ----------------------------------------------------
        # Approx name
        # ----------------------------------------------------

        name_results = (
            name_index.retrieve(
                record.get(
                    "business_name",
                    "",
                )
            )
        )

        name_candidates = {
            entity_id
            for entity_id, _, _
            in name_results
        }

        name_hits = (
            true_ids & name_candidates
        )

        name_pairs += len(
            name_hits
        )

        # ----------------------------------------------------
        # Approx address
        # ----------------------------------------------------

        address_results = (
            address_index.retrieve(
                record.get(
                    "business_address",
                    "",
                )
            )
        )

        address_candidates = {
            entity_id
            for entity_id, _, _
            in address_results
        }

        address_hits = (
            true_ids & address_candidates
        )

        address_pairs += len(
            address_hits
        )

        # ----------------------------------------------------
        # Combined
        # ----------------------------------------------------

        combined = (
            exact
            | name_candidates
            | address_candidates
        )

        combined_hits = (
            true_ids & combined
        )

        combined_pairs += len(
            combined_hits
        )

        # ----------------------------------------------------
        # Stats
        # ----------------------------------------------------

        candidate_counts.append(
            len(combined)
        )

        name_candidate_counts.append(
            len(name_candidates)
        )

        address_candidate_counts.append(
            len(address_candidates)
        )

        # How many matches were newly recovered
        # specifically through address retrieval?
        new_address_hits = (
            address_hits
            - exact_hits
            - name_hits
        )

        recovered_by_address += len(
            new_address_hits
        )

        new_name_hits = (
            name_hits
            - exact_hits
        )

        recovered_by_name += len(
            new_name_hits
        )

        # ----------------------------------------------------
        # Examples
        # ----------------------------------------------------

        if (
            new_address_hits
            and len(examples) < 15
        ):

            examples.append(
                (
                    s1_id,
                    record.get(
                        "business_name",
                        "",
                    ),
                    len(new_address_hits),
                    address_results[:5],
                )
            )

        if i % 500 == 0:

            print(
                f"  evaluated "
                f"{i:,}/{len(s1):,}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print("RESULTS")

    print(
        "=" * 70
    )

    def recall(value):

        if total_true_pairs == 0:
            return 0

        return (
            value
            / total_true_pairs
            * 100
        )

    print(
        f"\nTrue pairs                 : "
        f"{total_true_pairs:,}"
    )

    print(
        f"Exact recovered            : "
        f"{exact_pairs:,}"
    )

    print(
        f"Exact recall               : "
        f"{recall(exact_pairs):.2f}%"
    )

    print(
        f"\nApprox name recovered      : "
        f"{name_pairs:,}"
    )

    print(
        f"Approx name recall         : "
        f"{recall(name_pairs):.2f}%"
    )

    print(
        f"\nApprox address recovered   : "
        f"{address_pairs:,}"
    )

    print(
        f"Approx address recall      : "
        f"{recall(address_pairs):.2f}%"
    )

    print(
        f"\nCombined recovered         : "
        f"{combined_pairs:,}"
    )

    print(
        f"Combined recall            : "
        f"{recall(combined_pairs):.2f}%"
    )

    print(
        f"\nNew recoveries from name   : "
        f"{recovered_by_name:,}"
    )

    print(
        f"New recoveries from address: "
        f"{recovered_by_address:,}"
    )

    # --------------------------------------------------------
    # Candidate counts
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print("CANDIDATE COUNTS")

    print(
        "=" * 70
    )

    print(
        f"\nApprox name avg            : "
        f"{np.mean(name_candidate_counts):.2f}"
    )

    print(
        f"Approx name median         : "
        f"{np.median(name_candidate_counts):.0f}"
    )

    print(
        f"Approx address avg         : "
        f"{np.mean(address_candidate_counts):.2f}"
    )

    print(
        f"Approx address median      : "
        f"{np.median(address_candidate_counts):.0f}"
    )

    print(
        f"\nCombined avg               : "
        f"{np.mean(candidate_counts):.2f}"
    )

    print(
        f"Combined median            : "
        f"{np.median(candidate_counts):.0f}"
    )

    print(
        f"Combined max               : "
        f"{max(candidate_counts):,}"
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EXAMPLES RECOVERED BY ADDRESS"
    )

    print(
        "=" * 70
    )

    for (
        s1_id,
        name,
        count,
        results,
    ) in examples:

        print(
            f"\n{s1_id}"
        )

        print(
            f"  S1 name: {name}"
        )

        print(
            f"  Newly recovered: {count}"
        )

        for (
            entity_id,
            similarity,
            shared,
        ) in results:

            print(
                f"    {entity_id}: "
                f"similarity={similarity:.4f}, "
                f"shared_ngrams={shared}"
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