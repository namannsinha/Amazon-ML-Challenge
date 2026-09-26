import csv
import os
import sys
import time
import argparse
import multiprocessing as mp
from collections import defaultdict, Counter

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import preprocess_record
from src.features import compute_pair_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEST_S1 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source1.tsv")
TEST_S2 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source2.tsv")
TEST_S3 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source3.tsv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "generated", "combined_gradient_boosting_model.joblib")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

THRESHOLD = 0.875
NGRAM_SIZES = (3, 4)
NAME_MAX_POSTING = 5000
ADDRESS_MAX_POSTING = 2000
NAME_TOP_K = 10
ADDRESS_TOP_K = 10
BATCH_SIZE = 5000
WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

# These globals are populated once in the parent and inherited by forked workers.
G_S1 = None
G_PROCESSED_S1 = None
G_CANDIDATES = None
G_EXACT = None
G_NAME_INDEX = None
G_ADDRESS_INDEX = None
G_RECORDS = None
G_MODEL = None
G_FEATURE_COLUMNS = None


def normalize_text(value):
    if value is None:
        return ""
    import re
    value = str(value).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def ngrams(text):
    text = normalize_text(text)
    if not text:
        return set()
    text = f"^{text}$"
    grams = set()
    for n in NGRAM_SIZES:
        if len(text) >= n:
            grams.update(text[i:i+n] for i in range(len(text) - n + 1))
    return grams


class ApproxIndex:
    def __init__(self, max_posting, top_k, label):
        self.max_posting = max_posting
        self.top_k = top_k
        self.label = label
        self.postings = defaultdict(list)

    def add(self, entity_id, value):
        value = normalize_text(value)
        if not value:
            return
        for gram in ngrams(value):
            self.postings[gram].append(entity_id)

    def prune(self):
        self.postings = {
            gram: ids for gram, ids in self.postings.items()
            if len(ids) <= self.max_posting
        }

    def retrieve(self, query):
        query_grams = ngrams(query)
        if not query_grams:
            return []
        overlap = Counter()
        for gram in query_grams:
            for entity_id in self.postings.get(gram, ()):
                overlap[entity_id] += 1
        return [eid for eid, _ in overlap.most_common(self.top_k)]


def core_name(value):
    import re
    value = normalize_text(value)
    value = re.sub(
        r"\b(incorporated|inc|corporation|corp|limited|ltd|private|pvt|company|co|llc|llp)\b",
        "",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()


def build_indexes():
    print("Building test indexes / candidate records...")
    exact_name = defaultdict(set)
    exact_core = defaultdict(set)
    exact_address = defaultdict(set)

    name_index = ApproxIndex(NAME_MAX_POSTING, NAME_TOP_K, "name")
    address_index = ApproxIndex(ADDRESS_MAX_POSTING, ADDRESS_TOP_K, "address")
    records = {}

    total = 0
    for path in (TEST_S2, TEST_S3):
        print("Reading", path)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"]
                name = row.get("business_name", "")
                address = row.get("business_address", "")
                country = row.get("country", "")

                records[eid] = preprocess_record(eid, name, address, country)

                n = normalize_text(name)
                c = core_name(n)
                a = normalize_text(address)

                if n:
                    exact_name[n].add(eid)
                if c:
                    exact_core[c].add(eid)
                if a:
                    exact_address[a].add(eid)

                name_index.add(eid, name)
                address_index.add(eid, address)

                total += 1
                if total % 500000 == 0:
                    print(f"  {total:,}")

    print(f"Loaded {total:,} S2/S3 records")
    name_index.prune()
    address_index.prune()
    print(f"Name grams: {len(name_index.postings):,}")
    print(f"Address grams: {len(address_index.postings):,}")

    return (exact_name, exact_core, exact_address), name_index, address_index, records


def exact_candidates(row):
    name = normalize_text(row.get("business_name", ""))
    address = normalize_text(row.get("business_address", ""))
    c = core_name(name)

    out = set()
    if name:
        out.update(G_EXACT[0].get(name, ()))
    if c:
        out.update(G_EXACT[1].get(c, ()))
    if address:
        out.update(G_EXACT[2].get(address, ()))
    return out


def generate_candidates(row):
    out = exact_candidates(row)
    out.update(G_NAME_INDEX.retrieve(row.get("business_name", "")))
    out.update(G_ADDRESS_INDEX.retrieve(row.get("business_address", "")))
    return out


def process_batch(batch):
    batch_candidates = {}
    total_candidates = 0

    for row in batch:
        eid = row["entity_id"]
        cands = generate_candidates(row)
        batch_candidates[eid] = cands
        total_candidates += len(cands)

    pair_keys = []
    feature_rows = []

    for row in batch:
        s1id = row["entity_id"]
        s1p = G_PROCESSED_S1[s1id]
        for cid in batch_candidates[s1id]:
            candidate = G_RECORDS.get(cid)
            if candidate is None:
                continue
            feat = compute_pair_features(s1p, candidate)
            feature_rows.append([feat[col] for col in G_FEATURE_COLUMNS])
            pair_keys.append((s1id, cid))

    if feature_rows:
        X = pd.DataFrame(feature_rows, columns=G_FEATURE_COLUMNS)
        probs = G_MODEL.predict_proba(X)[:, 1]
    else:
        probs = []

    predictions = defaultdict(list)
    for (s1id, cid), p in zip(pair_keys, probs):
        if p >= THRESHOLD:
            predictions[s1id].append(cid)

    return batch_candidates, predictions, total_candidates


def init_worker(s1_records, processed_s1, exact, name_index, address_index, records, model, feature_columns):
    global G_S1, G_PROCESSED_S1, G_EXACT, G_NAME_INDEX, G_ADDRESS_INDEX
    global G_RECORDS, G_MODEL, G_FEATURE_COLUMNS
    G_S1 = s1_records
    G_PROCESSED_S1 = processed_s1
    G_EXACT = exact
    G_NAME_INDEX = name_index
    G_ADDRESS_INDEX = address_index
    G_RECORDS = records
    G_MODEL = model
    G_FEATURE_COLUMNS = feature_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-s1", type=int, default=None)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PARALLEL TEST SUBMISSION GENERATOR")
    print("=" * 70)
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")

    s1_records = []
    with open(TEST_S1, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_records.append(row)
            if args.sample_s1 and len(s1_records) >= args.sample_s1:
                break

    print(f"S1 records: {len(s1_records):,}")

    processed_s1 = {
        r["entity_id"]: preprocess_record(
            r["entity_id"],
            r.get("business_name", ""),
            r.get("business_address", ""),
            r.get("country", ""),
        )
        for r in s1_records
    }

    exact, name_index, address_index, records = build_indexes()

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    batches = [
        s1_records[i:i + args.batch_size]
        for i in range(0, len(s1_records), args.batch_size)
    ]

    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")

    # fork is important on macOS here: large read-only indexes can be shared
    # copy-on-write instead of serialized into every worker.
    ctx = mp.get_context("fork")
    workers = max(1, min(args.workers, len(batches)))

    with open(candidate_path, "w", encoding="utf-8", newline="") as cf, \
         open(matching_path, "w", encoding="utf-8", newline="") as mf:

        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")

        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        mw.writerow(["source1_entity_id", "matched_entity_ids"])

        total_candidates = 0
        total_matches = 0
        completed = 0

        # imap preserves batch order, so output remains deterministic.
        with ctx.Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(
                s1_records,
                processed_s1,
                exact,
                name_index,
                address_index,
                records,
                model,
                feature_columns,
            ),
        ) as pool:

            for batch_candidates, predictions, batch_candidate_count in pool.imap(
                process_batch, batches
            ):
                for row in batches[completed]:
                    sid = row["entity_id"]

                    cids = sorted(batch_candidates.get(sid, ()))
                    mids = sorted(set(predictions.get(sid, ())))

                    cw.writerow([sid, ",".join(cids)])
                    mw.writerow([sid, ",".join(mids)])

                    total_matches += len(mids)

                total_candidates += batch_candidate_count
                completed += 1

                print(
                    f"Completed {min(completed * args.batch_size, len(s1_records)):,}"
                    f"/{len(s1_records):,} S1 | "
                    f"candidates={total_candidates:,} | "
                    f"matches={total_matches:,} | "
                    f"elapsed={(time.time()-start)/60:.1f} min"
                )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"S1: {len(s1_records):,}")
    print(f"Candidates: {total_candidates:,}")
    print(f"Avg candidates/S1: {total_candidates / max(1,len(s1_records)):.2f}")
    print(f"Matches: {total_matches:,}")
    print(f"Runtime: {(time.time()-start)/60:.2f} minutes")
    print(candidate_path)
    print(matching_path)


if __name__ == "__main__":
    main()
