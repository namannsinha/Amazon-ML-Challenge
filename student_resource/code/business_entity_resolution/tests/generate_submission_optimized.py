# Optimized submission generator for Business Entity Resolution Challenge.
# Place this file at: tests/generate_submission_optimized.py

import csv
import os
import sys
import time
import argparse
import sqlite3
import re
from collections import defaultdict

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
DB_PATH = os.path.join(os.path.dirname(__file__), "generated", "test_blocking.sqlite")

THRESHOLD = 0.875
NGRAM_SIZES = (3, 4)
NAME_TOP_K = 10
ADDRESS_TOP_K = 10
BATCH_SIZE = 5000


def normalize_text(value):
    if value is None:
        return ""
    value = str(value).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def core_name(value):
    value = normalize_text(value)
    value = re.sub(r"\b(incorporated|inc|corporation|corp|limited|ltd|private|pvt|company|co|llc|llp)\b", "", value)
    return re.sub(r"\s+", " ", value).strip()


def ngrams(text):
    text = normalize_text(text)
    if not text:
        return ()
    text = f"^{text}$"
    grams = set()
    for n in NGRAM_SIZES:
        if len(text) >= n:
            grams.update(text[i:i+n] for i in range(len(text)-n+1))
    return tuple(grams)


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")
    return conn


def database_exists():
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        ok = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='records'").fetchone()
        conn.close()
        return ok is not None
    except Exception:
        return False


def build_database():
    print("=" * 70)
    print("BUILDING COMPACT TEST BLOCKING DATABASE")
    print("=" * 70)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    for suffix in ("-wal", "-shm"):
        p = DB_PATH + suffix
        if os.path.exists(p):
            os.remove(p)

    conn = connect_db()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE records (entity_id TEXT PRIMARY KEY, name TEXT, core_name TEXT, address TEXT, country TEXT);
        CREATE TABLE name_exact (key TEXT, entity_id TEXT);
        CREATE TABLE core_exact (key TEXT, entity_id TEXT);
        CREATE TABLE address_exact (key TEXT, entity_id TEXT);
        CREATE TABLE name_ngram (gram TEXT, entity_id TEXT);
        CREATE TABLE address_ngram (gram TEXT, entity_id TEXT);
    """)

    records = []
    name_exact = []
    core_exact_rows = []
    address_exact = []
    name_grams = []
    address_grams = []
    total = 0
    start = time.time()

    def flush():
        nonlocal records, name_exact, core_exact_rows, address_exact, name_grams, address_grams
        if records:
            cur.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?)", records)
        if name_exact:
            cur.executemany("INSERT INTO name_exact VALUES (?,?)", name_exact)
        if core_exact_rows:
            cur.executemany("INSERT INTO core_exact VALUES (?,?)", core_exact_rows)
        if address_exact:
            cur.executemany("INSERT INTO address_exact VALUES (?,?)", address_exact)
        if name_grams:
            cur.executemany("INSERT INTO name_ngram VALUES (?,?)", name_grams)
        if address_grams:
            cur.executemany("INSERT INTO address_ngram VALUES (?,?)", address_grams)
        conn.commit()
        records, name_exact, core_exact_rows, address_exact, name_grams, address_grams = [], [], [], [], [], []

    for path in (TEST_S2, TEST_S3):
        print("Reading", path)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"]
                name = normalize_text(row.get("business_name", ""))
                core = core_name(name)
                address = normalize_text(row.get("business_address", ""))
                country = normalize_text(row.get("country", ""))
                records.append((eid, name, core, address, country))
                if name:
                    name_exact.append((name, eid))
                if core:
                    core_exact_rows.append((core, eid))
                if address:
                    address_exact.append((address, eid))
                for g in ngrams(name):
                    name_grams.append((g, eid))
                for g in ngrams(address):
                    address_grams.append((g, eid))
                total += 1
                if total % 50000 == 0:
                    flush()
                if total % 500000 == 0:
                    print(f"  {total:,} records | {(time.time()-start)/60:.1f} min", flush=True)

    flush()
    print("Creating database indexes...", flush=True)
    cur.executescript("""
        CREATE INDEX idx_name_exact ON name_exact(key);
        CREATE INDEX idx_core_exact ON core_exact(key);
        CREATE INDEX idx_address_exact ON address_exact(key);
        CREATE INDEX idx_name_ngram ON name_ngram(gram);
        CREATE INDEX idx_address_ngram ON address_ngram(gram);
    """)
    conn.commit()
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    print(f"Loaded: {total:,} S2/S3 records")
    print(f"Database: {DB_PATH}")
    print(f"Database size: {os.path.getsize(DB_PATH)/(1024**3):.2f} GB")
    print(f"Build time: {(time.time()-start)/60:.1f} min")


def fetch_exact(cur, table, key):
    if not key:
        return []
    return [r[0] for r in cur.execute(f"SELECT entity_id FROM {table} WHERE key=?", (key,))]


def approximate_retrieve(cur, table, query, top_k):
    grams = ngrams(query)
    if not grams:
        return []
    placeholders = ",".join("?" for _ in grams)
    sql = f"SELECT entity_id, COUNT(*) AS overlap FROM {table} WHERE gram IN ({placeholders}) GROUP BY entity_id ORDER BY overlap DESC LIMIT ?"
    return [r[0] for r in cur.execute(sql, tuple(grams)+(top_k,))]


def get_candidate_record(cur, eid):
    row = cur.execute("SELECT entity_id,name,core_name,address,country FROM records WHERE entity_id=?", (eid,)).fetchone()
    if row is None:
        return None
    return preprocess_record(row[0], row[1], row[3], row[4])


def generate_candidates(cur, row):
    name = normalize_text(row.get("business_name", ""))
    address = normalize_text(row.get("business_address", ""))
    core = core_name(name)
    out = set()
    out.update(fetch_exact(cur, "name_exact", name))
    out.update(fetch_exact(cur, "core_exact", core))
    out.update(fetch_exact(cur, "address_exact", address))
    out.update(approximate_retrieve(cur, "name_ngram", name, NAME_TOP_K))
    out.update(approximate_retrieve(cur, "address_ngram", address, ADDRESS_TOP_K))
    return out


def process_batch(conn, batch, model, feature_columns):
    cur = conn.cursor()
    candidate_map = {}
    feature_rows = []
    pair_keys = []
    for row in batch:
        sid = row["entity_id"]
        cands = generate_candidates(cur, row)
        candidate_map[sid] = cands
        s1p = preprocess_record(sid, row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""))
        for cid in cands:
            candidate = get_candidate_record(cur, cid)
            if candidate is None:
                continue
            feat = compute_pair_features(s1p, candidate)
            feature_rows.append([feat[col] for col in feature_columns])
            pair_keys.append((sid, cid))

    if feature_rows:
        X = pd.DataFrame(feature_rows, columns=feature_columns)
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = []

    predictions = defaultdict(list)
    for (sid, cid), p in zip(pair_keys, probabilities):
        if p >= THRESHOLD:
            predictions[sid].append(cid)
    return candidate_map, predictions


def read_s1(sample_s1=None):
    rows = []
    with open(TEST_S1, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            if sample_s1 is not None and len(rows) >= sample_s1:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-s1", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    start = time.time()

    print("=" * 70)
    print("OPTIMIZED TEST SUBMISSION GENERATOR")
    print("=" * 70)
    print(f"Batch size: {args.batch_size}")
    print(f"Threshold: {THRESHOLD}")

    if args.rebuild_index or not database_exists():
        build_database()
    else:
        print(f"Using existing test database: {DB_PATH}")
        print(f"Database size: {os.path.getsize(DB_PATH)/(1024**3):.2f} GB")

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    print("Reading S1...", flush=True)
    s1_rows = read_s1(args.sample_s1)
    print(f"S1 records: {len(s1_rows):,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")

    conn = connect_db()
    total_candidates = 0
    total_matches = 0
    processed = 0

    with open(candidate_path, "w", encoding="utf-8", newline="") as cf, open(matching_path, "w", encoding="utf-8", newline="") as mf:
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        mw.writerow(["source1_entity_id", "matched_entity_ids"])

        for start_idx in range(0, len(s1_rows), args.batch_size):
            batch = s1_rows[start_idx:start_idx + args.batch_size]
            candidate_map, predictions = process_batch(conn, batch, model, feature_columns)

            for row in batch:
                sid = row["entity_id"]
                cids = sorted(candidate_map.get(sid, ()))
                mids = sorted(set(predictions.get(sid, ())))
                cw.writerow([sid, ",".join(cids)])
                mw.writerow([sid, ",".join(mids)])
                total_candidates += len(cids)
                total_matches += len(mids)

            processed += len(batch)
            print(f"Completed {processed:,}/{len(s1_rows):,} S1 | candidates={total_candidates:,} | matches={total_matches:,} | elapsed={(time.time()-start)/60:.1f} min", flush=True)

    conn.close()
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"S1: {len(s1_rows):,}")
    print(f"Candidates: {total_candidates:,}")
    print(f"Avg candidates/S1: {total_candidates / max(1,len(s1_rows)):.2f}")
    print(f"Matches: {total_matches:,}")
    print(f"Runtime: {(time.time()-start)/60:.2f} minutes")
    print(candidate_path)
    print(matching_path)


if __name__ == "__main__":
    main()
