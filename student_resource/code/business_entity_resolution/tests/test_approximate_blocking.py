import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.preprocessing import preprocess_record


S1_PATH = "../../samples/sample_train_source1.tsv"
S2_PATH = "../../samples/sample_train_source2.tsv"
S3_PATH = "../../samples/sample_train_source3.tsv"


def load_records(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    records = []

    for _, row in df.iterrows():
        records.append(
            preprocess_record(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"]
            )
        )

    return records


def build_name_index(records):
    """
    Build a character n-gram TF-IDF index over candidate names.
    """

    names = [
        record["name_normalized"]
        for record in records
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(names)

    return vectorizer, matrix


def retrieve_candidates(
    source1_records,
    candidate_records,
    vectorizer,
    candidate_matrix,
    top_k=10
):
    """
    Retrieve top-K approximate name candidates for each S1 record.
    """

    query_names = [
        record["name_normalized"]
        for record in source1_records
    ]

    query_matrix = vectorizer.transform(query_names)

    # cosine distance = 1 - cosine similarity
    nn = NearestNeighbors(
        n_neighbors=min(top_k, candidate_matrix.shape[0]),
        metric="cosine",
        algorithm="brute"
    )

    nn.fit(candidate_matrix)

    distances, indices = nn.kneighbors(query_matrix)

    results = {}

    for i, record in enumerate(source1_records):

        entity_id = record["entity_id"]

        candidates = []

        for distance, index in zip(
            distances[i],
            indices[i]
        ):
            candidate = candidate_records[index]

            similarity = 1.0 - distance

            candidates.append(
                {
                    "entity_id": candidate["entity_id"],
                    "similarity": float(similarity),
                    "name": candidate["name_normalized"],
                }
            )

        results[entity_id] = candidates

    return results


def print_examples(
    source1_records,
    results,
    num_examples=20
):
    """
    Print a few retrieval examples so we can inspect
    whether the approximate blocker is behaving sensibly.
    """

    print("\n" + "=" * 70)
    print("APPROXIMATE BLOCKING EXAMPLES")
    print("=" * 70)

    shown = 0

    for record in source1_records:

        entity_id = record["entity_id"]
        name = record["name_normalized"]

        print("\nS1:")
        print(f"  {name}")

        print("Candidates:")

        for candidate in results[entity_id]:
            print(
                f"  {candidate['similarity']:.3f}  "
                f"{candidate['name']}"
            )

        shown += 1

        if shown >= num_examples:
            break


def print_statistics(results):

    candidate_counts = [
        len(candidates)
        for candidates in results.values()
    ]

    similarities = [
        candidate["similarity"]
        for candidates in results.values()
        for candidate in candidates
    ]

    print("\n" + "=" * 70)
    print("APPROXIMATE BLOCKING STATISTICS")
    print("=" * 70)

    print(f"S1 records:             {len(results)}")

    print(
        f"Average candidates:     "
        f"{sum(candidate_counts) / len(candidate_counts):.2f}"
    )

    print(
        f"Maximum candidates:     "
        f"{max(candidate_counts)}"
    )

    if similarities:
        print(
            f"Average similarity:     "
            f"{sum(similarities) / len(similarities):.4f}"
        )

        print(
            f"Minimum similarity:     "
            f"{min(similarities):.4f}"
        )

        print(
            f"Maximum similarity:     "
            f"{max(similarities):.4f}"
        )


def main():

    print("Loading datasets...")

    s1 = load_records(S1_PATH)
    s2 = load_records(S2_PATH)
    s3 = load_records(S3_PATH)

    candidate_records = s2 + s3

    print(f"S1 records:       {len(s1)}")
    print(f"S2 records:       {len(s2)}")
    print(f"S3 records:       {len(s3)}")
    print(f"Candidate records:{len(candidate_records)}")

    print("\nBuilding TF-IDF index...")

    start = time.time()

    vectorizer, candidate_matrix = build_name_index(
        candidate_records
    )

    build_time = time.time() - start

    print(
        f"Index built in {build_time:.3f} seconds"
    )

    print("\nRetrieving approximate candidates...")

    start = time.time()

    results = retrieve_candidates(
        source1_records=s1,
        candidate_records=candidate_records,
        vectorizer=vectorizer,
        candidate_matrix=candidate_matrix,
        top_k=10
    )

    retrieval_time = time.time() - start

    print(
        f"Retrieval completed in "
        f"{retrieval_time:.3f} seconds"
    )

    print_statistics(results)

    print_examples(
        s1,
        results,
        num_examples=20
    )


if __name__ == "__main__":
    main()