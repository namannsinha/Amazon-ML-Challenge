import heapq

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_QUERY_BATCH_SIZE = 250
DEFAULT_TOP_K = 10


class ApproximateBlocker:

    def __init__(
        self,
        query_records,
        field,
        top_k=DEFAULT_TOP_K,
        chunk_size=DEFAULT_CHUNK_SIZE,
        query_batch_size=DEFAULT_QUERY_BATCH_SIZE,
    ):
        self.query_records = query_records
        self.field = field
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.query_batch_size = query_batch_size

        self.query_ids = []
        self.query_texts = []

        for entity_id, record in query_records.items():

            value = record.get(field, "")

            if value:
                self.query_ids.append(entity_id)
                self.query_texts.append(value)

        self.vectorizer = None
        self.query_matrix = None

        self._build_query_index()

    def _build_query_index(self):

        print(
            f"Building query TF-IDF index for "
            f"{self.field}..."
        )

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
        )

        self.query_matrix = self.vectorizer.fit_transform(
            self.query_texts
        )

        print(
            f"Query matrix shape: "
            f"{self.query_matrix.shape}"
        )

    def process_chunk(
        self,
        records,
        best_candidates,
    ):

        if not records:
            return

        candidate_texts = [
            record.get(self.field, "")
            for record in records
        ]

        candidate_matrix = self.vectorizer.transform(
            candidate_texts
        )

        total_queries = len(self.query_ids)

        for start in range(
            0,
            total_queries,
            self.query_batch_size,
        ):

            end = min(
                start + self.query_batch_size,
                total_queries,
            )

            query_matrix_batch = self.query_matrix[
                start:end
            ]

            similarities = cosine_similarity(
                query_matrix_batch,
                candidate_matrix,
            )

            for local_query_index in range(
                end - start
            ):

                query_index = start + local_query_index

                query_id = self.query_ids[
                    query_index
                ]

                row = similarities[
                    local_query_index
                ]

                k = min(
                    self.top_k,
                    len(row),
                )

                if k == 0:
                    continue

                if len(row) <= k:
                    indices = np.argsort(row)[::-1]
                else:
                    indices = np.argpartition(
                        row,
                        -k,
                    )[-k:]

                    indices = indices[
                        np.argsort(
                            row[indices]
                        )[::-1]
                    ]

                for candidate_index in indices:

                    similarity = float(
                        row[candidate_index]
                    )

                    if similarity <= 0:
                        continue

                    candidate_id = records[
                        candidate_index
                    ]["entity_id"]

                    self._add_candidate(
                        best_candidates,
                        query_id,
                        candidate_id,
                        similarity,
                    )

            del similarities

        del candidate_matrix

    def _add_candidate(
        self,
        best_candidates,
        query_id,
        candidate_id,
        similarity,
    ):

        heap = best_candidates.setdefault(
            query_id,
            [],
        )

        item = (
            similarity,
            candidate_id,
        )

        if len(heap) < self.top_k:

            heapq.heappush(
                heap,
                item,
            )

        elif similarity > heap[0][0]:

            heapq.heapreplace(
                heap,
                item,
            )

    def get_results(
        self,
        best_candidates,
    ):

        results = {}

        for query_id, heap in best_candidates.items():

            ranked = sorted(
                heap,
                key=lambda x: x[0],
                reverse=True,
            )

            results[query_id] = [
                (candidate_id, similarity)
                for similarity, candidate_id in ranked
            ]

        return results


def process_file_in_chunks(
    path,
    blocker,
    best_candidates,
    preprocess_function,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """
    Scan a source file without loading it completely.
    """

    total_rows = 0

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=usecols,
            chunksize=chunk_size,
        ),
        start=1,
    ):

        records = []

        for _, row in chunk.iterrows():

            name = row["business_name"]
            address = row["business_address"]
            country = row["country"]

            if pd.isna(name):
                name = ""

            if pd.isna(address):
                address = ""

            if pd.isna(country):
                country = ""

            record = preprocess_function(
                row["entity_id"],
                name,
                address,
                country,
            )

            if record.get(blocker.field, ""):
                records.append(record)

        blocker.process_chunk(
            records,
            best_candidates,
        )

        total_rows += len(chunk)

        print(
            f"  processed chunk {chunk_number} "
            f"({total_rows:,} rows)"
        )

    return total_rows