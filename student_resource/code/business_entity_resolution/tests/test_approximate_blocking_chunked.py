from src.approximate_blocking import ApproximateBlocker


def main():

    queries = {
        "S1-1": {
            "entity_id": "S1-1",
            "name_normalized": "innovative consultants",
        },
        "S1-2": {
            "entity_id": "S1-2",
            "name_normalized": "chiropractic care",
        },
    }

    candidates = [
        {
            "entity_id": "S2-1",
            "name_normalized": "innovative consultants",
        },
        {
            "entity_id": "S2-2",
            "name_normalized": "innovative consultants pvt",
        },
        {
            "entity_id": "S2-3",
            "name_normalized": "prime materials",
        },
        {
            "entity_id": "S2-4",
            "name_normalized": "chiropractic care llc",
        },
        {
            "entity_id": "S2-5",
            "name_normalized": "chiropractic clinic",
        },
    ]

    blocker = ApproximateBlocker(
        queries,
        field="name_normalized",
        top_k=3,
        query_batch_size=1,
    )

    results = {}

    blocker.process_chunk(
        candidates,
        results,
    )

    results = blocker.get_results(results)

    for query_id, matches in results.items():

        print(f"\n{query_id}")

        for candidate_id, similarity in matches:

            print(
                f"  {candidate_id}: "
                f"{similarity:.4f}"
            )


if __name__ == "__main__":
    main()