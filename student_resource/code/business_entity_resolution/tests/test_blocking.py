import pandas as pd

from src.preprocessing import preprocess_record
from src.blocking import (
    build_blocking_indexes,
    generate_candidates,
    generate_candidate_pairs,
)


def load_records(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    records = []

    for row in df.itertuples(index=False):

        records.append(
            preprocess_record(
                entity_id=row.entity_id,
                business_name=row.business_name,
                business_address=row.business_address,
                country=row.country,
            )
        )

    return records


# ------------------------------------------------------------
# Load samples
# ------------------------------------------------------------

source1 = load_records(
    "../../samples/sample_train_source1.tsv"
)

source2 = load_records(
    "../../samples/sample_train_source2.tsv"
)

source3 = load_records(
    "../../samples/sample_train_source3.tsv"
)


print("S1 records:", len(source1))
print("S2 records:", len(source2))
print("S3 records:", len(source3))


# ------------------------------------------------------------
# Build indexes
# ------------------------------------------------------------

indexes = build_blocking_indexes(
    source2,
    source3,
)


print("\nINDEX SIZES")
print("Name:     ", len(indexes["name"]))
print("Core name:", len(indexes["core_name"]))
print("Address:  ", len(indexes["address"]))


# ------------------------------------------------------------
# Generate candidates
# ------------------------------------------------------------

candidate_pairs = generate_candidate_pairs(
    source1,
    indexes,
)


candidate_counts = [
    len(candidates)
    for candidates in candidate_pairs.values()
]


print("\nCANDIDATE STATISTICS")
print("--------------------")

print(
    "S1 entities:",
    len(candidate_counts),
)

print(
    "With candidates:",
    sum(
        count > 0
        for count in candidate_counts
    ),
)

print(
    "Without candidates:",
    sum(
        count == 0
        for count in candidate_counts
    ),
)

print(
    "Average candidates:",
    sum(candidate_counts)
    / len(candidate_counts),
)

print(
    "Maximum candidates:",
    max(candidate_counts),
)


name_count = 0
core_name_count = 0
address_count = 0

name_candidates = 0
core_name_candidates = 0
address_candidates = 0

for record in source1:

    blocks = generate_candidates(
        record,
        indexes,
    )

    if blocks["name"]:
        name_count += 1
        name_candidates += len(blocks["name"])

    if blocks["core_name"]:
        core_name_count += 1
        core_name_candidates += len(blocks["core_name"])

    if blocks["address"]:
        address_count += 1
        address_candidates += len(blocks["address"])


print("\nBLOCK-BY-BLOCK STATISTICS")
print("-------------------------")

print(
    "Exact name:",
    name_count,
    "S1 records,",
    name_candidates,
    "candidates",
)

print(
    "Core name:",
    core_name_count,
    "S1 records,",
    core_name_candidates,
    "candidates",
)

print(
    "Exact address:",
    address_count,
    "S1 records,",
    address_candidates,
    "candidates",
)