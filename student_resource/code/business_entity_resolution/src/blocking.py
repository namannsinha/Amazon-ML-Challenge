"""
Candidate generation / blocking for Business Entity Resolution.

Blocking reduces the number of S1-S2/S3 pairs that need to be
evaluated by the matching model.

Version 1 uses exact normalized representations only.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Set


# ============================================================
# Index construction
# ============================================================

def build_index(
    records: Iterable[dict],
    field: str,
) -> Dict[str, List[str]]:
    """
    Build an inverted index.

    Parameters
    ----------
    records:
        Iterable of preprocessed records.

    field:
        Field to index, e.g.
        'name_normalized',
        'name_core',
        'address_normalized'.

    Returns
    -------
    dict:
        normalized value -> entity IDs
    """

    index = defaultdict(list)

    for record in records:
        value = record.get(field, "")
        entity_id = record.get("entity_id")

        if not value or not entity_id:
            continue

        index[value].append(entity_id)

    return dict(index)


def build_blocking_indexes(
    source2_records: Iterable[dict],
    source3_records: Iterable[dict],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build all v1 blocking indexes over Source 2 and Source 3.
    """

    candidate_sources = list(source2_records) + list(source3_records)

    return {
        "name": build_index(
            candidate_sources,
            "name_normalized",
        ),

        "core_name": build_index(
            candidate_sources,
            "name_core",
        ),

        "address": build_index(
            candidate_sources,
            "address_normalized",
        ),
    }


# ============================================================
# Candidate generation
# ============================================================

def generate_candidates(
    source1_record: dict,
    indexes: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Set[str]]:
    """
    Generate candidates separately for each blocking strategy.
    """

    candidates = {
        "name": set(),
        "core_name": set(),
        "address": set(),
    }

    # Exact normalized name
    name = source1_record.get("name_normalized", "")

    if name:
        candidates["name"].update(
            indexes["name"].get(name, [])
        )

    # Exact core name
    core_name = source1_record.get("name_core", "")

    if core_name:
        candidates["core_name"].update(
            indexes["core_name"].get(core_name, [])
        )

    # Exact normalized address
    address = source1_record.get("address_normalized", "")

    if address:
        candidates["address"].update(
            indexes["address"].get(address, [])
        )

    return candidates


def generate_candidate_pairs(
    source1_records: Iterable[dict],
    indexes: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Set[str]]:
    """
    Generate the union of all blocking candidates.
    """

    candidate_pairs = {}

    for record in source1_records:

        entity_id = record["entity_id"]

        blocks = generate_candidates(
            record,
            indexes,
        )

        candidates = (
            blocks["name"]
            | blocks["core_name"]
            | blocks["address"]
        )

        candidate_pairs[entity_id] = candidates

    return candidate_pairs