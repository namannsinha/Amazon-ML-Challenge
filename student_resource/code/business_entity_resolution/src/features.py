"""
Pairwise features for Business Entity Resolution.

These features describe how strongly two business records resemble
each other. They are designed to be interpretable and useful for
a later ML matching model.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List


# ---------------------------------------------------------
# String similarity
# ---------------------------------------------------------

def sequence_similarity(a: str, b: str) -> float:
    """
    Character-level similarity.

    Returns a value between 0 and 1.
    """

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------
# Token similarity
# ---------------------------------------------------------

def token_jaccard(
    tokens_a: List[str],
    tokens_b: List[str]
) -> float:
    """
    Jaccard similarity between token sets.
    """

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def token_overlap(
    tokens_a: List[str],
    tokens_b: List[str]
) -> float:
    """
    Fraction of the smaller token set that overlaps.

    Unlike Jaccard, this asks:
    'How much of the smaller representation is contained
    in the larger one?'
    """

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / min(
        len(set_a),
        len(set_b)
    )


# ---------------------------------------------------------
# Numeric address features
# ---------------------------------------------------------

def number_overlap(
    numbers_a: List[str],
    numbers_b: List[str]
) -> float:
    """
    Fraction of the smaller number-token set that overlaps.
    """

    set_a = set(numbers_a)
    set_b = set(numbers_b)

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / min(
        len(set_a),
        len(set_b)
    )


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extract numeric components from a string.

    Examples:
        '19320 1st place' -> ['19320', '1']
        '53/1 building'  -> ['53', '1']
    """

    if not text:
        return []

    return re.findall(r"\d+", text)


# ---------------------------------------------------------
# Address geographic components
# ---------------------------------------------------------

def extract_address_location_tokens(
    address: str
) -> List[str]:
    """
    Extract likely geographic/location tokens.

    This is intentionally conservative. We do not attempt
    to fully parse addresses because address formats differ
    substantially between countries.
    """

    if not address:
        return []

    tokens = address.split()

    # Very short tokens are often state/province/country
    # abbreviations, while longer tokens can represent
    # cities or regions.
    return [
        token
        for token in tokens
        if len(token) >= 2
    ]


# ---------------------------------------------------------
# Length features
# ---------------------------------------------------------

def normalized_length_difference(
    a: str,
    b: str
) -> float:
    """
    Relative difference in string lengths.

    Returns 0 when lengths are identical.
    """

    if not a or not b:
        return 1.0

    max_length = max(len(a), len(b))

    if max_length == 0:
        return 0.0

    return abs(len(a) - len(b)) / max_length


# ---------------------------------------------------------
# Country
# ---------------------------------------------------------

def country_match(
    country_a: str,
    country_b: str
) -> int:
    """
    Exact country match.
    """

    if not country_a or not country_b:
        return 0

    return int(country_a == country_b)


# ---------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------

def compute_pair_features(
    record_a: Dict,
    record_b: Dict
) -> Dict[str, float]:

    # =====================================================
    # NAME FEATURES
    # =====================================================

    name_a = record_a.get(
        "name_normalized",
        ""
    )

    name_b = record_b.get(
        "name_normalized",
        ""
    )

    core_a = record_a.get(
        "name_core",
        ""
    )

    core_b = record_b.get(
        "name_core",
        ""
    )

    name_tokens_a = record_a.get(
        "name_tokens",
        []
    )

    name_tokens_b = record_b.get(
        "name_tokens",
        []
    )

    name_exact = int(
        bool(name_a) and
        bool(name_b) and
        name_a == name_b
    )

    core_name_exact = int(
        bool(core_a) and
        bool(core_b) and
        core_a == core_b
    )

    name_sequence = sequence_similarity(
        name_a,
        name_b
    )

    core_name_sequence = sequence_similarity(
        core_a,
        core_b
    )

    name_jaccard = token_jaccard(
        name_tokens_a,
        name_tokens_b
    )

    name_overlap = token_overlap(
        name_tokens_a,
        name_tokens_b
    )

    name_length_difference = (
        normalized_length_difference(
            name_a,
            name_b
        )
    )

    # =====================================================
    # ADDRESS FEATURES
    # =====================================================

    address_a = record_a.get(
        "address_normalized",
        ""
    )

    address_b = record_b.get(
        "address_normalized",
        ""
    )

    address_tokens_a = address_a.split()
    address_tokens_b = address_b.split()

    address_exact = int(
        bool(address_a) and
        bool(address_b) and
        address_a == address_b
    )

    address_sequence = sequence_similarity(
        address_a,
        address_b
    )

    address_jaccard = token_jaccard(
        address_tokens_a,
        address_tokens_b
    )

    address_overlap = token_overlap(
        address_tokens_a,
        address_tokens_b
    )

    address_length_difference = (
        normalized_length_difference(
            address_a,
            address_b
        )
    )

    # =====================================================
    # ADDRESS NUMBER FEATURES
    # =====================================================

    numbers_a = record_a.get(
        "address_number_tokens",
        []
    )

    numbers_b = record_b.get(
        "address_number_tokens",
        []
    )

    address_number_overlap = number_overlap(
        numbers_a,
        numbers_b
    )

    numeric_tokens_a = extract_numeric_tokens(
        address_a
    )

    numeric_tokens_b = extract_numeric_tokens(
        address_b
    )

    numeric_component_overlap = number_overlap(
        numeric_tokens_a,
        numeric_tokens_b
    )

    # =====================================================
    # ADDRESS LOCATION FEATURES
    # =====================================================

    location_tokens_a = (
        extract_address_location_tokens(
            address_a
        )
    )

    location_tokens_b = (
        extract_address_location_tokens(
            address_b
        )
    )

    address_location_overlap = token_overlap(
        location_tokens_a,
        location_tokens_b
    )

    # =====================================================
    # COUNTRY
    # =====================================================

    same_country = country_match(
        record_a.get("country", ""),
        record_b.get("country", "")
    )

    # =====================================================
    # RETURN FEATURES
    # =====================================================

    return {

        # -----------------------------
        # Name
        # -----------------------------

        "name_exact":
            name_exact,

        "core_name_exact":
            core_name_exact,

        "name_sequence_similarity":
            name_sequence,

        "core_name_sequence_similarity":
            core_name_sequence,

        "name_token_jaccard":
            name_jaccard,

        "name_token_overlap":
            name_overlap,

        "name_length_difference":
            name_length_difference,

        # -----------------------------
        # Address
        # -----------------------------

        "address_exact":
            address_exact,

        "address_sequence_similarity":
            address_sequence,

        "address_token_jaccard":
            address_jaccard,

        "address_token_overlap":
            address_overlap,

        "address_length_difference":
            address_length_difference,

        # -----------------------------
        # Address numbers
        # -----------------------------

        "address_number_overlap":
            address_number_overlap,

        "address_numeric_component_overlap":
            numeric_component_overlap,

        # -----------------------------
        # Address location
        # -----------------------------

        "address_location_overlap":
            address_location_overlap,

        # -----------------------------
        # Country
        # -----------------------------

        "country_match":
            same_country,
    }