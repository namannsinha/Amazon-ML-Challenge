"""
Preprocessing utilities for Business Entity Resolution.

The preprocessing stage creates multiple conservative representations
of business names and addresses. We intentionally preserve the original
values because aggressive normalization can destroy useful information.
"""

import re
import unicodedata
from typing import List


# ============================================================
# Constants
# ============================================================

NAME_SUFFIX_MAP = {
    "incorporated": "inc",
    "inc": "inc",

    "corporation": "corp",
    "corp": "corp",

    "limited": "ltd",
    "ltd": "ltd",

    "private": "pvt",
    "pvt": "pvt",

    "company": "co",
    "co": "co",

    "llc": "llc",
    "llp": "llp",
}

LEGAL_SUFFIXES = {
    "inc",
    "corp",
    "ltd",
    "pvt",
    "co",
    "llc",
    "llp",
}

ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "st": "st",

    "road": "rd",
    "rd": "rd",

    "avenue": "ave",
    "av": "ave",
    "ave": "ave",

    "boulevard": "blvd",
    "blvd": "blvd",

    "drive": "dr",
    "dr": "dr",

    "lane": "ln",
    "ln": "ln",

    "court": "ct",
    "ct": "ct",

    "place": "pl",
    "pl": "pl",

    "highway": "hwy",
    "hwy": "hwy",

    "parkway": "pkwy",
    "pkwy": "pkwy",

    "apartment": "apt",
    "apt": "apt",

    "suite": "ste",
    "ste": "ste",

    "floor": "fl",
    "fl": "fl",

    "north": "n",
    "n": "n",

    "south": "s",
    "s": "s",

    "east": "e",
    "e": "e",

    "west": "w",
    "w": "w",
}


URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[^\s|,;]+\b",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"
)

MULTISPACE_PATTERN = re.compile(r"\s+")

DBA_PATTERN = re.compile(
    r"\bd\s*/\s*b\s*/\s*a\b",
    re.IGNORECASE,
)


# ============================================================
# Generic text utilities
# ============================================================

def unicode_normalize(text: str) -> str:
    """
    Normalize Unicode while preserving non-Latin scripts.

    NFKC handles compatibility characters without converting
    Indian scripts or other Unicode text into ASCII.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    # Normalize common punctuation variants.
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.casefold()


def tokenize(text: str) -> List[str]:
    """Split normalized text into whitespace-separated tokens."""
    if not text:
        return []

    return text.split()


def alphanumeric_form(tokens: List[str]) -> str:
    """
    Concatenate tokens while retaining Unicode letters and digits.
    """
    return "".join(tokens)


def clean_punctuation(
    text: str,
    preserve_address_symbols: bool = False,
) -> str:
    """
    Convert punctuation/symbols into spaces while preserving
    Unicode letters, combining marks and digits.

    For addresses, '/', '#', and '-' are retained because they
    can carry structural information.
    """
    result = []

    for char in text:
        category = unicodedata.category(char)

        if category[0] in {"L", "M", "N"} or char.isspace():
            result.append(char)
            continue

        if preserve_address_symbols and char in {"/", "#", "-"}:
            result.append(char)
            continue

        result.append(" ")

    return "".join(result)


# ============================================================
# Name preprocessing
# ============================================================

def normalize_name(name: str) -> dict:
    """
    Create multiple representations of a business name.

    Returns:
        original
        normalized
        core
        tokens
        alnum
        compact
    """

    original = "" if name is None else str(name)

    text = unicode_normalize(original)

    # Remove obvious email/URL artifacts from the normalized
    # representation. Original value remains untouched.
    text = EMAIL_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)

    # Convert DBA notation into a consistent token.
    text = DBA_PATTERN.sub(" dba ", text)

    # Remove bracketed metadata such as [INCORPORATED].
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Normalize punctuation.
    text = clean_punctuation(text)

    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    raw_tokens = tokenize(text)

    # Normalize legal-form tokens.
    tokens = [
        NAME_SUFFIX_MAP.get(token, token)
        for token in raw_tokens
    ]

    normalized = " ".join(tokens)

    # Remove legal suffixes ONLY for the core representation.
    #
    # We do not remove them from `normalized`, because whether
    # a legal suffix agrees can itself be useful to the model.
    core_tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    core = " ".join(core_tokens)

    return {
        "original": original,
        "normalized": normalized,
        "core": core,
        "tokens": tokens,
        "alnum": alphanumeric_form(tokens),
        "compact": alphanumeric_form(core_tokens),
    }


# ============================================================
# Address preprocessing
# ============================================================

def normalize_address(address: str) -> dict:
    """
    Create multiple representations of a business address.

    Address component ordering is intentionally NOT changed.
    We will handle order-independent comparison during feature
    engineering instead.
    """

    original = "" if address is None else str(address)

    text = unicode_normalize(original)

    text = EMAIL_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)

    # Preserve /, # and - because they can be meaningful in
    # addresses such as 19 1/2 or apartment/unit numbers.
    text = clean_punctuation(
        text,
        preserve_address_symbols=True,
    )

    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    raw_tokens = tokenize(text)

    tokens = [
        ADDRESS_ABBREVIATIONS.get(token, token)
        for token in raw_tokens
    ]

    normalized = " ".join(tokens)

    # Address numbers are retained separately.
    number_tokens = [
        token
        for token in tokens
        if any(char.isdigit() for char in token)
    ]

    return {
        "original": original,
        "normalized": normalized,
        "tokens": tokens,
        "alnum": alphanumeric_form(tokens),
        "number_tokens": number_tokens,
    }


# ============================================================
# Country preprocessing
# ============================================================

def normalize_country(country: str) -> str:
    """
    Normalize country labels without hard-coding allowed countries.

    The challenge explicitly requires country to be treated as
    an open set.
    """
    if not country:
        return ""

    return unicode_normalize(str(country)).strip()


# ============================================================
# Record-level preprocessing
# ============================================================

def preprocess_record(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str,
) -> dict:
    """
    Preprocess a single business record.
    """

    name = normalize_name(business_name)
    address = normalize_address(business_address)

    return {
        "entity_id": entity_id,
        "country": normalize_country(country),

        "name_normalized": name["normalized"],
        "name_core": name["core"],
        "name_alnum": name["alnum"],
        "name_compact": name["compact"],
        "name_tokens": name["tokens"],

        "address_normalized": address["normalized"],
        "address_alnum": address["alnum"],
        "address_number_tokens": address["number_tokens"],
    }