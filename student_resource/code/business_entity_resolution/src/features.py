"""Pairwise string features for the matching model."""
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
from rapidfuzz import fuzz, process
from .normalize import name_tokens, normalize_address, normalize_name, soundex


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _levenshtein_ratio(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left_char != right_char)))
        previous = current
    distance = previous[-1]
    return (len(left) + len(right) - distance) / (len(left) + len(right))


def pair_features(left: Dict[str, object], right: Dict[str, object]) -> Dict[str, float]:
    left_name = normalize_name(left.get("business_name", ""))
    right_name = normalize_name(right.get("business_name", ""))
    left_addr = normalize_address(left.get("business_address", ""))
    right_addr = normalize_address(right.get("business_address", ""))
    left_name_tokens, right_name_tokens = set(name_tokens(left_name)), set(name_tokens(right_name))
    left_addr_tokens, right_addr_tokens = set(left_addr.tokens), set(right_addr.tokens)
    return {
        "name_ratio": _levenshtein_ratio(left_name, right_name),
        "name_token_jaccard": _jaccard(left_name_tokens, right_name_tokens),
        "name_exact": float(bool(left_name) and left_name == right_name),
        "name_phonetic": float(bool(left_name) and soundex(left_name) == soundex(right_name)),
        "address_ratio": _levenshtein_ratio(left_addr.text, right_addr.text),
        "address_token_jaccard": _jaccard(left_addr_tokens, right_addr_tokens),
        "postal_exact": float(bool(left_addr.postal_code and right_addr.postal_code and left_addr.postal_code == right_addr.postal_code)),
        "country_exact": float(str(left.get("country", "")).casefold() == str(right.get("country", "")).casefold()),
        "name_length_delta": abs(len(left_name) - len(right_name)),
        "address_length_delta": abs(len(left_addr.text) - len(right_addr.text)),
        "name_token_count_delta": abs(len(left_name_tokens) - len(right_name_tokens)),
        "address_token_count_delta": abs(len(left_addr_tokens) - len(right_addr_tokens)),
    }


FEATURE_NAMES = ["name_ratio", "name_token_jaccard", "name_exact", "name_phonetic", "address_ratio", "address_token_jaccard", "postal_exact", "country_exact", "name_length_delta", "address_length_delta", "name_token_count_delta", "address_token_count_delta"]


def pair_feature_matrix(left_rows: Sequence[Dict[str, object]], right_rows: Sequence[Dict[str, object]]) -> np.ndarray:
    """Build paired features for a batch; edit distances run in rapidfuzz C code."""
    left_names = [normalize_name(row.get("business_name", "")) for row in left_rows]
    right_names = [normalize_name(row.get("business_name", "")) for row in right_rows]
    left_addresses = [normalize_address(row.get("business_address", "")) for row in left_rows]
    right_addresses = [normalize_address(row.get("business_address", "")) for row in right_rows]
    name_ratio = process.cpdist(left_names, right_names, scorer=fuzz.ratio, workers=-1, dtype=np.float32, score_multiplier=0.01)
    address_ratio = process.cpdist([address.text for address in left_addresses], [address.text for address in right_addresses], scorer=fuzz.ratio, workers=-1, dtype=np.float32, score_multiplier=0.01)
    rows = []
    for left_name, right_name, left_address, right_address, left, right, name_score, address_score in zip(left_names, right_names, left_addresses, right_addresses, left_rows, right_rows, name_ratio, address_ratio):
        left_name_tokens = set(name_tokens(left_name)); right_name_tokens = set(name_tokens(right_name))
        left_address_tokens = set(left_address.tokens); right_address_tokens = set(right_address.tokens)
        rows.append({
            "name_ratio": float(name_score),
            "name_token_jaccard": _jaccard(left_name_tokens, right_name_tokens),
            "name_exact": float(bool(left_name) and left_name == right_name),
            "name_phonetic": float(bool(left_name) and soundex(left_name) == soundex(right_name)),
            "address_ratio": float(address_score),
            "address_token_jaccard": _jaccard(left_address_tokens, right_address_tokens),
            "postal_exact": float(bool(left_address.postal_code and right_address.postal_code and left_address.postal_code == right_address.postal_code)),
            "country_exact": float(str(left.get("country", "")).casefold() == str(right.get("country", "")).casefold()),
            "name_length_delta": abs(len(left_name) - len(right_name)),
            "address_length_delta": abs(len(left_address.text) - len(right_address.text)),
            "name_token_count_delta": abs(len(left_name_tokens) - len(right_name_tokens)),
            "address_token_count_delta": abs(len(left_address_tokens) - len(right_address_tokens)),
        })
    return np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
