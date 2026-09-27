"""Country-agnostic normalization for names and addresses."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Set

LEGAL_SUFFIXES = {
    "corporation", "corp", "company", "co", "incorporated", "inc",
    "limited", "ltd", "private", "pvt", "llc", "plc", "sarl", "sas",
}
ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "bldg": "building", "ste": "suite", "apt": "apartment",
    "fl": "floor", "pkwy": "parkway", "ctr": "center", "ct": "court",
}
POSTAL_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d{5}(?:-\d{4})?|\d{6}|[A-Z]\d[A-Z][ -]?\d[A-Z]\d)(?![A-Za-z0-9])", re.I)
WORD_RE = re.compile(r"[a-z0-9]+")


def _ascii_fold(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def name_tokens(value: object) -> List[str]:
    text = _ascii_fold(value).replace("&", " and ")
    return WORD_RE.findall(text)


def normalize_name(value: object) -> str:
    tokens = [t for t in name_tokens(value) if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def soundex(value: object) -> str:
    letters = re.sub(r"[^a-z]", "", _ascii_fold(value))
    if not letters:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    first = letters[0].upper()
    result = []
    previous = codes.get(letters[0], "")
    for char in letters[1:]:
        code = codes.get(char, "")
        if code and code != previous:
            result.append(code)
        previous = code
    return (first + "".join(result) + "000")[:4]


@dataclass(frozen=True)
class NormalizedAddress:
    text: str
    tokens: List[str]
    postal_code: str
    locality_tokens: Set[str]


def normalize_address(value: object) -> NormalizedAddress:
    raw = _ascii_fold(value)
    postal = POSTAL_RE.search(raw)
    postal_code = re.sub(r"[^a-z0-9]", "", postal.group(0).lower()) if postal else ""
    without_postal = POSTAL_RE.sub(" ", raw)
    tokens = WORD_RE.findall(without_postal)
    expanded = [ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens]
    locality = {t for t in expanded if len(t) >= 4 and t not in ADDRESS_ABBREVIATIONS}
    return NormalizedAddress(" ".join(expanded), expanded, postal_code, locality)


def char_shingles(value: str, size: int = 3) -> Set[str]:
    compact = re.sub(r"\s+", " ", value.strip())
    if not compact:
        return set()
    padded = f"  {compact}  "
    return {padded[i : i + size] for i in range(max(1, len(padded) - size + 1))}
