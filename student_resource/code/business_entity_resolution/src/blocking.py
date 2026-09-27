"""Candidate generation using token, phonetic, postal, locality, and MinHash-LSH blocks."""
from __future__ import annotations
import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Set
from .normalize import char_shingles, name_tokens, normalize_address, normalize_name, soundex


def _hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(), "big")


def lsh_bands(text: str, bands: int = 4, rows: int = 4) -> List[str]:
    shingles = char_shingles(text)
    if not shingles:
        return []
    signature = [min(_hash(shingle, seed) for shingle in shingles) for seed in range(bands * rows)]
    return [f"{band}:{hash(tuple(signature[band * rows:(band + 1) * rows]))}" for band in range(bands)]


def build_blocks(records: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, Set[str]]]:
    blocks = {key: defaultdict(set) for key in ("token", "phonetic", "postal", "locality", "lsh")}
    for record in records:
        entity_id = record["entity_id"]
        name = normalize_name(record.get("business_name", ""))
        address = normalize_address(record.get("business_address", ""))
        for token in set(t for t in name_tokens(name) if len(t) >= 3): blocks["token"][token].add(entity_id)
        if name: blocks["phonetic"][soundex(name)].add(entity_id)
        if address.postal_code: blocks["postal"][address.postal_code].add(entity_id)
        for token in address.locality_tokens: blocks["locality"][token].add(entity_id)
        for band in lsh_bands(name): blocks["lsh"][band].add(entity_id)
    return blocks


def candidate_ids(record: Dict[str, str], blocks: Dict[str, Dict[str, Set[str]]], max_per_block: int = 250) -> Set[str]:
    name = normalize_name(record.get("business_name", "")); address = normalize_address(record.get("business_address", "")); keys = []
    keys.extend(("token", token) for token in set(name_tokens(name)) if len(token) >= 3)
    if name: keys.append(("phonetic", soundex(name)))
    if address.postal_code: keys.append(("postal", address.postal_code))
    keys.extend(("locality", token) for token in address.locality_tokens); keys.extend(("lsh", band) for band in lsh_bands(name))
    result: Set[str] = set()
    for block_type, key in keys: result.update(list(blocks[block_type].get(key, ()))[:max_per_block])
    return result
