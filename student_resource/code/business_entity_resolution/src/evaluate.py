"""Macro F_0.5 evaluation used by the challenge."""
from __future__ import annotations

from typing import Dict, Iterable, Set


def entity_f05(truth: Set[str], predicted: Set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(truth & predicted)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_f05(truth: Dict[str, Set[str]], predicted: Dict[str, Set[str]]) -> float:
    if not truth:
        return 0.0
    return sum(entity_f05(targets, predicted.get(entity_id, set()))
               for entity_id, targets in truth.items()) / len(truth)


def parse_ground_truth(rows: Iterable[tuple[str, str]]) -> Dict[str, Set[str]]:
    return {s1: {value.strip() for value in ids.split(",") if value.strip()}
            for s1, ids in rows}
