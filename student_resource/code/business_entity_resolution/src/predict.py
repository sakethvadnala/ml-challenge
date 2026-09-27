"""Inference helpers and exact submission writer."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Set
import csv


def write_submission(path: Path, source1_ids: Iterable[str], values: Dict[str, Set[str]], column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(["source1_entity_id", column])
        for source1_id in source1_ids:
            writer.writerow([source1_id, ",".join(sorted(values.get(source1_id, set())))])
