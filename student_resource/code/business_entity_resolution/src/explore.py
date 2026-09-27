"""Chunked data exploration command."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import pandas as pd

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "dataset"


def profile(path: Path, chunksize: int = 200_000) -> None:
    rows = 0; nulls = Counter(); countries = Counter(); lengths = {"business_name": [], "business_address": []}
    for frame in pd.read_csv(path, sep="\t", chunksize=chunksize):
        rows += len(frame); nulls.update(frame.isna().sum().to_dict()); countries.update(frame["country"].fillna("<NA>").astype(str))
        for column in lengths: lengths[column].extend(frame[column].fillna("").astype(str).str.len().tolist())
    quantiles = {column: tuple(round(float(pd.Series(values).quantile(q)), 2) for q in (0, .5, .95, 1)) for column, values in lengths.items()}
    print(path, "rows=", rows, "null_rates=", {c: round(nulls[c] / rows, 6) for c in nulls}, "length_quantiles=", quantiles, "countries=", dict(countries))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT); args = parser.parse_args()
    for path in sorted(args.data_root.glob("train/train_source*.tsv")) + sorted(args.data_root.glob("test/test_source*.tsv")): profile(path)


if __name__ == "__main__": main()
