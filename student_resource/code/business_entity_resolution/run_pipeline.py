"""Train once, then process test Source 1 in resumable columnar batches."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .src.evaluate import macro_f05
from .src.features import FEATURE_NAMES, pair_feature_matrix
from .src.index import RecordIndex
from .src.model import fit_classifier, tune_threshold

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "student_resource" / "dataset"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "student_resource" / "output"
DEFAULT_CACHE_ROOT = REPOSITORY_ROOT / "student_resource" / ".ber_cache"
BATCH_SIZE = 150_000


@contextmanager
def stage(name: str):
    started = time.perf_counter()
    print(f"[stage:start] {name}", flush=True)
    try:
        yield
    finally:
        print(f"[stage:end] {name} elapsed={time.perf_counter() - started:.2f}s", flush=True)


def ground_truth(path: Path):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row.source1_entity_id: {x.strip() for x in row.matched_entity_ids.split(",") if x.strip()} for row in frame.itertuples()}


def write_status(path: Path, payload: dict) -> None:
    completed = payload.get("completed_batches", [])
    lines = ["# Pipeline Status", "", f"State: **{payload.get('state', 'RUNNING')}**", f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "", f"Completed batches: {len(completed)}", f"Completed entities: {payload.get('completed_entities', 0):,}", f"Completed candidate pairs: {payload.get('completed_pairs', 0):,}", f"Elapsed seconds: {payload.get('elapsed_seconds', 0):.2f}", "", "## Batch Timings", ""]
    for batch in completed:
        lines.append(f"- Batch {batch['batch_id']}: entities={batch['entities']:,}, candidates={batch['candidate_pairs']:,}, elapsed={batch['elapsed_seconds']:.2f}s")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", []) or ["None"]
    lines.extend(f"- {warning}" for warning in warnings)
    if payload.get("final_summary"):
        lines.extend(["", "## RUN COMPLETE", "", payload["final_summary"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def batch_frames(path: Path, batch_size: int):
    for batch_id, frame in enumerate(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=batch_size)):
        yield batch_id, frame


def feature_rows_from_frame(frame: pd.DataFrame):
    rows = frame.to_dict("records")
    left_rows = [{"business_name": row["source1_business_name"], "business_address": row["source1_business_address"], "country": row["source1_country"]} for row in rows]
    right_rows = [{"business_name": row["business_name"], "business_address": row["business_address"], "country": row["country"]} for row in rows]
    return rows, pair_feature_matrix(left_rows, right_rows)


def train_model(train: Path, cache: Path, truth: dict, train_limit: int):
    train_index = RecordIndex(cache / "train.index")
    with stage("train target normalization"):
        train_index.build([train / "train_source2.tsv", train / "train_source3.tsv"], cache / "train.records.parquet")
    with stage("train blocking and capping"):
        train_index.build_candidates(train / "train_source1.tsv", cache / "train.candidates.parquet", train_limit)
    training_rows = []
    validation_scored = {}
    validation_truth = {}
    with stage("train feature engineering"):
        for frame in train_index.iter_candidate_records():
            rows, matrix = feature_rows_from_frame(frame)
            for row, values in zip(rows, matrix):
                features = dict(zip(FEATURE_NAMES, values))
                s1 = row["source1_entity_id"]
                if int(s1.split("-")[-1]) % 5 == 0:
                    validation_truth[s1] = truth.get(s1, set())
                    validation_scored.setdefault(s1, []).append((row["candidate_entity_id"], features))
                else:
                    training_rows.append((features, int(row["candidate_entity_id"] in truth.get(s1, set()))))
        print(f"[features] training_rows={len(training_rows)} validation_entities={len(validation_scored)}", flush=True)
    with stage("model training"):
        model = fit_classifier(training_rows)
    with stage("threshold selection"):
        validation_probabilities = {}
        for s1, pairs in validation_scored.items():
            matrix = np.asarray([[features[name] for name in FEATURE_NAMES] for _, features in pairs], dtype=float)
            probabilities = model.predict_proba(matrix)[:, 1]
            validation_probabilities[s1] = [(candidate, float(probability)) for (candidate, _), probability in zip(pairs, probabilities)]
        threshold, validation_score = tune_threshold(validation_probabilities, validation_truth)
        print(f"[threshold] value={threshold:.4f} validation_macro_f05={validation_score:.6f}", flush=True)
    train_index.close()
    return model, threshold, validation_score


def write_final_batches(output: Path, batch_files: list[Path]) -> tuple[int, int]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as candidate_handle, (output / "matching_results.tsv").open("w", encoding="utf-8", newline="") as matching_handle:
        candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        matching_writer = csv.writer(matching_handle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        entity_count = pair_count = 0
        for batch_file in batch_files:
            frame = pd.read_parquet(batch_file)
            for row in frame.itertuples(index=False):
                candidate_ids = row.candidate_entity_ids.split(",") if row.candidate_entity_ids else []
                matching_ids = row.matched_entity_ids.split(",") if row.matched_entity_ids else []
                candidate_writer.writerow([row.source1_entity_id, ",".join(candidate_ids)])
                matching_writer.writerow([row.source1_entity_id, ",".join(matching_ids)])
                entity_count += 1
                pair_count += len(candidate_ids)
    return entity_count, pair_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--train-limit", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prototype-s1", type=int, default=None)
    parser.add_argument("--prototype-target-rows", type=int, default=None)
    parser.add_argument("--stop-after-batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = REPOSITORY_ROOT / "STATUS.md"
    manifest_path = args.cache / "test_batch_manifest.json"
    started = time.perf_counter()
    train = args.data_root / "train"
    test = args.data_root / "test"
    truth = ground_truth(train / "train_ground_truth.tsv")
    model, threshold, validation_score = train_model(train, args.cache, truth, args.train_limit)
    test_index = RecordIndex(args.cache / "test.index")
    with stage("test target normalization"):
        test_index.build([test / "test_source2.tsv", test / "test_source3.tsv"], args.cache / "test.records.parquet", args.prototype_target_rows)
    manifest = {"batch_size": args.batch_size, "completed_batches": [], "warnings": [], "state": "RUNNING"}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_dir = args.cache / "test_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = {batch["batch_id"] for batch in manifest.get("completed_batches", [])}
    for batch_id, source1_frame in batch_frames(test / "test_source1.tsv", args.batch_size):
        if args.prototype_s1 is not None and batch_id * args.batch_size >= args.prototype_s1:
            break
        if args.prototype_s1 is not None:
            source1_frame = source1_frame.iloc[: max(0, args.prototype_s1 - batch_id * args.batch_size)]
            if source1_frame.empty:
                break
        batch_file = batch_dir / f"batch_{batch_id:05d}.parquet"
        if batch_id in completed_ids and batch_file.exists():
            continue
        batch_started = time.perf_counter()
        candidate_file = batch_dir / f"batch_{batch_id:05d}.candidates.parquet"
        test_index.build_candidate_batch(source1_frame, candidate_file)
        candidates = {s1: set() for s1 in source1_frame["entity_id"]}
        predictions = {s1: set() for s1 in source1_frame["entity_id"]}
        pair_count = 0
        for candidate_frame in test_index.iter_candidate_records():
            rows, matrix = feature_rows_from_frame(candidate_frame)
            probabilities = model.predict_proba(matrix)[:, 1]
            for row, probability in zip(rows, probabilities):
                s1 = row["source1_entity_id"]
                candidate = row["candidate_entity_id"]
                candidates[s1].add(candidate)
                if probability >= threshold:
                    predictions[s1].add(candidate)
                pair_count += 1
        result = pd.DataFrame({"source1_entity_id": list(candidates), "candidate_entity_ids": [",".join(sorted(candidates[s1])) for s1 in candidates], "matched_entity_ids": [",".join(sorted(predictions[s1])) for s1 in candidates]})
        result.to_parquet(batch_file, index=False)
        elapsed = round(time.perf_counter() - batch_started, 2)
        manifest.setdefault("completed_batches", []).append({"batch_id": batch_id, "entities": len(result), "candidate_pairs": pair_count, "elapsed_seconds": elapsed})
        manifest["completed_batches"] = sorted(manifest["completed_batches"], key=lambda item: item["batch_id"])
        manifest["completed_entities"] = sum(item["entities"] for item in manifest["completed_batches"])
        manifest["completed_pairs"] = sum(item["candidate_pairs"] for item in manifest["completed_batches"])
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_status(status_path, manifest)
        print(f"[batch] id={batch_id} entities={len(result)} candidates={pair_count} elapsed={elapsed:.2f}s", flush=True)
        if args.stop_after_batch is not None and batch_id >= args.stop_after_batch:
            manifest["state"] = "INTERRUPTED_AFTER_BATCH"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            write_status(status_path, manifest)
            test_index.close()
            return
    batch_files = sorted(batch_dir.glob("batch_*.parquet"))
    batch_files = [path for path in batch_files if ".candidates." not in path.name]
    entity_count, pair_count = write_final_batches(args.output, batch_files)
    metadata = {"threshold": threshold, "validation_macro_f05": validation_score, "feature_names": FEATURE_NAMES, "model": "XGBClassifier", "license": "XGBoost Apache-2.0; no hosted model", "blocking": {"joins": ["normalized_name_token", "postal_code"], "token_frequency_cutoff": 100, "max_candidates_per_entity": 250, "engine": "DuckDB hash joins", "checkpoints": "parquet", "batch_size": args.batch_size}}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest["state"] = "RUN COMPLETE"
    manifest["total_runtime_seconds"] = round(time.perf_counter() - started, 2)
    manifest["output_entities"] = entity_count
    manifest["output_candidate_pairs"] = pair_count
    manifest["validation_f05"] = validation_score
    manifest["final_summary"] = f"Entities={entity_count:,}; candidate pairs={pair_count:,}; validation F0.5={validation_score:.6f}; total runtime={manifest['total_runtime_seconds']:.2f}s; validator=run separately after output writing."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_status(status_path, manifest)
    test_index.close()
    print(f"RUN COMPLETE entities={entity_count} candidate_pairs={pair_count} total_seconds={manifest['total_runtime_seconds']}", flush=True)


if __name__ == "__main__":
    main()
