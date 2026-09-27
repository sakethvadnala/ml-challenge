# Up To Now

## Dataset and Path Audit

- Confirmed the real dataset is under `student_resource/dataset`.
- Ignored `__MACOSX/student_resource/dataset`; it contains macOS `._*` metadata files.
- Confirmed the real TSV files are tab-separated and load with `pandas.read_csv(..., sep="\t")`.
- Confirmed row counts:
  - Train Source 1: 2,206,821
  - Train Source 2: 5,034,616
  - Train Source 3: 5,285,603
  - Train ground truth: 2,206,821
  - Test Source 1: 1,732,544
  - Test Source 2: 4,887,273
  - Test Source 3: confirmed in the earlier full dataset audit
- The pipeline defaults were changed from the incorrect relative `dataset` path to a repository-relative `student_resource/dataset` path derived from `__file__`.
- The README reproduction commands were updated to use the corrected paths.

## Original Blocking Audit

- The original implementation used SQLite inserts and per-record SQL lookups.
- It did not use `hnswlib`, TF-IDF, or an ANN similarity stage.
- No full Source 1 x Source 2 x Source 3 cross join was found.
- The original implementation had a per-block limit but no final per-Source-1 candidate cap.
- A final cap of 250 candidates per Source 1 entity was added before the columnar rewrite.
- Stage and progress logging was added for loading, normalization, blocking, capping, features, training, threshold selection, prediction, and writing.

## Columnar Rewrite

- Added pinned dependency `duckdb==1.4.3` to `student_resource/code/business_entity_resolution/requirements.txt`.
- Replaced the SQLite-based `RecordIndex` with DuckDB and parquet checkpoints in `student_resource/code/business_entity_resolution/src/index.py`.
- Normalized target records into parquet using vectorized pandas string operations.
- Materialized normalized name tokens and postal codes as DuckDB relations.
- Implemented token blocking with a DuckDB hash join.
- Implemented postal blocking with a DuckDB hash join.
- Deduplicated candidate pairs and aggregated shared-token counts.
- Applied the 250-candidate cap with a vectorized `ROW_NUMBER() OVER (PARTITION BY source1_entity_id ...)` query.
- Reworked `run_pipeline.py` to consume DuckDB candidate batches instead of querying once per Source 1 record.
- Added parquet checkpoint paths for normalized target data and capped candidate data.
- Added `--prototype-s1` and `--prototype-target-rows` options for small-scale validation.
- DuckDB 1.4.3 import and both edited Python modules compile successfully.

## Prototype Run Evidence

The prototype was started with:

```powershell
python -m student_resource.code.business_entity_resolution.run_pipeline `
  --train-limit 20000 `
  --prototype-s1 20000 `
  --prototype-target-rows 500000 `
  --cache student_resource/.ber_cache_duckdb_prototype `
  --output student_resource/output_prototype
```

The prototype successfully completed loading and normalization before stopping during candidate capping.

Valid checkpoints and database state found afterward:

- `train.records_0.parquet`: valid, 500,000 rows.
- `train.records_1.parquet`: valid, 500,000 rows.
- `train.records.parquet`: valid, 1,000,000 rows.
- `train.candidates_source1.parquet`: valid, 20,000 rows.
- `train.duckdb`: readable and 6.7 GB.
- DuckDB `records`: 1,000,000 rows.
- DuckDB `target_tokens`: 2,477,674 rows.
- DuckDB `target_postal`: 73,374 rows.
- DuckDB `source1`: 20,000 rows.
- DuckDB `candidate_stage`: 254,848,989 rows.
- DuckDB `candidate_pairs`: 254,848,559 rows.

The `capped_candidates` table and capped candidate parquet checkpoint were not created. No model, prediction, or output files were written. The prototype validator therefore has not passed yet.

## Legacy Artifacts

- `.ber_cache/train.sqlite` is an old SQLite artifact and is corrupt.
- `PRAGMA integrity_check` reported invalid page numbers and out-of-order rowids.
- It must not be reused by the new DuckDB pipeline.
- `student_resource/output_prototype` is empty.
- `student_resource/output` does not exist.

## Current Stop Point

Completed:

1. Data loading.
2. Target normalization and parquet checkpointing.
3. Token and postal blocking joins.
4. Pre-cap candidate pair materialization.

Not completed:

1. Candidate capping.
2. Feature engineering.
3. Model training.
4. Threshold selection.
5. Test prediction.
6. Output writing.
7. Validator PASS.

The last process ended while materializing the candidate-cap query. No Python traceback was present in the available terminal output, and no Windows Resource Exhaustion or application crash event was found. The exact termination cause is therefore unproven, but it was an abnormal termination during candidate capping, likely due to resource pressure.

## Checkpoint Reuse Caveat

- Normalized target parquet checkpoints are detected and loaded on a rerun.
- Candidate checkpoint detection exists, but the current code does not reconstruct the DuckDB `capped_candidates` table from an existing candidate parquet file before prediction.
- Pre-cap DuckDB tables are not currently treated as resumable checkpoints.
- This checkpoint-resume behavior still needs to be fixed and tested.

## Remaining Work

1. Make candidate capping memory-safe and resumable, preferably by writing capped candidates in partitioned batches or using DuckDB settings/temporary parquet spill.
2. Ensure an existing capped candidate parquet checkpoint restores the DuckDB relation needed by prediction.
3. Rerun the 20k prototype end to end.
4. Run the validator and obtain `PASS` on the prototype.
5. Measure the full-data blocking stage before attempting the complete train/test run.
6. Run the full pipeline only after the prototype and blocking timing checks pass.
