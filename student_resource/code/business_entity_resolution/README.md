# Business Entity Resolution

Run from the repository root (the default data path is `student_resource/dataset`):

```powershell
pip install -r code/business_entity_resolution/requirements.txt
python -m student_resource.code.business_entity_resolution.src.explore --data-root student_resource/dataset
python -m student_resource.code.business_entity_resolution.run_pipeline --output student_resource/output
python student_resource/utils/validate_submission.py --matching student_resource/output/matching_results.tsv --candidate student_resource/output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

The pipeline streams TSV files with `sep="\t"`, normalizes text country-agnostically, builds SQLite-backed normalized-name-token and postal-code inverted blocks, caps each entity at 250 candidates after cheap ranking, computes pair features, trains an Apache-2.0-licensed XGBoost classifier, tunes precision-oriented matching, and writes the two required output files. There is no TF-IDF/hnswlib stage; candidate generation uses the bounded SQLite inverted index. No hosted model or external lookup is used.
