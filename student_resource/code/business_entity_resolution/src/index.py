"""Columnar DuckDB blocking with parquet checkpoints."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterator

import duckdb
import pandas as pd

from .normalize import LEGAL_SUFFIXES, POSTAL_RE

TOKEN_FREQUENCY_CUTOFF = 100


class RecordIndex:
    """Materialize targets once, then generate all candidates with hash joins."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path.with_suffix(".duckdb")))
        self.temp_directory = self.path.parent / f"{self.path.stem}_tmp"
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        self.connection.execute("SET memory_limit='5GB'")
        self.connection.execute("SET preserve_insertion_order=false")
        self.connection.execute("SET temp_directory=?", [str(self.temp_directory)])
        self.candidate_calls = 0
        self.candidate_pairs = 0
        self.capped_calls = 0
        self.max_raw_candidates = 0
        self.candidate_path: Path | None = None

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        names = (frame["business_name"].fillna("").astype(str).str.normalize("NFKD")
                 .str.encode("ascii", "ignore").str.decode("ascii").str.lower()
                 .str.replace("&", " and ", regex=False))
        suffix_pattern = r"\b(?:" + "|".join(sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b"
        tokens = names.str.replace(suffix_pattern, " ", regex=True).str.findall(r"[a-z0-9]+")
        frame = frame.copy()
        frame["name_tokens"] = tokens.map(lambda values: sorted({token for token in values if len(token) >= 3}))
        frame["postal_code"] = (frame["business_address"].fillna("").astype(str)
                                 .str.extract(f"({POSTAL_RE.pattern})", expand=False).fillna("")
                                 .str.lower().str.replace(r"[^a-z0-9]", "", regex=True))
        return frame

    def _load_sources(self, source_paths: list[Path], checkpoint: Path, max_rows: int | None = None) -> None:
        if checkpoint.exists():
            print(f"[checkpoint] using {checkpoint}", flush=True)
            self.connection.execute("CREATE OR REPLACE TABLE records AS SELECT * FROM read_parquet(?)", [str(checkpoint)])
        else:
            started = time.perf_counter()
            parts: list[Path] = []
            for number, source_path in enumerate(source_paths):
                source_parts = []
                for chunk_number, frame in enumerate(pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, nrows=max_rows, chunksize=250_000)):
                    part = checkpoint.with_name(f"{checkpoint.stem}_{number}_{chunk_number}.parquet")
                    if not part.exists():
                        self._normalize_frame(frame).to_parquet(part, index=False)
                    source_parts.append(part)
                parts.extend(source_parts)
            self.connection.execute("CREATE OR REPLACE TABLE records AS SELECT * FROM read_parquet(?)", [[str(part) for part in parts]])
            self.connection.execute("COPY records TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(checkpoint)])
            print(f"[stage:end] normalization elapsed={time.perf_counter() - started:.2f}s rows={self.connection.execute('SELECT COUNT(*) FROM records').fetchone()[0]}", flush=True)
        self.connection.execute("CREATE OR REPLACE TABLE target_tokens AS SELECT entity_id, unnest(name_tokens) AS token FROM records")
        self.connection.execute("CREATE OR REPLACE TABLE target_postal AS SELECT entity_id, postal_code FROM records WHERE postal_code <> ''")
        self.connection.execute("CREATE OR REPLACE TABLE token_frequency AS SELECT token, COUNT(*) AS frequency FROM target_tokens GROUP BY token")

    def build(self, source_paths, checkpoint: Path | None = None, max_rows: int | None = None) -> None:
        checkpoint = checkpoint or self.path.with_suffix(".records.parquet")
        self._load_sources(list(source_paths), checkpoint, max_rows)
        counts = self.connection.execute("SELECT COUNT(*), COUNT(DISTINCT entity_id) FROM records").fetchone()
        print(f"[columnar:index] records={counts[0]} unique_ids={counts[1]} signals=token,postal", flush=True)

    def build_candidates(self, source1_path: Path, checkpoint: Path, max_source1: int | None = None) -> Path:
        started = time.perf_counter()
        if checkpoint.exists():
            self.candidate_path = checkpoint
            self.connection.execute("CREATE OR REPLACE TABLE capped_candidates AS SELECT * FROM read_parquet(?)", [str(checkpoint)])
            source1_parquet = checkpoint.with_name(f"{checkpoint.stem}_source1.parquet")
            if source1_parquet.exists():
                self.connection.execute("CREATE OR REPLACE TABLE source1 AS SELECT * FROM read_parquet(?)", [str(source1_parquet)])
            print(f"[checkpoint] using {checkpoint}", flush=True)
            return checkpoint
        frame = pd.read_csv(source1_path, sep="\t", dtype=str, keep_default_na=False, nrows=max_source1)
        normalized = self._normalize_frame(frame)
        source1_parquet = checkpoint.with_name(f"{checkpoint.stem}_source1.parquet")
        normalized.to_parquet(source1_parquet, index=False)
        self.connection.execute("CREATE OR REPLACE TABLE source1 AS SELECT * FROM read_parquet(?)", [str(source1_parquet)])
        self.connection.execute("CREATE OR REPLACE TABLE source1_tokens AS SELECT entity_id AS source1_entity_id, unnest(name_tokens) AS token FROM source1")
        self.connection.execute("CREATE OR REPLACE TABLE source1_postal AS SELECT entity_id AS source1_entity_id, postal_code FROM source1 WHERE postal_code <> ''")
        self.connection.execute(f"""
            CREATE OR REPLACE TABLE capped_candidates AS
            SELECT source1_entity_id, candidate_entity_id
            FROM (
                SELECT source1_entity_id, candidate_entity_id,
                       ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY shared_token_count DESC, postal_match DESC, candidate_entity_id
                ) AS candidate_rank
                FROM (
                    SELECT source1_entity_id, candidate_entity_id,
                           SUM(shared_token_count) AS shared_token_count,
                           MAX(postal_match) AS postal_match
                    FROM (
                        SELECT s.source1_entity_id, t.entity_id AS candidate_entity_id,
                               COUNT(*) AS shared_token_count, 0 AS postal_match
                        FROM source1_tokens s
                        JOIN target_tokens t USING (token)
                        JOIN token_frequency f USING (token)
                        WHERE f.frequency <= {TOKEN_FREQUENCY_CUTOFF}
                        GROUP BY s.source1_entity_id, t.entity_id
                        UNION ALL
                        SELECT s.source1_entity_id, t.entity_id, 0, 1
                        FROM source1_postal s JOIN target_postal t USING (postal_code)
                    ) pairs
                    GROUP BY source1_entity_id, candidate_entity_id
                ) scored
            ) ranked
            WHERE candidate_rank <= 250
        """)
        self.connection.execute("COPY capped_candidates TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(checkpoint)])
        selected = self.connection.execute("SELECT COUNT(*) FROM capped_candidates").fetchone()[0]
        raw = self.connection.execute(f"SELECT COUNT(*) FROM (SELECT source1_entity_id, candidate_entity_id FROM (SELECT s.source1_entity_id, t.entity_id AS candidate_entity_id FROM source1_tokens s JOIN target_tokens t USING (token) JOIN token_frequency f USING (token) WHERE f.frequency <= {TOKEN_FREQUENCY_CUTOFF} UNION ALL SELECT s.source1_entity_id, t.entity_id FROM source1_postal s JOIN target_postal t USING (postal_code)) GROUP BY 1, 2)").fetchone()[0]
        capped = self.connection.execute(f"SELECT COUNT(*) FROM (SELECT source1_entity_id FROM (SELECT DISTINCT s.source1_entity_id, t.entity_id AS candidate_entity_id FROM source1_tokens s JOIN target_tokens t USING (token) JOIN token_frequency f USING (token) WHERE f.frequency <= {TOKEN_FREQUENCY_CUTOFF} UNION ALL SELECT DISTINCT s.source1_entity_id, t.entity_id FROM source1_postal s JOIN target_postal t USING (postal_code)) pairs GROUP BY source1_entity_id HAVING COUNT(*) > 250)").fetchone()[0]
        self.candidate_pairs = selected
        self.capped_calls = capped
        self.candidate_path = checkpoint
        print(f"[blocking] token+postal hash joins raw_pairs={raw} selected_pairs={selected} capped_entities={capped} elapsed={time.perf_counter() - started:.2f}s", flush=True)
        return checkpoint

    def iter_candidate_records(self, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
        query = """
            SELECT c.source1_entity_id, c.candidate_entity_id,
                   s.business_name AS source1_business_name, s.business_address AS source1_business_address, s.country AS source1_country,
                   r.business_name, r.business_address, r.country
            FROM capped_candidates c
            JOIN source1 s ON s.entity_id = c.source1_entity_id
            JOIN records r ON r.entity_id = c.candidate_entity_id
            ORDER BY c.source1_entity_id, c.candidate_entity_id
        """
        self.connection.execute(query)
        while True:
            batch = self.connection.fetch_df_chunk(batch_size)
            if batch.empty:
                break
            yield batch

    def build_candidate_batch(self, frame: pd.DataFrame, checkpoint: Path) -> dict[str, int | float]:
        """Build and persist one Source 1 candidate batch using DuckDB joins."""
        started = time.perf_counter()
        normalized = self._normalize_frame(frame)
        source1_parquet = checkpoint.with_name(f"{checkpoint.stem}_source1.parquet")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(source1_parquet, index=False)
        self.connection.execute("CREATE OR REPLACE TABLE source1 AS SELECT * FROM read_parquet(?)", [str(source1_parquet)])
        self.connection.execute("CREATE OR REPLACE TABLE source1_tokens AS SELECT entity_id AS source1_entity_id, unnest(name_tokens) AS token FROM source1")
        self.connection.execute("CREATE OR REPLACE TABLE source1_postal AS SELECT entity_id AS source1_entity_id, postal_code FROM source1 WHERE postal_code <> ''")
        self.connection.execute(f"""
            CREATE OR REPLACE TABLE capped_candidates AS
            SELECT source1_entity_id, candidate_entity_id
            FROM (
                SELECT source1_entity_id, candidate_entity_id,
                       ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY shared_token_count DESC, postal_match DESC, candidate_entity_id) AS candidate_rank
                FROM (
                    SELECT source1_entity_id, candidate_entity_id, SUM(shared_token_count) AS shared_token_count, MAX(postal_match) AS postal_match
                    FROM (
                        SELECT s.source1_entity_id, t.entity_id AS candidate_entity_id, COUNT(*) AS shared_token_count, 0 AS postal_match
                        FROM source1_tokens s JOIN target_tokens t USING (token) JOIN token_frequency f USING (token)
                        WHERE f.frequency <= {TOKEN_FREQUENCY_CUTOFF}
                        GROUP BY s.source1_entity_id, t.entity_id
                        UNION ALL
                        SELECT s.source1_entity_id, t.entity_id, 0, 1
                        FROM source1_postal s JOIN target_postal t USING (postal_code)
                    ) pairs
                    GROUP BY source1_entity_id, candidate_entity_id
                ) scored
            ) ranked
            WHERE candidate_rank <= 250
        """)
        self.connection.execute("COPY capped_candidates TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(checkpoint)])
        selected = int(self.connection.execute("SELECT COUNT(*) FROM capped_candidates").fetchone()[0])
        entities = int(self.connection.execute("SELECT COUNT(DISTINCT entity_id) FROM source1").fetchone()[0])
        return {"source1_entities": entities, "candidate_pairs": selected, "elapsed_seconds": round(time.perf_counter() - started, 2)}

    def candidates(self, record: Dict[str, str], **_: int) -> Iterator[Dict[str, str]]:
        if self.candidate_path is None:
            return
        rows = self.connection.execute("""
            SELECT r.entity_id, r.business_name, r.business_address, r.country
            FROM capped_candidates c JOIN records r ON r.entity_id = c.candidate_entity_id
            WHERE c.source1_entity_id = ?
        """, [record["entity_id"]]).fetchall()
        self.candidate_calls += 1
        self.candidate_pairs += len(rows)
        yield from (dict(zip(("entity_id", "business_name", "business_address", "country"), row)) for row in rows)

    def close(self) -> None:
        self.connection.close()
