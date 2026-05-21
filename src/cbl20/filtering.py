from pathlib import Path
import duckdb as db

# Robust project root when running from src/cbl20/filtering.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / "crimes_filtered_for_model.parquet"

PARQUET_PATH = PROCESSED_DIR / "crimes_clean_dedup_all_years.parquet"

print(f"Project root: {PROJECT_ROOT}")
print(f"Crime parquet path: {PARQUET_PATH}")
print(f"Parquet exists: {PARQUET_PATH.exists()}")

if not PARQUET_PATH.exists():
    raise FileNotFoundError(
        "Could not find crimes_clean_dedup_all_years.parquet. "
        "Expected it under data/processed/."
    )

con = db.connect()

con.execute(f"""
    COPY (
        SELECT *
        FROM read_parquet('{PARQUET_PATH.as_posix()}') AS crimes
        WHERE
            Month >= '2012-01'
            AND (
                Month < '2020-03'
                OR Month >= '2021-03'
            )
    )
    TO '{OUTPUT_PATH.as_posix()}'
    (FORMAT parquet, COMPRESSION zstd);
""")

print(f"Saved filtered data to: {OUTPUT_PATH}")
print(f"Output exists: {OUTPUT_PATH.exists()}")