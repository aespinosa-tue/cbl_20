import duckdb
import pandas as pd
from pathlib import Path


def read_parquet_query(parquet_path: Path, query: str) -> pd.DataFrame:
    """
    Runs SQL on the crime Parquet file, but when writing the SQL query u should refer to the crime table as crimes.
    """
    parquet_path = Path(parquet_path).as_posix()

    full_query = f"""
    SELECT *
    FROM read_parquet('{parquet_path}') AS crimes
    """

    if query.strip():
        full_query = query.replace("{parquet_path}", parquet_path)

    return duckdb.sql(full_query).df()


def check_crime_data(parquet_path: Path) -> pd.DataFrame:
    """
    Quick data validation summary for the crime parquet
    """
    parquet_path = Path(parquet_path).as_posix()

    return duckdb.sql(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT "Crime ID") AS n_unique_crime_ids,
            COUNT(DISTINCT "LSOA code") AS n_lsoas,
            COUNT(DISTINCT "Falls within") AS n_forces,
            MIN("Month") AS min_month,
            MAX("Month") AS max_month
        FROM read_parquet('{parquet_path}')
    """).df()