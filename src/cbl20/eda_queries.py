import duckdb
import pandas as pd
from pathlib import Path

## this file just has reusable queries we might use a lot

def monthly_total_crime(parquet_path: Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path).as_posix()

    return duckdb.sql(f"""
        SELECT
            "Month" AS month,
            COUNT(*) AS total_crimes
        FROM read_parquet('{parquet_path}')
        GROUP BY "Month"
        ORDER BY "Month"
    """).df()


def monthly_crime_by_type(parquet_path: Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path).as_posix()

    return duckdb.sql(f"""
        SELECT
            "Month" AS month,
            "Crime type" AS crime_type,
            COUNT(*) AS total_crimes
        FROM read_parquet('{parquet_path}')
        GROUP BY "Month", "Crime type"
        ORDER BY "Month", "Crime type"
    """).df()


def lsoa_crime_counts(parquet_path: Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path).as_posix()

    return duckdb.sql(f"""
        SELECT
            "LSOA code" AS lsoa_code,
            COUNT(*) AS total_crimes
        FROM read_parquet('{parquet_path}')
        WHERE "LSOA code" IS NOT NULL
        GROUP BY "LSOA code"
        ORDER BY total_crimes DESC
    """).df()


def force_crime_totals(parquet_path: Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path).as_posix()

    return duckdb.sql(f"""
        SELECT
            "Falls within" AS police_force,
            COUNT(*) AS total_crimes
        FROM read_parquet('{parquet_path}')
        GROUP BY "Falls within"
        ORDER BY total_crimes DESC
    """).df()


def force_crime_type_counts(parquet_path: Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path).as_posix()

    return duckdb.sql(f"""
        SELECT
            "Falls within" AS police_force,
            "Crime type" AS crime_type,
            COUNT(*) AS total_crimes
        FROM read_parquet('{parquet_path}')
        GROUP BY "Falls within", "Crime type"
        ORDER BY police_force, total_crimes DESC
    """).df()


def monthly_force_counts(parquet_path: Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path).as_posix()

    df = duckdb.sql(f"""
        SELECT
            "Month" AS month,
            "Falls within" AS police_force,
            COUNT(*) AS total_crimes
        FROM read_parquet('{parquet_path}')
        GROUP BY "Month", "Falls within"
        ORDER BY "Month", "Falls within"
    """).df()

    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    df["month_name"] = df["month"].dt.strftime("%b")

    return df