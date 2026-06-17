import duckdb as db
import pandas as pd
from pyarrow.compute import index
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

crime_types = [
    "Violence and sexual offences",
    "Criminal damage and arson",
    "Drugs",
    "Violent crime",
    "Burglary",
    "Other theft",
    "Vehicle crime",
    "Public order",
    "Other crime",
    "Shoplifting",
    "Robbery",
    "Bicycle theft",
    "Theft from the person",
    "Possession of weapons",
    "Public disorder and weapons",
    "Anti-social behaviour",
]

PATH_FORCE_CRIME_WEIGHT_MATRIX = PROCESSED_DIR / "force_crime_weight_matrix.parquet"


def generate_index(police_force: str, crime_counts: dict[str, int]) -> float:
    #sanity check:
    missing = set(crime_types) - set(crime_counts)
    extra = set(crime_counts) - set(crime_types)

    if missing:
        raise ValueError(f"Missing crime types: {missing}")

    if extra:
        raise ValueError(f"Invalid crime types: {extra}")
    

    df_crime_weights = db.sql(f"""
        SELECT *
        FROM '{PATH_FORCE_CRIME_WEIGHT_MATRIX}'
        WHERE "Force Name" = '{police_force}'
    """).df()

    if df_crime_weights.empty:
        raise ValueError(f"No weights found for police force: {police_force}")

    row = df_crime_weights.iloc[0]

    index_value = sum(
        crime_counts[crime_type] * row[crime_type]
        for crime_type in crime_types
    )

    return index_value

print(generate_index(
    police_force="Bedfordshire",
    crime_counts={
        "Violence and sexual offences": 1,
        "Criminal damage and arson": 0,
        "Drugs": 0,
        "Violent crime": 0,
        "Burglary": 0,
        "Other theft": 0,
        "Vehicle crime": 0,
        "Public order": 0,
        "Other crime": 0,
        "Shoplifting": 0,
        "Robbery": 0,
        "Bicycle theft": 1,
        "Theft from the person": 0,
        "Possession of weapons": 0,
        "Public disorder and weapons": 0,
        "Anti-social behaviour": 1,
    }
))

print(db.sql(f"""
    SELECT *
    FROM '{PATH_FORCE_CRIME_WEIGHT_MATRIX}'
    WHERE "Force Name" = 'Bedfordshire'
    """).df())