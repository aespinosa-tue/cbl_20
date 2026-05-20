from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] # gets this file's path, and standardizes dir for below

# this file here ensures that we all can work reusing the same variable names for directionaries and/or data

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
DASHBOARD_OUTPUT_DIR = OUTPUT_DIR / "dashboard"

CRIME_PARQUET_PATH = PROCESSED_DIR / "crimes_clean_dedup_all_years.parquet"