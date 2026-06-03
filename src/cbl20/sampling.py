import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 20260603

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crimes_filtered_for_model.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_MONTH = "2017-01"
END_MONTH = "2019-12"

N_FORCES = 5
N_CRIME_TYPES = 5

EXCLUDED_FORCES = {"British Transport Police"}

rng = np.random.default_rng(SEED)

available = duckdb.sql(f"""
    SELECT
        "Falls within" AS police_force,
        "Crime type" AS crime_type,
        COUNT(*) AS n_records
    FROM read_parquet('{DATA_PATH.as_posix()}')
    WHERE "Falls within" IS NOT NULL
      AND "Crime type" IS NOT NULL
      AND "Month" BETWEEN '{START_MONTH}' AND '{END_MONTH}'
    GROUP BY
        "Falls within",
        "Crime type"
""").df()

available = available[~available["police_force"].isin(EXCLUDED_FORCES)].copy()

forces = sorted(available["police_force"].unique())

selected_forces = sorted(
    rng.choice(forces, size=N_FORCES, replace=False).tolist()
)

available_for_selected_forces = available[
    available["police_force"].isin(selected_forces)
].copy()

crime_force_counts = (
    available_for_selected_forces
    .groupby("crime_type")["police_force"]
    .nunique()
    .reset_index(name="n_forces_available")
)

eligible_crimes = sorted(
    crime_force_counts.loc[
        crime_force_counts["n_forces_available"] == N_FORCES,
        "crime_type"
    ].tolist()
)

if len(eligible_crimes) < N_CRIME_TYPES:
    raise ValueError(
        f"Only {len(eligible_crimes)} crime types are available in all selected forces. "
        "Try another seed or lower N_CRIME_TYPES."
    )

selected_crimes = sorted(
    rng.choice(eligible_crimes, size=N_CRIME_TYPES, replace=False).tolist()
)

manifest = pd.MultiIndex.from_product(
    [selected_forces, selected_crimes],
    names=["police_force", "crime_type"]
).to_frame(index=False)

manifest = manifest.merge(
    available,
    on=["police_force", "crime_type"],
    how="left"
)

manifest["seed"] = SEED
manifest["start_month"] = START_MONTH
manifest["end_month"] = END_MONTH

manifest_path = OUTPUT_DIR / f"shared_model_sample_seed_{SEED}.csv"
config_path = OUTPUT_DIR / f"shared_model_sample_seed_{SEED}.json"

manifest.to_csv(manifest_path, index=False)

config = {
    "seed": SEED,
    "start_month": START_MONTH,
    "end_month": END_MONTH,
    "n_forces": N_FORCES,
    "n_crime_types": N_CRIME_TYPES,
    "selected_forces": selected_forces,
    "selected_crime_types": selected_crimes,
    "excluded_forces": sorted(EXCLUDED_FORCES),
    "manifest_path": str(manifest_path),
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("Selected police forces:")
for force in selected_forces:
    print("-", force)

print("\nSelected crime types:")
for crime in selected_crimes:
    print("-", crime)

print("\nSaved manifest to:")
print(manifest_path)

manifest