import duckdb
import pandas as pd
from pathlib import Path

# testing pre-using the crime index but i made a file in data/external to store the actual weights
DEFAULT_HARM_WEIGHTS = {
    "Anti-social behaviour": 1,
    "Bicycle theft": 2,
    "Burglary": 5,
    "Drugs": 3,
    "Other crime": 2
}


def build_lsoa_month_demand(
    parquet_path: Path,
    harm_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build LSOA-month harm-weighted police demand: LSOA code + police force + month + crime type + count + hamr weight + weighted demand
    """
    harm_weights = harm_weights or DEFAULT_HARM_WEIGHTS
    parquet_path = Path(parquet_path).as_posix()

    counts = duckdb.sql(f"""
        SELECT
            "LSOA code" AS lsoa_code,
            "LSOA name" AS lsoa_name,
            "Falls within" AS police_force,
            "Month" AS month,
            "Crime type" AS crime_type,
            COUNT(*) AS crime_count
        FROM read_parquet('{parquet_path}')
        WHERE "LSOA code" IS NOT NULL
        GROUP BY
            "LSOA code",
            "LSOA name",
            "Falls within",
            "Month",
            "Crime type"
    """).df()

    weights_df = pd.DataFrame(
        list(harm_weights.items()),
        columns=["crime_type", "harm_weight"]
    )

    demand = counts.merge(weights_df, on="crime_type", how="left")
    demand["harm_weight"] = demand["harm_weight"].fillna(1)
    demand["harm_weighted_demand"] = (
        demand["crime_count"] * demand["harm_weight"]
    )

    return demand


def aggregate_force_month_demand(lsoa_month_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate LSOA-month harm demand to force-month level
        essentially groups the data by police force + month (output is one row per police force per month)
    """
    return (
        lsoa_month_demand
        .groupby(["police_force", "month"], as_index=False)
        .agg(
            total_crime=("crime_count", "sum"),
            harm_weighted_demand=("harm_weighted_demand", "sum"),
            n_lsoas=("lsoa_code", "nunique"),
        )
    )


def aggregate_force_total_demand(lsoa_month_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate harm demand to police force level
        this groups only by police force disregarding month
    """
    force = (
        lsoa_month_demand
        .groupby("police_force", as_index=False)
        .agg(
            total_crime=("crime_count", "sum"),
            harm_weighted_demand=("harm_weighted_demand", "sum"),
            n_lsoas=("lsoa_code", "nunique"),
        )
    )

    force["crime_share"] = force["total_crime"] / force["total_crime"].sum()
    force["harm_demand_share"] = (
        force["harm_weighted_demand"] / force["harm_weighted_demand"].sum()
    )

    return force.sort_values("harm_weighted_demand", ascending=False)