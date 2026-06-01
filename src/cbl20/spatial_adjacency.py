from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import BallTree
except ImportError:  # pragma: no cover
    BallTree = None


REQUIRED_LSOA_CSV_COLUMNS = {
    "LSOA21CD",
    "LSOA21NM",
    "BNG_E",
    "BNG_N",
    "LAT",
    "LONG",
}


def load_lsoa_centroids(
    csv_path: str | Path,
    *,
    code_col: str = "LSOA21CD",
    name_col: str = "LSOA21NM",
    easting_col: str = "BNG_E",
    northing_col: str = "BNG_N",
    lat_col: str = "LAT",
    lon_col: str = "LONG",
) -> pd.DataFrame:
    """
    Load the ONS LSOA 2021 BGC CSV and standardise the useful columns.

    Parameters
    ----------
    csv_path:
        Path to the ONS LSOA CSV file.
    code_col, name_col, easting_col, northing_col, lat_col, lon_col:
        Column names in the source file.

    Returns
    -------
    pd.DataFrame
        Columns:
        - lsoa_code
        - lsoa_name
        - bng_e
        - bng_n
        - lat
        - lon
        - shape_area, if available
        - shape_length, if available

    Notes
    -----
    This file contains centroid coordinates, not polygon geometry.
    Therefore, it supports centroid-distance neighbours, not true
    touching-boundary adjacency.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"LSOA centroid CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_LSOA_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "The LSOA CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = pd.DataFrame(
        {
            "lsoa_code": df[code_col].astype(str).str.strip(),
            "lsoa_name": df[name_col].astype(str).str.strip(),
            "bng_e": pd.to_numeric(df[easting_col], errors="coerce"),
            "bng_n": pd.to_numeric(df[northing_col], errors="coerce"),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )

    if "Shape__Area" in df.columns:
        out["shape_area"] = pd.to_numeric(df["Shape__Area"], errors="coerce")

    if "Shape__Length" in df.columns:
        out["shape_length"] = pd.to_numeric(df["Shape__Length"], errors="coerce")

    out = out.dropna(subset=["lsoa_code", "bng_e", "bng_n"]).drop_duplicates(
        subset=["lsoa_code"]
    )

    return out.reset_index(drop=True)


def validate_lsoa_code_coverage(
    model_lsoas: Iterable[str],
    lsoa_reference: pd.DataFrame,
    *,
    code_col: str = "lsoa_code",
) -> dict:
    """
    Check how many model LSOA codes are covered by the spatial reference file.
    """
    model_codes = set(pd.Series(list(model_lsoas)).dropna().astype(str))
    ref_codes = set(lsoa_reference[code_col].dropna().astype(str))

    matched = model_codes & ref_codes
    missing = model_codes - ref_codes

    return {
        "n_model_lsoas": len(model_codes),
        "n_reference_lsoas": len(ref_codes),
        "n_matched_lsoas": len(matched),
        "n_missing_lsoas": len(missing),
        "coverage_rate": np.nan if not model_codes else len(matched) / len(model_codes),
        "missing_lsoas_sample": sorted(list(missing))[:20],
    }


def build_knn_adjacency_from_centroids(
    lsoa_reference: pd.DataFrame,
    *,
    lsoa_codes: Optional[Sequence[str]] = None,
    k: int = 5,
    code_col: str = "lsoa_code",
    easting_col: str = "bng_e",
    northing_col: str = "bng_n",
) -> pd.DataFrame:
    """
    Build a centroid-distance k-nearest-neighbour table.

    Returns columns:
    - lsoa_code
    - neighbour_lsoa_code
    - neighbour_rank
    - distance_m
    - adjacency_method

    This is NOT true boundary adjacency. It is a distance-based proxy using
    British National Grid centroid coordinates.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    ref = lsoa_reference.copy()

    if lsoa_codes is not None:
        keep = set(pd.Series(lsoa_codes).dropna().astype(str))
        ref = ref[ref[code_col].astype(str).isin(keep)].copy()

    ref = ref.dropna(subset=[code_col, easting_col, northing_col]).drop_duplicates(
        subset=[code_col]
    )

    if len(ref) <= k:
        raise ValueError(
            f"Need more LSOAs than k. Got {len(ref)} LSOAs and k={k}."
        )

    coords = ref[[easting_col, northing_col]].to_numpy(dtype=float)
    codes = ref[code_col].astype(str).to_numpy()

    # k + 1 because the nearest neighbour of each point is itself.
    if BallTree is not None:
        tree = BallTree(coords, metric="euclidean")
        distances, indices = tree.query(coords, k=k + 1)
    else:
        # Fallback for environments without sklearn. Fine for small subsets.
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt((diff**2).sum(axis=2))
        indices = np.argsort(dist_matrix, axis=1)[:, : k + 1]
        distances = np.take_along_axis(dist_matrix, indices, axis=1)

    rows = []
    for i, focal_code in enumerate(codes):
        rank = 0
        for distance, neighbour_idx in zip(distances[i], indices[i]):
            neighbour_code = codes[neighbour_idx]
            if neighbour_code == focal_code:
                continue

            rank += 1
            rows.append(
                {
                    "lsoa_code": focal_code,
                    "neighbour_lsoa_code": neighbour_code,
                    "neighbour_rank": rank,
                    "distance_m": float(distance),
                    "adjacency_method": f"centroid_knn_k{k}",
                }
            )

            if rank == k:
                break

    return pd.DataFrame(rows)


def build_radius_adjacency_from_centroids(
    lsoa_reference: pd.DataFrame,
    *,
    lsoa_codes: Optional[Sequence[str]] = None,
    radius_m: float = 1500.0,
    max_neighbours: Optional[int] = None,
    code_col: str = "lsoa_code",
    easting_col: str = "bng_e",
    northing_col: str = "bng_n",
) -> pd.DataFrame:
    """
    Build a centroid-distance radius-neighbour table.

    Returns columns:
    - lsoa_code
    - neighbour_lsoa_code
    - neighbour_rank
    - distance_m
    - adjacency_method
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")

    ref = lsoa_reference.copy()

    if lsoa_codes is not None:
        keep = set(pd.Series(lsoa_codes).dropna().astype(str))
        ref = ref[ref[code_col].astype(str).isin(keep)].copy()

    ref = ref.dropna(subset=[code_col, easting_col, northing_col]).drop_duplicates(
        subset=[code_col]
    )

    if len(ref) < 2:
        raise ValueError("Need at least 2 LSOAs to build radius adjacency.")

    coords = ref[[easting_col, northing_col]].to_numpy(dtype=float)
    codes = ref[code_col].astype(str).to_numpy()

    if BallTree is not None:
        tree = BallTree(coords, metric="euclidean")
        neighbour_indices, neighbour_distances = tree.query_radius(
            coords,
            r=radius_m,
            return_distance=True,
            sort_results=True,
        )
    else:
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt((diff**2).sum(axis=2))
        neighbour_indices = []
        neighbour_distances = []
        for i in range(len(coords)):
            idx = np.where(dist_matrix[i] <= radius_m)[0]
            order = np.argsort(dist_matrix[i, idx])
            neighbour_indices.append(idx[order])
            neighbour_distances.append(dist_matrix[i, idx][order])

    rows = []
    for i, focal_code in enumerate(codes):
        rank = 0
        for neighbour_idx, distance in zip(neighbour_indices[i], neighbour_distances[i]):
            neighbour_code = codes[neighbour_idx]
            if neighbour_code == focal_code:
                continue

            rank += 1
            rows.append(
                {
                    "lsoa_code": focal_code,
                    "neighbour_lsoa_code": neighbour_code,
                    "neighbour_rank": rank,
                    "distance_m": float(distance),
                    "adjacency_method": f"centroid_radius_{int(radius_m)}m",
                }
            )

            if max_neighbours is not None and rank >= max_neighbours:
                break

    return pd.DataFrame(rows)


def summarize_adjacency(adjacency: pd.DataFrame) -> dict:
    """Summarise an adjacency/neighbour table."""
    if adjacency.empty:
        return {
            "n_edges": 0,
            "n_lsoas_with_neighbours": 0,
            "mean_neighbours": 0,
            "min_neighbours": 0,
            "max_neighbours": 0,
        }

    counts = adjacency.groupby("lsoa_code")["neighbour_lsoa_code"].nunique()

    return {
        "n_edges": len(adjacency),
        "n_lsoas_with_neighbours": int(counts.shape[0]),
        "mean_neighbours": float(counts.mean()),
        "min_neighbours": int(counts.min()),
        "max_neighbours": int(counts.max()),
    }


def add_previous_month_count(panel: pd.DataFrame) -> pd.DataFrame:
    """Add previous-month own-LSOA count."""
    out = panel.copy()
    out = out.sort_values(["lsoa_code", "month"])
    out["prev_month_count"] = (
        out.groupby("lsoa_code")["crime_count"].shift(1).fillna(0.0)
    )
    return out


def add_neighbour_previous_month_feature(
    panel: pd.DataFrame,
    adjacency: pd.DataFrame,
    *,
    neighbour_value_col: str = "prev_month_count",
    output_col: str = "neighbour_prev_month_count",
    scale_output: bool = True,
) -> pd.DataFrame:
    """
    Add an adjacency-aware previous-month neighbour feature.

    For each LSOA-month row, this computes the average previous-month crime
    count among neighbouring LSOAs.
    """
    required_panel_cols = {"lsoa_code", "month", "crime_count"}
    missing_panel = required_panel_cols - set(panel.columns)
    if missing_panel:
        raise ValueError(
            "Panel is missing required columns: "
            + ", ".join(sorted(missing_panel))
        )

    required_adj_cols = {"lsoa_code", "neighbour_lsoa_code"}
    missing_adj = required_adj_cols - set(adjacency.columns)
    if missing_adj:
        raise ValueError(
            "Adjacency table is missing required columns: "
            + ", ".join(sorted(missing_adj))
        )

    out = add_previous_month_count(panel)

    neighbour_values = out[
        ["lsoa_code", "month", neighbour_value_col]
    ].rename(
        columns={
            "lsoa_code": "neighbour_lsoa_code",
            neighbour_value_col: "neighbour_value",
        }
    )

    expanded = adjacency[["lsoa_code", "neighbour_lsoa_code"]].merge(
        neighbour_values,
        on="neighbour_lsoa_code",
        how="left",
    )

    neighbour_avg = (
        expanded
        .groupby(["lsoa_code", "month"], as_index=False)["neighbour_value"]
        .mean()
        .rename(columns={"neighbour_value": output_col})
    )

    out = out.merge(neighbour_avg, on=["lsoa_code", "month"], how="left")
    out[output_col] = out[output_col].fillna(0.0)

    if scale_output:
        scaled_col = f"{output_col}_scaled"
        std = out[output_col].std()
        if std == 0 or np.isnan(std):
            out[scaled_col] = 0.0
        else:
            out[scaled_col] = (out[output_col] - out[output_col].mean()) / std

    return out


def save_adjacency(adjacency: pd.DataFrame, output_path: str | Path) -> Path:
    """Save adjacency table to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adjacency.to_csv(output_path, index=False)
    return output_path


def load_adjacency(adjacency_path: str | Path) -> pd.DataFrame:
    """Load a previously saved adjacency table."""
    adjacency_path = Path(adjacency_path)
    if not adjacency_path.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adjacency_path}")
    return pd.read_csv(
        adjacency_path,
        dtype={"lsoa_code": str, "neighbour_lsoa_code": str},
    )
