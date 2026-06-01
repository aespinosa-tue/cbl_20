
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Police Demand & Resource Allocation Dashboard",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

USE_DEMO_DATA = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REAL_FILES = {
    "predictions": TABLES_DIR / "all_forces_adjacency_no_offset_dashboard_lsoa_month_predictions.csv",
    "force_month": TABLES_DIR / "all_forces_adjacency_no_offset_dashboard_force_month_crime_type.csv",
    "lsoa_totals": TABLES_DIR / "all_forces_adjacency_no_offset_dashboard_lsoa_totals.csv",
    "summary": TABLES_DIR / "all_forces_adjacency_no_offset_scaling_summary.csv",
}

POLICE_FORCES = [
    "Greater Manchester Police",
    "Kent Police",
    "Metropolitan Police Service",
    "West Midlands Police",
    "West Yorkshire Police",
]

CRIME_TYPES = [
    "Anti-social behaviour",
    "Burglary",
    "Criminal damage and arson",
    "Public order",
    "Vehicle crime",
    "Violence and sexual offences",
]

FORCE_CENTRES = {
    "Greater Manchester Police": (53.4808, -2.2426),
    "Kent Police": (51.2787, 0.5217),
    "Metropolitan Police Service": (51.5072, -0.1276),
    "West Midlands Police": (52.4862, -1.8904),
    "West Yorkshire Police": (53.8008, -1.5491),
}

LSOA_NAMES = {
    "Greater Manchester Police": ["Manchester 021A", "Salford 014C", "Bolton 009B", "Rochdale 011D"],
    "Kent Police": ["Maidstone 007B", "Canterbury 004A", "Dartford 008C", "Thanet 012D"],
    "Metropolitan Police Service": ["Barking and Dagenham 014C", "Camden 021B", "Croydon 033A", "Newham 027D"],
    "West Midlands Police": ["Birmingham 040B", "Coventry 018A", "Wolverhampton 012C", "Dudley 006D"],
    "West Yorkshire Police": ["Leeds 034A", "Bradford 019B", "Wakefield 011C", "Kirklees 008D"],
}

PRESSURE_ORDER = ["Lower", "Moderate", "High", "Very high"]

CSS = """
<style>
    :root {
        --bg: #111418;
        --panel: #191d23;
        --panel-2: #20252d;
        --panel-3: #262c35;
        --text: #f1f5f9;
        --muted: #a8b3c2;
        --line: #343b46;
        --accent: #ff6a3d;
        --accent-2: #ff9b70;
        --good: #43d17a;
        --warn: #f5c542;
        --bad: #ff6b6b;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] {
        background: rgba(17,20,24,0.9);
    }
    [data-testid="stSidebar"] {
        background: #0d1014;
        border-right: 1px solid var(--line);
    }
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span, div {
        color: var(--text);
    }
    .hero-box {
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(255,106,61,0.65);
        border-radius: 20px;
        background: linear-gradient(135deg, #1a1f27 0%, #15191f 55%, #221a16 100%);
        box-shadow: 0 14px 34px rgba(0,0,0,0.28);
        margin-bottom: 1.1rem;
    }
    .hero-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }
    .hero-text {
        font-size: 1.02rem;
        color: var(--muted);
        line-height: 1.45;
    }
    .section-note {
        padding: 0.9rem 1rem;
        background: var(--panel);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        margin: 0.4rem 0 1rem 0;
        color: var(--muted);
    }
    .kpi-card {
        padding: 1rem 1.1rem;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        min-height: 118px;
    }
    .kpi-label {
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-help {
        color: #7f8b9a;
        font-size: 0.78rem;
        margin-top: 0.45rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.22rem 0.62rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.25rem;
    }
    .pill-high { background: rgba(255,107,107,0.16); color: #ff9a9a; border: 1px solid rgba(255,107,107,0.35); }
    .pill-moderate { background: rgba(245,197,66,0.16); color: #ffe08a; border: 1px solid rgba(245,197,66,0.35); }
    .pill-low { background: rgba(67,209,122,0.16); color: #8ff0b0; border: 1px solid rgba(67,209,122,0.35); }
    .pill-caution { background: rgba(170,120,255,0.16); color: #c7abff; border: 1px solid rgba(170,120,255,0.35); }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--panel);
        border-radius: 12px 12px 0 0;
        color: var(--muted);
        padding: 0.65rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--panel-3);
        color: #ffffff;
        border-bottom: 2px solid var(--accent);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stAlert"] {
        background: var(--panel);
        border: 1px solid var(--line);
        color: var(--text);
    }
    .small-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.35;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

@dataclass
class DashboardData:
    predictions: pd.DataFrame
    force_month: pd.DataFrame
    lsoa_totals: pd.DataFrame
    model_summary: pd.DataFrame
    funding: pd.DataFrame
    deprivation: pd.DataFrame
    kde_points: pd.DataFrame
    metadata: dict


def theme_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#191d23",
        font=dict(color="#f1f5f9"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=55, b=25),
    )
    fig.update_xaxes(gridcolor="#303741", zerolinecolor="#303741")
    fig.update_yaxes(gridcolor="#303741", zerolinecolor="#303741")
    return fig


def month_range(start="2019-01", periods=12):
    return pd.date_range(start=start, periods=periods, freq="MS")


def pressure_category(series):
    q60 = series.quantile(0.60)
    q80 = series.quantile(0.80)
    q93 = series.quantile(0.93)
    return pd.cut(series, [-np.inf, q60, q80, q93, np.inf], labels=["Lower", "Moderate", "High", "Very high"]).astype(str)


def confidence_from_metrics(abs_total_pct_error, coverage_90, max_rhat, min_ess):
    if max_rhat > 1.05 or min_ess < 50 or coverage_90 < 0.80 or abs_total_pct_error > 0.30:
        return "Use with caution"
    if abs_total_pct_error > 0.18 or coverage_90 < 0.88:
        return "Moderate"
    return "High"


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def jitter_force_point(force, rng, scale=0.18):
    lat, lon = FORCE_CENTRES.get(force, (52.5, -1.5))
    return lat + rng.normal(0, scale), lon + rng.normal(0, scale)


@st.cache_data(show_spinner=False)
def generate_demo_data(seed=42):
    rng = np.random.default_rng(seed)
    months = month_range("2019-01", 12)
    force_scale = {
        "Greater Manchester Police": 1.65,
        "Kent Police": 0.75,
        "Metropolitan Police Service": 1.35,
        "West Midlands Police": 1.00,
        "West Yorkshire Police": 0.95,
    }
    crime_scale = {
        "Anti-social behaviour": 1.25,
        "Burglary": 0.80,
        "Criminal damage and arson": 0.70,
        "Public order": 0.65,
        "Vehicle crime": 0.85,
        "Violence and sexual offences": 1.45,
    }

    prediction_rows, summary_rows, kde_rows = [], [], []

    for force in POLICE_FORCES:
        lsoa_list = LSOA_NAMES[force]
        for crime in CRIME_TYPES:
            baseline_mae = rng.uniform(0.7, 2.2) * force_scale[force] * crime_scale[crime]
            model_edge = rng.normal(0.08, 0.16)
            if force in {"Kent Police", "West Yorkshire Police"}:
                model_edge -= rng.uniform(0.05, 0.18)
            if force == "Greater Manchester Police":
                model_edge += rng.uniform(0.05, 0.35)
            model_mae = max(0.05, baseline_mae - model_edge)
            abs_total_pct_error = float(np.clip(abs(rng.normal(0.12, 0.12)), 0.01, 0.65))
            coverage_90 = float(np.clip(rng.normal(0.91, 0.07), 0.55, 0.99))
            max_rhat = float(np.clip(rng.normal(1.02, 0.025), 1.00, 1.12))
            min_ess = float(np.clip(rng.normal(110, 45), 20, 220))
            confidence = confidence_from_metrics(abs_total_pct_error, coverage_90, max_rhat, min_ess)
            recommended_model = "Forecasting model" if model_mae < baseline_mae else "Historical baseline"

            summary_rows.append(
                {
                    "police_force": force,
                    "crime_type": crime,
                    "baseline_mae": baseline_mae,
                    "model_mae": model_mae,
                    "mae_improvement": baseline_mae - model_mae,
                    "abs_total_pct_error": abs_total_pct_error,
                    "coverage_90": coverage_90,
                    "max_rhat": max_rhat,
                    "min_ess": min_ess,
                    "recommended_model": recommended_model,
                    "confidence_level": confidence,
                    "model_note": "Prototype estimate" if confidence != "Use with caution" else "Use with caution",
                }
            )

            for lsoa_idx, lsoa_name in enumerate(lsoa_list):
                lsoa_code = f"DEMO{POLICE_FORCES.index(force)+1:02d}{lsoa_idx+1:03d}"
                lat, lon = jitter_force_point(force, rng, 0.12)
                local_scale = rng.uniform(0.65, 1.45)
                trend = rng.normal(0.02, 0.035)
                seasonal = np.sin(np.linspace(0, 2 * np.pi, len(months), endpoint=False))
                base_level = 10 * force_scale[force] * crime_scale[crime] * local_scale

                for t, month in enumerate(months):
                    expected = max(0.2, base_level * (1 + trend * t) * (1 + 0.12 * seasonal[t]))
                    observed = rng.poisson(expected)
                    predicted = max(0.0, expected * (1 + rng.normal(0, 0.12)))
                    baseline = max(0.0, base_level)
                    spread = max(1.0, predicted * rng.uniform(0.20, 0.42))
                    low = max(0.0, predicted - spread)
                    high = predicted + spread
                    recommended_prediction = predicted if recommended_model == "Forecasting model" else baseline

                    prediction_rows.append(
                        {
                            "police_force": force,
                            "lsoa_code": lsoa_code,
                            "lsoa_name": lsoa_name,
                            "month": month,
                            "crime_type": crime,
                            "observed_count": observed,
                            "predicted_count": predicted,
                            "lower_ci": low,
                            "upper_ci": high,
                            "baseline_predicted_count": baseline,
                            "recommended_prediction": recommended_prediction,
                            "recommended_model": recommended_model,
                            "confidence_level": confidence,
                            "lat": lat,
                            "lon": lon,
                        }
                    )

                    weight = max(0.1, recommended_prediction)
                    n_points = int(np.clip(round(weight / 4), 1, 18))
                    for _ in range(n_points):
                        p_lat, p_lon = lat + rng.normal(0, 0.028), lon + rng.normal(0, 0.038)
                        kde_rows.append(
                            {
                                "police_force": force,
                                "crime_type": crime,
                                "month": month,
                                "lat": p_lat,
                                "lon": p_lon,
                                "weight": rng.uniform(0.7, 1.4) * weight,
                            }
                        )

    predictions = pd.DataFrame(prediction_rows)
    model_summary = pd.DataFrame(summary_rows)
    kde_points = pd.DataFrame(kde_rows)

    force_month = (
        predictions.groupby(["police_force", "month", "crime_type"], as_index=False)
        .agg(
            observed_count=("observed_count", "sum"),
            predicted_count=("predicted_count", "sum"),
            lower_ci=("lower_ci", "sum"),
            upper_ci=("upper_ci", "sum"),
            baseline_predicted_count=("baseline_predicted_count", "sum"),
            recommended_prediction=("recommended_prediction", "sum"),
        )
    )

    lsoa_totals = (
        predictions.groupby(["police_force", "lsoa_code", "lsoa_name", "crime_type", "lat", "lon"], as_index=False)
        .agg(
            observed_count=("observed_count", "sum"),
            predicted_count=("predicted_count", "sum"),
            baseline_predicted_count=("baseline_predicted_count", "sum"),
            recommended_prediction=("recommended_prediction", "sum"),
        )
    )

    funding_rows = []
    for force in POLICE_FORCES:
        total_pred = force_month.loc[force_month["police_force"] == force, "recommended_prediction"].sum()
        funding_index = total_pred * rng.uniform(0.82, 1.22)
        funding_rows.append(
            {
                "police_force": force,
                "current_funding_index": funding_index,
                "estimated_demand_index": total_pred,
                "funding_gap_index": funding_index - total_pred,
            }
        )
    funding = pd.DataFrame(funding_rows)
    funding["funding_alignment"] = np.where(
        funding["funding_gap_index"] > funding["estimated_demand_index"] * 0.08,
        "Potentially over-funded",
        np.where(
            funding["funding_gap_index"] < -funding["estimated_demand_index"] * 0.08,
            "Potentially under-funded",
            "Broadly aligned",
        ),
    )

    deprivation_rows = []
    for force in POLICE_FORCES:
        for decile in range(1, 11):
            demand = (11 - decile) * rng.uniform(80, 140) * force_scale[force]
            deprivation_rows.append(
                {
                    "police_force": force,
                    "imd_decile": decile,
                    "predicted_demand": demand,
                    "context_note": "Placeholder IMD relationship",
                }
            )
    deprivation = pd.DataFrame(deprivation_rows)

    force_total = force_month.groupby("police_force", as_index=False)["recommended_prediction"].sum().rename(columns={"recommended_prediction": "total_recommended_prediction"})
    force_total["demand_pressure"] = pressure_category(force_total["total_recommended_prediction"])
    force_month = force_month.merge(force_total[["police_force", "demand_pressure"]], on="police_force", how="left")
    funding = funding.merge(force_total[["police_force", "demand_pressure"]], on="police_force", how="left")

    return DashboardData(
        predictions=predictions,
        force_month=force_month,
        lsoa_totals=lsoa_totals,
        model_summary=model_summary,
        funding=funding,
        deprivation=deprivation,
        kde_points=kde_points,
        metadata={"source": "demo", "message": "Using placeholder data. Replace with final model, KDE, funding, and IMD outputs when frozen."},
    )


@st.cache_data(show_spinner=False)
def load_real_data():
    missing = [name for name, path in REAL_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(f"{name}: {REAL_FILES[name]}" for name in missing))

    predictions = pd.read_csv(REAL_FILES["predictions"])
    force_month = pd.read_csv(REAL_FILES["force_month"])
    lsoa_totals = pd.read_csv(REAL_FILES["lsoa_totals"])
    model_summary = pd.read_csv(REAL_FILES["summary"])

    rename_map = {
        "force": "police_force",
        "crime": "crime_type",
        "pred": "predicted_count",
        "obs": "observed_count",
        "lo": "lower_ci",
        "hi": "upper_ci",
        "base": "baseline_predicted_count",
        "base_pred": "baseline_predicted_count",
    }
    for df in [predictions, force_month, lsoa_totals, model_summary]:
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    for df in [predictions, force_month]:
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")

    if "recommended_model" not in model_summary.columns:
        if {"baseline_mae", "model_mae"}.issubset(model_summary.columns):
            model_summary["recommended_model"] = np.where(model_summary["model_mae"] < model_summary["baseline_mae"], "Forecasting model", "Historical baseline")
        else:
            model_summary["recommended_model"] = "Forecasting model"

    if "confidence_level" not in model_summary.columns:
        model_summary = safe_numeric(model_summary, ["abs_total_pct_error", "coverage_90", "max_rhat", "min_ess"])
        model_summary["confidence_level"] = model_summary.apply(
            lambda r: confidence_from_metrics(
                float(r.get("abs_total_pct_error", 0.20) if pd.notna(r.get("abs_total_pct_error", np.nan)) else 0.20),
                float(r.get("coverage_90", 0.90) if pd.notna(r.get("coverage_90", np.nan)) else 0.90),
                float(r.get("max_rhat", 1.01) if pd.notna(r.get("max_rhat", np.nan)) else 1.01),
                float(r.get("min_ess", 100) if pd.notna(r.get("min_ess", np.nan)) else 100),
            ),
            axis=1,
        )

    if {"police_force", "crime_type"}.issubset(model_summary.columns):
        pred_meta = model_summary[["police_force", "crime_type", "recommended_model", "confidence_level"]].drop_duplicates()
        predictions = predictions.merge(pred_meta, on=["police_force", "crime_type"], how="left")

    if "recommended_prediction" not in predictions.columns:
        predictions["recommended_prediction"] = np.where(
            predictions.get("recommended_model", "Forecasting model").eq("Historical baseline"),
            predictions.get("baseline_predicted_count", predictions.get("predicted_count", np.nan)),
            predictions.get("predicted_count", np.nan),
        )

    if "recommended_prediction" not in force_month.columns:
        force_month = (
            predictions.groupby(["police_force", "month", "crime_type"], as_index=False)
            .agg(
                observed_count=("observed_count", "sum"),
                predicted_count=("predicted_count", "sum"),
                lower_ci=("lower_ci", "sum"),
                upper_ci=("upper_ci", "sum"),
                baseline_predicted_count=("baseline_predicted_count", "sum"),
                recommended_prediction=("recommended_prediction", "sum"),
            )
        )

    if "lat" not in predictions.columns or "lon" not in predictions.columns:
        rng = np.random.default_rng(101)
        coords = {}
        for key in predictions[["police_force", "lsoa_code"]].drop_duplicates().itertuples(index=False):
            coords[(key.police_force, key.lsoa_code)] = jitter_force_point(key.police_force, rng, 0.15)
        predictions["lat"] = [coords.get((f, l), (52.5, -1.5))[0] for f, l in zip(predictions["police_force"], predictions["lsoa_code"])]
        predictions["lon"] = [coords.get((f, l), (52.5, -1.5))[1] for f, l in zip(predictions["police_force"], predictions["lsoa_code"])]

    if "lat" not in lsoa_totals.columns or "lon" not in lsoa_totals.columns:
        lsoa_coords = predictions.groupby(["police_force", "lsoa_code"], as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"))
        lsoa_totals = lsoa_totals.merge(lsoa_coords, on=["police_force", "lsoa_code"], how="left")

    kde_points = predictions.dropna(subset=["lat", "lon"]).copy()
    kde_points["weight"] = kde_points.get("recommended_prediction", kde_points.get("predicted_count", 1.0))

    force_total = force_month.groupby("police_force", as_index=False)["recommended_prediction"].sum().rename(columns={"recommended_prediction": "estimated_demand_index"})
    force_total["current_funding_index"] = force_total["estimated_demand_index"]
    force_total["funding_gap_index"] = 0.0
    force_total["funding_alignment"] = "Funding data not loaded"
    force_total["demand_pressure"] = pressure_category(force_total["estimated_demand_index"])
    funding = force_total

    deprivation = pd.DataFrame({"police_force": [], "imd_decile": [], "predicted_demand": [], "context_note": []})

    if "demand_pressure" not in force_month.columns:
        force_month = force_month.merge(funding[["police_force", "demand_pressure"]], on="police_force", how="left")

    return DashboardData(predictions, force_month, lsoa_totals, model_summary, funding, deprivation, kde_points, {"source": "real", "message": "Using current model output files from outputs/tables."})


@st.cache_data(show_spinner=False)
def load_dashboard_data(use_demo):
    return generate_demo_data() if use_demo else load_real_data()


def format_number(value, decimals=0):
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}" if decimals == 0 else f"{value:,.{decimals}f}"


def kpi(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_by_sidebar(data):
    force_options = sorted(data.force_month["police_force"].dropna().unique().tolist())
    crime_options = sorted(data.force_month["crime_type"].dropna().unique().tolist())

    with st.sidebar:
        st.title("Controls")
        st.caption("Policy-facing filters for demand estimates.")
        selected_forces = st.multiselect("Police forces", force_options, default=force_options)
        selected_crimes = st.multiselect("Crime types", crime_options, default=crime_options[: min(4, len(crime_options))])
        all_months = pd.to_datetime(data.force_month["month"].dropna()).sort_values().unique()
        if len(all_months) > 0:
            selected_months = st.slider(
                "Time period",
                min_value=pd.Timestamp(all_months.min()).to_pydatetime(),
                max_value=pd.Timestamp(all_months.max()).to_pydatetime(),
                value=(pd.Timestamp(all_months.min()).to_pydatetime(), pd.Timestamp(all_months.max()).to_pydatetime()),
                format="YYYY-MM",
            )
        else:
            selected_months = None

        estimate_mode = st.radio(
            "Estimate shown",
            ["Recommended estimate", "Forecasting model", "Historical baseline"],
            index=0,
        )

        st.divider()
        if data.metadata["source"] == "demo":
            st.warning("Using placeholder data", icon="⚠️")
        else:
            st.success("Using model output files", icon="✅")

    return selected_forces, selected_crimes, selected_months, estimate_mode


def select_estimate_column(mode):
    return {
        "Recommended estimate": "recommended_prediction",
        "Forecasting model": "predicted_count",
        "Historical baseline": "baseline_predicted_count",
    }.get(mode, "recommended_prediction")


def apply_common_filters(df, forces, crimes, months):
    out = df.copy()
    if "police_force" in out.columns and forces:
        out = out[out["police_force"].isin(forces)]
    if "crime_type" in out.columns and crimes:
        out = out[out["crime_type"].isin(crimes)]
    if months is not None and "month" in out.columns:
        start, end = pd.to_datetime(months[0]), pd.to_datetime(months[1])
        out = out[(pd.to_datetime(out["month"]) >= start) & (pd.to_datetime(out["month"]) <= end)]
    return out


def plot_monthly_trend(df, estimate_col, title):
    if df.empty:
        st.info("No data available for the selected filters.")
        return
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            observed=("observed_count", "sum"),
            estimate=(estimate_col, "sum"),
            low=("lower_ci", "sum") if "lower_ci" in df.columns else (estimate_col, "sum"),
            high=("upper_ci", "sum") if "upper_ci" in df.columns else (estimate_col, "sum"),
        )
        .sort_values("month")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["observed"], mode="lines+markers", name="Observed demand", line=dict(color="#ff9b70")))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["estimate"], mode="lines+markers", name="Estimated demand", line=dict(color="#43d17a")))
    fig.add_trace(
        go.Scatter(
            x=pd.concat([monthly["month"], monthly["month"][::-1]]),
            y=pd.concat([monthly["high"], monthly["low"][::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            name="Expected range",
            opacity=0.22,
        )
    )
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Demand count", hovermode="x unified")
    st.plotly_chart(theme_plotly(fig), use_container_width=True)


def render_status_legend():
    st.markdown(
        """
        <div class="small-note">
        <b>Confidence labels:</b><br>
        <span class="status-pill pill-low">High</span> Estimate appears stable in validation.<br>
        <span class="status-pill pill-moderate">Moderate</span> Useful, but interpret with some caution.<br>
        <span class="status-pill pill-caution">Use with caution</span> Estimate should not be used alone.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(data, forces, crimes, months, estimate_mode):
    estimate_col = select_estimate_column(estimate_mode)
    fm = apply_common_filters(data.force_month, forces, crimes, months)
    funding = data.funding[data.funding["police_force"].isin(forces)].copy()

    st.header("National demand overview")
    st.markdown('<div class="section-note">Identify police forces showing the highest estimated demand, demand pressure, and potential funding misalignment.</div>', unsafe_allow_html=True)

    total_observed = fm["observed_count"].sum() if "observed_count" in fm else np.nan
    total_estimated = fm[estimate_col].sum() if estimate_col in fm else np.nan
    highest_force = "—"
    if not fm.empty and estimate_col in fm:
        by_force = fm.groupby("police_force", as_index=False)[estimate_col].sum()
        highest_force = by_force.sort_values(estimate_col, ascending=False).iloc[0]["police_force"]
    caution_count = 0
    if not data.model_summary.empty and "confidence_level" in data.model_summary.columns:
        ms = data.model_summary[data.model_summary["police_force"].isin(forces)]
        caution_count = int((ms["confidence_level"] == "Use with caution").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Observed demand", format_number(total_observed), "Recorded demand in selected period")
    with c2: kpi("Estimated demand", format_number(total_estimated), "Selected estimate type")
    with c3: kpi("Highest-pressure force", highest_force, "Largest selected demand estimate")
    with c4: kpi("Caution flags", format_number(caution_count), "Estimates needing care")

    plot_monthly_trend(fm, estimate_col, "Observed vs estimated demand over time")

    left, right = st.columns([1.1, 1])
    with left:
        if not fm.empty:
            force_rank = fm.groupby("police_force", as_index=False)[estimate_col].sum().sort_values(estimate_col, ascending=False)
            force_rank["demand_pressure"] = pressure_category(force_rank[estimate_col])
            fig = px.bar(
                force_rank,
                x=estimate_col,
                y="police_force",
                color="demand_pressure",
                category_orders={"demand_pressure": PRESSURE_ORDER},
                orientation="h",
                title="Estimated demand by police force",
                labels={estimate_col: "Estimated demand", "police_force": "Police force"},
                color_discrete_map={"Lower": "#43d17a", "Moderate": "#f5c542", "High": "#ff9b70", "Very high": "#ff6b6b"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(theme_plotly(fig), use_container_width=True)
    with right:
        st.subheader("Funding alignment")
        st.caption("Placeholder until current funding data is joined.")
        if not funding.empty:
            st.dataframe(funding[["police_force", "funding_alignment", "demand_pressure"]], use_container_width=True, hide_index=True)
        else:
            st.info("Funding data has not been loaded yet.")


def page_compare_forces(data, forces, crimes, months, estimate_mode):
    estimate_col = select_estimate_column(estimate_mode)
    fm = apply_common_filters(data.force_month, forces, crimes, months)
    ms = data.model_summary[data.model_summary["police_force"].isin(forces)].copy()
    if crimes:
        ms = ms[ms["crime_type"].isin(crimes)]

    st.header("Compare police forces")
    st.markdown('<div class="section-note">Compare estimated demand and demand pressure across territorial police forces.</div>', unsafe_allow_html=True)

    if fm.empty:
        st.info("No data available for selected filters.")
        return

    force_view = fm.groupby("police_force", as_index=False).agg(observed_demand=("observed_count", "sum"), estimated_demand=(estimate_col, "sum"))
    force_view["demand_pressure"] = pressure_category(force_view["estimated_demand"])

    if not ms.empty:
        conf = ms.groupby("police_force", as_index=False).agg(
            avg_model_mae=("model_mae", "mean"),
            avg_baseline_mae=("baseline_mae", "mean"),
            caution_share=("confidence_level", lambda x: (x == "Use with caution").mean()),
        )
        conf["confidence_level"] = np.where(conf["caution_share"] > 0.30, "Use with caution", np.where(conf["caution_share"] > 0, "Moderate", "High"))
        force_view = force_view.merge(conf, on="police_force", how="left")
    else:
        force_view["confidence_level"] = "Moderate"

    fig = px.bar(
        force_view.sort_values("estimated_demand", ascending=False),
        x="police_force",
        y="estimated_demand",
        color="demand_pressure",
        category_orders={"demand_pressure": PRESSURE_ORDER},
        title="Estimated demand by force",
        labels={"estimated_demand": "Estimated demand", "police_force": "Police force"},
        color_discrete_map={"Lower": "#43d17a", "Moderate": "#f5c542", "High": "#ff9b70", "Very high": "#ff6b6b"},
    )
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(theme_plotly(fig), use_container_width=True)

    table = force_view.sort_values("estimated_demand", ascending=False).copy()
    table["observed_demand"] = table["observed_demand"].round(0).astype(int)
    table["estimated_demand"] = table["estimated_demand"].round(0).astype(int)
    st.dataframe(table[["police_force", "observed_demand", "estimated_demand", "demand_pressure", "confidence_level"]], use_container_width=True, hide_index=True)
    render_status_legend()


def page_explore_force(data, forces, crimes, months, estimate_mode):
    estimate_col = select_estimate_column(estimate_mode)
    selected_force = st.selectbox("Select one force for detailed exploration", forces or POLICE_FORCES)
    fm = apply_common_filters(data.force_month, [selected_force], crimes, months)
    preds = apply_common_filters(data.predictions, [selected_force], crimes, months)
    lsoa = data.lsoa_totals[data.lsoa_totals["police_force"] == selected_force].copy()
    if crimes and "crime_type" in lsoa:
        lsoa = lsoa[lsoa["crime_type"].isin(crimes)]

    st.header(f"Explore a force: {selected_force}")
    st.markdown('<div class="section-note">Inspect what drives estimated demand in the selected force: trends, crime-type composition, and high-demand local areas.</div>', unsafe_allow_html=True)

    plot_monthly_trend(fm, estimate_col, f"Demand trend for {selected_force}")

    left, right = st.columns(2)
    with left:
        if not fm.empty:
            crime_comp = fm.groupby("crime_type", as_index=False)[estimate_col].sum().sort_values(estimate_col, ascending=False)
            fig = px.pie(crime_comp, names="crime_type", values=estimate_col, title="Selected crime-type contribution", hole=0.42)
            st.plotly_chart(theme_plotly(fig), use_container_width=True)
    with right:
        if not lsoa.empty:
            value_col = "recommended_prediction" if "recommended_prediction" in lsoa.columns else "predicted_count"
            lsoa_rank = lsoa.groupby(["lsoa_code", "lsoa_name"], as_index=False).agg(estimated_demand=(value_col, "sum")).sort_values("estimated_demand", ascending=False).head(10)
            fig = px.bar(
                lsoa_rank,
                x="estimated_demand",
                y="lsoa_name",
                orientation="h",
                title="Top local areas by estimated demand",
                labels={"estimated_demand": "Estimated demand", "lsoa_name": "Local area"},
                color_discrete_sequence=["#ff6a3d"],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(theme_plotly(fig), use_container_width=True)

    with st.expander("Show local-area table"):
        cols = ["police_force", "lsoa_code", "lsoa_name", "month", "crime_type", "observed_count", "predicted_count", "baseline_predicted_count", "recommended_prediction", "confidence_level"]
        available = [c for c in cols if c in preds.columns]
        st.dataframe(preds[available].head(1000), use_container_width=True, hide_index=True)


def page_hotspot_map(data, forces, crimes, months, estimate_mode):
    st.header("Hotspot map")
    st.markdown('<div class="section-note">Prototype spatial view for KDE-style hotspot exploration. Current map uses placeholder/demo points unless final KDE outputs are loaded.</div>', unsafe_allow_html=True)

    points = apply_common_filters(data.kde_points, forces, crimes, months)
    lsoa = apply_common_filters(data.lsoa_totals, forces, crimes, None)

    if points.empty:
        st.info("No hotspot points available for the selected filters.")
        return

    mode = st.radio("Map layer", ["KDE-style heatmap", "Local demand points"], horizontal=True)

    if mode == "KDE-style heatmap":
        fig = px.density_mapbox(
            points,
            lat="lat",
            lon="lon",
            z="weight",
            radius=26,
            center=dict(lat=52.8, lon=-1.6),
            zoom=5,
            mapbox_style="carto-darkmatter",
            hover_data=["police_force", "crime_type"],
            title="Prototype KDE-style demand intensity",
        )
    else:
        if lsoa.empty or "lat" not in lsoa.columns or "lon" not in lsoa.columns:
            st.info("Local-area point coordinates are not available.")
            return
        value_col = "recommended_prediction" if "recommended_prediction" in lsoa.columns else "predicted_count"
        plot_df = lsoa.groupby(["police_force", "lsoa_code", "lsoa_name", "lat", "lon"], as_index=False)[value_col].sum()
        fig = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            size=value_col,
            color=value_col,
            color_continuous_scale="Oranges",
            center=dict(lat=52.8, lon=-1.6),
            zoom=5,
            mapbox_style="carto-darkmatter",
            hover_name="lsoa_name",
            hover_data=["police_force", "lsoa_code"],
            title="Local areas by estimated demand",
        )

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), margin=dict(l=0, r=0, t=45, b=0), height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("The final KDE layer should be replaced with the team’s standardised KDE output once available.")


def page_demand_context(data, forces, crimes, months, estimate_mode):
    st.header("Demand context")
    st.markdown('<div class="section-note">Reserved for sociological and spatial context. The final version can combine IMD/deprivation analysis and KDE hotspot outputs.</div>', unsafe_allow_html=True)

    st.subheader("Deprivation context")
    if data.deprivation.empty:
        st.info("IMD/deprivation data has not been loaded yet.")
    else:
        dep = data.deprivation[data.deprivation["police_force"].isin(forces)].copy()
        fig = px.line(
            dep.groupby("imd_decile", as_index=False)["predicted_demand"].sum(),
            x="imd_decile",
            y="predicted_demand",
            markers=True,
            title="Placeholder: estimated demand by deprivation decile",
            labels={"imd_decile": "IMD decile (1 = most deprived)", "predicted_demand": "Estimated demand"},
            color_discrete_sequence=["#ff6a3d"],
        )
        st.plotly_chart(theme_plotly(fig), use_container_width=True)
        st.caption("Interpretation note: this page is descriptive and should not be presented as causal evidence without further analysis.")


def page_method_notes(data):
    st.header("Method notes")
    st.markdown('<div class="section-note">Plain-English transparency for policy users. Full technical validation belongs in the written report.</div>', unsafe_allow_html=True)

    st.subheader("What the dashboard estimates")
    st.write("The dashboard estimates police demand from recorded crime data. The current prototype works at local-area month level and aggregates to police-force level for policy comparison.")

    st.subheader("How to interpret estimates")
    st.markdown(
        """
        - **Predicted demand** estimates expected recorded crime demand for the selected force, crime type, and period.
        - **Expected range** communicates uncertainty around the estimate.
        - **Demand pressure** highlights forces or local areas with relatively high estimated demand.
        - **Use with caution** means the estimate should be supported by additional evidence.
        """
    )

    st.subheader("Current prototype assumptions")
    st.markdown(
        """
        - Current frontend data can be placeholder data until final model outputs are frozen.
        - The current prediction prototype uses nearby-area historical patterns, not true polygon adjacency.
        - Population offset was tested but is not used as the primary raw-demand estimate.
        - IMD/deprivation and KDE hotspot layers are planned as contextual layers.
        - COVID handling should be agreed by the team before being added to the final modelling pipeline.
        """
    )

    with st.expander("Technical details for analysts"):
        st.markdown("The current modelling prototype is a Bayesian Negative Binomial count model with LSOA-level effects, month effects, time trend, and a centroid-distance spatial-lag feature.")
        if not data.model_summary.empty:
            st.dataframe(data.model_summary.head(100), use_container_width=True, hide_index=True)


def main():
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Police Demand & Resource Allocation Dashboard</div>
            <div class="hero-text">
            A decision-support prototype for the Home Office to explore estimated police demand across territorial police forces, identify pressure points, and support evidence-based resource allocation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        data = load_dashboard_data(USE_DEMO_DATA)
    except Exception as exc:
        st.error("Could not load real dashboard files. Falling back to demo data.")
        st.exception(exc)
        data = generate_demo_data()

    forces, crimes, months, estimate_mode = filter_by_sidebar(data)
    st.caption(data.metadata.get("message", ""))

    tabs = st.tabs(["Overview", "Compare Forces", "Explore a Force", "Hotspot Map", "Demand Context", "Method Notes"])

    with tabs[0]:
        page_overview(data, forces, crimes, months, estimate_mode)
    with tabs[1]:
        page_compare_forces(data, forces, crimes, months, estimate_mode)
    with tabs[2]:
        page_explore_force(data, forces, crimes, months, estimate_mode)
    with tabs[3]:
        page_hotspot_map(data, forces, crimes, months, estimate_mode)
    with tabs[4]:
        page_demand_context(data, forces, crimes, months, estimate_mode)
    with tabs[5]:
        page_method_notes(data)


if __name__ == "__main__":
    main()
