"""
Cached data/model loaders shared by every page.

Paths are relative to streamlit_app/, expecting the folder layout from the
phase-1 zip:

    deliverable/
        data/processed/full_engineered.pkl, weather_cache.csv, month=*.csv.gz
        models/demand_model.pkl, demand_model_features.json
        streamlit_app/   <- this file lives here
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent  # deliverable/
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

# Same dtypes 01_build_pipeline.py uses when writing the engineered data.
# Needed here too: without them, pandas' default dtypes (object for every
# string column, float64/int64 for numbers) blow up memory ~2.5x for a
# dataset this size -- confirmed to peak at ~1.9GB without this, which is
# almost certainly what was causing 503/connection-reset errors on
# Streamlit Community Cloud's free tier. With dtypes applied, the same
# data loads at ~700MB.
_ENGINEERED_DTYPES = {
    "FEEDER_NAME": "category", "ADDRESS": "category", "LATITUDE": "float32",
    "LONGITUDE": "float32", "Maintenance Status": "category", "BAND": "float32",
    "ACTUAL_HOURS": "float32", "SHORTFALL": "float32", "AVAILABILITY_PERCENTAGE": "float32",
    "SHORTFALL_PERCENTAGE": "float32", "MONTH": "category", "YEAR": "int16",
    "DAY_OF_WEEK": "category", "WEEKEND": "int8", "DAY_OF_YEAR": "int16",
    "WX_TEMP_MAX": "float32", "WX_TEMP_MIN": "float32", "WX_TEMP_AVG": "float32",
    "WX_HUMIDITY": "float32", "WX_CLOUD_COVER": "float32", "WX_RAINFALL_MM": "float32",
    "WX_WIND_KMH": "float32", "WX_SOLAR_KWH_M2": "float32", "ROLL_MEAN_7D": "float32",
    "ROLL_MEAN_14D": "float32", "ROLL_MEAN_30D": "float32", "VOLATILITY_7D": "float32",
    "LAG_1D": "float32", "LAG_2D": "float32", "LAG_3D": "float32", "LAG_7D": "float32",
    "LAG_14D": "float32", "LAG_30D": "float32", "TREND_30D": "float32",
    "CUSTOMER_DENSITY": "int32", "DOW_NUM": "int8", "MONTH_NUM": "int8", "SEASON": "category",
}


@st.cache_data(show_spinner="Loading historical data...")
def load_engineered_data() -> pd.DataFrame:
    pkl_path = DATA_DIR / "full_engineered.pkl"
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    # Fallback: reassemble from the gzip-CSV month partitions if the pickle
    # wasn't shipped (e.g. you regenerated data without re-running with the
    # pickle writer, or copied only the CSVs to save space).
    parts = sorted(DATA_DIR.glob("month=*.csv.gz"))
    if not parts:
        raise FileNotFoundError(
            f"No data found in {DATA_DIR}. Run src/01_build_pipeline.py first."
        )
    # Only pass dtypes for columns actually present (older/regenerated CSVs
    # might not have every column) so this doesn't hard-fail on a mismatch.
    sample_cols = set(pd.read_csv(parts[0], nrows=0).columns)
    dtypes = {k: v for k, v in _ENGINEERED_DTYPES.items() if k in sample_cols}
    return pd.concat(
        [pd.read_csv(p, parse_dates=["DATE"], dtype=dtypes) for p in parts],
        ignore_index=True,
    )


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = MODEL_DIR / "demand_model.pkl"
    if not model_path.exists():
        # No terminal access on platforms like Streamlit Community Cloud, so
        # train it here on first load rather than erroring out. Takes ~2-3
        # min the first time; cached (via @st.cache_resource) after that.
        with st.spinner("First-time setup: training the forecasting model (2-3 minutes)..."):
            status_box = st.empty()
            from utils.model_training import train_demand_model
            train_demand_model(
                data_dir=str(DATA_DIR),
                model_dir=str(MODEL_DIR),
                progress_callback=lambda msg: status_box.text(msg),
            )
            status_box.empty()
    model = joblib.load(model_path)
    with open(MODEL_DIR / "demand_model_features.json") as f:
        meta = json.load(f)
    return model, meta["features"]


@st.cache_data(show_spinner=False)
def load_weather_cache() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "weather_cache.csv", parse_dates=["DATE"])


@st.cache_data(show_spinner=False)
def address_feeder_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (FEEDER_NAME, ADDRESS) with static attributes -- used for
    search/autosuggest and the network map, without needing the full
    2.7M-row table in memory for those views."""
    cols = ["FEEDER_NAME", "ADDRESS", "LATITUDE", "LONGITUDE",
            "Maintenance Status", "BAND", "CUSTOMER_DENSITY"]
    lookup = (
        df[df["ADDRESS"] != "NO_ADDRESS_MATCH"][cols]
        .drop_duplicates(subset=["FEEDER_NAME", "ADDRESS"])
        .reset_index(drop=True)
    )
    return lookup


@st.cache_data(show_spinner=False)
def feeder_reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feeder average availability -- used for the network map color
    coding and the 'top areas needing solar' home page insight."""
    summary = (
        df[df["ADDRESS"] != "NO_ADDRESS_MATCH"]
        .groupby("FEEDER_NAME", observed=True)
        .agg(
            avg_actual_hours=("ACTUAL_HOURS", "mean"),
            avg_availability_pct=("AVAILABILITY_PERCENTAGE", "mean"),
            avg_shortfall=("SHORTFALL", "mean"),
            band=("BAND", "first"),
            latitude=("LATITUDE", "mean"),
            longitude=("LONGITUDE", "mean"),
            n_addresses=("ADDRESS", "nunique"),
        )
        .reset_index()
        .dropna(subset=["latitude", "longitude"])
    )
    return summary


def reliability_color(avg_actual_hours: float) -> str:
    """Green/yellow/orange/red banding for map markers and badges.

    Deliberately based on absolute hours of supply out of a 24h day, NOT on
    AVAILABILITY_PERCENTAGE (which is relative to each feeder's BAND). A
    feeder on a low band (e.g. Band D, 8h max) hitting 100% of its band is
    still only delivering 8h/day -- that should read as poor, not green.
    """
    if pd.isna(avg_actual_hours):
        return "gray"
    if avg_actual_hours >= 18:
        return "green"
    if avg_actual_hours >= 12:
        return "yellow"
    if avg_actual_hours >= 6:
        return "orange"
    return "red"
