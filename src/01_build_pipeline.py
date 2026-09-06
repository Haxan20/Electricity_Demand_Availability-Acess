"""
Data pipeline: load raw daily availability data, attach synthetic weather,
engineer ML features, and write partitioned output.

NOTE ON ENVIRONMENT: this box has no internet access and no pyarrow/fastparquet,
so output is written as gzip-compressed CSV partitioned by month instead of
Parquet. In your real environment (where pyarrow is installed), convert with:

    import pandas as pd, glob
    for f in glob.glob("../data/processed/month=*.csv.gz"):
        pd.read_csv(f).to_parquet(f.replace(".csv.gz", ".parquet"), partition_cols=None)

or just re-run this script after `pip install pyarrow` and swap the writer
(swap_to_parquet() below shows the one-line change).
"""
import numpy as np
import pandas as pd
import os

RAW_PATH = "../PREPARED_DATA.csv"  # place your raw CSV in deliverable/ alongside src/, data/, models/
OUT_DIR = "../data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

FLOAT_COLS = ["LATITUDE", "LONGITUDE", "BAND", "ACTUAL_HOURS", "SHORTFALL",
              "AVAILABILITY_PERCENTAGE", "SHORTFALL_PERCENTAGE"]
CAT_COLS = ["FEEDER_NAME", "ADDRESS", "Maintenance Status", "MONTH", "DAY_OF_WEEK"]

print("Loading raw data...")
# 12 known feeders have no DT-List match, so LATITUDE/LONGITUDE/BAND/etc are
# blank for them -- read those columns as plain floats (NaN-tolerant) rather
# than forcing dtype at parse time, then downcast after.
df = pd.read_csv(RAW_PATH, parse_dates=["DATE"])
for c in FLOAT_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
for c in CAT_COLS:
    df[c] = df[c].astype("category")
df["YEAR"] = df["YEAR"].astype("int16")
df["WEEKEND"] = df["WEEKEND"].astype("int8")
df["DAY_OF_YEAR"] = df["DAY_OF_YEAR"].astype("int16")
print(f"Loaded {len(df):,} rows, {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
n_no_address = df["ADDRESS"].isna().sum()
print(f"  ({n_no_address:,} rows have no ADDRESS/LATITUDE/LONGITUDE -- feeders with no DT List match)")
# Give these a placeholder so groupby doesn't silently drop them as NaN groups
if "NO_ADDRESS_MATCH" not in df["ADDRESS"].cat.categories:
    df["ADDRESS"] = df["ADDRESS"].cat.add_categories(["NO_ADDRESS_MATCH"])
df["ADDRESS"] = df["ADDRESS"].fillna("NO_ADDRESS_MATCH")

# ---------------------------------------------------------------------------
# Synthetic weather (Lagos climate profile). Cached per unique DATE, with a
# small deterministic spatial jitter so nearby feeders aren't bit-identical.
# Replace generate_weather() with a real OpenWeatherMap/NOAA call when you
# have API access -- same output schema (WX_* columns) so downstream code
# doesn't change.
# ---------------------------------------------------------------------------
def generate_weather(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    doy = dates.dayofyear.values
    # Lagos: rainy season ~Apr-Oct (peak Jun/Sep), dry/harmattan Nov-Mar
    rainy_weight = np.clip(np.sin((doy - 60) / 365 * 2 * np.pi), -1, 1)
    rainy_weight = (rainy_weight + 1) / 2  # 0..1, higher = more rainy-season-like

    temp_max = 33 - 4 * rainy_weight + rng.normal(0, 1.2, len(dates))
    temp_min = 24 - 2 * rainy_weight + rng.normal(0, 1.0, len(dates))
    temp_avg = (temp_max + temp_min) / 2
    humidity = 55 + 30 * rainy_weight + rng.normal(0, 5, len(dates))
    humidity = np.clip(humidity, 30, 98)
    cloud_cover = 30 + 55 * rainy_weight + rng.normal(0, 8, len(dates))
    cloud_cover = np.clip(cloud_cover, 5, 100)
    rain_prob = rainy_weight
    rainfall = np.where(
        rng.random(len(dates)) < rain_prob * 0.6,
        rng.gamma(2.0, 8.0 * (0.3 + rainy_weight), len(dates)),
        0.0,
    )
    wind_speed = 8 + 4 * (1 - rainy_weight) + rng.normal(0, 1.5, len(dates))
    wind_speed = np.clip(wind_speed, 2, 30)
    # Solar irradiation drops with cloud cover
    solar_irradiation = np.clip(6.2 - 0.045 * cloud_cover + rng.normal(0, 0.2, len(dates)), 1.0, 6.5)

    return pd.DataFrame({
        "DATE": dates,
        "WX_TEMP_MAX": temp_max.astype("float32"),
        "WX_TEMP_MIN": temp_min.astype("float32"),
        "WX_TEMP_AVG": temp_avg.astype("float32"),
        "WX_HUMIDITY": humidity.astype("float32"),
        "WX_CLOUD_COVER": cloud_cover.astype("float32"),
        "WX_RAINFALL_MM": rainfall.astype("float32"),
        "WX_WIND_KMH": wind_speed.astype("float32"),
        "WX_SOLAR_KWH_M2": solar_irradiation.astype("float32"),
    })

print("Generating synthetic weather (Lagos profile)...")
unique_dates = pd.DatetimeIndex(sorted(df["DATE"].unique()))
weather = generate_weather(unique_dates)
weather.to_csv(f"{OUT_DIR}/weather_cache.csv", index=False)
print(f"Weather cached for {len(weather)} unique dates -> {OUT_DIR}/weather_cache.csv")

df = df.merge(weather, on="DATE", how="left")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
print("Sorting by FEEDER_NAME/ADDRESS/DATE...")
df = df.sort_values(["FEEDER_NAME", "ADDRESS", "DATE"]).reset_index(drop=True)

grp_key = ["FEEDER_NAME", "ADDRESS"]
g = df.groupby(grp_key, observed=True)["ACTUAL_HOURS"]

print("Rolling statistics (7/14/30-day)...")
for w in (7, 14, 30):
    df[f"ROLL_MEAN_{w}D"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

print("Volatility (7-day std)...")
df["VOLATILITY_7D"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())

print("Lag features (1,2,3,7,14,30 days)...")
for lag in (1, 2, 3, 7, 14, 30):
    df[f"LAG_{lag}D"] = g.transform(lambda s, lag=lag: s.shift(lag))

print("Trend (slope of ACTUAL_HOURS over trailing 30 days)...")
def rolling_slope(s: pd.Series, window: int = 30) -> pd.Series:
    y = s.shift(1).values.astype("float64")
    n = len(y)
    out = np.full(n, np.nan)
    x = np.arange(window, dtype="float64")
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(window, n + 1):
        window_vals = y[i - window:i]
        if np.isnan(window_vals).sum() > window * 0.5:
            continue
        yv = np.nan_to_num(window_vals, nan=np.nanmean(window_vals))
        y_mean = yv.mean()
        cov = ((x - x_mean) * (yv - y_mean)).sum()
        out[i - 1] = cov / x_var if x_var else 0.0
    return pd.Series(out, index=s.index)

df["TREND_30D"] = df.groupby(grp_key, observed=True)["ACTUAL_HOURS"].transform(rolling_slope)

print("Ratio / demand features...")
df["DEMAND_RATIO"] = (df["ACTUAL_HOURS"] / df["BAND"]).astype("float32")
df["SHORTFALL_RATIO"] = (df["SHORTFALL"] / df["BAND"]).astype("float32")

print("Customer density (addresses per feeder)...")
density = df.groupby("FEEDER_NAME", observed=True)["ADDRESS"].transform("nunique")
df["CUSTOMER_DENSITY"] = density.astype("int32")

print("Calendar features...")
df["DOW_NUM"] = df["DATE"].dt.dayofweek.astype("int8")  # Monday=0
df["MONTH_NUM"] = df["DATE"].dt.month.astype("int8")
df["SEASON"] = np.where(df["MONTH_NUM"].isin([11, 12, 1, 2, 3]), "Dry", "Rainy")

for c in ["ROLL_MEAN_7D", "ROLL_MEAN_14D", "ROLL_MEAN_30D", "VOLATILITY_7D",
          "LAG_1D", "LAG_2D", "LAG_3D", "LAG_7D", "LAG_14D", "LAG_30D", "TREND_30D"]:
    df[c] = df[c].astype("float32")

print(f"Final shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

# ---------------------------------------------------------------------------
# Write partitioned output (by MONTH). Swap to parquet with one line once
# pyarrow is installed: df_month.to_parquet(path, index=False)
# ---------------------------------------------------------------------------
for month, df_month in df.groupby("MONTH", observed=True):
    path = f"{OUT_DIR}/month={month}.csv.gz"
    df_month.to_csv(path, index=False, compression="gzip")
    print(f"  wrote {path}  ({len(df_month):,} rows)")

df.to_pickle(f"{OUT_DIR}/full_engineered.pkl")
print(f"Also wrote full engineered dataset -> {OUT_DIR}/full_engineered.pkl (for local model training)")

schema = {c: str(t) for c, t in df.dtypes.items()}
import json
with open(f"{OUT_DIR}/schema.json", "w") as f:
    json.dump(schema, f, indent=2)
print("Schema written to schema.json")
