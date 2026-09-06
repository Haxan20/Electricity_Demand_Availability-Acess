"""
Recursive N-day-ahead forecasting for a single FEEDER_NAME + ADDRESS.
Same logic as src/03_forecast.py, adapted to take pre-loaded data/model
(via data_loader's cached loaders) instead of reading from disk each call.
"""
import numpy as np
import pandas as pd


def _weather_for_date(weather: pd.DataFrame, d: pd.Timestamp) -> dict:
    row = weather[weather["DATE"] == d]
    if not row.empty:
        return row.iloc[0].to_dict()
    doy = d.dayofyear
    tmp = weather.copy()
    tmp["doy"] = tmp["DATE"].dt.dayofyear
    near = tmp.iloc[(tmp["doy"] - doy).abs().argsort()[:7]]
    return near.drop(columns=["doy", "DATE"]).mean().to_dict()


def forecast_n_days(df: pd.DataFrame, weather: pd.DataFrame, model, features: list,
                     feeder_name: str, address: str, n_days: int = 7,
                     as_of: pd.Timestamp = None) -> pd.DataFrame:
    hist = df[(df["FEEDER_NAME"] == feeder_name) & (df["ADDRESS"] == address)].sort_values("DATE")
    if hist.empty:
        raise ValueError(f"No history found for feeder={feeder_name!r} address={address!r}")

    if as_of is None:
        as_of = hist["DATE"].max()
    hist = hist[hist["DATE"] <= as_of].copy()

    band = hist["BAND"].iloc[-1]
    lat = hist["LATITUDE"].iloc[-1]
    lon = hist["LONGITUDE"].iloc[-1]
    density = hist["CUSTOMER_DENSITY"].iloc[-1]

    series = hist.set_index("DATE")["ACTUAL_HOURS"].copy()

    results = []
    for step in range(1, n_days + 1):
        target_date = as_of + pd.Timedelta(days=step)
        wx = _weather_for_date(weather, target_date)

        lag_1 = series.iloc[-1] if len(series) >= 1 else np.nan
        lag_2 = series.iloc[-2] if len(series) >= 2 else np.nan
        lag_3 = series.iloc[-3] if len(series) >= 3 else np.nan
        lag_7 = series.iloc[-7] if len(series) >= 7 else np.nan
        lag_14 = series.iloc[-14] if len(series) >= 14 else np.nan
        lag_30 = series.iloc[-30] if len(series) >= 30 else np.nan
        roll_7 = series.iloc[-7:].mean() if len(series) >= 1 else np.nan
        roll_14 = series.iloc[-14:].mean() if len(series) >= 1 else np.nan
        roll_30 = series.iloc[-30:].mean() if len(series) >= 1 else np.nan
        vol_7 = series.iloc[-7:].std() if len(series) >= 2 else np.nan
        if len(series) >= 30:
            y = series.iloc[-30:].values.astype("float64")
            x = np.arange(30, dtype="float64")
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = np.nan

        row = {
            "BAND": band, "LATITUDE": lat, "LONGITUDE": lon,
            "WX_TEMP_MAX": wx["WX_TEMP_MAX"], "WX_TEMP_MIN": wx["WX_TEMP_MIN"],
            "WX_TEMP_AVG": wx["WX_TEMP_AVG"], "WX_HUMIDITY": wx["WX_HUMIDITY"],
            "WX_CLOUD_COVER": wx["WX_CLOUD_COVER"], "WX_RAINFALL_MM": wx["WX_RAINFALL_MM"],
            "WX_WIND_KMH": wx["WX_WIND_KMH"], "WX_SOLAR_KWH_M2": wx["WX_SOLAR_KWH_M2"],
            "ROLL_MEAN_7D": roll_7, "ROLL_MEAN_14D": roll_14, "ROLL_MEAN_30D": roll_30,
            "VOLATILITY_7D": vol_7, "LAG_1D": lag_1, "LAG_2D": lag_2, "LAG_3D": lag_3,
            "LAG_7D": lag_7, "LAG_14D": lag_14, "LAG_30D": lag_30, "TREND_30D": slope,
            "CUSTOMER_DENSITY": density, "DOW_NUM": target_date.dayofweek,
            "MONTH_NUM": target_date.month, "WEEKEND": int(target_date.dayofweek >= 5),
            "SEASON_Dry": 1 if target_date.month in (11, 12, 1, 2, 3) else 0,
            "SEASON_Rainy": 1 if target_date.month not in (11, 12, 1, 2, 3) else 0,
        }
        X_row = pd.DataFrame([row])[features]
        pred = float(model.predict(X_row)[0])
        pred = max(0.0, min(24.0, pred))

        series.loc[target_date] = pred

        shortfall = max(0.0, band - pred) if pd.notna(band) else None
        avail_pct = (pred / band * 100) if pd.notna(band) and band else None

        results.append({
            "DATE": target_date.date(),
            "PREDICTED_ACTUAL_HOURS": round(pred, 2),
            "BAND": band,
            "PREDICTED_SHORTFALL": round(shortfall, 2) if shortfall is not None else None,
            "PREDICTED_AVAILABILITY_PCT": round(avail_pct, 2) if avail_pct is not None else None,
        })

    return pd.DataFrame(results)


def recent_history(df: pd.DataFrame, feeder_name: str, address: str, n_days: int = 30) -> pd.DataFrame:
    hist = df[(df["FEEDER_NAME"] == feeder_name) & (df["ADDRESS"] == address)].sort_values("DATE")
    cols = ["DATE", "ACTUAL_HOURS", "SHORTFALL", "AVAILABILITY_PERCENTAGE", "BAND"]
    return hist[cols].tail(n_days).reset_index(drop=True)
