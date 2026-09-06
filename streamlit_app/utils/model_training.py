"""
Core model-training logic, factored out so it can be called both from the
CLI script (src/02_train_demand_model.py) and automatically by the
Streamlit app on first load if no trained model exists yet (needed for
platforms like Streamlit Community Cloud, which give you no terminal to
run a setup script manually).
"""
import glob
import json
import os
import time

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES_BASE = [
    "BAND", "LATITUDE", "LONGITUDE",
    "WX_TEMP_MAX", "WX_TEMP_MIN", "WX_TEMP_AVG", "WX_HUMIDITY",
    "WX_CLOUD_COVER", "WX_RAINFALL_MM", "WX_WIND_KMH", "WX_SOLAR_KWH_M2",
    "ROLL_MEAN_7D", "ROLL_MEAN_14D", "ROLL_MEAN_30D", "VOLATILITY_7D",
    "LAG_1D", "LAG_2D", "LAG_3D", "LAG_7D", "LAG_14D", "LAG_30D",
    "TREND_30D", "CUSTOMER_DENSITY", "DOW_NUM", "MONTH_NUM", "WEEKEND",
]
TARGET = "ACTUAL_HOURS"


def load_engineered_dataframe(data_dir: str) -> pd.DataFrame:
    pkl_path = os.path.join(data_dir, "full_engineered.pkl")
    if os.path.exists(pkl_path):
        return pd.read_pickle(pkl_path)
    parts = sorted(glob.glob(os.path.join(data_dir, "month=*.csv.gz")))
    if not parts:
        raise FileNotFoundError(
            f"No engineered data found in {data_dir}. Run 01_build_pipeline.py first."
        )
    return pd.concat([pd.read_csv(p, parse_dates=["DATE"]) for p in parts], ignore_index=True)


def train_demand_model(data_dir: str, model_dir: str, progress_callback=None) -> dict:
    """
    Trains the demand model and writes demand_model.pkl + metadata JSON
    files into model_dir. Returns the metrics dict.

    progress_callback, if given, is called with short status strings --
    used by the Streamlit app to show a spinner/status message during
    first-time setup. Safe to leave as None for CLI use (falls back to print).
    """
    def status(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    try:
        import xgboost as xgb
        backend = "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        backend = "sklearn_hgb"

    status(f"Model backend: {backend}")
    os.makedirs(model_dir, exist_ok=True)

    status("Loading engineered dataset...")
    df = load_engineered_dataframe(data_dir)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    df = pd.get_dummies(df, columns=["SEASON"], prefix="SEASON")
    season_cols = [c for c in df.columns if c.startswith("SEASON_")]
    features = FEATURES_BASE + season_cols

    X = df[features]
    y = df[TARGET]

    train_mask = df["MONTH_NUM"] <= 6
    val_mask = df["MONTH_NUM"] == 7
    test_mask = df["MONTH_NUM"] == 8

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    status(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    t0 = time.time()
    if backend == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=-1, early_stopping_rounds=30,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=400, max_depth=6, learning_rate=0.05,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=None, random_state=42,
        )
        model.fit(X_train, y_train)

    status(f"Training took {time.time()-t0:.1f}s")

    def evaluate(name, X_, y_):
        if X_.empty:
            return None
        pred = model.predict(X_)
        mae = mean_absolute_error(y_, pred)
        rmse = mean_squared_error(y_, pred) ** 0.5
        r2 = r2_score(y_, pred)
        status(f"  {name}: MAE={mae:.3f}h  RMSE={rmse:.3f}h  R2={r2:.4f}")
        return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(X_)}

    status("Evaluating...")
    metrics = {
        "backend": backend,
        "train": evaluate("Train", X_train, y_train),
        "val": evaluate("Val (Jul)", X_val, y_val),
        "test": evaluate("Test (Aug)", X_test, y_test),
    }

    if backend == "xgboost":
        importances = dict(zip(features, model.feature_importances_.tolist()))
    else:
        from sklearn.inspection import permutation_importance
        sample = X_val.sample(min(20000, len(X_val)), random_state=42) if len(X_val) else X_train.sample(20000, random_state=42)
        sample_y = y.loc[sample.index]
        perm = permutation_importance(model, sample, sample_y, n_repeats=3, random_state=42, n_jobs=-1)
        importances = dict(zip(features, perm.importances_mean.tolist()))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    joblib.dump(model, os.path.join(model_dir, "demand_model.pkl"))
    with open(os.path.join(model_dir, "demand_model_features.json"), "w") as f:
        json.dump({"features": features, "target": TARGET, "backend": backend}, f, indent=2)
    with open(os.path.join(model_dir, "demand_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(model_dir, "demand_model_importance.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    status(f"Model saved -> {model_dir}/demand_model.pkl")
    return metrics
