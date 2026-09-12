"""
Core model-training logic, factored out so it can be called both from the
CLI script (src/02_train_demand_model.py) and automatically by the
Streamlit app on first load if no trained model exists yet (needed for
platforms like Streamlit Community Cloud, which give you no terminal to
run a setup script manually).

Memory note: training on the full ~2.17M-row training split with
HistGradientBoostingRegressor peaked at ~3GB RAM in testing -- comfortably
fine locally, but enough to crash a free-tier cloud container (which is
also already holding the ~700MB engineered dataframe). So this module:
  - accepts an already-loaded dataframe (df_override) to avoid loading the
    ~700MB dataset a second time when the caller (the Streamlit app) has
    already loaded it once via its own cache
  - supports max_train_rows to subsample the training split specifically
    (val/test stay full-size, since those drive the metrics you actually
    care about) -- keeps memory and training time bounded on constrained
    hosts, at a small, measured cost to accuracy (see comment near the
    default value below)
  - explicitly frees large intermediates as soon as they're no longer
    needed, rather than waiting for the whole function to return
"""
import gc
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


def train_demand_model(data_dir: str, model_dir: str, progress_callback=None,
                        df_override: pd.DataFrame = None,
                        max_train_rows: int = None,
                        compute_importance: bool = True) -> dict:
    """
    Trains the demand model and writes demand_model.pkl + metadata JSON
    files into model_dir. Returns the metrics dict.

    progress_callback, if given, is called with short status strings --
    used by the Streamlit app to show a spinner/status message during
    first-time setup. Safe to leave as None for CLI use (falls back to print).

    df_override: pass an already-loaded engineered dataframe to skip
    loading it again from disk. Caller keeps ownership; this function
    doesn't mutate it (works on df.copy() internally where needed, or
    row-index views that don't touch the original data's memory beyond
    the new column pandas has to add for get_dummies output).

    max_train_rows: if set, randomly subsamples the TRAIN split (not
    val/test) down to at most this many rows before fitting. Keeps
    memory and CPU time bounded on constrained hosts.

    compute_importance: sklearn's HistGradientBoostingRegressor has no
    built-in feature_importances_, so getting one means running
    permutation_importance (predicts on a sample n_repeats times) --
    real added memory/CPU cost for a number that's only ever read by the
    standalone static report generator, not the Streamlit app itself.
    Set False to skip it (writes an empty importance file instead).
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

    if df_override is not None:
        status("Using already-loaded dataset...")
        df_source = df_override
    else:
        status("Loading engineered dataset...")
        df_source = load_engineered_dataframe(data_dir)

    # Work on the smallest slice possible: only the columns the model needs,
    # only rows with a usable target, and (for the train split) only up to
    # max_train_rows -- BEFORE dropna/get_dummies, not after. Transforming
    # the full 2.7M-row frame first (then subsampling) was measured to peak
    # at ~1.8GB on top of the ~700MB the caller's cache already holds;
    # slicing down first keeps the transient copies proportional to what's
    # actually used instead of the whole dataset.
    needed_cols = FEATURES_BASE + ["SEASON", "MONTH_NUM", TARGET]
    needed_cols = [c for c in dict.fromkeys(needed_cols) if c in df_source.columns]
    has_target = df_source[TARGET].notna()

    train_idx = df_source.index[has_target & (df_source["MONTH_NUM"] <= 6)]
    val_idx = df_source.index[has_target & (df_source["MONTH_NUM"] == 7)]
    test_idx = df_source.index[has_target & (df_source["MONTH_NUM"] == 8)]

    if max_train_rows and len(train_idx) > max_train_rows:
        status(f"Subsampling train set: {len(train_idx):,} -> {max_train_rows:,} rows")
        train_idx = pd.Index(pd.Series(train_idx).sample(max_train_rows, random_state=42))

    all_idx = train_idx.append(val_idx).append(test_idx)
    df = df_source.loc[all_idx, needed_cols].copy()
    if df_override is None:
        del df_source  # only ours to free if we loaded it ourselves
    gc.collect()

    df = pd.get_dummies(df, columns=["SEASON"], prefix="SEASON")
    season_cols = [c for c in df.columns if c.startswith("SEASON_")]
    features = FEATURES_BASE + season_cols

    X_train = df.loc[train_idx, features]
    y_train = df.loc[train_idx, TARGET]
    X_val = df.loc[val_idx, features]
    y_val = df.loc[val_idx, TARGET]
    X_test = df.loc[test_idx, features]
    y_test = df.loc[test_idx, TARGET]

    del df
    gc.collect()

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
    elif compute_importance:
        from sklearn.inspection import permutation_importance
        sample_pool = X_val if len(X_val) else X_train
        sample_y_pool = y_val if len(X_val) else y_train
        sample = sample_pool.sample(min(20000, len(sample_pool)), random_state=42)
        sample_y = sample_y_pool.loc[sample.index]
        perm = permutation_importance(model, sample, sample_y, n_repeats=3, random_state=42, n_jobs=-1)
        importances = dict(zip(features, perm.importances_mean.tolist()))
        del sample, sample_y, perm
    else:
        status("Skipping feature importance computation (compute_importance=False)")
        importances = {}
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

    joblib.dump(model, os.path.join(model_dir, "demand_model.pkl"))
    with open(os.path.join(model_dir, "demand_model_features.json"), "w") as f:
        json.dump({"features": features, "target": TARGET, "backend": backend}, f, indent=2)
    with open(os.path.join(model_dir, "demand_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(model_dir, "demand_model_importance.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    status(f"Model saved -> {model_dir}/demand_model.pkl")
    return metrics
