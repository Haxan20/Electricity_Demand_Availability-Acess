# Electricity Availability Forecasting — Phase 1

Data pipeline + feature engineering + trained demand-forecasting model.
Built and tested against your 2,747,030-row PREPARED_DATA.csv.

## What's here

```
project/
├── src/
│   ├── 01_build_pipeline.py     # load raw data, synth weather, engineer features
│   ├── 02_train_demand_model.py # train + evaluate the demand model
│   └── 03_forecast.py           # recursive N-day-ahead forecast for one address
├── data/processed/
│   ├── month=Jan.csv.gz ... month=Aug.csv.gz   # engineered features, gzip CSV, partitioned by month
│   ├── full_engineered.pkl      # same data as a single pandas pickle (fastest to reload)
│   ├── weather_cache.csv        # synthetic daily weather, one row per date
│   └── schema.json              # column -> dtype
├── models/
│   ├── demand_model.pkl              # trained model
│   ├── demand_model_features.json    # exact feature list + backend used
│   ├── demand_model_metrics.json     # train/val/test MAE, RMSE, R2
│   └── demand_model_importance.json  # top feature importances
└── requirements.txt
```

## Environment note

This was built in a sandbox with **no internet access** and only
`pandas`/`numpy`/`scikit-learn` installed — `xgboost`, `pyarrow`, and
`streamlit` aren't available here. So:

- **Model backend**: the training script tries XGBoost first and falls back
  to scikit-learn's `HistGradientBoostingRegressor` (same algorithm family —
  histogram-based gradient boosting) if XGBoost isn't importable. In this
  sandbox it used the sklearn fallback. Install `xgboost` in your real
  environment and re-run `02_train_demand_model.py` — no code changes
  needed, it'll pick up XGBoost automatically and should get similar or
  better accuracy.
- **Storage format**: output is gzip-CSV partitioned by month instead of
  Parquet (no `pyarrow` here). Once you `pip install pyarrow`, swap the one
  line noted in `01_build_pipeline.py` to write `.parquet` instead.
- **Weather**: synthetic, generated from a Lagos seasonal profile (dry
  Nov–Mar, rainy Apr–Oct) as you specified as the fallback. Swap
  `generate_weather()` for a real OpenWeatherMap/NOAA call later — the
  output schema (`WX_*` columns) is what the rest of the pipeline expects,
  so nothing downstream needs to change.

## How to run

```bash
pip install -r requirements.txt
cd src
python 01_build_pipeline.py      # ~4 min, writes data/processed/
python 02_train_demand_model.py  # ~2-3 min, writes models/
python 03_forecast.py            # demo: 7-day forecast for one address
```

## Model results (time-based split: train Jan–Jun, validate Jul, test Aug)

| Split | MAE (hours) | RMSE (hours) | R² |
|---|---|---|---|
| Train | 1.76 | 2.54 | 0.85 |
| Validation (Jul) | 2.41 | 3.44 | 0.71 |
| Test (Aug) | 1.96 | 2.86 | 0.76 |

Top predictive features: yesterday's actual hours (LAG_1D) dominates, followed
by 7/30-day rolling averages and BAND. This is expected for this kind of
series — electricity availability is highly autocorrelated day-to-day.

Caveat: weather features in this run are synthetic, so their real-world
predictive value is unverified — once real weather is wired in, re-check
feature importance; it may shift.

## Using `forecast_n_days()`

```python
from importlib import import_module
import sys
sys.path.insert(0, "src")
forecast_mod = import_module("03_forecast")  # or just copy the function into your app
fc = forecast_mod.forecast_n_days("7UP", "16,IKOSI ROAD", n_days=7)
```

Returns a DataFrame with `DATE`, `PREDICTED_ACTUAL_HOURS`, `BAND`,
`PREDICTED_SHORTFALL`, `PREDICTED_AVAILABILITY_PCT` — everything the
Streamlit forecast page needs to render directly.

## What's NOT built yet (per your original spec)

- Shortfall model (currently derived as BAND − predicted hours, not a
  separately trained model — same result for a linear relationship like
  this, but say if you want it modeled independently, e.g. if shortfall
  should account for uncertainty differently than the demand forecast)
- Solar irradiation physics model, anomaly detection
- Streamlit UI (all 5 pages), recommendation engine, cost/ROI calculators
- Real weather API integration, database, Docker/deployment files

Ready to keep going on any of these next.
