# Running the Streamlit app

## Setup

```bash
cd deliverable
pip install -r requirements.txt
```

**Training the model.** The engineered data (`data/processed/`) is included
and ready to use, but the trained model (`models/demand_model.pkl`) is NOT
included in this zip. That's deliberate: scikit-learn model pickles only
reliably load with the *exact* scikit-learn version that created them
(internal module paths like `sklearn.ensemble._hist_gradient_boosting._loss`
change between versions), so shipping a pre-trained pickle would likely fail
with something like `ModuleNotFoundError: No module named '_loss'` the
moment your local scikit-learn version differs from ours -- which it will.

For local development, train it explicitly:

```bash
cd src
python 02_train_demand_model.py    # ~2-3 min, writes models/
cd ..
```

**On a platform with no terminal (Streamlit Community Cloud, etc.):** you
don't need to do this manually. `utils/data_loader.py`'s `load_model()`
automatically trains the model on first app load if `models/demand_model.pkl`
doesn't exist yet, using whatever scikit-learn/XGBoost version got installed
from `requirements.txt` in that environment -- so there's no version
mismatch risk there, since the same environment trains and loads it. First
load takes ~2-3 minutes with a "First-time setup..." spinner; every load
after that is instant (Streamlit caches the loaded model in memory for the
life of the app instance).

You don't need to re-run `01_build_pipeline.py` -- the engineered dataset
it produces is already in `data/processed/`.

If you ever see `ModuleNotFoundError: No module named '_loss'` (or any
similarly cryptic unpickle error), it means the model on disk was pickled by
a different library version than what's currently installed -- delete
`models/demand_model.pkl` and either re-run `02_train_demand_model.py` or
just reload the app (the auto-train fallback will recreate it).

## Run

```bash
cd streamlit_app
streamlit run app.py
```

Opens at http://localhost:8501

## Pages

- **Home** (`app.py`) — search by address or feeder, network-wide quick
  stats, top 5 least/most reliable areas.
- **Forecast** (`pages/1_Forecast.py`) — 1-14 day forecast chart (history vs
  predicted), band cap line, daily detail table.
- **Recommendations** (`pages/2_Recommendations.py`) — enter your daily need
  (kWh, hours, or an appliance checklist), get a solar/generator/battery
  sizing recommendation with indicative Naira costs and payback estimate.
- **Network Map** (`pages/3_Network_Map.py`) — all feeders on a Folium map,
  color-coded green/yellow/orange/red by average availability, with a
  popup/details panel per feeder. Falls back to a plain point map if
  `folium`/`streamlit-folium` aren't installed.
- **Reports** (`pages/4_Reports.py`) — download an Excel or PDF forecast +
  recommendation report for one address, or compare up to 5 addresses
  side-by-side and export that as Excel.

Selecting an address on Home stores it in `st.session_state`, which
Forecast/Recommendations/Reports read — so the normal flow is Home →
Forecast → Recommendations → Reports, though each page will compute its own
forecast if you land there directly without going through Home first.

## Known limitations / what's next

- No separately-trained shortfall/solar/anomaly models -- shortfall is
  derived as `BAND - predicted hours`.
- Weather is synthetic (see main README) -- you mentioned you don't have a
  weather data source right now, so this stays synthetic until you do.
- No database/caching layer -- every page recomputes from the in-memory
  DataFrame each run. Fine for one user testing locally; for 100 concurrent
  users (per your Part 8 spec) you'd want to precompute/cache forecasts,
  probably in Postgres or Redis, rather than recomputing per request.
- No Docker/deployment files yet.
- Not yet tested inside an actual Streamlit runtime (not installed in the
  sandbox this was built in) -- the data/model/forecast/report logic
  underneath every page *is* tested against your real dataset (see chat),
  including full round-trip Excel/PDF generation, but the UI layer itself
  should get a smoke-test in your environment before you trust it fully.
