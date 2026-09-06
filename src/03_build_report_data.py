"""
Regenerates report/data/report_data.json and report/data.js from the
engineered dataset and the trained model's saved metrics. Run this after
retraining the model or refreshing the underlying data, so the static
report reflects reality again.

Run from src/:  python 03_build_report_data.py
"""
import glob
import json
import os

import pandas as pd

DATA_DIR = "../data/processed"
MODEL_DIR = "../models"
REPORT_DIR = "../report"
os.makedirs(f"{REPORT_DIR}/data", exist_ok=True)


def hours_color(h):
    if pd.isna(h):
        return "gray"
    if h >= 18:
        return "green"
    if h >= 12:
        return "yellow"
    if h >= 6:
        return "orange"
    return "red"


usecols = ["FEEDER_NAME", "ADDRESS", "LATITUDE", "LONGITUDE", "BAND", "ACTUAL_HOURS",
           "SHORTFALL", "AVAILABILITY_PERCENTAGE", "MONTH", "DAY_OF_WEEK"]
dtypes = {
    "FEEDER_NAME": "category", "ADDRESS": "category", "LATITUDE": "float32",
    "LONGITUDE": "float32", "BAND": "float32", "ACTUAL_HOURS": "float32",
    "SHORTFALL": "float32", "AVAILABILITY_PERCENTAGE": "float32",
    "MONTH": "category", "DAY_OF_WEEK": "category",
}

print("Loading engineered data...")
parts = sorted(glob.glob(f"{DATA_DIR}/month=*.csv.gz"))
if not parts:
    raise FileNotFoundError(f"No data found in {DATA_DIR}. Run 01_build_pipeline.py first.")
df = pd.concat([pd.read_csv(p, usecols=usecols, dtype=dtypes) for p in parts], ignore_index=True)
df_real = df[df["ADDRESS"] != "NO_ADDRESS_MATCH"]

overview = {
    "total_rows": int(len(df)),
    "n_feeders": int(df["FEEDER_NAME"].nunique()),
    "n_addresses": int(df_real["ADDRESS"].nunique()),
    "avg_actual_hours": round(float(df_real["ACTUAL_HOURS"].mean()), 2),
    "avg_shortfall": round(float(df_real["SHORTFALL"].mean()), 2),
    "avg_availability_pct": round(float(df_real["AVAILABILITY_PERCENTAGE"].mean()), 2),
}

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
monthly_trend = (
    df_real.groupby("MONTH", observed=True)
    .agg(avg_hours=("ACTUAL_HOURS", "mean"), avg_shortfall=("SHORTFALL", "mean"))
    .reindex(month_order).round(2).reset_index().to_dict("records")
)

band_distribution = (
    df_real.drop_duplicates("FEEDER_NAME")["BAND"].value_counts().sort_index(ascending=False).to_dict()
)
band_distribution = {str(int(k)): int(v) for k, v in band_distribution.items() if pd.notna(k)}

rel = (
    df_real.groupby("FEEDER_NAME", observed=True)
    .agg(avg_actual_hours=("ACTUAL_HOURS", "mean"), avg_availability_pct=("AVAILABILITY_PERCENTAGE", "mean"),
         avg_shortfall=("SHORTFALL", "mean"), band=("BAND", "first"),
         latitude=("LATITUDE", "mean"), longitude=("LONGITUDE", "mean"),
         n_addresses=("ADDRESS", "nunique")).reset_index().dropna(subset=["latitude", "longitude"])
)
rel["color"] = rel["avg_actual_hours"].apply(hours_color)
reliability_color_counts = {k: int(v) for k, v in rel["color"].value_counts().to_dict().items()}
feeders_geo = rel.round(3).to_dict("records")
worst5 = rel.nsmallest(5, "avg_actual_hours")[["FEEDER_NAME", "avg_actual_hours", "band", "n_addresses"]].round(2).to_dict("records")
best5 = rel.nlargest(5, "avg_actual_hours")[["FEEDER_NAME", "avg_actual_hours", "band", "n_addresses"]].round(2).to_dict("records")

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_of_week_rhythm = (
    df_real.groupby("DAY_OF_WEEK", observed=True)["ACTUAL_HOURS"].mean()
    .reindex(dow_order).round(2).reset_index().rename(columns={"ACTUAL_HOURS": "avg_hours"})
    .to_dict("records")
)

metrics_path = f"{MODEL_DIR}/demand_model_metrics.json"
importance_path = f"{MODEL_DIR}/demand_model_importance.json"
if not os.path.exists(metrics_path) or not os.path.exists(importance_path):
    raise FileNotFoundError(
        f"Model metrics not found at {metrics_path}. Run 02_train_demand_model.py first."
    )
with open(metrics_path) as f:
    model_metrics = json.load(f)
with open(importance_path) as f:
    importance = json.load(f)
feature_importance = sorted(importance.items(), key=lambda x: -x[1])[:8]

payload = {
    "overview": overview,
    "monthly_trend": monthly_trend,
    "band_distribution": band_distribution,
    "reliability_color_counts": reliability_color_counts,
    "feeders_geo": feeders_geo,
    "worst5": worst5,
    "best5": best5,
    "day_of_week_rhythm": day_of_week_rhythm,
    "model_metrics": model_metrics,
    "feature_importance": feature_importance,
}

json_path = f"{REPORT_DIR}/data/report_data.json"
with open(json_path, "w") as f:
    json.dump(payload, f, indent=2, default=str)
print(f"Wrote {json_path}")

with open(json_path) as f:
    json_content = f.read()
js_path = f"{REPORT_DIR}/data.js"
with open(js_path, "w") as f:
    f.write(f"const REPORT_DATA = {json_content};\n")
print(f"Wrote {js_path}")

print("\nOverview:", overview)
print("Reliability color counts:", reliability_color_counts)
