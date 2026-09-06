"""
CLI wrapper around the shared training logic in
streamlit_app/utils/model_training.py -- run this to (re)train the model
locally. The Streamlit app also calls the same function automatically on
first load if no model exists yet (needed for cloud deployment, where you
don't get a terminal to run this script manually).

Run from src/:  python 02_train_demand_model.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))
from utils.model_training import train_demand_model

DATA_DIR = "../data/processed"
MODEL_DIR = "../models"

if __name__ == "__main__":
    metrics = train_demand_model(DATA_DIR, MODEL_DIR)
    print("\nFinal metrics:", metrics)
