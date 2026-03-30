"""
Train regression model for popularity_score and save sklearn pipeline + metadata.
Run after generate_car_market_data.py (or this script will call it if CSV is missing).
"""
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "car_market_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "vehicle_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")
GEN_SCRIPT = os.path.join(BASE_DIR, "generate_car_market_data.py")


def ensure_csv():
    if not os.path.exists(CSV_PATH):
        subprocess.check_call([sys.executable, GEN_SCRIPT], cwd=BASE_DIR)


def main():
    ensure_csv()
    df = pd.read_csv(CSV_PATH)

    feature_cols = [
        "brand",
        "model",
        "year",
        "engine_hp",
        "engine_cylinders",
        "fuel_type",
        "transmission",
        "driven_wheels",
        "doors",
        "market_category",
        "price",
    ]
    cat_cols = [
        "brand",
        "model",
        "fuel_type",
        "transmission",
        "driven_wheels",
        "market_category",
    ]
    num_cols = ["year", "engine_hp", "engine_cylinders", "doors", "price"]

    X = df[feature_cols]
    y = df["popularity_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        [
            ("cat", enc, cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=160, max_depth=16, random_state=42, n_jobs=-1
    )
    pipeline = Pipeline([("prep", preprocessor), ("reg", model)])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    q1, q2 = df["popularity_score"].quantile([0.34, 0.67]).tolist()
    pop_min = float(df["popularity_score"].min())
    pop_max = float(df["popularity_score"].max())

    prep = pipeline.named_steps["prep"]
    reg = pipeline.named_steps["reg"]
    cat_encoder = prep.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    num_names = num_cols
    all_names = cat_names + num_names
    importances = reg.feature_importances_
    order = np.argsort(importances)[::-1][:12]
    feature_importance = {
        all_names[i]: round(float(importances[i]), 4) for i in order
    }

    meta = {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "level_thresholds": {"q1": float(q1), "q2": float(q2)},
        "popularity_score_min": pop_min,
        "popularity_score_max": pop_max,
        "categories": {c: sorted(df[c].astype(str).unique().tolist()) for c in cat_cols},
        "numeric_ranges": {
            "year": [int(df["year"].min()), int(df["year"].max())],
            "engine_hp": [int(df["engine_hp"].min()), int(df["engine_hp"].max())],
            "engine_cylinders": sorted(df["engine_cylinders"].astype(int).unique().tolist()),
            "doors": sorted(df["doors"].astype(int).unique().tolist()),
            "price": [int(df["price"].min()), int(df["price"].max())],
        },
        "feature_importance_top": feature_importance,
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {MODEL_PATH} and {META_PATH} (R2={r2:.3f}, MAE={mae:.1f}).")


if __name__ == "__main__":
    main()
