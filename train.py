import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data2.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "car_price_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "training_metrics.json")
CURRENT_YEAR = 2026


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    out["Accident History"] = out["Accident History"].replace({"Unknown": "Not Reported"})
    out["Insurance Claims"] = out["Insurance Claims"].replace({"None": np.nan, "Unknown": np.nan})
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Car_Age"] = (CURRENT_YEAR - out["Manufacturing Year"]).clip(lower=0, upper=35)
    out["kms_per_year"] = out["Kilometers Driven"] / np.maximum(out["Car_Age"], 1)

    quality_map = {"Poor": 0, "Average": 1, "Good": 2}
    condition_cols = [
        "Body Panels",
        "Rust",
        "Glass Condition",
        "Lights Condition",
        "Tyres",
        "Seats Condition",
        "Electronics Condition",
        "Smell",
        "Engine Condition",
        "Transmission Condition",
        "Brakes Condition",
        "Steering Condition",
        "Suspension Condition",
        "Battery Condition",
        "AC Cooling",
        "Idle Vibration",
        "Acceleration Condition",
        "Brakes Drive",
    ]
    out["condition_score"] = out[condition_cols].apply(lambda s: s.map(quality_map)).mean(axis=1)
    out["has_airbags"] = out["Airbags Present"].map({"Yes": 1, "No": 0})
    out["is_engine_start_ok"] = out["Engine Start"].map({"Yes": 1, "No": 0})
    out["rattling_present"] = out["Rattling Noises"].map({"Present": 1, "Occasional": 0})
    return out


def build_preprocessor(categorical_features, numeric_features) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                categorical_features,
            ),
        ]
    )


def evaluate_model(pipe, X_train, X_test, y_train_log, y_test_log):
    pipe.fit(X_train, y_train_log)
    pred_log = pipe.predict(X_test)
    pred = np.expm1(pred_log)
    true = np.expm1(y_test_log)
    mae = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    r2 = float(r2_score(true, pred))
    return mae, rmse, r2


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = normalize_categories(df)

    numeric_base = ["Manufacturing Year", "Kilometers Driven", "Mileage", "Owners", "Airbags Count"]
    for col in numeric_base + ["Selling Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Selling Price", "Manufacturing Year", "Kilometers Driven", "Mileage"]).copy()
    df = df[df["Selling Price"] > 10000].copy()

    price_lo, price_hi = df["Selling Price"].quantile([0.005, 0.995])
    kms_lo, kms_hi = df["Kilometers Driven"].quantile([0.005, 0.995])
    df["Selling Price"] = df["Selling Price"].clip(price_lo, price_hi)
    df["Kilometers Driven"] = df["Kilometers Driven"].clip(kms_lo, kms_hi)

    df = add_features(df)

    feature_cols = [
        "Make",
        "Model",
        "Variant",
        "City",
        "Color",
        "Fuel",
        "Transmission Type",
        "RC Status",
        "Loan Clearance",
        "Accident History",
        "Insurance Claims",
        "Service Records",
        "Body Panels",
        "Rust",
        "Glass Condition",
        "Lights Condition",
        "Tyres",
        "Seats Condition",
        "Electronics Condition",
        "Smell",
        "Engine Condition",
        "Transmission Condition",
        "Brakes Condition",
        "Steering Condition",
        "Suspension Condition",
        "Battery Condition",
        "AC Cooling",
        "Rattling Noises",
        "Engine Start",
        "Idle Vibration",
        "Acceleration Condition",
        "Brakes Drive",
        "Airbags Present",
        "Safety Systems",
        "Manufacturing Year",
        "Kilometers Driven",
        "Mileage",
        "Owners",
        "Airbags Count",
        "Car_Age",
        "kms_per_year",
        "condition_score",
        "has_airbags",
        "is_engine_start_ok",
        "rattling_present",
    ]

    X = df[feature_cols].copy()
    y_log = np.log1p(df["Selling Price"].values)

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Explicit missing-value handling to keep serialized pipeline simple/stable.
    for col in cat_cols:
        X[col] = X[col].astype(str).str.strip().replace({"nan": "Unknown", "": "Unknown"})
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    preprocessor = build_preprocessor(cat_cols, num_cols)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    candidates = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "lightgbm": LGBMRegressor(
            objective="regression",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if XGB_AVAILABLE:
        candidates["xgboost"] = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.2,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

    results = []
    trained_pipelines = {}

    for name, reg in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", reg)])
        mae, rmse, r2 = evaluate_model(pipe, X_train, X_test, y_train_log, y_test_log)
        trained_pipelines[name] = pipe
        results.append({"model": name, "mae_holdout": mae, "rmse_holdout": rmse, "r2_holdout": r2})
        print(f"{name}: MAE={mae:,.2f} RMSE={rmse:,.2f} R2={r2:.4f}")

    results_sorted = sorted(results, key=lambda x: x["mae_holdout"])
    best_overall = results_sorted[0]["model"]

    preferred = [r for r in results if r["model"] in ("lightgbm", "xgboost")]
    if not preferred:
        raise RuntimeError("Neither LightGBM nor XGBoost is available to deploy.")
    deployed = sorted(preferred, key=lambda x: x["mae_holdout"])[0]["model"]
    deployed_pipe = trained_pipelines[deployed]

    joblib.dump(deployed_pipe, MODEL_PATH)

    metrics = {
        "rows": int(len(df)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost_available": XGB_AVAILABLE,
        "results_holdout": [
            {
                "model": r["model"],
                "mae_holdout": round(float(r["mae_holdout"]), 2),
                "rmse_holdout": round(float(r["rmse_holdout"]), 2),
                "r2_holdout": round(float(r["r2_holdout"]), 4),
            }
            for r in results_sorted
        ],
        "best_overall_model": best_overall,
        "deployed_model": deployed,
        "deployment_rule": "best model among {'lightgbm','xgboost'} by holdout MAE",
        "saved_model_path": MODEL_PATH,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete")
    print(f"Best overall: {best_overall}")
    print(f"Deployed model (preferred set): {deployed}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
