"""
Inference module: load a trained model and run predictions on new data.
Used for standalone CLI inference outside the FastAPI context.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from ml.preprocessing import encode_protocol, get_feature_columns, clean_dataframe


def load_model_and_scaler(model_name: str, model_dir: str = "./models"):
    """Load a trained model and its preprocessor."""
    base = Path(model_dir) / model_name
    model, scaler, feature_cols = None, None, None

    for fname in ["model.joblib", "model.pkl"]:
        p = base / fname
        if p.exists():
            model = joblib.load(p)
            break

    scaler_path = base / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    fc_path = base / "feature_cols.joblib"
    if fc_path.exists():
        feature_cols = joblib.load(fc_path)

    return model, scaler, feature_cols


def predict_from_csv(csv_path: str, model_name: str = "random_forest", model_dir: str = "./models") -> pd.DataFrame:
    """Run inference on a CSV file and return predictions."""
    from ml.data_loader import load_dataset
    df, labels = load_dataset(csv_path)
    df = clean_dataframe(df)
    df = encode_protocol(df)

    model, scaler, feature_cols = load_model_and_scaler(model_name, model_dir)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' not found in {model_dir}")

    cols = feature_cols or get_feature_columns(df)
    X = df[[c for c in cols if c in df.columns]].fillna(0).values

    if scaler:
        X = scaler.transform(X)

    preds = model.predict(X)
    df["prediction"] = preds

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df["anomaly_score"] = proba[:, -1]

    logger.info(f"Predictions complete. Anomalies: {(preds != 0).sum()} / {len(preds)}")
    return df


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if csv:
        result = predict_from_csv(csv)
        print(result[["prediction", "anomaly_score"]].value_counts())
    else:
        print("Usage: python ml/inference.py path/to/data.csv")
