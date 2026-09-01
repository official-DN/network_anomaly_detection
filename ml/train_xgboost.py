"""
Train an XGBoost classifier for network anomaly detection.
"""

import joblib
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from loguru import logger

from ml.data_loader import generate_sample_dataset
from ml.preprocessing import preprocess_dataset

MODEL_DIR = Path("./models/xgboost")


def train_xgboost(
    data_path: Optional[str] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    save: bool = True,
) -> dict:
    """
    Train an XGBoost classifier.

    Args:
        data_path: CSV dataset path; uses synthetic if None
        n_estimators: number of boosting rounds
        max_depth: maximum tree depth
        learning_rate: step size shrinkage
        random_state: seed
        save: persist model
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed — pip install xgboost")
        return {}

    # Load data
    if data_path:
        from ml.data_loader import load_dataset
        df, labels = load_dataset(data_path)
    else:
        logger.info("No dataset provided — using synthetic data")
        df, labels = generate_sample_dataset(n_samples=10000)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    splits = preprocess_dataset(
        df, labels, save_path=str(MODEL_DIR) if save else None
    )
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    if y_train is None:
        logger.error("Labels required for supervised XGBoost training")
        return {}

    # Calculate class weight for imbalanced datasets
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    logger.info(f"Training XGBoost ({n_estimators} estimators, lr={learning_rate})...")
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    train_start = time.perf_counter()
    clf.fit(
        X_train, y_train,
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=False,
    )
    training_time = time.perf_counter() - train_start
    logger.info(f"XGBoost training complete in {training_time:.2f}s")

    # Evaluate
    pred_start = time.perf_counter()
    y_pred = clf.predict(X_test)
    latency_ms = (time.perf_counter() - pred_start) / len(X_test) * 1000

    metrics = {
        "model_name": "xgboost",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "training_time_s": round(training_time, 2),
        "prediction_latency_ms": round(latency_ms, 4),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
    }

    proba = clf.predict_proba(X_test)[:, 1]
    metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    logger.info(
        f"XGBoost Metrics → Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1_score']:.4f} | AUC: {metrics['roc_auc']:.4f}"
    )
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["NORMAL", "ANOMALY"]))

    if save:
        joblib.dump(clf, MODEL_DIR / "model.joblib")
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"XGBoost model saved to {MODEL_DIR}")

    return metrics


if __name__ == "__main__":
    train_xgboost()
