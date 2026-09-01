"""
Train an Isolation Forest for unsupervised network anomaly detection.
"""

import joblib
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from loguru import logger

from ml.data_loader import generate_sample_dataset
from ml.preprocessing import preprocess_dataset

MODEL_DIR = Path("./models/isolation_forest")


def train_isolation_forest(
    data_path: Optional[str] = None,
    contamination: float = 0.1,
    n_estimators: int = 100,
    max_samples: str = "auto",
    random_state: int = 42,
    save: bool = True,
) -> dict:
    """
    Train an Isolation Forest model.

    Args:
        data_path: CSV dataset path; uses synthetic if None
        contamination: expected fraction of anomalies in data
        n_estimators: number of isolation trees
        max_samples: number of samples to draw per tree
        random_state: seed
        save: persist model
    """
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
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    # Use contamination from label proportion if available
    if y_test is not None and splits["y_train"] is not None:
        measured_contamination = float(splits["y_train"].mean())
        contamination = min(max(measured_contamination, 0.01), 0.5)
        logger.info(f"Using contamination={contamination:.3f} from training labels")

    # Train on normal traffic only (unsupervised: fit on all, detect anomalies)
    logger.info(f"Training Isolation Forest (contamination={contamination}, n_estimators={n_estimators})...")
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )

    train_start = time.perf_counter()
    clf.fit(X_train)
    training_time = time.perf_counter() - train_start
    logger.info(f"Isolation Forest training complete in {training_time:.2f}s")

    # Evaluate
    pred_start = time.perf_counter()
    raw_pred = clf.predict(X_test)  # 1=normal, -1=anomaly
    latency_ms = (time.perf_counter() - pred_start) / max(len(X_test), 1) * 1000

    # Convert to binary labels: 0=normal, 1=anomaly
    y_pred_binary = (raw_pred == -1).astype(int)
    anomaly_scores = -clf.score_samples(X_test)  # higher = more anomalous

    metrics = {
        "model_name": "isolation_forest",
        "contamination": contamination,
        "training_time_s": round(training_time, 2),
        "prediction_latency_ms": round(latency_ms, 4),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "detected_anomalies": int(y_pred_binary.sum()),
        "anomaly_rate_detected": round(float(y_pred_binary.mean()), 4),
    }

    if y_test is not None:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred_binary))
        metrics["precision"] = float(precision_score(y_test, y_pred_binary, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred_binary, zero_division=0))
        metrics["f1_score"] = float(f1_score(y_test, y_pred_binary, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(y_test, anomaly_scores))
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred_binary).tolist()
        logger.info(
            f"IF Metrics → Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | AUC: {metrics['roc_auc']:.4f}"
        )

    if save:
        joblib.dump(clf, MODEL_DIR / "model.joblib")
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Isolation Forest saved to {MODEL_DIR}")

    return metrics


if __name__ == "__main__":
    train_isolation_forest()
