"""
Train a Random Forest classifier for network anomaly detection.
"""

import joblib
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from loguru import logger

from ml.data_loader import generate_sample_dataset
from ml.preprocessing import preprocess_dataset

MODEL_DIR = Path("./models/random_forest")


def train_random_forest(
    data_path: Optional[str] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    save: bool = True,
) -> dict:
    """
    Train a Random Forest model.

    Args:
        data_path: path to CSV dataset; uses synthetic data if None
        n_estimators: number of trees
        max_depth: maximum tree depth
        random_state: reproducibility seed
        save: whether to save model and metrics
    """
    # Load data
    if data_path:
        from ml.data_loader import load_dataset
        df, labels = load_dataset(data_path)
    else:
        logger.info("No dataset provided — using synthetic data")
        df, labels = generate_sample_dataset(n_samples=10000)

    # Preprocess
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    splits = preprocess_dataset(
        df, labels, save_path=str(MODEL_DIR) if save else None
    )
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    if y_train is None:
        logger.error("Labels required for supervised Random Forest training")
        return {}

    # Train
    logger.info(f"Training Random Forest ({n_estimators} trees)...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    train_start = time.perf_counter()
    clf.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start
    logger.info(f"Training complete in {training_time:.2f}s")

    # Evaluate
    pred_start = time.perf_counter()
    y_pred = clf.predict(X_test)
    latency_ms = (time.perf_counter() - pred_start) / len(X_test) * 1000

    metrics = {
        "model_name": "random_forest",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "training_time_s": round(training_time, 2),
        "prediction_latency_ms": round(latency_ms, 4),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
    }

    # ROC-AUC
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    logger.info(
        f"RF Metrics → Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1_score']:.4f} | "
        f"AUC: {metrics.get('roc_auc', 'N/A')}"
    )
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["NORMAL", "ANOMALY"]))

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_DIR / "model.joblib")
        with open(MODEL_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Model saved to {MODEL_DIR}")

    return metrics


if __name__ == "__main__":
    train_random_forest()
