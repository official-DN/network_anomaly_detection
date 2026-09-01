"""
Model evaluation and comparison across all trained models.
Outputs a complete performance report.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

MODEL_DIR = Path("./models")
MODELS = ["random_forest", "xgboost", "isolation_forest", "autoencoder"]


def load_metrics(model_name: str) -> Optional[Dict]:
    """Load saved metrics for a model."""
    path = MODEL_DIR / model_name / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def evaluate_all_models() -> pd.DataFrame:
    """
    Load and compare metrics from all trained models.
    Returns a comparison DataFrame.
    """
    rows = []
    for model_name in MODELS:
        metrics = load_metrics(model_name)
        if metrics:
            rows.append({
                "Model": model_name.replace("_", " ").title(),
                "Accuracy": round(metrics.get("accuracy", 0) * 100, 2),
                "Precision": round(metrics.get("precision", 0) * 100, 2),
                "Recall": round(metrics.get("recall", 0) * 100, 2),
                "F1-Score": round(metrics.get("f1_score", 0) * 100, 2),
                "ROC-AUC": round(metrics.get("roc_auc", 0) * 100, 2),
                "Train Time (s)": metrics.get("training_time_s", "N/A"),
                "Latency (ms)": metrics.get("prediction_latency_ms", "N/A"),
            })
        else:
            rows.append({"Model": model_name.replace("_", " ").title(), "Status": "Not trained"})

    df = pd.DataFrame(rows)
    if not df.empty and "Accuracy" in df.columns:
        logger.info("\nModel Comparison:\n" + df.to_string(index=False))
    return df


def select_best_model(metric: str = "f1_score") -> str:
    """Select the best model by a given metric (from saved JSON metrics)."""
    best_score = -1.0
    best_model = "random_forest"

    for model_name in MODELS:
        metrics = load_metrics(model_name)
        if metrics and metric in metrics:
            score = metrics[metric]
            if score > best_score:
                best_score = score
                best_model = model_name

    logger.info(f"Best model by {metric}: {best_model} ({best_score:.4f})")
    return best_model


def print_evaluation_report() -> None:
    """Print a formatted evaluation report for all models."""
    print("\n" + "=" * 70)
    print("  AI NETWORK ANOMALY DETECTION — MODEL EVALUATION REPORT")
    print("=" * 70)

    for model_name in MODELS:
        metrics = load_metrics(model_name)
        print(f"\n{'─' * 40}")
        print(f"  Model: {model_name.replace('_', ' ').title()}")
        print(f"{'─' * 40}")
        if not metrics:
            print("  ⚠  Not trained yet")
            continue
        for k, v in metrics.items():
            if k == "confusion_matrix":
                print(f"  Confusion Matrix:")
                for row in v:
                    print(f"    {row}")
            elif isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")
            else:
                print(f"  {k:30s}: {v}")

    best = select_best_model()
    print(f"\n{'=' * 70}")
    print(f"  RECOMMENDED MODEL: {best.replace('_', ' ').title()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_evaluation_report()
    df = evaluate_all_models()
