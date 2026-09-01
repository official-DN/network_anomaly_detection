"""
Model service: retrieve model info, metrics, and performance comparisons.
"""

from datetime import datetime
from typing import List, Dict, Optional
import joblib
from pathlib import Path

from backend.config import settings
from backend.database import get_db
from backend.schemas.prediction import ModelInfo, ModelPerformance
from backend.utils.logger import logger

_metrics_store: List[Dict] = []


async def get_all_model_info() -> List[ModelInfo]:
    """Return information about all available models."""
    model_names = ["random_forest", "xgboost", "isolation_forest", "autoencoder"]
    infos = []
    for name in model_names:
        model_dir = settings.model_path / name
        exists = any((model_dir / f).exists() for f in ["model.joblib", "model.pkl", "autoencoder.pt"])
        metrics = await get_model_metrics(name)

        infos.append(ModelInfo(
            name=name.replace("_", " ").title(),
            version="1.0.0",
            type="supervised" if name in ("random_forest", "xgboost") else "unsupervised",
            status="loaded" if exists else "not_found",
            last_trained=metrics.created_at.isoformat() if metrics and metrics.created_at else None,
            accuracy=metrics.accuracy if metrics else None,
            f1_score=metrics.f1_score if metrics else None,
        ))
    return infos


async def get_model_metrics(model_name: str) -> Optional[ModelPerformance]:
    """Get the most recent metrics for a specific model."""
    db = get_db()
    if db is not None:
        try:
            doc = await db.model_metrics.find_one(
                {"model_name": model_name},
                sort=[("created_at", -1)]
            )
            if doc:
                doc.pop("_id", None)
                return ModelPerformance(**doc)
        except Exception as e:
            logger.error(f"DB metrics query failed: {e}")

    for m in reversed(_metrics_store):
        if m.get("model_name") == model_name:
            return ModelPerformance(**m)
    return None


async def get_all_model_performance() -> List[ModelPerformance]:
    """Return performance metrics for all models."""
    model_names = ["random_forest", "xgboost", "isolation_forest", "autoencoder"]
    results = []
    for name in model_names:
        metrics = await get_model_metrics(name)
        if metrics:
            results.append(metrics)
        else:
            results.append(ModelPerformance(model_name=name))
    return results


async def save_model_metrics(metrics: ModelPerformance) -> None:
    """Persist model performance metrics."""
    metrics_dict = metrics.model_dump()
    if not metrics_dict.get("created_at"):
        metrics_dict["created_at"] = datetime.utcnow()

    db = get_db()
    if db is not None:
        try:
            await db.model_metrics.insert_one(metrics_dict)
            logger.info(f"Saved metrics for {metrics.model_name}")
            return
        except Exception as e:
            logger.error(f"DB metrics save failed: {e}")

    _metrics_store.append(metrics_dict)
