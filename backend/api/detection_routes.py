"""
Detection API routes: single prediction and batch inference.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime

from backend.schemas.traffic import TrafficRecord, TrafficEvent
from backend.schemas.prediction import PredictionResult, BatchPredictionResult
from backend.schemas.alert import Alert, RiskLevel, AlertStatus
from backend.services.detection_service import detection_service
from backend.services.alert_service import create_alert
from backend.services.traffic_service import store_traffic_event
from backend.utils.logger import logger

router = APIRouter(prefix="/api/detection", tags=["Detection"])


def _should_create_alert(result: PredictionResult) -> bool:
    return result.prediction == "ANOMALY" and result.risk_score >= 30


@router.post("/predict", response_model=PredictionResult)
async def predict_single(record: TrafficRecord, model: str = "random_forest"):
    """Run anomaly detection on a single traffic record."""
    if model not in ["random_forest", "xgboost", "isolation_forest", "autoencoder", "auto"]:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    model_name = model
    if model == "auto":
        available = detection_service.get_available_models()
        model_name = available[0] if available else "random_forest"

    result = detection_service.predict_single(record, model_name)

    # Store traffic event
    event = TrafficEvent(
        **record.model_dump(),
        prediction=result.prediction,
        anomaly_score=result.anomaly_score,
        confidence=result.confidence,
        risk_level=result.risk_level,
        model_used=result.model_used,
    )
    await store_traffic_event(event)

    # Generate alert for significant anomalies
    if _should_create_alert(result):
        alert = Alert(
            timestamp=datetime.utcnow(),
            attack_type=result.attack_type or "Unknown Anomaly",
            src_ip=record.src_ip,
            dst_ip=record.dst_ip,
            src_port=record.src_port,
            dst_port=record.dst_port,
            protocol=record.protocol,
            anomaly_score=result.anomaly_score,
            confidence=result.confidence,
            risk_level=RiskLevel(result.risk_level),
            model_used=result.model_used,
            status=AlertStatus.NEW,
        )
        await create_alert(alert)
        logger.info(f"Alert generated: {alert.attack_type} [{alert.risk_level}]")

    return result


@router.post("/batch", response_model=BatchPredictionResult)
async def predict_batch(records: List[TrafficRecord], model: str = "random_forest"):
    """Run batch anomaly detection on multiple traffic records."""
    if not records:
        raise HTTPException(status_code=400, detail="No records provided")
    if len(records) > 10000:
        raise HTTPException(status_code=400, detail="Batch size exceeds 10,000 records")

    result = detection_service.predict_batch(records, model)

    # Persist events and create alerts for high-risk items
    for record, pred in zip(records, result.results):
        event = TrafficEvent(
            **record.model_dump(),
            prediction=pred.prediction,
            anomaly_score=pred.anomaly_score,
            confidence=pred.confidence,
            risk_level=pred.risk_level,
            model_used=pred.model_used,
        )
        await store_traffic_event(event)

        if _should_create_alert(pred):
            alert = Alert(
                timestamp=datetime.utcnow(),
                attack_type=pred.attack_type or "Unknown Anomaly",
                src_ip=record.src_ip,
                dst_ip=record.dst_ip,
                src_port=record.src_port,
                dst_port=record.dst_port,
                protocol=record.protocol,
                anomaly_score=pred.anomaly_score,
                confidence=pred.confidence,
                risk_level=RiskLevel(pred.risk_level),
                model_used=pred.model_used,
                status=AlertStatus.NEW,
            )
            await create_alert(alert)

    return result
