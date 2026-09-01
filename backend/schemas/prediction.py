"""Pydantic schemas for ML predictions."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class PredictionResult(BaseModel):
    prediction: str  # NORMAL or ANOMALY
    attack_type: Optional[str] = None
    anomaly_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: int = Field(ge=0, le=100)
    model_used: str
    explanation: Optional[Dict[str, float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionResult(BaseModel):
    total: int
    anomalies: int
    normal: int
    anomaly_rate: float
    results: List[PredictionResult]
    processing_time_ms: float


class ModelInfo(BaseModel):
    name: str
    version: str
    type: str  # supervised, unsupervised
    status: str  # loaded, not_found
    last_trained: Optional[str] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None


class ModelPerformance(BaseModel):
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    training_time: Optional[float] = None
    prediction_latency_ms: Optional[float] = None
    created_at: Optional[datetime] = None
