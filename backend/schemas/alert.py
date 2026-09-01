"""Pydantic schemas for security alerts."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    NEW = "NEW"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    IGNORED = "IGNORED"


class Alert(BaseModel):
    id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    attack_type: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    anomaly_score: float
    confidence: float
    risk_level: RiskLevel
    model_used: str
    status: AlertStatus = AlertStatus.NEW
    description: Optional[str] = None
    features: Optional[dict] = None


class AlertUpdate(BaseModel):
    status: AlertStatus
    notes: Optional[str] = None


class AlertFilter(BaseModel):
    risk_level: Optional[RiskLevel] = None
    status: Optional[AlertStatus] = None
    attack_type: Optional[str] = None
    src_ip: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    skip: int = Field(default=0, ge=0)
