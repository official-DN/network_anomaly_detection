"""
Pydantic schemas for traffic events and analysis requests.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TrafficRecord(BaseModel):
    """Single network traffic flow record for analysis."""
    src_ip: str = Field(default="0.0.0.0", description="Source IP address")
    dst_ip: str = Field(default="0.0.0.0", description="Destination IP address")
    src_port: int = Field(default=0, ge=0, le=65535)
    dst_port: int = Field(default=0, ge=0, le=65535)
    protocol: str = Field(default="TCP", description="Network protocol")
    packets: int = Field(default=1, ge=0)
    bytes: int = Field(default=0, ge=0)
    duration: float = Field(default=0.0, ge=0.0, description="Flow duration in seconds")
    packets_per_sec: Optional[float] = None
    bytes_per_sec: Optional[float] = None
    avg_packet_size: Optional[float] = None
    syn_count: Optional[int] = None
    ack_count: Optional[int] = None
    fin_count: Optional[int] = None
    rst_count: Optional[int] = None
    fwd_packets: Optional[int] = None
    bwd_packets: Optional[int] = None
    timestamp: Optional[datetime] = None


class TrafficEvent(TrafficRecord):
    """Traffic event stored in the database."""
    id: Optional[str] = None
    prediction: Optional[str] = None
    anomaly_score: Optional[float] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    model_used: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrafficStatistics(BaseModel):
    """Aggregate traffic statistics."""
    total_events: int
    total_packets: int
    total_bytes: int
    anomaly_count: int
    normal_count: int
    anomaly_percentage: float
    protocol_distribution: dict
    top_src_ips: list
    top_dst_ips: list
    avg_packets_per_flow: float
    avg_bytes_per_flow: float
