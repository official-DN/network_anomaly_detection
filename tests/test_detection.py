"""Tests for the detection service."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.schemas.traffic import TrafficRecord
from backend.services.detection_service import DetectionService

def test_detection_service_init():
    svc = DetectionService()
    assert svc is not None

def test_fallback_predict_normal():
    svc = DetectionService()
    record = TrafficRecord(src_ip="192.168.1.1", dst_ip="10.0.0.1", src_port=12345, dst_port=80, protocol="TCP", packets=10, bytes=5000, duration=1.0)
    result = svc.predict_single(record, "nonexistent_model")
    assert result.prediction in ("NORMAL", "ANOMALY")
    assert 0.0 <= result.anomaly_score <= 1.0
    assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

def test_risk_score_range():
    svc = DetectionService()
    score = svc._calculate_risk_score(0.85, "random_forest")
    assert 0 <= score <= 100

def test_get_risk_level():
    svc = DetectionService()
    assert svc._get_risk_level(10) == "LOW"
    assert svc._get_risk_level(45) == "MEDIUM"
    assert svc._get_risk_level(70) == "HIGH"
    assert svc._get_risk_level(90) == "CRITICAL"
