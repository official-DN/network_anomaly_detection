"""Tests for feature extraction and traffic record processing."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.schemas.traffic import TrafficRecord

def test_traffic_record_defaults():
    r = TrafficRecord()
    assert r.src_port == 0
    assert r.protocol == "TCP"

def test_traffic_record_valid():
    r = TrafficRecord(src_ip="192.168.1.1", dst_ip="10.0.0.1", src_port=12345, dst_port=80, protocol="TCP", packets=100, bytes=50000, duration=1.5)
    assert r.src_ip == "192.168.1.1"
    assert r.packets == 100

def test_traffic_record_invalid_port():
    with pytest.raises(Exception):
        TrafficRecord(src_port=99999)

def test_traffic_record_negative_packets():
    with pytest.raises(Exception):
        TrafficRecord(packets=-1)
