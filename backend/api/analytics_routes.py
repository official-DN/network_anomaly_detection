"""Analytics and visualization data API routes."""

from fastapi import APIRouter
from backend.services.traffic_service import get_traffic_statistics, get_threat_analytics, get_recent_events
from backend.services.alert_service import get_alert_statistics
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/traffic")
async def traffic_analytics():
    """Return traffic volume and protocol analytics."""
    stats = await get_traffic_statistics()
    return {"success": True, "data": stats.model_dump()}


@router.get("/threats")
async def threat_analytics():
    """Return threat distribution and attack type analytics."""
    threat_data = await get_threat_analytics()
    alert_stats = await get_alert_statistics()
    return {
        "success": True,
        "data": {
            **threat_data,
            "alert_statistics": alert_stats,
        }
    }


@router.get("/protocols")
async def protocol_analytics():
    """Return protocol distribution analytics."""
    stats = await get_traffic_statistics()
    return {"success": True, "data": {"protocol_distribution": stats.protocol_distribution}}


@router.get("/timeline")
async def traffic_timeline(hours: int = 24):
    """Return traffic and alert counts bucketed by hour for charting."""
    events = await get_recent_events(limit=10000)
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=hours)

    hourly: Dict[str, Dict] = {}
    for h in range(hours):
        ts = now - timedelta(hours=h)
        key = ts.strftime("%Y-%m-%dT%H:00:00")
        hourly[key] = {"timestamp": key, "total": 0, "anomalies": 0, "normal": 0}

    for event in events:
        ts_raw = event.get("created_at")
        if not ts_raw:
            continue
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw)
            except Exception:
                continue
        else:
            ts = ts_raw
        if ts < cutoff:
            continue
        key = ts.strftime("%Y-%m-%dT%H:00:00")
        if key in hourly:
            hourly[key]["total"] += 1
            if event.get("prediction") == "ANOMALY":
                hourly[key]["anomalies"] += 1
            else:
                hourly[key]["normal"] += 1

    return {
        "success": True,
        "data": sorted(hourly.values(), key=lambda x: x["timestamp"])
    }
