"""
Traffic service: store and retrieve network traffic events.
"""

from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter
import uuid

from backend.database import get_db
from backend.schemas.traffic import TrafficEvent, TrafficStatistics
from backend.utils.logger import logger

_event_store: List[Dict] = []


async def store_traffic_event(event: TrafficEvent) -> Dict:
    """Persist a traffic event to the database."""
    event_dict = event.model_dump()
    event_dict["_id"] = str(uuid.uuid4())

    db = get_db()
    if db is not None:
        try:
            result = await db.traffic_events.insert_one(event_dict)
            event_dict["id"] = str(result.inserted_id)
            return event_dict
        except Exception as e:
            logger.error(f"DB event insert failed: {e}")

    _event_store.append(event_dict)
    return event_dict


async def get_recent_events(limit: int = 100) -> List[Dict]:
    """Retrieve most recent traffic events."""
    db = get_db()
    if db is not None:
        try:
            cursor = db.traffic_events.find({}).sort("created_at", -1).limit(limit)
            events = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id", ""))
                events.append(doc)
            return events
        except Exception as e:
            logger.error(f"DB events query failed: {e}")

    return sorted(
        _event_store, key=lambda x: x.get("created_at", datetime.min), reverse=True
    )[:limit]


async def get_traffic_statistics() -> TrafficStatistics:
    """Compute aggregate traffic statistics."""
    events = await get_recent_events(limit=10000)

    if not events:
        return TrafficStatistics(
            total_events=0, total_packets=0, total_bytes=0,
            anomaly_count=0, normal_count=0, anomaly_percentage=0.0,
            protocol_distribution={}, top_src_ips=[], top_dst_ips=[],
            avg_packets_per_flow=0.0, avg_bytes_per_flow=0.0,
        )

    total_packets = sum(e.get("packets", 0) for e in events)
    total_bytes = sum(e.get("bytes", 0) for e in events)
    anomaly_count = sum(1 for e in events if e.get("prediction") == "ANOMALY")
    normal_count = len(events) - anomaly_count

    protocols = Counter(e.get("protocol", "UNKNOWN") for e in events)
    src_ips = Counter(e.get("src_ip", "") for e in events)
    dst_ips = Counter(e.get("dst_ip", "") for e in events)

    return TrafficStatistics(
        total_events=len(events),
        total_packets=total_packets,
        total_bytes=total_bytes,
        anomaly_count=anomaly_count,
        normal_count=normal_count,
        anomaly_percentage=round(anomaly_count / max(len(events), 1) * 100, 2),
        protocol_distribution=dict(protocols.most_common(10)),
        top_src_ips=[{"ip": ip, "count": cnt} for ip, cnt in src_ips.most_common(10)],
        top_dst_ips=[{"ip": ip, "count": cnt} for ip, cnt in dst_ips.most_common(10)],
        avg_packets_per_flow=round(total_packets / max(len(events), 1), 2),
        avg_bytes_per_flow=round(total_bytes / max(len(events), 1), 2),
    )


async def get_threat_analytics() -> Dict:
    """Return threat distribution analytics."""
    events = await get_recent_events(limit=10000)
    attack_types = Counter(
        e.get("attack_type", "Unknown")
        for e in events
        if e.get("prediction") == "ANOMALY"
    )
    return {
        "attack_distribution": dict(attack_types.most_common(20)),
        "total_anomalies": sum(attack_types.values()),
    }
