"""
Alert service: create, retrieve, and manage security alerts.
"""

from datetime import datetime
from typing import List, Optional, Dict
from bson import ObjectId
import uuid

from backend.database import get_db
from backend.schemas.alert import Alert, AlertUpdate, AlertFilter, RiskLevel, AlertStatus
from backend.utils.logger import logger

# In-memory fallback when MongoDB is unavailable
_alert_store: List[Dict] = []


def _generate_alert_description(attack_type: str, src_ip: str, dst_ip: str, score: float) -> str:
    """Generate human-readable alert description."""
    return (
        f"Potential {attack_type} detected from {src_ip} targeting {dst_ip}. "
        f"Anomaly score: {score:.2f}. Immediate investigation recommended."
    )


async def create_alert(alert: Alert) -> Dict:
    """Persist a new security alert."""
    alert_dict = alert.model_dump()
    alert_dict["_id"] = str(uuid.uuid4())
    alert_dict["description"] = alert_dict.get("description") or _generate_alert_description(
        alert.attack_type, alert.src_ip, alert.dst_ip, alert.anomaly_score
    )

    db = get_db()
    if db is not None:
        try:
            result = await db.alerts.insert_one(alert_dict)
            alert_dict["id"] = str(result.inserted_id)
        except Exception as e:
            logger.error(f"DB alert insert failed: {e}")
            _alert_store.append(alert_dict)
    else:
        _alert_store.append(alert_dict)

    logger.info(f"Alert created: {alert.attack_type} from {alert.src_ip} [{alert.risk_level}]")
    return alert_dict


async def get_alerts(filters: AlertFilter) -> List[Dict]:
    """Retrieve alerts with optional filtering."""
    db = get_db()
    if db is not None:
        try:
            query: Dict = {}
            if filters.risk_level:
                query["risk_level"] = filters.risk_level.value
            if filters.status:
                query["status"] = filters.status.value
            if filters.attack_type:
                query["attack_type"] = {"$regex": filters.attack_type, "$options": "i"}
            if filters.src_ip:
                query["src_ip"] = filters.src_ip
            if filters.start_time or filters.end_time:
                ts_filter = {}
                if filters.start_time:
                    ts_filter["$gte"] = filters.start_time
                if filters.end_time:
                    ts_filter["$lte"] = filters.end_time
                query["timestamp"] = ts_filter

            cursor = db.alerts.find(query).sort("timestamp", -1).skip(filters.skip).limit(filters.limit)
            alerts = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id", doc.get("id", "")))
                alerts.append(doc)
            return alerts
        except Exception as e:
            logger.error(f"DB alert query failed: {e}")

    # Fallback to in-memory store
    alerts = list(_alert_store)
    if filters.risk_level:
        alerts = [a for a in alerts if a.get("risk_level") == filters.risk_level.value]
    if filters.status:
        alerts = [a for a in alerts if a.get("status") == filters.status.value]
    return sorted(alerts, key=lambda x: x.get("timestamp", datetime.min), reverse=True)[
        filters.skip: filters.skip + filters.limit
    ]


async def get_alert_by_id(alert_id: str) -> Optional[Dict]:
    """Get a single alert by its ID."""
    db = get_db()
    if db is not None:
        try:
            doc = await db.alerts.find_one({"_id": ObjectId(alert_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return doc
        except Exception:
            pass

    return next((a for a in _alert_store if a.get("_id") == alert_id), None)


async def update_alert(alert_id: str, update: AlertUpdate) -> Optional[Dict]:
    """Update alert status."""
    db = get_db()
    update_data = {"status": update.status.value}
    if update.notes:
        update_data["notes"] = update.notes
    update_data["updated_at"] = datetime.utcnow().isoformat()

    if db is not None:
        try:
            await db.alerts.update_one({"_id": ObjectId(alert_id)}, {"$set": update_data})
            return await get_alert_by_id(alert_id)
        except Exception as e:
            logger.error(f"DB alert update failed: {e}")

    for alert in _alert_store:
        if alert.get("_id") == alert_id:
            alert.update(update_data)
            return alert
    return None


async def get_alert_statistics() -> Dict:
    """Return alert count statistics by risk level and status."""
    db = get_db()
    if db is not None:
        try:
            pipeline = [
                {"$group": {"_id": {"risk": "$risk_level", "status": "$status"}, "count": {"$sum": 1}}}
            ]
            cursor = db.alerts.aggregate(pipeline)
            stats: Dict = {"by_risk": {}, "by_status": {}, "total": 0}
            async for doc in cursor:
                risk = doc["_id"]["risk"]
                status = doc["_id"]["status"]
                count = doc["count"]
                stats["by_risk"][risk] = stats["by_risk"].get(risk, 0) + count
                stats["by_status"][status] = stats["by_status"].get(status, 0) + count
                stats["total"] += count
            return stats
        except Exception as e:
            logger.error(f"Alert stats query failed: {e}")

    # Fallback
    stats: Dict = {"by_risk": {}, "by_status": {}, "total": len(_alert_store)}
    for a in _alert_store:
        r = a.get("risk_level", "UNKNOWN")
        s = a.get("status", "NEW")
        stats["by_risk"][r] = stats["by_risk"].get(r, 0) + 1
        stats["by_status"][s] = stats["by_status"].get(s, 0) + 1
    return stats
