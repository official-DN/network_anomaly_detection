"""Alert management API routes."""

from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.schemas.alert import AlertFilter, AlertUpdate, RiskLevel, AlertStatus
from backend.services.alert_service import get_alerts, get_alert_by_id, update_alert, get_alert_statistics

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("")
async def list_alerts(
    risk_level: Optional[str] = None,
    status: Optional[str] = None,
    attack_type: Optional[str] = None,
    src_ip: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
):
    """List alerts with optional filters."""
    filters = AlertFilter(
        risk_level=RiskLevel(risk_level) if risk_level else None,
        status=AlertStatus(status) if status else None,
        attack_type=attack_type,
        src_ip=src_ip,
        limit=limit,
        skip=skip,
    )
    alerts = await get_alerts(filters)
    stats = await get_alert_statistics()
    return {"success": True, "data": alerts, "count": len(alerts), "statistics": stats}


@router.get("/{alert_id}")
async def get_alert(alert_id: str):
    """Get a single alert by ID."""
    alert = await get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"success": True, "data": alert}


@router.patch("/{alert_id}")
async def patch_alert(alert_id: str, update: AlertUpdate):
    """Update alert status (NEW → INVESTIGATING → RESOLVED/IGNORED)."""
    updated = await update_alert(alert_id, update)
    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"success": True, "data": updated}
